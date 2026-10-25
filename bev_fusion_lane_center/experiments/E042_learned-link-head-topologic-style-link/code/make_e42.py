"""Bericht E42: gelernter Verbindungskopf (explizite Abzweige / Zusammenfuehrungen). Aufruf (gcar): make_e42.py [figs|report]
Liest out/loop2/e38_link/{link_eval.json}, stageB.out (Trainingskennzahlen), e43/graphs_rule_e37_final_<menge>.pkl und e43/graphs_e37_final_<menge>.pkl (final_system.py --dense e37_final --no_prop --out e43). Text: loop2/text_e37_e42.py TEXT['e42']."""
import json
import os
import pickle
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs2 as AF2  # noqa: E402
import dense_report_lib as DR  # noqa: E402
import graph_eval as GE  # noqa: E402
import viz_graph as VG  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e37_e42.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e37_e42.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT", {})
COLS = [("F1@1.0", "F1@1m"), ("route_rec", "route rec"), ("route_prec", "route prec"), ("route_f", "route F"), ("split_F1", "split F1 3m"), ("split6_F1", "split F1 6m"), ("merge_F1", "merge F1 3m"), ("merge6_F1", "merge F1 6m"), ("dir_acc", "dir acc")]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main(mode):
    lk = jl(f"{OUT}/e38_link/link_eval.json") or {}
    head = ["links", "set"] + [c[1] for c in COLS] + ["links / sample (pred)"]
    rows = []
    for st, tag in (("VAL", "refined_val"), ("TEST", "refined_test")):
        if tag not in lk:
            continue
        r = lk[tag]
        if "rule" in r:
            rows.append(["rule: inherited grid edges + distance / heading links", st] + [f3(r["rule"].get(k)) for k, _ in COLS] + ["-"])
        for tau in ("0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"):
            if tau in r:
                rows.append([f"learned link head, tau_l {tau}", st] + [f3(r[tau].get(k)) for k, _ in COLS] + ["-"])
    # Trainingskennzahlen aus dem Protokoll
    h2 = ["item", "value"]
    r2 = []
    so = open(f"{OUT}/stageB.out", encoding="utf-8", errors="ignore").read() if os.path.exists(f"{OUT}/stageB.out") else ""
    for m in re.finditer(r"(refined_oof\d): (\d+) Samples, (\d+) Kandidaten, ([\d.]+) Anteil wahr", so):
        r2.append([f"training candidates from {m.group(1)}", f"{m.group(2)} samples, {m.group(3)} candidate links, {float(m.group(4)):.1%} true"])
    for m in re.finditer(r"Training: tau ([\d.]+): Praezision ([\d.]+) Ausbeute ([\d.]+)", so):
        r2.append([f"training-set precision / recall at tau_l {m.group(1)}", f"{m.group(2)} / {m.group(3)}"])
    images = []
    # Beispiele
    try:
        d0 = pickle.load(open(f"{OUT}/e43/graphs_rule_e37_final_noprop_val.pkl", "rb"))
        d1 = pickle.load(open(f"{OUT}/e43/graphs_e37_final_noprop_val.pkl", "rb"))
        ids = d0["ids"]
        gts = DR.gt_graphs("val", ids)
        # Samples mit Abzweigen / Zusammenfuehrungen im Soll und deutlichem Unterschied der Graphen
        cand = []
        for n in range(len(ids)):
            g = gts[n]
            o = np.zeros(len(g["pts"]), int)
            i_ = np.zeros(len(g["pts"]), int)
            for a, b in g["edges"]:
                o[a] += 1
                i_[b] += 1
            if (o >= 2).sum() + (i_ >= 2).sum() >= 2:
                cand.append(n)
        sel = [cand[int(q)] for q in np.linspace(5, len(cand) - 6, 4)] if len(cand) > 10 else cand[:4]
        fig, axs = plt.subplots(len(sel), 3, figsize=(11.5, 3.8 * len(sel)), squeeze=False)
        for r, n in enumerate(sel):
            VG.panel(axs[r, 0], d0["graphs"][n], gts[n], title="rule links" if r == 0 else "")
            VG.panel(axs[r, 1], d1["graphs"][n], gts[n], title="learned link head" if r == 0 else "")
            VG.panel(axs[r, 2], None, gts[n], title=f"ground truth - sample {int(ids[n])}" if r == 0 else f"sample {int(ids[n])}")
        fig.suptitle("VAL samples with junctions: red = prediction (arrows = direction, triangles = splits, diamonds = merges), green = ground truth", fontsize=9)
        fig.tight_layout()
        p = f"{FIG}/e42_examples.png"
        fig.savefig(p, dpi=90)
        plt.close(fig)
        images.append((p, "Rule links vs learned link head on VAL samples that contain splits and merges."))
    except Exception as ex:  # noqa: BLE001
        print("Beispielbild:", ex)
    if lk.get("refined_val"):
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
        taus = [t for t in ("0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7") if t in lk["refined_val"]]
        for ax, keys, ttl in ((axs[0], ("route_f", "F1@1.0"), "route F and point F1 (VAL)"), (axs[1], ("split6_F1", "merge6_F1"), "split / merge F1 at 6 m (VAL)")):
            for k in keys:
                ax.plot([float(t) for t in taus], [lk["refined_val"][t][k] for t in taus], "-o", label=k)
                if "rule" in lk["refined_val"]:
                    ax.axhline(lk["refined_val"]["rule"][k], ls="--", color="gray", lw=0.8)
            ax.set_xlabel("link threshold tau_l")
            ax.set_title(ttl + " (dashed = rule)", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        p = f"{FIG}/e42_tau.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Operating point of the link head."))
    if mode == "figs":
        for r in rows + r2:
            print(r)
        return
    t = TEXT.get("e42", {})
    make_report2("E42", t.get("title", "E42"), t.get("question", ""), t.get("setup", []),
                 [("Rule links vs learned link head on the refined pieces of E39 (final dense head)", head, rows), ("Training data and fit of the link head", h2, r2)],
                 images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF2.link_arch)
    print("E42 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
