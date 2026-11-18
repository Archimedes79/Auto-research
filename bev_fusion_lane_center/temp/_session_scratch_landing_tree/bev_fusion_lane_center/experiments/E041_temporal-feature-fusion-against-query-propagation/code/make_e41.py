"""Bericht E41: zeitliche Verfolgung, zwei Varianten: A = Fusion der BEV-Merkmale ueber Bilder (dichter Kopf mit gefaltetem Vorgaenger-Bild), B = Weitergabe der Anfragen (verfeinerte Linien des Vorgaengers als zusaetzliche Vorschlaege).
Aufruf (gcar): make_e41.py [figs|report]. Liest out/loop2/e41_A/metrics.json, e40_nomap_ft, sys_e41_A/metrics.json, sys_e37_final, e41_B*/tb_<lauf>_<menge>.json. Text: loop2/text_e37_e42.py TEXT['e41']."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs2 as AF2  # noqa: E402
import dense_report_lib as DR  # noqa: E402
import viz_graph as VG  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e37_e42.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e37_e42.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT", {})


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main(mode):
    # Tabelle 1: Variante A auf Rasterebene
    h1 = ["dense head", "TRAIN_EVAL F1@1m", "VAL F1@0.5m", "VAL F1@1m", "VAL F1@1.5m", "VAL P / R @1m", "TEST F1@1m", "TEST P / R @1m", "d VAL F1@1m"]
    r1 = []
    base = DR.load("e40_nomap_ft")
    fb = None
    for n, lab in (("e37_final", "E37 final head (reference)"), ("e40_nomap_ft", "single frame, same extra fine-tuning (control)"), ("e41_A", "A: + warped previous-frame features (BEV feature fusion)")):
        r = DR.load(n)
        if not r:
            continue
        bt = DR.best_tau(r)
        v, t = r["val"][bt], r["test"][bt]
        if n == "e40_nomap_ft":
            fb = v["F1@1.0"]
        d = f"{v['F1@1.0'] - fb:+.3f}" if fb is not None and n == "e41_A" else "-"
        r1.append([lab, f3(r["train_eval"][bt]["F1@1.0"]) if "train_eval" in r else "-", f3(v["F1@0.5"]), f3(v["F1@1.0"]), f3(v["F1@1.5"]), f"{f3(v['P@1.0'])} / {f3(v['R@1.0'])}", f3(t["F1@1.0"]), f"{f3(t['P@1.0'])} / {f3(t['R@1.0'])}", d])
    # Tabelle 2: Systemebene (Vorschlaege + Verfeinerer), beide Varianten + Einzelbild
    h2 = ["system (graph after the refiner)", "set", "F1@0.5m", "F1@1m", "P@1m", "R@1m", "split F1 6m", "merge F1 6m", "route F", "stability (F1 of warped previous vs current graph)", "flicker (1 - stability)"]
    r2 = []

    def add(lab, st, m, stab):
        r2.append([lab, st, f3(m["F1@0.5"]), f3(m["F1@1.0"]), f3(m["P@1.0"]), f3(m["R@1.0"]), f3(m["split6_F1"]), f3(m["merge6_F1"]), f3(m["route_f"]), f3(stab["pred"]["F1"]) if stab else "-", f3(1 - stab["pred"]["F1"]) if stab else "-"])

    gt_ref = {}
    for st in ("val", "test"):
        s0 = jl(f"{OUT}/sys_e37_final/metrics.json")
        if s0 and st in s0:
            add("single frame (E39 main)", st.upper(), s0[st]["refined"], s0[st]["stability"])
            gt_ref[st] = s0[st]["stability"]["gt_reference"]["F1"]
    for st in ("val", "test"):
        sA = jl(f"{OUT}/sys_e41_A/metrics.json")
        if sA and st in sA:
            add("A: feature fusion (dense) -> proposals -> refiner", st.upper(), sA[st]["refined"], sA[st]["stability"])
    labB = {"plain": "B: propagated pieces added as extra proposals (all)", "gate": "B': propagated pieces added only where the grid proposals do not already cover them", "gate_tau": "B'': like B' and a stricter existence threshold (0.5) for propagated pieces"}
    for st in ("val", "test"):
        tb = jl(f"{OUT}/e41_B/tb_e37_final_{st}.json")
        if not tb:
            continue
        for k in ("single", "plain", "gate", "gate_tau"):
            if k in tb and k != "single":
                add(labB[k], st.upper(), tb[k]["metrics"], tb[k]["stability"])
    for st in ("val", "test"):
        sB = jl(f"{OUT}/sys_e41_Btr/metrics.json")
        if sB and st in sB:
            add("B trained: refiner re-trained with the previous frame's pieces as extra proposals", st.upper(), sB[st]["refined"], sB[st]["stability"])
    r2.append(["ground-truth graphs (reference of the stability measure)", "VAL", "", "", "", "", "", "", "", f3(gt_ref.get("val")), f3(1 - gt_ref["val"]) if "val" in gt_ref else "-"])
    images = []
    # Abbildung: Balken Stabilitaet / F1
    rows = [r for r in r2 if r[1] == "VAL" and r[9] != "-" and r[0].startswith(("single", "A:", "B"))]
    if rows:
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
        names = [r[0][:34] for r in rows]
        for ax, col, ttl in ((axs[0], 3, "F1 @ 1 m (VAL)"), (axs[1], 8, "route F (VAL)"), (axs[2], 9, "temporal stability (VAL, higher = steadier)")):
            vals = [float(r[col]) for r in rows]
            ax.bar(range(len(rows)), vals, color=["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][: len(rows)])
            ax.set_xticks(range(len(rows)))
            ax.set_xticklabels([n[:16] for n in names], rotation=35, ha="right", fontsize=7)
            ax.set_ylim(max(min(vals) - 0.05, 0), min(max(vals) + 0.02, 1.0))
            ax.set_title(ttl, fontsize=9)
            ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = f"{FIG}/e41_bars.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Accuracy and stability of the temporal variants (VAL)."))
    # Abbildung: Einzelbild vs Variante A (Stichproben mit der groessten Aenderung des Punkt-F1), Vorbild der Vorhersage blau gestrichelt
    try:
        import pickle

        import graph_eval as GE
        import lf_inputs as LI

        d0 = pickle.load(open(f"{OUT}/e43/graphs_rule_e37_final_noprop_val.pkl", "rb"))
        d1p = f"{OUT}/e43/graphs_rule_e41_A_noprop_val.pkl"
        d1 = pickle.load(open(d1p if os.path.exists(d1p) else f"{OUT}/e43/graphs_rule_e41_A_val.pkl", "rb"))
        ids = d0["ids"]
        gts = DR.gt_graphs("val", ids)
        pose = LI.PoseTable("b123", 1)
        pos = {int(i): n for n, i in enumerate(ids)}

        def f1s(g, gt, n):
            r_ = GE.eval_sample(g, gt, n, with_route=False, directed=True)
            P, Rc = r_["cnt"][1, 0] / max(r_["cnt"][1, 1], 1), r_["cnt"][1, 2] / max(r_["cnt"][1, 3], 1)
            return 2 * P * Rc / max(P + Rc, 1e-9)

        diff = np.array([f1s(d1["graphs"][n], gts[n], n) - f1s(d0["graphs"][n], gts[n], n) for n in range(len(ids))])
        has_prev = np.array([pose.prev("val", int(i)) in pos for i in ids])
        order = [n for n in np.argsort(-np.abs(diff)) if has_prev[n]][:3]
        fig, axs = plt.subplots(len(order), 3, figsize=(11.5, 3.8 * len(order)), squeeze=False)
        for r, n in enumerate(order):
            j = pos[pose.prev("val", int(ids[n]))]
            Q, s_, _ = pose.rel("val", int(ids[n]))
            prev_w = [np.asarray(p_) @ Q.reshape(2, 2).T + s_ for p_ in d0["graphs"][j]["pts"]]
            VG.panel(axs[r, 0], d0["graphs"][n], gts[n], title=(("single frame | " if r == 0 else "") + f"sample {int(ids[n])}  (F1 change with A: {diff[n]:+.2f})"))
            for p_ in prev_w:
                axs[r, 0].plot(p_[:, 0], p_[:, 1], "--", color="#1f77b4", lw=1.0)
            VG.panel(axs[r, 1], d1["graphs"][n], gts[n], title="A: feature fusion across frames" if r == 0 else "")
            VG.panel(axs[r, 2], None, gts[n], title="ground truth" if r == 0 else "")
        fig.suptitle("VAL samples with the largest change in point F1 (red = prediction at t; blue dashed (left) = previous frame's single-frame prediction moved into the current frame)", fontsize=8)
        fig.tight_layout()
        p = f"{FIG}/e41_examples.png"
        fig.savefig(p, dpi=90)
        plt.close(fig)
        images.append((p, "Temporal examples (VAL): the three samples where variant A changes the point F1 most (positive or negative)."))
    except Exception as ex:  # noqa: BLE001
        print("Beispielbild:", ex)
    if mode == "figs":
        for r in r1 + r2:
            print(r)
        return
    t = TEXT.get("e41", {})
    make_report2("E41", t.get("title", "E41"), t.get("question", ""), t.get("setup", []),
                 [("Variant A at the grid: previous-frame features as extra input channels (dense head level)", h1, r1), ("System level: both variants and the single-frame system (refiner E39 fixed)", h2, r2)],
                 images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF2.temporal_grid_arch)
    print("E41 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
