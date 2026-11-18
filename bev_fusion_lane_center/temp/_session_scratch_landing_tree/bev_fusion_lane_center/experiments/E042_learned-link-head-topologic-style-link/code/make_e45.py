"""Bericht E45 (Schleife 4): TopoLogic-artiger Verbindungskopf. Liest out/loop2/e45_link/link_eval2.json, e38_link/link_eval.json (E42), Text loop2/loop4_text_e45.py (TEXT45). Aufruf (gcar): make_e45.py [report]"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs4 as AF4  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e45.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e45.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT45", {})


def main(mode):
    d2 = json.load(open(f"{OUT}/e45_link/link_eval2.json"))
    d1 = json.load(open(f"{OUT}/e38_link/link_eval.json"))
    head = ["link decision", "set", "F1@1m", "route rec", "route prec", "route F", "split F1 3 / 6 m", "merge F1 3 / 6 m", "dir acc"]
    rows = []

    def add(lab, st, m):
        rows.append([lab, st, f3(m["F1@1.0"]), f3(m["route_rec"]), f3(m["route_prec"]), f3(m["route_f"]), f"{f3(m['split_F1'])} / {f3(m['split6_F1'])}", f"{f3(m['merge_F1'])} / {f3(m['merge6_F1'])}", f3(m.get("dir_acc"))])

    for st, tag, lab in (("VAL", "refined_val", "val"), ("TEST", "refined_test", "test")):
        add("rule links (inherited grid edges + distance / heading)", st, d2[tag]["rule"])
        for tau in ("0.05", "0.3", "0.5"):
            if tau in d1.get(tag, {}):
                add(f"E42 MLP link head, tau_l {tau}", st, d1[tag][tau])
        for tau in ("0.05", "0.1", "0.3", "0.5"):
            add(f"E45 geometry x semantic link head, tau_l {tau}", st, d2[tag][tau])
    images = []
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    taus = ["0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]
    for ax, key, ttl in ((axs[0], "route_f", "route F (VAL)"), (axs[1], "split6_F1", "split F1 @ 6 m (VAL)")):
        ax.plot([float(t) for t in taus], [d2["refined_val"][t][key] for t in taus], "-o", label="E45 geometry x semantic")
        ax.plot([float(t) for t in taus if t in d1["refined_val"]], [d1["refined_val"][t][key] for t in taus if t in d1["refined_val"]], "-s", label="E42 MLP")
        ax.axhline(d2["refined_val"]["rule"][key], color="k", ls=":", label="rule links")
        ax.set_xlabel("link threshold tau_l")
        ax.set_title(ttl, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = f"{FIG}/e45_links.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    images.append((p, "Link head operating point on VAL: the TopoLogic-style head follows the E42 head and the rule."))
    if mode == "figs":
        for r in rows:
            print(r)
        return
    t = TEXT
    make_report2("E45", t.get("title", "E45"), t.get("question", ""), t.get("setup", []), [("Link decisions on the refined pieces of the single-frame system (E39); thresholds on VAL", head, rows)], images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF4.link2_arch)
    print("E45 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "report")
