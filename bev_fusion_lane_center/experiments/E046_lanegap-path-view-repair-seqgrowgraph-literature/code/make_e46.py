"""Bericht E46 (Schleife 4): LaneGAP-Sicht - Pfade statt Stuecke: Pfad-Ausbeute / -Praezision unserer Graphen, innere Sackgassen, Wirkung der Knoten-gefuehrten Konfigurationen.
Liest out/loop2/e46/path_eval_*.json (path_eval.py); Text: loop2/loop4_text_e46.py (TEXT46).  Aufruf (gcar): make_e46.py [figs|report]"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e46.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e46.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT46", {})

LAB = {
    "e43/graphs_rule_e37_final_noprop": "single frame: refiner + rule links (S2)",
    "e43/graphs_rule_e41_A": "final (temporal A + refiner + rule links)",
    "e46_repair/graphs": "final + continuity repair (dead ends joined within 12 m)",
    "e46_repair2/graphs": "final + looser continuity repair (20 m, heading 66 deg)",
    "e44_cons_p/graphs": "final + key-point nodes: conservative cut + prune (E44 default)",
    "e44_cons/graphs": "final + key-point nodes: conservative cut, no prune",
    "e44_cut/graphs": "final + key-point nodes: loose cut + prune",
    "e44_cut_np/graphs": "final + key-point nodes: loose cut, no prune",
    "e44_balanced/graphs": "final + key-point nodes: snap + create + prune, balanced",
    "e44_assemble/graphs": "final + key-point nodes: snap, node priority",
    "e53_f0.50/graphs": "final + key-point nodes (default) + Bezier connectors (E53, heat >= 0.5)",
    "e44_cut_oracle/graphs": "upper bound: GROUND-TRUTH nodes, loose cut + prune",
    "e51/graphs_rule_e51_A_nodes_noprop": "E51 node-conditioned dense head (no key-point insertion)",
    "e51_cons/graphs": "E51 + key-point nodes (default)",
    "e43/graphs_rule_e43_joint_noprop": "final with map prior (exact pose; joint head)",
    "e44_cons_map/graphs": "final with map prior + key-point nodes (conservative cut + prune)",
    "e44_cut_map/graphs": "final with map prior + key-point nodes (loose cut + prune)",
}


def main(mode):
    res = {}
    res.update(json.load(open(f"{OUT}/e46/path_eval_all.json")))
    head = ["graph", "set", "path recall", "path precision", "path F1", "paths / sample (GT | pred)", "interior dead ends / sample (GT | pred)", "pieces / sample (GT | pred)"]
    rows = []
    for key, lab in LAB.items():
        for st in ("val", "test"):
            r = res.get(f"{key}_{st}")
            if r:
                rows.append([lab, st.upper(), f3(r["path_recall"]), f3(r["path_precision"]), f3(r["path_f1"]), f"{r['paths_gt']:.1f} | {r['paths_pred']:.1f}", f"{r['dead_ends_gt']:.2f} | {r['dead_ends_pred']:.2f}", f"{r['pieces_gt']:.1f} | {r['pieces_pred']:.1f}"])
    images = []
    keys = [k for k in LAB if f"{k}_val" in res]
    if keys:
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
        w = 0.38
        for ax, st in ((axs[0], "val"), (axs[1], "test")):
            x = np.arange(len(keys))
            ax.bar(x - w / 2, [res[f"{k}_{st}"]["path_recall"] if f"{k}_{st}" in res else 0 for k in keys], w, label="path recall", color="#1f77b4")
            ax.bar(x + w / 2, [res[f"{k}_{st}"]["path_precision"] if f"{k}_{st}" in res else 0 for k in keys], w, label="path precision", color="#ff7f0e")
            ax.set_xticks(x)
            ax.set_xticklabels([LAB[k].replace("final + key-point nodes: ", "nodes: ")[:34] for k in keys], rotation=40, ha="right", fontsize=6)
            ax.set_ylim(0, 0.9)
            ax.set_title(f"path-level recall / precision ({st.upper()})", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        p = f"{FIG}/e46_paths.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Path-level recall and precision of the graph variants (a ground-truth path counts as found if >= 90 % of its points lie within 1.5 m of ONE predicted path with the right direction)."))
    if mode == "figs":
        for r in rows:
            print(r)
        return
    t = TEXT
    make_report2("E46", t.get("title", "E46"), t.get("question", ""), t.get("setup", []), [("Path-level quality of the lane graphs (LaneGAP's view)", head, rows)], images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=None)
    print("E46 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
