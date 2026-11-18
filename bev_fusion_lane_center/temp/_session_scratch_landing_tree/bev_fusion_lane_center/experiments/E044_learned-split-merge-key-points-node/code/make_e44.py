"""Bericht E44 (Schleife 4): gelernte Abzweig-Schluesselpunkte (SAM-Road-artig) und Knoten-gefuehrter Graphaufbau (Verschieben / geometrieerhaltender Schnitt).
Liest out/loop2/e44_nodes/metrics.json, e44_assemble (Verschieben aggressiv), e44_balanced, e44_cut, e44_cut_np, e44_cut_map, e44_oracle, e44_cut_oracle (assemble.json, graphs_*.pkl), e44_assemble/diag.json.
Text: loop2/loop4_text_e44.py (TEXT44).  Aufruf (gcar): make_e44.py [figs|report]"""
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs4 as AF4  # noqa: E402
import graph_eval as GE  # noqa: E402
import graph_from_dense as GD  # noqa: E402
import viz_graph as VG  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e44.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e44.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT44", {})


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def row(lab, st, m):
    return [lab, st, f3(m["F1@1.0"]), f3(m["route_f"]), f"{f3(m['split_F1'])} / {f3(m['split6_F1'])}", f"{f3(m['split6_P'])} / {f3(m['split6_R'])}", f"{f3(m['merge_F1'])} / {f3(m['merge6_F1'])}", f"{f3(m['merge6_P'])} / {f3(m['merge6_R'])}", f3(m.get("dir_acc"))]


def main(mode):
    nm = jl(f"{OUT}/e44_nodes/metrics.json")
    A = {k: jl(f"{OUT}/{k}/assemble.json") for k in ("e44_assemble", "e44_balanced", "e44_cut", "e44_cut_np", "e44_cut_map", "e44_oracle", "e44_cut_oracle", "e44_cons", "e44_cons_p", "e44_cons_map")}
    diag = jl(f"{OUT}/e44_assemble/diag.json")
    h1 = ["source of the split / merge nodes", "set", "split P", "split R", "split F1 @3 m", "split F1 @6 m", "merge P", "merge R", "merge F1 @3 m", "merge F1 @6 m"]
    rows1, rows2, rows3, rows4 = [], [], [], []
    base = A["e44_cut"]["base"] if A["e44_cut"] else None
    if base:
        for st, lab in (("val", "VAL"), ("test", "TEST")):
            m = base[st]
            rows1.append(["nodes derived from the final graph (A + refiner + rule links)", lab, f3(m["split6_P"]), f3(m["split6_R"]), f3(m["split_F1"]), f3(m["split6_F1"]), f3(m["merge6_P"]), f3(m["merge6_R"]), f3(m["merge_F1"]), f3(m["merge6_F1"])])
    if nm:
        bt = max(nm["val"].items(), key=lambda kv: kv[1]["6.0"]["split"]["F1"] + kv[1]["6.0"]["merge"]["F1"])[0]
        for st, lab in (("train_eval", "TRAIN_EVAL (seen)"), ("val", "VAL"), ("test", "TEST")):
            if st in nm:
                r = nm[st][bt]
                rows1.append([f"learned key-point head (tau {bt}, chosen on VAL)", lab, f3(r["6.0"]["split"]["P"]), f3(r["6.0"]["split"]["R"]), f3(r["3.0"]["split"]["F1"]), f3(r["6.0"]["split"]["F1"]), f3(r["6.0"]["merge"]["P"]), f3(r["6.0"]["merge"]["R"]), f3(r["3.0"]["merge"]["F1"]), f3(r["6.0"]["merge"]["F1"])])
        for tau, r in nm["val"].items():
            rows2.append([tau, f3(r["6.0"]["split"]["P"]), f3(r["6.0"]["split"]["R"]), f3(r["6.0"]["split"]["F1"]), f"{r['6.0']['split']['n_pred'] / 873:.2f}", f3(r["6.0"]["merge"]["P"]), f3(r["6.0"]["merge"]["R"]), f3(r["6.0"]["merge"]["F1"]), f"{r['6.0']['merge']['n_pred'] / 873:.2f}"])
    for st, lab in (("val", "VAL"), ("test", "TEST")):
        for key, name in (("e44_cons_p", "nodes of the conservative cut graph + prune (default)"), ("e44_cut", "nodes of the loose cut graph + prune (node priority)"), ("e44_assemble", "nodes of the snap-based graph, node-priority configuration")):
            if A[key] and st in A[key]:
                m = A[key][st]
                rows1.append([name, lab, f3(m["split6_P"]), f3(m["split6_R"]), f3(m["split_F1"]), f3(m["split6_F1"]), f3(m["merge6_P"]), f3(m["merge6_R"]), f3(m["merge_F1"]), f3(m["merge6_F1"])])
    rows1.append(["ceiling: ground-truth field through the rule-based assembler (E38)", "VAL", "-", "-", "0.236", "0.340", "-", "-", "0.242", "0.450"])
    rows1.append(["ceiling: ground-truth field through the rule-based assembler (E38)", "TEST", "-", "-", "0.249", "0.368", "-", "-", "0.271", "0.449"])
    h3 = ["variant", "set", "F1@1m", "route F", "split F1 3 / 6 m", "split P / R @6m", "merge F1 3 / 6 m", "merge P / R @6m", "dir acc"]
    pareto = []
    if base:
        for st, lab in (("val", "VAL"), ("test", "TEST")):
            rows3.append(row("final graph without key points (temporal A + refiner + rule links)", lab, base[st]))
        pareto.append(("no key points", base["val"]))
    for key, name in (("e44_cons", "key points, CONSERVATIVE cut (branch search 8 m, heading >= 0.5), no pruning"), ("e44_cons_p", "key points, CONSERVATIVE cut + prune (default)"), ("e44_cut_np", "key points, LOOSE cut (branch search 16 m, any heading), no pruning"), ("e44_cut", "key points, LOOSE cut + prune (node priority; best sum on VAL)"), ("e44_balanced", "key points, SNAP + create + prune, balanced"), ("e44_assemble", "key points, SNAP + create + prune, node priority (chosen on VAL by the sum of four metrics)"), ("e44_cut_oracle", "UPPER BOUND: ground-truth nodes instead of key points, CUT configuration"), ("e44_oracle", "UPPER BOUND: ground-truth nodes, SNAP node-priority configuration")):
        if A[key]:
            for st, lab in (("val", "VAL"), ("test", "TEST")):
                if st in A[key]:
                    rows3.append(row(name, lab, A[key][st]))
            if "val" in A[key]:
                pareto.append(({"e44_cons": "conservative cut", "e44_cons_p": "conservative cut + prune", "e44_cut_np": "loose cut, no prune", "e44_cut": "loose cut + prune", "e44_balanced": "snap, balanced", "e44_assemble": "snap, node priority", "e44_cut_oracle": "upper bound: GT nodes, cut", "e44_oracle": "upper bound: GT nodes, snap"}[key], A[key]["val"]))
    if A["e44_cut_map"]:
        for st, lab in (("val", "VAL"), ("test", "TEST")):
            rows3.append(row("WITH MAP PRIOR (exact pose, joint head): final graph without key points", lab, A["e44_cut_map"]["base"][st]))
        for st, lab in (("val", "VAL"), ("test", "TEST")):
            if A["e44_cons_map"] and st in A["e44_cons_map"]:
                rows3.append(row("WITH MAP PRIOR: + key points, CONSERVATIVE cut + prune (default)", lab, A["e44_cons_map"][st]))
        for st, lab in (("val", "VAL"), ("test", "TEST")):
            rows3.append(row("WITH MAP PRIOR: + key points, LOOSE cut + prune (node priority)", lab, A["e44_cut_map"][st]))
    if diag:
        for kind in ("split", "merge"):
            for key, c in diag.items():
                k_, rb_, cm_ = key.split("|")
                if k_ == kind:
                    rows4.append([kind, f"{float(rb_):.0f} m, cos >= {float(cm_):.1f}", f"{c['detected']} / {c['total']}", f"{100 * c['D'] / max(c['detected'], 1):.0f} %", f"{100 * c['C'] / max(c['detected'], 1):.0f} %", f"{100 * c['B'] / max(c['detected'], 1):.0f} %", f"{100 * c['A'] / max(c['detected'], 1):.0f} %"])
    images = []
    if nm:
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
        for ax, nme in ((axs[0], "split"), (axs[1], "merge")):
            for st, c in (("val", "#1f77b4"), ("test", "#ff7f0e"), ("train_eval", "#2ca02c")):
                if st not in nm:
                    continue
                ts = sorted(nm[st], key=float)
                ax.plot([float(t) for t in ts], [nm[st][t]["6.0"][nme]["F1"] for t in ts], "-o", color=c, label=f"{st} F1@6m")
                ax.plot([float(t) for t in ts], [nm[st][t]["3.0"][nme]["F1"] for t in ts], "--", color=c, alpha=0.6, label=f"{st} F1@3m")
            if base:
                ax.axhline(base["val"][f"{nme}6_F1"], color="#d62728", ls=":", label="graph-derived (VAL, 6 m)")
            ax.set_xlabel("key-point threshold tau")
            ax.set_ylabel("location F1")
            ax.set_title(f"{nme} nodes: learned key points vs threshold", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        p = f"{FIG}/e44_f1_vs_tau.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Location F1 of the learned split / merge key points against the detection threshold (solid: 6 m tolerance, dashed: 3 m); dotted red: nodes derived from the final graph."))
    if pareto:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.2))
        for lab, m in pareto:
            ax.scatter(m["route_f"], 0.5 * (m["split6_F1"] + m["merge6_F1"]), s=45)
            ax.annotate(lab, (m["route_f"], 0.5 * (m["split6_F1"] + m["merge6_F1"])), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("route F (VAL)")
        ax.set_ylabel("mean of split and merge F1 @ 6 m (VAL)")
        ax.set_title("node quality vs connectivity of the graph", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = f"{FIG}/e44_pareto.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Trade-off between the quality of the split / merge nodes and the route F of the graph for the node-guided variants (VAL)."))
    try:
        import node_assemble as NA

        d0 = pickle.load(open(f"{OUT}/e43/graphs_rule_e41_A_val.pkl", "rb"))
        d1 = pickle.load(open(f"{OUT}/e44_cut/graphs_val.pkl", "rb"))
        ids = d0["ids"]
        gts = GD.load_gt_graphs("val", ids)
        prob = np.load(f"{OUT}/e44_nodes/prob_val.npy", mmap_mode="r")
        nid = {int(i): k for k, i in enumerate(np.load(f"{OUT}/e44_nodes/ids_val.npy"))}

        def hits(g, gt):
            a, b = GE.junctions(g), GE.junctions(gt)
            return GE._match_points(a[0], b[0], 6.0) + GE._match_points(a[1], b[1], 6.0)

        diff = np.array([hits(d1["graphs"][k], gts[k]) - hits(d0["graphs"][k], gts[k]) for k in range(len(ids))])
        order = list(np.argsort(-diff))[:3] + [int(len(ids) * 0.3)]
        fig, axs = plt.subplots(len(order), 3, figsize=(11.5, 3.7 * len(order)), squeeze=False)
        for r, k in enumerate(order):
            pk = NA.keypoints(np.asarray(prob[nid[int(ids[k])]], np.float32), 0.5)
            for c, (g, ttl) in enumerate(((d0["graphs"][k], "final graph (before)"), (d1["graphs"][k], "cut-based node insertion (after)"), (None, "ground truth + key points"))):
                VG.panel(axs[r, c], g, gts[k], title=(ttl + " | " if r == 0 else "") + f"sample {int(ids[k])} (node hits {hits(g, gts[k]) if g is not None else '-'})")
                if c == 2:
                    for P_, mk in ((pk[0], "x"), (pk[1], "+")):
                        if len(P_):
                            axs[r, c].scatter(P_[:, 0], P_[:, 1], marker=mk, color="#1f77b4", s=60, zorder=5)
        fig.suptitle("VAL: red = graph (triangles = splits, diamonds = merges), green = ground truth; blue x / + = learned split / merge key points", fontsize=8)
        fig.tight_layout()
        p = f"{FIG}/e44_examples.png"
        fig.savefig(p, dpi=90)
        plt.close(fig)
        images.append((p, "VAL examples: the three samples with the largest gain in node hits and one typical sample; graph before / after the cut-based node insertion; ground truth with the learned key points."))
    except Exception as ex:  # noqa: BLE001
        print("Beispielbild:", ex)
    if mode == "figs":
        for r in rows1 + rows3:
            print(r)
        return
    t = TEXT
    tabs = [("Split / merge node quality: learned key points vs nodes derived from the graph", h1, rows1), ("Key-point head: threshold sweep on VAL (6 m tolerance)", ["tau", "split P", "split R", "split F1", "pred / sample", "merge P", "merge R", "merge F1", "pred / sample"], rows2), ("Node-guided assembly on the final graph: variants, upper bounds and the system with map prior", h3, rows3)]
    if rows4:
        tabs.append(("Why the graph cannot simply adopt every key point: for ground-truth nodes with a matching key point (6 m), what the predicted graph contains nearby (VAL; r_b = branch search radius, cos = heading limit)", ["node type", "branch criterion (r_b m, cos)", "key point found / GT nodes", "node already in graph", "creatable from existing pieces", "branch piece missing", "lane not covered"], rows4))
    make_report2("E44", t.get("title", "E44"), t.get("question", ""), t.get("setup", []), tabs, images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF4.node_arch)
    print("E44 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
