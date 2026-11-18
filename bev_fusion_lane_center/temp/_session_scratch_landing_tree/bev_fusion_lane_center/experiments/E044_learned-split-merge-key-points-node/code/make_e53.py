"""Bericht E53 (Schleife 4): Bezier-Verbinder zwischen Schluesselpunkten (SeqGrowGraph-Geometrie, SAM-Road-Knoten).
Liest out/loop2/e44_cons_p, e53_t0.10 ... e53_t0.70 (VAL), e53_f0.50 / e53_f0.60 (VAL + TEST) assemble.json, e46/path_eval_all.json; Text loop2/loop4_text_e53.py (TEXT53). Aufruf (gcar): make_e53.py [figs|report]"""
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
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e53.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop4_text_e53.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT53", {})


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main(mode):
    head = ["variant", "set", "F1@1m", "P / R @1m", "lanes found", "route rec / prec / F", "split F1 3 / 6 m", "split P / R @6m", "merge F1 3 / 6 m", "merge P / R @6m", "dir acc", "path F1"]
    rows = []
    paths = jl(f"{OUT}/e46/path_eval_all.json") or {}

    def add(lab, st, m, pk=None):
        pf = paths.get(f"{pk}_{st.lower()}", {}).get("path_f1") if pk else None
        rows.append([lab, st, f3(m["F1@1.0"]), f"{f3(m['P@1.0'])} / {f3(m['R@1.0'])}", f3(m["lanes_found"]), f"{f3(m['route_rec'])} / {f3(m['route_prec'])} / {f3(m['route_f'])}", f"{f3(m['split_F1'])} / {f3(m['split6_F1'])}", f"{f3(m['split6_P'])} / {f3(m['split6_R'])}", f"{f3(m['merge_F1'])} / {f3(m['merge6_F1'])}", f"{f3(m['merge6_P'])} / {f3(m['merge6_R'])}", f3(m.get("dir_acc")), f3(pf)])

    d0 = jl(f"{OUT}/e44_cons_p/assemble.json")
    if d0:
        for st in ("val", "test"):
            add("previous final system (no key points)", st.upper(), d0["base"][st], "e43/graphs_rule_e41_A")
        for st in ("val", "test"):
            add("DEFAULT: key-point nodes (conservative cut + prune)", st.upper(), d0[st], "e44_cons_p/graphs")
    sweep = []
    for th in ("0.10", "0.20", "0.30", "0.50", "0.60", "0.70"):
        d = jl(f"{OUT}/e53_f{th}/assemble.json") or jl(f"{OUT}/e53_t{th}/assemble.json")
        if not d:
            continue
        for st in ("val", "test"):
            if st in d:
                add(f"+ Bezier connectors between key points, heat >= {th}", st.upper(), d[st], f"e53_f{th}/graphs")
        if "val" in d:
            sweep.append((float(th), d["val"]))
    dm = jl(f"{OUT}/e44_cons_map/assemble.json")
    if dm:
        for st in ("val", "test"):
            add("MAP PRIOR (exact pose): previous final system", st.upper(), dm["base"][st], "e43/graphs_rule_e43_joint_noprop")
        for st in ("val", "test"):
            add("MAP PRIOR: DEFAULT key-point nodes", st.upper(), dm[st], "e44_cons_map/graphs")
        for th in ("0.50", "0.60"):
            d = jl(f"{OUT}/e53_map_f{th}/assemble.json")
            for st in ("val", "test"):
                if d and st in d:
                    add(f"MAP PRIOR: + Bezier connectors, heat >= {th}", st.upper(), d[st], f"e53_map_f{th}/graphs")
    dr = jl(f"{OUT}/e53_reg_cons/assemble.json")
    if dr:
        for st in ("val", "test"):
            add("REGISTERED PRIOR (1 m pose noise, correlation + learned fine stage, E48): previous final system", st.upper(), dr["base"][st])
        for st in ("val", "test"):
            add("REGISTERED PRIOR: DEFAULT key-point nodes", st.upper(), dr[st], "e53_reg_cons/graphs")
        d = jl(f"{OUT}/e53_reg_conn50/assemble.json")
        for st in ("val", "test"):
            if d and st in d:
                add("REGISTERED PRIOR: + Bezier connectors, heat >= 0.50", st.upper(), d[st], "e53_reg_conn50/graphs")
    images = []
    if sweep:
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
        xs = [x for x, _ in sweep]
        axs[0].plot(xs, [m["split6_F1"] for _, m in sweep], "-o", label="split F1 @6 m")
        axs[0].plot(xs, [m["merge6_F1"] for _, m in sweep], "-s", label="merge F1 @6 m")
        if d0:
            axs[0].axhline(d0["val"]["split6_F1"], color="C0", ls=":", label="default split")
            axs[0].axhline(d0["val"]["merge6_F1"], color="C1", ls=":", label="default merge")
        axs[0].set_xlabel("minimum mean heat along the connector")
        axs[0].set_title("node quality (VAL)", fontsize=9)
        axs[0].legend(fontsize=7)
        axs[0].grid(alpha=0.3)
        axs[1].plot(xs, [m["route_f"] for _, m in sweep], "-o", label="route F")
        axs[1].plot(xs, [m["F1@1.0"] for _, m in sweep], "-s", label="point F1@1 m")
        axs[1].plot(xs, [m["lanes_found"] for _, m in sweep], "-^", label="lanes found")
        if d0:
            axs[1].axhline(d0["val"]["route_f"], color="C0", ls=":")
        axs[1].set_xlabel("minimum mean heat along the connector")
        axs[1].set_title("graph quality (VAL)", fontsize=9)
        axs[1].legend(fontsize=7)
        axs[1].grid(alpha=0.3)
        fig.tight_layout()
        p = f"{FIG}/e53_sweep.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Effect of the heat threshold of the connector acceptance (VAL). Dotted: default graph without connectors."))
    try:
        import node_assemble as NA

        d0g = pickle.load(open(f"{OUT}/e44_cons_p/graphs_val.pkl", "rb"))
        d1g = pickle.load(open(f"{OUT}/e53_f0.50/graphs_val.pkl", "rb"))
        ids = d0g["ids"]
        gts = GD.load_gt_graphs("val", ids)
        prob = np.load(f"{OUT}/e44_nodes/prob_val.npy", mmap_mode="r")
        nid = {int(i): k for k, i in enumerate(np.load(f"{OUT}/e44_nodes/ids_val.npy"))}

        def stats(g, gt):
            a, b = GE.junctions(g), GE.junctions(gt)
            return GE._match_points(a[0], b[0], 6.0) + GE._match_points(a[1], b[1], 6.0), len(a[0]) + len(a[1])

        s0 = [stats(d0g["graphs"][k], gts[k]) for k in range(len(ids))]
        s1 = [stats(d1g["graphs"][k], gts[k]) for k in range(len(ids))]
        dh = np.array([s1[k][0] - s0[k][0] for k in range(len(ids))])
        df = np.array([(s1[k][1] - s1[k][0]) - (s0[k][1] - s0[k][0]) for k in range(len(ids))])
        order = [int(k) for k in np.argsort(-dh)[:2]] + [int(k) for k in np.argsort(-df)[:2]]
        fig, axs = plt.subplots(len(order), 3, figsize=(11.5, 3.7 * len(order)), squeeze=False)
        for r, k in enumerate(order):
            pk = NA.keypoints(np.asarray(prob[nid[int(ids[k])]], np.float32), 0.5)
            for c, (g, ttl) in enumerate(((d0g["graphs"][k], "default (no connectors)"), (d1g["graphs"][k], "+ Bezier connectors"), (None, "ground truth + key points"))):
                cnt = f" (hits {s0[k][0] if c == 0 else s1[k][0]}, false nodes {(s0[k][1] - s0[k][0]) if c == 0 else (s1[k][1] - s1[k][0])})" if g is not None else ""
                VG.panel(axs[r, c], g, gts[k], title=(ttl + " | " if r == 0 else "") + f"sample {int(ids[k])}{cnt}")
                if c == 2:
                    for P_, mk in ((pk[0], "x"), (pk[1], "+")):
                        if len(P_):
                            axs[r, c].scatter(P_[:, 0], P_[:, 1], marker=mk, color="#1f77b4", s=60, zorder=5)
        fig.suptitle("VAL: rows 1-2 = largest gain in node hits, rows 3-4 = largest increase of false nodes; red = graph, green = ground truth, blue x / + = split / merge key points", fontsize=8)
        fig.tight_layout()
        p = f"{FIG}/e53_examples.png"
        fig.savefig(p, dpi=90)
        plt.close(fig)
        images.append((p, "VAL examples (tau_h 0.5): two samples with the largest gain in correct split / merge nodes (rows 1-2) and the two with the largest increase of false nodes (rows 3-4); 'hits' = nodes within 6 m of a ground-truth node."))
    except Exception as ex:  # noqa: BLE001
        print("Beispielbild:", ex)
    if mode == "figs":
        for r in rows:
            print(r)
        return
    t = TEXT
    make_report2("E53", t.get("title", "E53"), t.get("question", ""), t.get("setup", []), [("Bezier connectors between split and merge key points added to the default graph", head, rows)], images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF4.synth_arch)
    print("E53 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
