# scratch: radius curve of E275 (thr 0.5) from the curve files and the 1.5 m card
import json
for s in (0, 1):
    cv = json.load(open("runs/e275/e273_p_s%d.curve.json" % s))["curve"]
    c15 = json.load(open("runs/e275/e273_p_s%d.close1.5_t0.5/card.json" % s))["p"]
    c0 = json.load(open("runs/e275/e273_p_s%d.plain_t0.5/card.json" % s))["p"]
    print("seed", s)
    def line(r, g, n):
        print("  R %-5s topo %.3f F1@1 %.3f junc %.3f lanes %.3f vdp %.1f nodeP %.3f nodeR %.3f" % (r, g["topo_F1"], g["F1@1.0"], g["junction_F1"], g["lanes_found"], g["vert100_dp"], n["node_P"], n["node_R"]))
    line("none", c0["graph"] | {"node_P": 0} if False else c0["graph"], c0["nodes"])
    for r in ("1", "1.25"): line(r, cv[r], cv[r])
    line("1.5", c15["graph"], c15["nodes"])
    for r in ("1.75", "2", "2.0"):
        if r in cv: line(r, cv[r], cv[r])
for n in ("e273_p_s0", "e273_p_s1", "e274_r_s0", "e274_r_s1"):
    a = json.load(open("temp/e275/analysis_%s.json" % n))
    print(n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in a.items() if k in ("dropped_share","dropped_counted_true_share","dropped_in_intersection_share","delivered_in_intersection_share","gt_nodes_with_gt_neighbour_within_R_share","true_nodes_lost_explained_by_gt_pair_share","node_recall_before","node_recall_after")})
