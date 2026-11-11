# scratch: table of the E276 scoring cards (numbers read from card.json): B (c) against arm t / s, per seed
import json, glob, os, sys
rows = sorted(glob.glob("runs/e276/*/card.json"))
cols = [("topo_F1","graph","topo_F1"),("F1@1.0","graph","F1@1.0"),("junc_F1","graph","junction_F1"),("lanes","graph","lanes_found"),("Q","graph","Q"),("vdp","graph","vert100_dp"),
        ("n/100m","nodes","nodes_per_100m"),("nodeR","nodes","node_R"),("nodeP","nodes","node_P"),("t12F1","tier","t12_F1_1"),("t1R","tier","t1_R_1"),("t2R","tier","t2_R_1"),("t3R","tier","t3_R_1"),
        ("z1P","tier","z1_P_1"),("z3P","tier","z3_P_1")]
print("run".ljust(14) + " ".join(c[0].rjust(7) for c in cols) + "  z1nodeFalse z3nodeFalse  exact  err")
for p in rows:
    c = json.load(open(p))
    vals = []
    for n, a, b in cols:
        v = c["p"].get(a, {}).get(b)
        vals.append(("%.3f" % v if isinstance(v, (int, float)) else "-").rjust(7))
    t = c["p"].get("tier", {})
    zf = lambda k: (t.get(f"z{k}_node_false", 0) / t.get(f"z{k}_node_n", 1)) if t.get(f"z{k}_node_n") else float("nan")
    print(os.path.basename(os.path.dirname(p)).ljust(14) + " ".join(vals) + "  %.3f %.3f  %+.4f %+.4f" % (zf(1), zf(3), c["diff"]["exact"]["mean"], c["diff"]["error"]["mean"]))
