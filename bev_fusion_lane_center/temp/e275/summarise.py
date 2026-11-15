# scratch: table of every E275 card (numbers read from card.json, nothing typed by hand)
import json, glob, os
rows = sorted(glob.glob("runs/e275/*/card.json"))
cols = [("topo_F1","graph","topo_F1"),("F1@1.0","graph","F1@1.0"),("junc_F1","graph","junction_F1"),("lanes","graph","lanes_found"),("Q","graph","Q"),
        ("dup","graph","dup_frac"),("vdp","graph","vert100_dp"),("nodeR","nodes","node_R"),("nodeP","nodes","node_P"),("lnkP","nodes","link_long_P"),("lnkR","nodes","link_long_R"),
        ("t12F1","tier","t12_F1_1"),("t1R","tier","t1_R_1"),("t3R","tier","t3_R_1")]
print("run".ljust(26) + " ".join(c[0].rjust(7) for c in cols) + "  exact(p-g) err(p-g) dropped  ms")
for p in rows:
    c = json.load(open(p))
    vals = []
    for n, a, b in cols:
        v = c["p"].get(a, {}).get(b)
        vals.append(("%.3f" % v if isinstance(v, (int, float)) else "-").rjust(7))
    ro = c["readout"]
    print(os.path.basename(os.path.dirname(p)).ljust(26) + " ".join(vals) + "  %+.4f   %+.4f  %.3f  %.2f" % (c["diff"]["exact"]["mean"], c["diff"]["error"]["mean"], ro.get("dropped_frac", 0.0), ro.get("ms_per_frame", 0.0)))
print()
for p in sorted(glob.glob("runs/e275/*.curve.json")):
    print(os.path.basename(p), json.dumps(json.load(open(p)))[:900])
