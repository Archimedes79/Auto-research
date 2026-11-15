import json, glob, os
cols = [("topo_F1","graph","topo_F1"),("F1@1.0","graph","F1@1.0"),("junc_F1","graph","junction_F1"),("lanes","graph","lanes_found"),("Q","graph","Q"),("dup","graph","dup_frac"),("vdp","graph","vert100_dp"),("nodeR","nodes","node_R"),("nodeP","nodes","node_P"),("lnkP","nodes","link_long_P"),("lnkR","nodes","link_long_R")]
print("run".ljust(30) + " ".join(c[0].rjust(7) for c in cols) + "  exact   err")
paths = sorted(glob.glob("runs/e278/*/card.json")) + sorted(glob.glob("runs/e273/e273_p_s?/card.json")) + sorted(glob.glob("runs/e275/e273_p_s?.close1.5_t0.5/card.json"))
for p in paths:
    c = json.load(open(p))
    v = [("%.3f" % c["p"][a][b]).rjust(7) if isinstance(c["p"].get(a, {}).get(b), (int, float)) else "      -" for n, a, b in cols]
    print(os.path.basename(os.path.dirname(p)).ljust(30) + " ".join(v) + "  %+.4f %+.4f" % (c["diff"]["exact"]["mean"], c["diff"]["error"]["mean"]))
