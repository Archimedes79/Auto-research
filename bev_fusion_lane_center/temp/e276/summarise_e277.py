import json, glob, os
cols = [("topo_F1","graph","topo_F1"),("F1@1.0","graph","F1@1.0"),("junc_F1","graph","junction_F1"),("lanes","graph","lanes_found"),("Q","graph","Q"),("dup","graph","dup_frac"),("vdp","graph","vert100_dp"),("nodeR","nodes","node_R"),("nodeP","nodes","node_P"),("lnkP","nodes","link_long_P"),("lnkR","nodes","link_long_R"),("n/100m","nodes","nodes_per_100m")]
print("run".ljust(30) + " ".join(c[0].rjust(7) for c in cols) + "  exact   err   ms")
paths = []
for s in (0, 1):
    paths += [f"runs/e273/e273_p_s{s}/card.json", f"runs/e275/e273_p_s{s}.close1.5_t0.5/card.json", f"runs/e277/e277_u_s{s}.plain_t0.5/card.json", f"runs/e277/e277_u_s{s}.close1.5_t0.5/card.json"]
for p in paths:
    if not os.path.exists(p): print(p, "pending"); continue
    c = json.load(open(p))
    v = [("%.3f" % c["p"][a][b]).rjust(7) if isinstance(c["p"].get(a, {}).get(b), (int, float)) else "      -" for n, a, b in cols]
    print(p.split("/")[-2].ljust(30) + " ".join(v) + "  %+.4f %+.4f  %s" % (c["diff"]["exact"]["mean"], c["diff"]["error"]["mean"], c.get("readout", {}).get("ms_per_frame", "")))
