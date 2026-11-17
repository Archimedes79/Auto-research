import json
K = ("topo_F1", "F1@1.0", "dup_frac", "vert100_dp", "junction_F1", "lanes_found")
def card(run, tag):
    return json.load(open(f"runs/{run}/ends_card_{tag}thr0.1.json"))["numbers"]["head"]
for tagname, tag in (("snapped", "snap_"), ("snap-nogeom", "snap_nogeom_"), ("unsnapped", "")):
    for S in (0, 1):
        u, g = card(f"e318_u_s{S}", tag), card(f"e317_g_s{S}", tag)
        print(f"{tagname:12s} s{S}  u topo {u['topo_F1']:.4f} g {g['topo_F1']:.4f} | " + " ".join(f"{k} {u[k]-g[k]:+.4f}" for k in K))
