import json, glob, sys
ks = ("topo_F1", "topo_P", "topo_R", "F1@1.0", "lanes_found", "junction_F1", "dup_frac", "vert100_dp")
for r in sys.argv[1:]:
    for f in sorted(glob.glob(f"runs/{r}/token_card_rec*.json")):
        d = json.load(open(f)); m = d["metrics"]; g = m.get("graph", m); info = d.get("info", {})
        print(r, f.split("/")[-1][11:-5], {k: round(g[k], 4) for k in ks if k in g})
        pv = info.get("paired_vs_plain_b")
        if pv and "topo_F1" in pv: print("      vs plain B", {k: round(v, 4) for k, v in pv["topo_F1"].items() if k in ("diff", "lo", "hi")})
