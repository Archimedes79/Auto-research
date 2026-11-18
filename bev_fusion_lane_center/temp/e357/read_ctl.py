import json, os, sys
keys = ("topo_F1", "topo_P", "topo_R", "junction_F1", "lanes_found", "F1@1.0", "dup_frac", "vert100_dp", "link_long_P", "link_long_R", "link_long_F1", "link_junction_P", "link_junction_R", "link_junction_F1", "link_short_P", "link_short_R", "link_lane_long_n_gt")
for n in sys.argv[1:]:
    p = f"runs/eval/{n}/val.json"
    if not os.path.exists(p):
        print(n, "missing"); continue
    m = json.load(open(p))["metrics"]
    print(n, {k: round(m[k], 4) for k in keys if k in m})
