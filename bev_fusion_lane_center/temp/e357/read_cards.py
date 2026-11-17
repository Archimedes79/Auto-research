import json, os, sys
keys = ("topo_F1", "topo_P", "topo_R", "junction_F1", "lanes_found", "F1@1.0", "dup_frac", "vert100_dp", "link_junction_P", "link_junction_R", "link_junction_F1", "link_short_P", "link_short_R", "link_short_F1")
for p in sys.argv[1:]:
    if not os.path.exists(p):
        print(p, "missing"); continue
    m = json.load(open(p))["metrics"]
    print(os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p).replace("tok_card_", "").replace(".json", ""), {k: round(m[k], 4) for k in keys if k in m})
