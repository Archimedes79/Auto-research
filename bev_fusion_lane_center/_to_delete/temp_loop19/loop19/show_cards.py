import json, sys
for n in sys.argv[1:]:
    d = json.load(open(f"runs/eval/{n}/val.json", encoding="utf-8-sig"))
    m = d.get("metrics", d)
    print(n, {k: round(m[k], 4) for k in ("Q", "lanes_found", "R@1_far", "topo_F1", "vert100", "F1@1.0", "false_len_frac") if k in m})
