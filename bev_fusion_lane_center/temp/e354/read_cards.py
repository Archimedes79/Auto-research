import json, os, sys
runs = sys.argv[1:]
for r in runs:
    for f in ("ends_card_thr0.1.json", "ends_card_snap_thr0.1.json"):
        p = f"runs/{r}/{f}"
        if not os.path.exists(p):
            print(r, f, "missing"); continue
        h = json.load(open(p))["numbers"]["head"]
        print(r, f.replace("ends_card_", "").replace(".json", ""), {k: round(h[k], 4) for k in ("topo_F1", "topo_P", "junction_F1", "lanes_found", "dup_frac", "F1@1.0", "vert100_dp")})
