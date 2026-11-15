import json
for S in (0, 1):
    for r, name in ((f"e312_p_s{S}", "ends_card_snap_thr0.1.json"), (f"e316_p_s{S}", "ends_card_snap_thr0.1.json"), (f"e312_p_s{S}", "ends_card_thr0.1.json"), (f"e316_p_s{S}", "ends_card_thr0.1.json")):
        n = json.load(open(f"runs/{r}/{name}"))["numbers"]["head"]
        print(S, r, name, {k: round(n[k], 4) for k in ("topo_F1", "topo_P", "topo_R", "lanes_found", "junction_F1", "dup_frac", "F1@1.0", "vert100_dp")})
