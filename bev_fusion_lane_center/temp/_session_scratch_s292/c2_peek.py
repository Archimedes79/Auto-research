import json, os
R = os.environ["BFLC_ROOT"] + "/runs/"
K = ("topo_F1", "dup_frac", "F1@1.0", "lanes_found", "junction_F1", "vert100_dp")
for s in (0, 1):
    for nm in ("ends_card_snap_thr0.1.json", "ends_card_c2_snap_thr0.1.json"):
        p = f"{R}e312_p_s{s}/{nm}"
        if os.path.exists(p):
            h = json.load(open(p))["numbers"]["head"]; print(s, nm[:24].ljust(24), {k: round(h[k], 4) for k in K})
