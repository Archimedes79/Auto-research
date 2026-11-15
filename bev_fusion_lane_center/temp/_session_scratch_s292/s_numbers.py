import json, os
R = os.environ["BFLC_ROOT"] + "/runs/"
K = ("topo_F1", "dup_frac", "F1@1.0", "lanes_found", "junction_F1", "vert100_dp")
for s in (0, 1):
    for nm, run, f in (("s own kp snapped", "e324_s_s", "ends_card_snap_thr0.1.json"), ("c2 snapped", "e312_p_s", "ends_card_c2_snap_thr0.1.json"), ("c (E312)", "e312_p_s", "ends_card_snap_thr0.1.json")):
        h = json.load(open(f"{R}{run}{s}/{f}"))["numbers"]["head"]
        print(s, nm.ljust(18), {k: round(h[k], 4) for k in K})
