import json, os
R = os.environ["BFLC_ROOT"] + "/runs/"
K = ("topo_F1", "dup_frac", "F1@1.0", "lanes_found", "junction_F1", "vert100_dp")
for s in (0, 1):
    c = json.load(open(f"{R}e312_p_s{s}/ends_card_snap_thr0.1.json"))["numbers"]["head"]
    c2 = json.load(open(f"{R}e312_p_s{s}/ends_card_c2_snap_thr0.1.json"))["numbers"]["head"]
    print(f"seed {s} c2 {[round(c2[k], 4) for k in K]}  c {[round(c[k], 4) for k in K]}  diff {[round(c2[k]-c[k], 4) for k in K]}")
