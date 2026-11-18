import json, os
R = os.environ["BFLC_ROOT"] + "/runs/e324_s_s%d/"
K = ("topo_F1", "dup_frac", "F1@1.0", "junction_F1", "lanes_found", "vert100_dp")
for s in (0, 1):
    for tag, suf in (("e360c", "snap_nogeom_attach"), ("e360a", "snap_nogeom_attach"), ("e360ck", "snap_nogeom"), ("e360ak", "snap_nogeom")):
        c = json.load(open(R % s + f"ends_card_{tag}_{suf}_thr0.1.json"))
        h = c["numbers"]["head"]
        w = c["ends_within_1m_of_a_delivered_piece_end"]
        print(s, tag, {k: round(h[k], 4) for k in K}, "curves", c["curves"], "wired", c["wired_both_ends"], "within1m att-kp", round(w["entry"] - w["keypoint_entry"], 4), round(w["exit"] - w["keypoint_exit"], 4))
