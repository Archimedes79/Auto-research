import json, os
R = os.environ["BFLC_ROOT"] + "/runs/"
K = ("topo_F1", "dup_frac", "F1@1.0", "lanes_found", "junction_F1", "vert100_dp")
for s in (0, 1):
    for nm, f in (("sc own snapped (e338)", "ends_card_e338_snap_thr0.1.json"), ("own nogeom on c2 (e338n)", "ends_card_e338n_snap_nogeom_thr0.1.json"), ("PRIMARY attached on c2 (e338a)", "ends_card_e338a_snap_nogeom_attach_thr0.1.json"), ("s own snapped (B)", "ends_card_snap_thr0.1.json"), ("s own nogeom (B)", "ends_card_snap_nogeom_thr0.1.json"), ("s PRIMARY (B)", "ends_card_snap_nogeom_attach_thr0.1.json")):
        h = json.load(open(f"{R}e324_s_s{s}/{f}"))["numbers"]["head"]
        print(s, nm.ljust(32), {k: round(h[k], 4) for k in K})
