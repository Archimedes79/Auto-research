import json, os
R = os.environ["BFLC_ROOT"]
for s in (0, 1):
    for nm in ("ends_card_thr0.1.json", "ends_card_attach_thr0.1.json", "ends_card_snap_nogeom_attach_thr0.1.json", "ends_card_snap_thr0.1.json"):
        p = os.path.join(R, "runs", f"e324_s_s{s}", nm)
        if os.path.exists(p):
            h = json.load(open(p))["numbers"]["head"]
            print(s, nm.replace("ends_card_", "").replace("_thr0.1.json", ""), "topo", round(h["topo_F1"], 4), "dup", round(h["dup_frac"], 4), "F1@1.0", round(h["F1@1.0"], 4), "junction", round(h["junction_F1"], 4), "vert", round(h["vert100_dp"], 2))
