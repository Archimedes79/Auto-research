import json, os, pickle
R = os.environ["BFLC_ROOT"] + "/runs/e324_s_s0/"
x = json.load(open(R + "ends_card_e360t_snap_nogeom_attach_thr0.1.json")); y = json.load(open(R + "ends_card_e338a_snap_nogeom_attach_thr0.1.json"))
for k in ("numbers", "curves", "wired_both_ends", "long_links_dropped", "ends_within_1m_of_a_delivered_piece_end"):
    print(k, x[k] == y[k])
cx = pickle.load(open(R + "ends_card_e360t_snap_nogeom_attach_thr0.1.counts.pkl", "rb")); cy = pickle.load(open(R + "ends_card_e338a_snap_nogeom_attach_thr0.1.counts.pkl", "rb"))
print("counts", repr(cx) == repr(cy))
print({k: round(v, 4) for k, v in x["numbers"]["head"].items() if k in ("topo_F1", "dup_frac")})
