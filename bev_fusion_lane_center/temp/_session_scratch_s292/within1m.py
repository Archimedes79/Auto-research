import json, os
R = os.environ["BFLC_ROOT"] + "/runs/"
for s in (0, 1):
    d = json.load(open(f"{R}e324_s_s{s}/ends_card_snap_nogeom_attach_thr0.1.json"))["ends_within_1m_of_a_delivered_piece_end"]
    print(s, {k: round(v, 3) for k, v in d.items()})
