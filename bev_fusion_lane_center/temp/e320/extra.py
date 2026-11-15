import json, os
for S in (0, 1):
    for run, f in ((f"e312_p_s{S}", "ends_card_snap_thr0.1.json"), (f"e312_p_s{S}", "ends_card_snap_nogeom_thr0.1.json"), (f"e320_a_s{S}", "ends_card_snap_nogeom_attach_thr0.1.json"), (f"e320_a_s{S}", "ends_card_snap_thr0.1.json"), (f"e320_a_s{S}", "ends_card_attach_thr0.1.json")):
        p = f"runs/{run}/{f}"
        if not os.path.exists(p): continue
        c = json.load(open(p))
        print(S, run, f, {k: c.get(k) for k in ("snap", "attach", "attach_r", "long_links_dropped", "ends_within_1m_of_a_delivered_piece_end", "curves", "wired_both_ends")})
kp = lambda r: sum(json.load(open(f"runs/{r}/jmaps_val.json"))["keypoints"][f"kp_ours_tol3.0_noang_{k}"]["f1"] for k in ("ent", "ext")) / 2
for S in (0, 1):
    try:
        a, b = kp(f"e320_a_s{S}_maps"), kp(f"e307_n_s{S}")
        print(f"seed {S}: maps after the joint training kp F1 {a:.4f}, E307 {b:.4f}, difference {a - b:+.4f}; E312 maps {kp(f'e312_p_s{S}_maps'):.4f}; E316 maps {kp(f'e316_p_s{S}_maps'):.4f}")
    except Exception as e:
        print(S, "maps not yet", e)
