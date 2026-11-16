import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/jwire_verdict.py", [
("""control {a.control_prefix}<S>, head {a.head_prefix}<S>, tag {a.tag}"]""", """control {a.control_prefix}<S>, head {a.head_prefix}<S>, tag {a.tag}, read-out {a.read_out}"]"""),
("""        out.append(f"   pilot VAL keypoint error median {wa['pilot_val']['err_median']:.2f} m (TRAIN / VAL {pt['err_median'] / wa['pilot_val']['err_median']:.2f}, shift-dominated below {SHIFT_LIMIT}); provenance: pair run {wa.get('pair_run', 'not recorded (E312)')}, seed {wa.get('seed', s)}")""",
 """        out.append(f"   pilot VAL keypoint error median {wa['pilot_val']['err_median']:.2f} m (TRAIN / VAL {pt['err_median'] / wa['pilot_val']['err_median']:.2f}, shift-dominated below {SHIFT_LIMIT}); labelled ends within 0.5 m of a piece end: TRAIN {pt.get('near05', float('nan')):.3f}, VAL {wa['pilot_val'].get('near05', float('nan')):.3f} (shift-dominated above a difference of {SHARE_LIMIT}); provenance: pair run {wa.get('pair_run', 'not recorded (E312)')}, ends {wa.get('ends', 'kp (not recorded)')}, seed {wa.get('seed', s)}")"""),
("""        kv, kt = kills(wa)
        out.append(f"   KILL fired:""", """        nj = wa.get("no_jitter_head")
        if nj:
            v, t = nj["val"], nj["train"]
            out.append(f"   no-jitter head (wn, reported only): VAL overall {v['head_same'] / v['oracle']:.3f} (rule {v['rule_same'] / v['oracle']:.3f}), real {v['head_multi_same'] / v['multi']:.3f} (rule {v['rule_multi_same'] / v['multi']:.3f}) | TRAIN (unperturbed) real {t['head_multi_same'] / t['multi']:.3f} (rule {t['rule_multi_same'] / t['multi']:.3f})")
        kv, kt = kills(wa)
        out.append(f"   KILL fired:"""),
("""            out.append(f"   head {hh} snapped: " +""", """            out.append(f"   head {hh} {'primary' if a.read_out == 'primary' else 'snapped'}: " +"""),
("""    ap.add_argument("--boot", default=None, help=""", """    ap.add_argument("--read-out", default="snap", choices=tuple(READ_OUT_ENDS), help="snap (default, E325 / E350): the snapped cards; primary (E358): the end-to-end read-out of E324 s, the cards `..._snap_nogeom_attach_thr0.1.json`")
    ap.add_argument("--boot", default=None, help="""),
])
print("ok")
