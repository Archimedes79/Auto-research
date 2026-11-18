import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/jwire_verdict.py", [
("""    wsl bash tools/env/_py14.sh jwire_verdict.py [--card E325] [--control-prefix e312_p_s] [--head-prefix e325_w_s] [--tag e325w] [--tag-x e350x --x-head-prefix e325_w_s] [--boot <pattern>]     # the defaults are E325's:
                                                                                  # reads the cards `runs/<control prefix><S>/ends_card_snap_thr0.1.json` (c), `ends_card_<tag>[1|2]_snap_thr0.1.json` (head seeds""",
 """    wsl bash tools/env/_py14.sh jwire_verdict.py [--card E325] [--control-prefix e312_p_s] [--head-prefix e325_w_s] [--tag e325w] [--tag-x e350x --x-head-prefix e325_w_s] [--read-out snap|primary] [--boot <pattern>]     # the defaults are E325's:
                                                                                  # reads the cards `runs/<control prefix><S>/ends_card_snap_thr0.1.json` (c), `ends_card_<tag>[1|2]_snap_thr0.1.json` (head seeds"""),
("""    E350: --card E350 --control-prefix e324_s_s --head-prefix e350_w_s --tag e350w --tag-x e350x
""", """    E350: --card E350 --control-prefix e324_s_s --head-prefix e350_w_s --tag e350w --tag-x e350x
    E358: --card E358 --control-prefix e324_s_s --head-prefix e358_w_s --tag e358w --read-out primary
The READ-OUT (E358): `snap` (default, E325 / E350) reads the snapped cards `ends_card_[<tag>_]snap_thr0.1.json` and wants `attach` and `snap_nogeom` false in them; `primary` (the end-to-end read-out of E324 s: attached ends, snap wiring without the
snap geometry) reads `ends_card_[<tag>_]snap_nogeom_attach_thr0.1.json` (the control `ends_card_snap_nogeom_attach_thr0.1.json`) and wants `attach` true and `snap_nogeom` (in the card's `snap_options`) true. The head's `ends`
(`wire_agreement.json`, no field = kp) must be the read-out's kind (primary: att, snap: kp): a head of the other kind is INVALID.
"""),
("""limit: a TRAIN median keypoint error below 0.7 x the VAL median labels the result 'shift-dominated'.""", """limit: a TRAIN median keypoint error below 0.7 x the VAL median, or a TRAIN share of the labelled ends whose nearest piece end lies within 0.5 m (`near05` of the pilot, E358) that differs from the VAL share by more than 0.10 (absolute), labels the result 'shift-dominated'."""),
("""KILL_MARGIN, SHIFT_LIMIT, POINT_TOL = 0.02, 0.7, 1e-4
""", """KILL_MARGIN, SHIFT_LIMIT, SHARE_LIMIT, POINT_TOL = 0.02, 0.7, 0.10, 1e-4
READ_OUT_ENDS = dict(snap="kp", primary="att")                                                       # the read-outs and the end kind a head must have been trained on for each
"""),
("""def finite(x):""", """def stem(read_out):
    \"\"\"the part of a card's file name after the tag: `ends_card_[<tag>_]<stem>thr0.1.json` (the name rule of `qends_eval.card_name` for the two read-outs)\"\"\"
    return "snap_nogeom_attach_" if read_out == "primary" else "snap_"


def finite(x):"""),
("""        p = rd + f"ends_card_{tag + '_' if tag else ''}snap_thr0.1.json"
""", """        p = rd + f"ends_card_{tag + '_' if tag else ''}{stem(a.read_out)}thr0.1.json"
"""),
("""    run, cards, wa, bt = sd["run"], sd["cards"], sd["wa"], sd["boot"]
    named =""", """    run, cards, wa, bt = sd["run"], sd["cards"], sd["wa"], sd["boot"]
    primary = a.read_out == "primary"
    named ="""),
("""        for key, ok in (("run", c.get("run") == run), ("seed", c.get("seed") == s), ("snap", c.get("snap") is True), ("attach", not c.get("attach")), ("thr", c.get("thr") == 0.1)):
            if not ok:
                return f"seed {s}: the {what} says {key} = {c.get(key)!r} (the snapped read-out at p >= 0.1 of {run} seed {s} is meant)\"""", """        got = dict(c, snap_nogeom="snap_nogeom" in (c.get("snap_options") or []))
        for key, ok in (("run", c.get("run") == run), ("seed", c.get("seed") == s), ("snap", c.get("snap") is True), ("attach", bool(c.get("attach")) == primary), ("snap_nogeom", got["snap_nogeom"] == primary), ("thr", c.get("thr") == 0.1)):
            if not ok:
                return f"seed {s}: the {what} says {key} = {got.get(key)!r} (the {'primary (attached ends, snap wiring without the snap geometry)' if primary else 'snapped'} read-out at p >= 0.1 of {run} seed {s} is meant)\""""),
("""    if bt is not None:
        if "error" in bt:""", """    if wa.get("ends", "kp") != READ_OUT_ENDS[a.read_out]:
        return f"seed {s}: the head was trained on {wa.get('ends', 'kp')!r} ends (no field = kp), the {a.read_out} read-out needs a head trained on {READ_OUT_ENDS[a.read_out]!r} ends"
    if bt is not None:
        if "error" in bt:"""),
("""        want_a, want_b = (run, f"ends_card_{a.tag}_snap_thr0.1"), (run, "ends_card_snap_thr0.1")""", """        want_a, want_b = (run, f"ends_card_{a.tag}_{stem(a.read_out)}thr0.1"), (run, f"ends_card_{stem(a.read_out)}thr0.1")"""),
("""        if kt and not kv[s]:
            notes.append(""", """        near = [wa[k].get("near05") for k in ("pilot_train", "pilot_val")]
        if None not in near and abs(near[0] - near[1]) > SHARE_LIMIT:
            notes.append(f"seed {s}: SHIFT-DOMINATED (TRAIN share of the labelled ends within 0.5 m of a piece end {near[0]:.3f} differs from VAL {near[1]:.3f} by more than {SHARE_LIMIT})")
        if kt and not kv[s]:
            notes.append("""),
])
print("ok")
