import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply

R = "/mnt/c/Auto-research/bev_fusion_lane_center/"
apply(R + "evaluation/jwire_verdict.py", [
    # docstring
    ("""[--read-out snap|primary] [--boot <pattern>]     # the defaults are E325's:""", """[--read-out snap|primary] [--tag-wn e358wn] [--prediction 0.001 --window 0.003] [--boot <pattern>]     # the defaults are E325's:"""),
    ("""    E358: --card E358 --control-prefix e324_s_s --head-prefix e358_w_s --tag e358w --read-out primary
""", """    E358: --card E358 --control-prefix e324_s_s --head-prefix e358_w_s --tag e358w --read-out primary --tag-wn e358wn --prediction 0.001 --window 0.003
Reported next to the rule, never in it (tokens and exit codes do not depend on them): `--tag-wn` = the cards of the no-jitter head (`ends_card_<tag-wn>_<stem>thr0.1.json`, scored with `--wire-run <head prefix><S>:model_wn.pt`; optional, a missing
card prints 'not scored'): its topo F1 and its difference to c and to w; `--prediction` / `--window` = the prediction of the card (w - c) and its hit window: per seed w - c and HIT when |w - c| <= the window on BOTH seeds, else MISS.
"""),
    ("""labels the result 'shift-dominated'. INCOMPLETE""", """labels the result 'shift-dominated'; this rule judges the UNPERTURBED TRAIN records against VAL (`pilot_train.near05`), the share of the JITTERED draw the head trains on (`pilot_train_jittered.near05`, with the jitter scale) is printed
as a NOTE next to them and is no rule. INCOMPLETE"""),
    # read_seed / check_inputs
    ("""x=card(a.tag_x) if a.tag_x else None)
    num =""", """x=card(a.tag_x) if a.tag_x else None, wn=card(a.tag_wn) if a.tag_wn else None)
    num ="""),
    ("""x=num(cards["x"]), cards=cards,""", """x=num(cards["x"]), wn=num(cards["wn"]), cards=cards,"""),
    ("""[("arm x card", cards["x"])]
    for what, c in named:""", """[("arm x card", cards["x"]), ("no-jitter card", cards["wn"])]
    for what, c in named:"""),
    ("""        heads.append(("arm x card", cards["x"], a.tag_x, f"{a.x_head_prefix}{s}"))
""", """        heads.append(("arm x card", cards["x"], a.tag_x, f"{a.x_head_prefix}{s}"))
    if a.tag_wn:
        heads.append(("no-jitter card", cards["wn"], a.tag_wn, f"{a.head_prefix}{s}:model_wn.pt"))
"""),
    # judge: the note
    ("""        if kt and not kv[s]:
            notes.append(f"seed {s}: SHIFT FLAG""", """        pj = wa.get("pilot_train_jittered")
        if pj and None not in near:
            notes.append(f"seed {s}: labelled ends within 0.5 m of a piece end: VAL {near[1]:.3f}, TRAIN unperturbed {near[0]:.3f} (the shift rule above judges this one), TRAIN jittered {pj['near05']:.3f} (jitter scale {pj['jitter_scale']:.2f} m, {pj['draws']} draws: what the head trains on; a note, no rule)")
        if kt and not kv[s]:
            notes.append(f"seed {s}: SHIFT FLAG"""),
    # render: wn line, prediction line
    ("""    token, why, notes = judge(data)
    out += ["price:""", """        if a.tag_wn:
            wn, w0 = sd["wn"], sd["w"][0]
            out.append(f"   no-jitter head wn ({a.tag_wn}, reported only): " + ("not scored" if wn is None or c is None else f"topo_F1 {wn['topo_F1']:.4f} (vs c {wn['topo_F1'] - c['topo_F1']:+.4f}" + (f", vs w {wn['topo_F1'] - w0['topo_F1']:+.4f})" if w0 else ")")))
    if a.prediction is not None:
        d = [None if data[s]["w"][0] is None or data[s]["c"] is None else data[s]["w"][0]["topo_F1"] - data[s]["c"]["topo_F1"] for s in SEEDS]
        hit = None if None in d else all(abs(x) <= a.window + 1e-9 for x in d)
        out.append(f"prediction of the card: w - c = {a.prediction:+.4f}, hit window |w - c| <= {a.window}: " + ", ".join(f"seed {s} " + ("n/a" if x is None else f"{x:+.4f}") for s, x in zip(SEEDS, d)) + " -> " + ("not computable" if hit is None else "HIT" if hit else "MISS") + " (reported, not in the rule)")
    token, why, notes = judge(data)
    out += ["price:"""),
    # main
    ("""    ap.add_argument("--x-head-prefix", default="e325_w_s",""", """    ap.add_argument("--tag-wn", default=None, help="E358: the tag of the cards of the no-jitter head (--wire-run <head prefix><seed>:model_wn.pt), reported only; a missing card prints 'not scored'")
    ap.add_argument("--prediction", type=float, default=None, help="E358: the prediction of the card (w - c, topo F1): prints the per-seed w - c and HIT / MISS against --window")
    ap.add_argument("--window", type=float, default=0.003, help="the hit window of --prediction: HIT when |w - c| <= window on both seeds")
    ap.add_argument("--x-head-prefix", default="e325_w_s","""),
    # selftest: the planted world and the NS
    ("""            if sp.get("x"):
                json.dump(card({k: c[k] - 0.001 for k in K}, "x", tag="tx", wire="head", wire_run=f"xs{s}"), open(f"{rd}/ends_card_tx_{st}thr0.1.json", "w"))""", """            if sp.get("x"):
                json.dump(card({k: c[k] - 0.001 for k in K}, "x", tag="tx", wire="head", wire_run=f"xs{s}"), open(f"{rd}/ends_card_tx_{st}thr0.1.json", "w"))
            if sp.get("wn") is not None:                                                                      # the no-jitter head: its differences to c
                json.dump(card({k: c[k] + sp["wn"].get(k, 0.0) for k in K}, "wn", tag="tn", wire="head", wire_run=f"hs{s}:model_wn.pt"), open(f"{rd}/ends_card_tn_{st}thr0.1.json", "w"))"""),
    ("""tag_x=kw.get("tag_x"), x_head_prefix="xs", boot="temp/e000/boot_w_vs_c_s{seed}.log", read_out=ro)
            data =""", """tag_x=kw.get("tag_x"), tag_wn=kw.get("tag_wn"), prediction=kw.get("prediction"), window=kw.get("window", 0.003), x_head_prefix="xs", boot="temp/e000/boot_w_vs_c_s{seed}.log", read_out=ro)
            data ="""),
    ("""tag_x=None, x_head_prefix="xs", boot="temp/e000/boot_w_vs_c_s{seed}.log", read_out="primary")""", """tag_x=None, tag_wn=None, prediction=None, window=0.003, x_head_prefix="xs", boot="temp/e000/boot_w_vs_c_s{seed}.log", read_out="primary")"""),
    # selftest: the new tests (before the command line block)
    ("""    # the command line: the default read-out is the snapped one""", """    # the no-jitter head and the prediction window are reported next to the rule: they change no token (the same cards give the same token with and without them), a missing card is 'not scored', a card of another head is INVALID
    wn_d = dict(topo_F1=0.0015)
    both = {s: dict(gp, wn=wn_d) for s in SEEDS}
    t, lines, _ = run_case(both, ro="primary", tag_wn="tn")
    text = "\\n".join(lines)
    assert t == ptok({0: gp, 1: gp}) == "KEEP+" and text.count("no-jitter head wn (tn, reported only): topo_F1 0.7015 (vs c +0.0015, vs w -0.0005)") == 2, text            # w = c + 0.002, wn = c + 0.0015
    t, lines, _ = run_case({0: dict(gp, wn=wn_d), 1: gp}, ro="primary", tag_wn="tn")
    assert t == "KEEP+" and sum("wn (tn, reported only): not scored" in x for x in lines) == 1 and sum("wn (tn, reported only): topo_F1" in x for x in lines) == 1, lines          # seed 1 has no card of wn
    assert not any("wn (tn" in x for x in run_case(both, ro="primary")[1]), "no tag-wn, no line: E325 / E350 print what they printed"
    for over in (dict(wire_run="hs9:model_wn.pt"), dict(wire_run="hs0"), dict(wire="twin"), dict(tag="tw"), dict(run="other"), dict(attach=False)):
        for s in SEEDS:
            t, _, (_, reason, _) = run_case({s: dict(gp, wn=wn_d, over={"wn": over}), 1 - s: dict(gp, wn=wn_d)}, ro="primary", tag_wn="tn")
            assert t == "INVALID" and f"seed {s}" in reason and "no-jitter card" in reason, (over, s, t, reason)
    for d0, d1, want in ((0.002, 0.002, "HIT"), (0.003, 0.0029, "HIT"), (0.0031, 0.002, "MISS"), (0.002, -0.0031, "MISS"), (-0.003, 0.0, "HIT")):             # |w - c| <= 0.003 on BOTH seeds (0.003 itself is a hit)
        sp = {0: dict(gp, d=dict(topo_F1=d0)), 1: dict(gp, d=dict(topo_F1=d1))}
        t, lines, _ = run_case(sp, ro="primary", prediction=0.001)
        line = [x for x in lines if x.startswith("prediction of the card")]
        assert len(line) == 1 and line[0].endswith(f"-> {want} (reported, not in the rule)") and f"seed 0 {d0:+.4f}, seed 1 {d1:+.4f}" in line[0] and "w - c = +0.0010, hit window |w - c| <= 0.003" in line[0], (d0, d1, line)
        assert t == run_case(sp, ro="primary")[0], "the prediction line changes no token"
    assert not any(x.startswith("prediction of the card") for x in run_case({0: gp, 1: gp}, ro="primary")[1]), "no --prediction, no line"
    assert "window |w - c| <= 0.005" in [x for x in run_case({0: gp, 1: gp}, ro="primary", prediction=0.001, window=0.005)[1] if x.startswith("prediction")][0]
    line = [x for x in run_case({0: gp, 1: dict(gp, d=None)}, ro="primary", prediction=0.001)[1] if x.startswith("prediction")][0]
    assert "seed 1 n/a -> not computable" in line, line
    # the jittered share is a note, never a rule: the three shares are printed, the token does not change, the 0.10 shift rule judges the unperturbed ones
    for near, jit in (((0.45, 0.44), dict(near05=0.80, jitter_scale=0.6, draws=3)), ((0.45, 0.56), dict(near05=0.45, jitter_scale=0.6, draws=3))):
        sp = {s: dict(gp, wa=agree(ends="att", near=near, pilot_train_jittered=jit)) for s in SEEDS}
        t, _, (_, _, notes) = run_case(sp, ro="primary")
        mine = [n for n in notes if "TRAIN jittered" in n]
        assert t == "KEEP+" and len(mine) == 2 and f"VAL {near[1]:.3f}, TRAIN unperturbed {near[0]:.3f} (the shift rule above judges this one), TRAIN jittered {jit['near05']:.3f} (jitter scale 0.60 m, 3 draws" in mine[0], (t, mine)
        assert sum("SHIFT-DOMINATED (TRAIN share" in n for n in notes) == (2 if abs(near[0] - near[1]) > 0.10 else 0), "the jittered share does not enter the shift rule"
    t, _, (_, _, notes) = run_case({0: gp, 1: gp}, ro="primary")
    assert not any("TRAIN jittered" in n for n in notes), "a file without the jittered share (E325 / E350) says nothing"
    t, _, (_, _, notes) = run_case({s: dict(gp, wa=agree(ends="att", pilot_train_jittered=dict(near05=0.5, jitter_scale=0.6, draws=3))) for s in SEEDS}, ro="primary")
    assert not any("TRAIN jittered" in n for n in notes), "without the unperturbed and VAL shares there is nothing to put it next to"
    # the command line: the default read-out is the snapped one"""),
])
