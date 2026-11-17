import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply

R = "/mnt/c/Auto-research/bev_fusion_lane_center/"
S = R + "temp/_session_scratch_e358/"
# the verdict: the command line with --tag-wn / --prediction / --window
apply(R + "evaluation/jwire_verdict.py", [
    ("""    finally:
        ROOT = old_root
    print("jwire_verdict selftest ok")""", """        with tempfile.TemporaryDirectory() as t:                                                           # --tag-wn, --prediction and --window reach the report (the defaults of the window: 0.003)
            world(t, {s: dict(gp, wn=wn_d) for s in SEEDS}, "cs", "primary")
            ROOT = t
            base = ["--card", "E000", "--control-prefix", "cs", "--head-prefix", "hs", "--tag", "tw", "--boot", "temp/e000/boot_w_vs_c_s{seed}.log", "--read-out", "primary"]
            for args, want in (([], []), (["--tag-wn", "tn"], ["no-jitter head wn (tn, reported only): topo_F1 0.7015"]), (["--prediction", "0.001"], ["hit window |w - c| <= 0.003", "-> HIT"]), (["--prediction", "0.001", "--window", "0.001"], ["hit window |w - c| <= 0.001", "-> MISS"])):
                with contextlib.redirect_stdout(io.StringIO()) as buf:
                    main(base + args)
                assert buf.getvalue().splitlines()[-1].startswith("TOKEN (E000): KEEP+") and all(x in buf.getvalue() for x in want) and (args or "no-jitter head wn" not in buf.getvalue() and "prediction of the card" not in buf.getvalue()), (args, buf.getvalue())
    finally:
        ROOT = old_root
    print("jwire_verdict selftest ok")"""),
])
# the mutation harness: the new mutants and the staged modules
apply(S + "mutate.py", [
    ("""FILES = dict(jwire_data="evaluation/jwire_data.py", jwire_train="training/train/jwire_train.py", jwire_verdict="evaluation/jwire_verdict.py", jwire_check="evaluation/jwire_check.py")""", """FILES = dict(jwire_data="evaluation/jwire_data.py", jwire_train="training/train/jwire_train.py", jwire_verdict="evaluation/jwire_verdict.py", jwire_check="evaluation/jwire_check.py",
             qends_eval="temp/_session_scratch_e358/stage/qends_eval.py", jwire_head="temp/_session_scratch_e358/stage/jwire_head.py")                       # the last two: the STAGED copies, tested by test_staged.py"""),
    ("""        ("main val: --ends default att", \"\"\"    s.add_argument("--ends", default="kp", \"\"\", \"\"\"    s.add_argument("--ends", default="att", \"\"\"),
    ],""", """        ("main val: --ends default att", \"\"\"    s.add_argument("--ends", default="kp", \"\"\", \"\"\"    s.add_argument("--ends", default="att", \"\"\"),
        ("is_near: distance not scaled", "return d_feat * JH.REACH <= NEAR_M", "return d_feat <= NEAR_M"),
        ("is_near: threshold in the feature units", "return d_feat * JH.REACH <= NEAR_M", "return d_feat * JH.REACH <= NEAR_M * 2"),
        ("pilot_stats: near05 without is_near (distance not scaled)", "near += [is_near(float(i[\\"feat\\"][0, 0]))", "near += [float(i[\\"feat\\"][0, 0]) <= NEAR_M"),
        ("record_of: kept ends of the other kind", "for en in kept], float).reshape(-1, 8), attrs=attrs,", "for en in pairs], float).reshape(-1, 8), attrs=attrs,"),
    ],"""),
    ("""        ("main run: --ends default att", \"\"\"    r.add_argument("--ends", default="kp", \"\"\", \"\"\"    r.add_argument("--ends", default="att", \"\"\"),
    ],""", """        ("main run: --ends default att", \"\"\"    r.add_argument("--ends", default="kp", \"\"\", \"\"\"    r.add_argument("--ends", default="att", \"\"\"),
        ("near_share: every end, not the labelled ones", "feat[orc][:, 0, 0]", "feat[:, 0, 0]"),
        ("near_share: place 1", "feat[orc][:, 0, 0]", "feat[orc][:, 1, 0]"),
        ("near_share: no NaN without labelled ends", "if bool(orc.any()) else float(\\"nan\\")", "if True else float(\\"nan\\")"),
        ("run_pipeline: jittered share of the unperturbed records", "near05=near_share(feat, orc), jitter_scale=scale", "near05=near_share(*[tensors(train)[i] for i in (0, 3)]), jitter_scale=scale"),
        ("run_pipeline: jittered share without the scale", "jitter_scale=scale, draws=draws)    #", "jitter_scale=0.0, draws=draws)    #"),
        ("run_pipeline: jittered share with one draw", "jitter_scale=scale, draws=draws)    #", "jitter_scale=scale, draws=1)    #"),
        ("run_pipeline: no jittered share", "    info[\\"pilot_train_jittered\\"] = dict(", "    _ = dict("),
        ("run_pipeline: wn epochs 1", "nl.repeat(draws), epochs, seed=0)", "nl.repeat(draws), 1, seed=0)"),
        ("run_pipeline: wn epochs default", "nl.repeat(draws), epochs, seed=0)", "nl.repeat(draws), 60, seed=0)"),
        ("run_pipeline: wn learning rate", "nl.repeat(draws), epochs, seed=0)", "nl.repeat(draws), epochs, lr=1e-3, seed=0)"),
        ("run_pipeline: wn weight decay", "nl.repeat(draws), epochs, seed=0)", "nl.repeat(draws), epochs, wd=0.0, seed=0)"),
        ("run_pipeline: wn batch size", "nl.repeat(draws), epochs, seed=0)", "nl.repeat(draws), epochs, bs=64, seed=0)"),
    ],"""),
    ("""        ("main: --read-out default primary", \"\"\"ap.add_argument("--read-out", default="snap",\"\"\", \"\"\"ap.add_argument("--read-out", default="primary",\"\"\"),
    ],""", """        ("main: --read-out default primary", \"\"\"ap.add_argument("--read-out", default="snap",\"\"\", \"\"\"ap.add_argument("--read-out", default="primary",\"\"\"),
        ("note: jittered share not printed", "        if pj and None not in near:", "        if False:"),
        ("note: the unperturbed number in the jittered place", "TRAIN jittered {pj['near05']:.3f}", "TRAIN jittered {near[0]:.3f}"),
        ("note: wrong number of draws", "{pj['draws']} draws", "{pj['draws'] + 1} draws"),
        ("note: wrong scale", "jitter scale {pj['jitter_scale']:.2f} m", "jitter scale {pj['jitter_scale'] * 2:.2f} m"),
        ("note: printed also without the other shares", "        if pj and None not in near:", "        if pj:"),
        ("shift rule judges the jittered share", "abs(near[0] - near[1]) > SHARE_LIMIT", "abs((wa.get('pilot_train_jittered') or {}).get('near05', near[0]) - near[1]) > SHARE_LIMIT"),
        ("wn: cards not read", "wn=card(a.tag_wn) if a.tag_wn else None)", "wn=None)"),
        ("wn: cards not named", ", (\\"no-jitter card\\", cards[\\"wn\\"])]", "]"),
        ("wn: head not checked", "    if a.tag_wn:\\n        heads.append(", "    if False:\\n        heads.append("),
        ("wn: checkpoint name not checked", "f\\"{a.head_prefix}{s}:model_wn.pt\\"", "f\\"{a.head_prefix}{s}\\""),
        ("wn: tag not checked", "heads.append((\\"no-jitter card\\", cards[\\"wn\\"], a.tag_wn,", "heads.append((\\"no-jitter card\\", cards[\\"wn\\"], a.tag,"),
        ("wn: difference to c reversed", "{wn['topo_F1'] - c['topo_F1']:+.4f}", "{c['topo_F1'] - wn['topo_F1']:+.4f}"),
        ("wn: difference to w against head 1", "wn['topo_F1'] - w0['topo_F1']", "wn['topo_F1'] - sd['w'][1]['topo_F1']"),
        ("wn: always scored", "(\\"not scored\\" if wn is None or c is None else", "(\\"not scored\\" if False else"),
        ("prediction: any seed", "all(abs(x) <= a.window + 1e-9 for x in d)", "any(abs(x) <= a.window + 1e-9 for x in d)"),
        ("prediction: strict window", "abs(x) <= a.window + 1e-9", "abs(x) < a.window"),
        ("prediction: no tolerance", "abs(x) <= a.window + 1e-9", "abs(x) <= a.window"),
        ("prediction: signed", "abs(x) <= a.window + 1e-9", "x <= a.window + 1e-9"),
        ("prediction: c - w", "data[s][\\"w\\"][0][\\"topo_F1\\"] - data[s][\\"c\\"][\\"topo_F1\\"] for s in SEEDS]", "data[s][\\"c\\"][\\"topo_F1\\"] - data[s][\\"w\\"][0][\\"topo_F1\\"] for s in SEEDS]"),
        ("prediction: missing card is a hit", "hit = None if None in d else", "hit = True if None in d else"),
        ("prediction: window default 0.005", "ap.add_argument(\\"--window\\", type=float, default=0.003,", "ap.add_argument(\\"--window\\", type=float, default=0.005,"),
        ("prediction: option not parsed", "ap.add_argument(\\"--prediction\\", type=float, default=None,", "ap.add_argument(\\"--prediction\\", type=float, default=0.001,"),
    ],"""),
    ("""        ("main: --ends without choices", \"\"\"choices=tuple(READ_OUT_ENDS.values()), help\"\"\", \"\"\"help\"\"\"),
    ],
)""", """        ("main: --ends without choices", \"\"\"choices=tuple(READ_OUT_ENDS.values()), help\"\"\", \"\"\"help\"\"\"),
    ],
    qends_eval=[
        ("check_wire: attached read-out without nogeom accepted", "or getattr(a, \\"snap_nogeom\\", False), \\"--wire twin / head with --attach needs", "or True, \\"--wire twin / head with --attach needs"),
        ("check_wire: nogeom also required without attach", "not getattr(a, \\"attach\\", False) or getattr(a, \\"snap_nogeom\\", False), \\"--wire twin / head with", "getattr(a, \\"snap_nogeom\\", False), \\"--wire twin / head with"),
        ("check_wire: attach refused again", "for k in (\\"snap_all\\", \\"snap_guard\\")) and getattr(a, \\"attach_r\\"", "for k in (\\"snap_all\\", \\"snap_guard\\", \\"attach\\")) and getattr(a, \\"attach_r\\""),
        ("check_wire: attach_r accepted", "and getattr(a, \\"attach_r\\", None) is None), \\"--wire twin / head needs --snap", "), \\"--wire twin / head needs --snap"),
        ("check_wire: snap_all accepted with attach", "for k in (\\"snap_all\\", \\"snap_guard\\")) and getattr(a, \\"attach_r\\"", "for k in (\\"snap_guard\\",)) and getattr(a, \\"attach_r\\""),
        ("cmd_score: head not checked", "        JH.check_head(head, a.run, a.seed, bool(getattr(a, \\"attach\\", False)))", "        pass"),
        ("cmd_score: attach not passed to the head check", "JH.check_head(head, a.run, a.seed, bool(getattr(a, \\"attach\\", False)))", "JH.check_head(head, a.run, a.seed, False)"),
        ("cmd_score: seed not passed to the head check", "JH.check_head(head, a.run, a.seed, bool(", "JH.check_head(head, a.run, 0, bool("),
    ],
    jwire_head=[
        ("check_head: kind not checked", "    assert ends == want, f\\"the head was trained on", "    assert True or ends == want, f\\"the head was trained on"),
        ("check_head: no field read as att", "m.get(\\"ends\\", \\"kp\\")", "m.get(\\"ends\\", \\"att\\")"),
        ("check_head: wanted kind inverted", "want = \\"att\\" if attach else \\"kp\\"", "want = \\"kp\\" if attach else \\"att\\""),
        ("check_head: seed not checked", "    assert hs == seed, ", "    assert True, "),
        ("check_head: pair run not checked", "    assert hp == run, ", "    assert True, "),
        ("check_head: no pair run field read as the run asked for", "m.get(\\"pair_run\\") or f\\"e312_p_s{hs}\\"", "m.get(\\"pair_run\\") or run"),
        ("check_head: no seed field read as seed 0", "hs = m.get(\\"seed\\", seed)", "hs = m.get(\\"seed\\", 0)"),
        ("load_wire: args not kept", "    w.meta = dict(ck.get(\\"args\\") or {})", "    w.meta = {}"),
    ],
)"""),
    ("""def run_tests(name, mod):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if name == "jwire_train":""", """def run_tests(name, mod):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if name in ("qends_eval", "jwire_head"):                                                          # the staged copies: the tests of test_staged.py against the mutant
            import test_staged as TS
            if name == "qends_eval":
                TS.QE = mod
                TS.test_check_wire()
                TS.test_cmd_score_refuses_before_anything_long()
            else:
                TS.JH = mod
                TS.test_check_head()
                TS.test_load_wire_keeps_the_args()
                TS.test_cmd_score_refuses_before_anything_long()
        elif name == "jwire_train":"""),
    ("""    import jwire_verdict  # noqa: F401
    real = sys.modules[name]""", """    import jwire_verdict  # noqa: F401
    sys.path.insert(0, os.path.join(ROOT, "temp", "_session_scratch_e358"))
    import test_staged as TS  # noqa: F401  (imports the staged qends_eval / jends_eval / jwire_head: they come first on the path)
    real = sys.modules[name]
    real_qe, real_jh = TS.QE, TS.JH"""),
])
