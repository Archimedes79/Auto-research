def _complete_point(runs_dir, run, pid):
    d = os.path.join(runs_dir, PT.point_run(run, pid))
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "point.json"), "w").write("{}")
    return d


def test_score_point_calls_the_stored_scorer():
    calls, old = [], JEV.score
    JEV.score = lambda a: calls.append(a)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for run, pid in (("e312_p_s1", 12), ("e324_s_s0", 101), ("e324_s_s0", 0), ("e324_s_s0", PT.NULL2), ("e312_p_s1", 14)):
                _complete_point(tmp, run, pid)
            PT.score_point("e312_p_s1", "e312", 12, tmp)
            PT.score_point("e324_s_s0", "e324", 101, tmp)                    # a point: the two judged read-outs only (F1 of the critic: the third is for the null)
            PT.score_point("e324_s_s0", "e324", 0, tmp)                      # the nulls: every read-out of the model, for the stored-card check
            PT.score_point("e324_s_s0", "e324", PT.NULL2, tmp)
            open(os.path.join(tmp, "e345_p012_e312_p_s1", "ends_card_e345012_snap_thr0.1.counts.pkl"), "w").write("x")      # scored already: skipped
            PT.score_point("e312_p_s1", "e312", 12, tmp)
            os.makedirs(os.path.join(tmp, "e345_p013_e312_p_s1"))                                                          # an unfinished dump (no point.json): refused
            try:
                PT.score_point("e312_p_s1", "e312", 13, tmp)
                raise RuntimeError("an unfinished dump must be refused")
            except AssertionError as ex:
                assert "no point.json" in str(ex)
            n = len(calls)
            open(os.path.join(tmp, "e345_p014_e312_p_s1", "ends_card_e345014_snap_thr0.1.json"), "w").write("{}")           # a crash between the card and its counts: reported, nothing is deleted, the scorer is not called
            try:
                PT.score_point("e312_p_s1", "e312", 14, tmp)
                raise RuntimeError("a card without counts must be reported")
            except RuntimeError as ex:
                assert "exists without" in str(ex) and "Move or delete" in str(ex)
            assert len(calls) == n and os.path.exists(os.path.join(tmp, "e345_p014_e312_p_s1", "ends_card_e345014_snap_thr0.1.json"))
    finally:
        JEV.score = old
    got = [(a.run, a.seed, a.thr, a.tag, bool(getattr(a, "snap", False)), bool(getattr(a, "attach", False)), bool(getattr(a, "snap_nogeom", False))) for a in calls]
    r12, r101, r0, r900 = "e345_p012_e312_p_s1", "e345_p101_e324_s_s0", "e345_p000_e324_s_s0", "e345_p900_e324_s_s0"
    assert got == [(r12, 1, 0.1, "e345012", True, False, False), (r12, 1, 0.1, "e345012", False, False, False),
                   (r101, 0, 0.1, "e345101", True, False, False), (r101, 0, 0.1, "e345101", True, True, True),
                   (r0, 0, 0.1, "e345000", True, False, False), (r0, 0, 0.1, "e345000", True, True, True), (r0, 0, 0.1, "e345000", True, False, True),
                   (r900, 0, 0.1, "e345900", True, False, False), (r900, 0, 0.1, "e345900", True, True, True), (r900, 0, 0.1, "e345900", True, False, True),
                   (r12, 1, 0.1, "e345012", False, False, False)], got
    assert PT.card_stem(7, PT.READOUTS["e324"]["primary"]) == "ends_card_e345007_snap_nogeom_attach_thr0.1" and PT.card_stem(7, {}) == "ends_card_e345007_thr0.1"


# ---------------------------------------------------------------- the CPU side on synthetic sweeps and on stored cards
def _synthetic_run(units, cost, loss, extra=None, n=4):
    """a RunData without disk: n frames (half 0, 1, 0, 1); one pass per point whose pairs_hit is lowered so that ends_mean6 loses exactly `loss(unit, f)` (`extra`: {pass key: loss} for more passes)"""
    r = PT.RunData.__new__(PT.RunData)
    r.half, r.family, r.run, r.runs_dir = np.array([0, 1] * (n // 2)), "e312", "e312_p_s0", "/nonexistent"
    pts = PP.points(units)
    base = [100, 100, 80, 80, 100, 100, 80, 80, 1000, 1000, 800, 800]                                             # (n, kp, ok, hit) of entries, exits, pairs
    pair = {k: dict(ap=0.5, p_at_r05=0.5) for k in ("all", "sel", "jud")}
    r.rec = dict(run=r.run, attach=False, rows=list(range(n)), controls={}, points=[dict(id=i, unit=u.name, f=f, group=u.group, kind=u.kind) for i, u, f in pts])
    r.units = {u.name: [p for p in r.rec["points"] if p["unit"] == u.name] for u in units}

    def pas(L, sv):
        c = list(base)
        c[PP.COUNTS.index("pairs_hit")] -= 6 * L * base[PP.COUNTS.index("pairs_n")]                                # ends_mean6 is the mean of six numbers: -6 L x n_pairs hits = -L
        return dict(counts=[c] * n, kp=[[10, 20.0, 30.0, 0.0, 0.0]] * n, pair=pair, saved_macs=sv["macs"], saved_params=sv["params"])
    r.rec["passes"] = {"null": pas(0.0, dict(macs=0.0, params=0))}
    for i, u, f in pts:
        r.rec["passes"][f"p{i:03d}"] = pas(loss(u.name, f), cost.saved(PP.count_edits(u, f, cost.C)))
    for k, L in (extra or {}).items():
        r.rec["passes"][k] = pas(L, dict(macs=0.0, params=0))
    return r


def _set_kp(r, key, sel, jud):
    """the keypoint columns of a synthetic pass: sel / jud = (mean position error, mean attached-end error) of the frames of that half (two matched ends per frame)"""
    r.rec["passes"][key]["kp"] = [[2, 2 * (sel if h == 0 else jud)[0], 20.0, 2 * (sel if h == 0 else jud)[1], 24.0] for h in r.half]


def test_sweep_position_columns_and_dpos():
    f = fx()
    r = _synthetic_run(f["units"], f["cost"], lambda u, fr: 0.0)
    _set_kp(r, "null", (2.0, 1.5), (5.0, 2.5))
    _set_kp(r, "p001", (4.0, 1.0), (7.0, 2.5))                                                                       # selection: the keypoint error +2.0, the attached end -0.5
    _set_kp(r, "p002", (1.0, 2.2), (5.0, 2.5))                                                                       # the keypoint better by 1.0, the attached end worse by 0.7: the larger increase counts
    _set_kp(r, "p003", (1.0, 1.2), (5.0, 2.5))                                                                       # both better: dpos is negative, not clipped
    s = r.sweep("null", "sel"), r.sweep("null", "jud"), r.sweep("null", "all")
    assert [round(x["pos"], 9) for x in s] == [2.0, 5.0, 3.5] and [round(x["apos"], 9) for x in s] == [1.5, 2.5, 2.0], s                  # the halves are the frames of the half, the pooled one the sum over the matched ends
    assert abs(r.delta("p001", "sel")["dpos"] - 2.0) < 1e-9 and abs(r.delta("p001", "jud")["dpos"] - 2.0) < 1e-9
    assert abs(r.delta("p002", "sel")["dpos"] - 0.7) < 1e-9 and abs(r.delta("p003", "sel")["dpos"] - (-0.3)) < 1e-9
    assert PT.sweep_free(dict(d6=0, dap=0, dpos=r.delta("p003", "sel")["dpos"])) and not PT.sweep_free(dict(d6=0, dap=0, dpos=r.delta("p001", "sel")["dpos"]))


def test_worse_seed_in_klass_and_free_fraction():
    f = fx()
    units, cost = f["units"], f["cost"]
    A = "jm.blocks.0.hid"
    r0 = _synthetic_run(units, cost, lambda u, fr: 0.001 if u == A and fr <= 0.5 else 0.5)
    r1 = _synthetic_run(units, cost, lambda u, fr: (0.012 if fr == 0.5 else 0.001) if u == A and fr <= 0.5 else 0.5)      # seed 1 loses 0.012 at f = 0.5
    mod = PT.Model([r0, r1], units, cost)
    pid = mod.pid[(A, 0.5)]
    assert mod.klass(pid) == ("CHEAP", "sweep", mod.klass(pid)[2]) and abs(mod.klass(pid)[2] - 0.012) < 1e-9, "the class of the WORSE seed"
    assert PT.Model([r1, r0], units, cost).klass(pid)[:2] == ("CHEAP", "sweep")
    assert mod.free_fraction(A) == 0.25 and mod.free_fraction(A, [r0]) == 0.5 and mod.free_fraction(A, [r1]) == 0.25, "FREE on every run that is asked"
    # the same with full scoring: seed 0 loses 0.003 (FREE), seed 1 0.04 (COSTLY); the interval guard of the card: FREE needs the upper end of the loss interval <= 0.01 on every seed
    def score(run, loss, up):
        run.full_loss = lambda pid_, half, key="topo_F1", readouts=None: loss
        run.loss_upper = lambda pid_, key="topo_F1": up
    for l0, l1, u0, u1, want in ((0.003, 0.04, 0.005, 0.02, ("COSTLY", "full", 0.04)), (0.003, 0.0045, 0.008, 0.0099, ("FREE", "full + interval", 0.0045)), (0.003, 0.0045, 0.008, 0.0101, ("CHEAP", "full + interval", 0.0045)),
                                 (0.003, 0.0045, 0.008, None, ("FREE", "full, point estimate only", 0.0045)), (0.0045, 0.003, None, 0.002, ("FREE", "full, point estimate only", 0.0045)), (0.01, 0.003, 0.001, 0.001, ("CHEAP", "full", 0.01))):
        score(r0, l0, u0)
        score(r1, l1, u1)
        got = mod.klass(pid)
        assert got[:2] == want[:2] and abs(got[2] - want[2]) < 1e-12, (l0, l1, u0, u1, got, want)


def test_classes_greedy_prefixes_sizes():
    f = fx()
    units, cost = f["units"], f["cost"]
    assert [PT.classify(x, PT.FULL_LIMITS) for x in (0.005, 0.0051, 0.02, 0.0201, 0.05, 0.0501)] == ["FREE", "CHEAP", "CHEAP", "COSTLY", "COSTLY", "CRITICAL"]
    assert [PT.classify(x, PT.SWEEP_LIMITS) for x in (-0.1, 0.003, 0.0031, 0.03, 0.031)] == ["FREE", "FREE", "CHEAP", "COSTLY", "CRITICAL"]
    A, B, C = "jm.blocks.0.hid", "jm.blocks.1.skip", "je.blocks.0.att.h0"                                         # a ladder unit, a whole-block unit, a unit that saves no MACs (parameters only)
    L = {(A, 0.25): 0.006, (A, 0.5): 0.007, (A, 0.75): 0.030, (B, 1.0): 0.005, (C, 1.0): 0.0001}
    r = _synthetic_run(units, cost, lambda u, fr: L.get((u, fr), 0.5))
    mod = PT.Model([r, r], units, cost)
    d = r.delta(f"p{mod.pid[(A, 0.5)]:03d}", "sel")
    assert abs(d["d6"] - 0.007) < 1e-9 and d["dap"] == 0 and d["dpos"] == 0
    assert PT.sweep_free(dict(d6=0.003, dap=0.003, dpos=0.05)) and not PT.sweep_free(dict(d6=0.0031, dap=0, dpos=0)) and not PT.sweep_free(dict(d6=0, dap=0.0031, dpos=0)) and not PT.sweep_free(dict(d6=0, dap=0, dpos=0.051))
    k = mod.klass(mod.pid[(B, 1.0)])
    assert k[:2] == ("CHEAP", "sweep") and abs(k[2] - 0.005) < 1e-9 and mod.free_fraction(B) == 0.0 and mod.free_fraction(C) == 1.0 and mod.klass(mod.pid[(C, 1.0)])[0] == "FREE"
    order = mod.greedy(("maps", "pair"), r)
    names = [(i["unit"], i["f"]) for i in order]
    dm = lambda fr: mod.saved({A: fr})["macs"] / 1e9  # noqa: E731
    assert 0.006 / dm(0.25) > (0.007 - 0.006) / (dm(0.5) - dm(0.25)), "the test needs the second ladder step to look cheaper per GMAC than the first"
    assert names.index((A, 0.25)) < names.index((A, 0.5)) < names.index((A, 0.75)), "a unit's fraction comes only after its lower one"
    small = [k for k, i in enumerate(order) if i["dmacs"] < PT.SMALL_GMACS]
    assert small == list(range(len(order) - len(small), len(order))) and (C, 1.0) in names[-len(small):], "the increments that save less than 0.01 GMACs come last"
    # prefixes: the cut closest to a target share of the heads' MACs (joint savings); the end of the curve when the units run out
    tot = cost.all["macs_maps"] + cost.all["macs_pair"]
    cum, cands = {}, []
    for i in order:
        cum[i["unit"]] = i["f"]
        s = mod.saved(cum)
        cands.append(s["macs_maps"] + s["macs_pair"])
    for t, spec, sv in mod.prefixes(order, (0.10, 0.50, 5.0), tot, "heads"):
        assert abs(sv["macs_maps"] + sv["macs_pair"] - t * tot) == min(abs(c - t * tot) for c in cands)
    assert abs(mod.prefixes(order, (5.0,), tot, "heads")[0][2]["macs_maps"] + mod.prefixes(order, (5.0,), tot, "heads")[0][2]["macs_pair"] - cands[-1]) == 0, "units run out before the target: the end of the curve"
    s = PT.sizes([(0.3, 0.002), (0.6, 0.004), (0.9, 0.012), (1.2, 0.060)])                                         # slopes 0.0067, 0.0267, 0.16: the loss per GMAC more than doubles after the second prefix
    assert s["S_a"] == (0.6, 0.004) and s["S_b"] == (0.9, 0.012) and s["S_c"] == (0.6, 0.004), s
    assert PT.sizes([(0.3, 0.02)])["S_a"] is None
    assert PT.sizes([(0.3, 0.004), (0.6, 0.003), (0.9, 0.0035), (1.2, 0.0036)])["S_c"] is None, "noise (a rise of 0.0017 per GMAC after a fall) is no knee: the previous slope is floored at the 0.01 band per GMAC"
    assert PT.sizes([(0.3, 0.001), (0.6, 0.0011), (0.9, 0.0065), (1.2, 0.0066)])["S_c"] == (0.6, 0.0011), "a real rise (0.0003 -> 0.018 per GMAC) still gives the knee"


def test_greedy_orders_by_loss_per_gmac():
    f = fx()
    units, cost = f["units"], f["cost"]
    Ba, Bb, Bc, M = "jm.blocks.0.skip", "jm.blocks.1.skip", "jm.blocks.2.skip", "jm.m05.hid"
    L = {(Ba, 1.0): 0.004, (Bb, 1.0): 0.002, (Bc, 1.0): 0.006, (M, 0.25): 0.007}                                  # every other point loses 0.5: far behind
    r = _synthetic_run(units, cost, lambda u, fr: L.get((u, fr), 0.5))
    mod = PT.Model([r, r], units, cost)
    cand = {(u, fr): (v, mod.saved({u: fr})["macs"] / 1e9) for (u, fr), v in L.items()}
    by_ratio = sorted(cand, key=lambda k: cand[k][0] / cand[k][1])
    by_loss = sorted(cand, key=lambda k: cand[k][0])
    assert by_ratio != by_loss and by_ratio == [(Bb, 1.0), (M, 0.25), (Ba, 1.0), (Bc, 1.0)], (by_ratio, by_loss)    # the test has power: the order by loss alone differs
    assert [(i["unit"], i["f"]) for i in mod.greedy(("maps", "pair"), r)][:4] == by_ratio, "the cumulative curve takes the increments by loss per saved GMAC"


def test_plan_top_savers_are_heads_and_v_composition():
    f = fx()
    units, cost = f["units"], f["cost"]
    free = {("jm.blocks.0.hid", 0.25): 0.0, ("jm.blocks.0.hid", 0.5): 0.0, ("jm.blocks.0.hid", 0.75): 0.01, ("je.pair.1", 0.25): 0.001, ("jm.blocks.2.skip", 1.0): 0.0, ("je.paint", 1.0): 0.0, ("pyr.e3", 0.75): 0.0, ("pyr.s05+m05", 0.75): 0.0,
            ("pyr.s05+m05", 0.5): 0.0, ("pyr.s05+m05", 0.25): 0.0, ("jm.m05.hid", 0.25): 0.0, ("jm.m05.hid", 0.5): 0.0, ("jm.stream", 0.25): 0.0, ("je.scale.1", 0.5): 0.4}        # the pyramid stem 0.5 m is free and saves the most: no top-saver
    r = _synthetic_run(units, cost, lambda u, fr: free.get((u, fr), 0.5))
    mod = PT.Model([r, r], units, cost)
    pl = PT.plan(mod)
    top = [p for p in pl if p["name"].startswith("top-saver")]
    assert [list(p["set"].items())[0] for p in top] == [("jm.m05.hid", 0.5), ("jm.stream", 0.25), ("jm.blocks.2.skip", 1.0), ("jm.blocks.0.hid", 0.5)], "the four HEAD units with the largest saved GMACs among the sweep-FREE ones, at their largest FREE fraction"
    assert not any(u.startswith("pyr.") for p in top for u in p["set"]) and mod.free_fraction("pyr.s05+m05") == 0.75
    assert not any(p["id"] == 120 for p in pl), "v needs the handle-scale points scored in full"
    scale = [mod.pid[("je.scale.1", fr)] for fr in (0.25, 0.5, 0.75)]
    full = {scale[0]: 0.001, scale[1]: 0.004, scale[2]: 0.02}                                                        # scored in full on seed 0: 0.5 is the largest FREE fraction, whatever the sweep says (it says 0.4 there)
    r.full_loss = lambda pid, half, key="topo_F1", readouts=None: full.get(pid)
    v = [p for p in PT.plan(mod) if p["id"] == 120][0]["set"]
    assert v == {"jm.blocks.0.hid": 0.5, "jm.blocks.2.skip": 1.0, "jm.m05.hid": 0.5, "jm.stream": 0.25, "je.pair.1": 0.25, "je.paint": 1.0, "je.scale.1": 0.5}, v      # head units at their largest sweep-FREE fraction; no pyramid unit, no unit that is not FREE
    full[scale[0]] = 0.001
    full[scale[1]] = 0.006
    assert "je.scale.1" not in [p for p in PT.plan(mod) if p["id"] == 120][0]["set"], "no handle-scale fraction is FREE in full: the unit stays out of v"


def test_plan_ids_second_null_and_sets_report():
    f = fx()
    units, cost = f["units"], f["cost"]
    r = _synthetic_run(units, cost, lambda u, fr: 0.0 if u.startswith(("jm.", "je.blocks")) else 0.5)
    mod = PT.Model([r, r], units, cost)
    pl = PT.plan(mod)
    ids = [p["id"] for p in pl]
    assert ids[0] == 0 and pl[0]["set"] == {} and len(set(ids)) == len(ids) and {101, 102, 103, 104, 111, 112} <= set(ids), ids
    assert pl[-1] == dict(id=PT.NULL2, name="null (second pass)", set={}), "the second null is scored too (the spread control)"
    assert len({tuple(sorted(p["set"].items())) for p in pl[:-1]}) == len(pl) - 1, "a point set named twice is scored once"
    # the sets of the plan with synthetic full losses: S1 / S2 / v, the rule clauses of v (junction F1 loses 0.05 > 0.03) and the sizes from the prefixes
    A = "jm.blocks.0.hid"
    ent = [dict(id=101, name="S1 (heads-only, 10 %)", set={A: 0.25}), dict(id=102, name="S2 (heads-only, 20 %)", set={A: 0.5}), dict(id=120, name="v (all-FREE heads)", set={A: 0.25})]
    lm = {101: 0.002, 102: 0.02, 120: 0.004}
    for run in mod.runs:
        run.full_loss = lambda pid, half, key="topo_F1", readouts=None: lm.get(pid) if key == "topo_F1" else (0.0 if key == "dup_frac" else 0.05)
        run.loss_upper = lambda pid, key="topo_F1": 0.009
    rows, sz = PT.sets_report(mod, ent)
    assert rows[0]["s0_L_jud"] == 0.002 and rows[0]["s0_L_hi"] == 0.009 and rows[2]["rule_holds"]["s0"] == [True, True, False] and sz["S_a"][1] == 0.002 and sz["S_b"][1] == 0.02 and sz["S_c"] is None, (rows, sz)


def test_control_report_shares_differ():
    f = fx()
    units, cost = f["units"], f["cost"]
    lad = [u.name for u in units if len(u.fracs) > 1][:6]
    # (ranked loss, the three random draws, the reverse loss) at f = 0.5: ranked beats the MEAN of the draws in 3 of the 5 non-tied units (not the draws one by one: u0 loses to its first draw), the reverse is worse than ranked in 4 of 5; the sixth is tied
    spec = [(0.015, (0.005, 0.03, 0.02), 0.05), (0.01, (0.02, 0.02, 0.02), 0.05), (0.01, (0.005, 0.005, 0.005), 0.05), (0.02, (0.01, 0.01, 0.01), 0.001), (0.01, (0.03, 0.03, 0.03), 0.04), (0.0, (0.0, 0.0, 0.0), 0.0)]
    extra = {}
    for u, (a, draws, rv) in zip(lad, spec):
        extra.update({f"rand{s}_{u}": d for s, d in enumerate(draws)}, **{f"rev_{u}": rv})
    ranked = {u: s[0] for u, s in zip(lad, spec)}
    r = _synthetic_run(units, cost, lambda u, fr: ranked.get(u, 0.0) if fr == 0.5 else 0.0, extra)
    r.units = {u: p for u, p in r.units.items() if u in lad}
    rep = PT.control_report(r)
    assert rep["units"] == 5 and rep["tied"] == 1 and abs(rep["ranked_beats_random"] - 0.6) < 1e-9 and abs(rep["reverse_worse"] - 0.8) < 1e-9, rep
    # the proxy check uses rank ties: Spearman with the average rank (known answer 0.9487) and nan without variance
    assert abs(PT.spearman([1, 2, 2, 3], [1, 2, 3, 4]) - 4.5 / (4.5 * 5) ** 0.5) < 1e-12 and PT.spearman([1, 2, 3], [3, 2, 1]) == -1.0 and np.isnan(PT.spearman([1, 1, 1], [1, 2, 3])) and np.isnan(PT.spearman([1, 2], [1, 2]))


def test_run_pass_pair_curves_by_half():
    f = fx()
    half = np.array([0] * 5 + [1] * 11)                                                                              # the halves of the pass are the ones it is given (the real halves put these 16 frames in one scene)
    r, _ = PP.run_pass(f["model"], f["cache"], "cpu", half)
    stats = []
    for out, b in PP.forward_batches(f["model"], f["cache"], "cpu"):
        stats += PP.frame_stats(out, b)
    for name, idx in (("all", range(16)), ("sel", range(5)), ("jud", range(5, 16))):
        sc, lb = np.concatenate([stats[i]["scores"] for i in idx]), np.concatenate([stats[i]["labels"] for i in idx])
        assert r["pair"][name] == PP.pair_curve(sc, lb, sum(stats[i]["counts"][PP.COUNTS.index("pairs_n")] for i in idx)), name
    assert r["pair"]["sel"]["ap"] != r["pair"]["jud"]["ap"] and r["counts"] == [s["counts"] for s in stats] and r["kp"] == [s["kp"] for s in stats]


# ---------------------------------------------------------------- stored cards: the loss on the worse read-out, dup_frac, the halves, the bootstrap
STORED_E312 = {0: dict(snap="ends_card_snap_thr0.1", unsnap="ends_card_cnt_thr0.1"), 5: dict(snap="ends_card_snap_nogeom_thr0.1", unsnap="ends_card_a_snap_thr0.1")}      # the null (snapped / unsnapped) and an ablated point: other graphs of the same frames


def _stored_folders(tmp):
    """point folders of E312 s0 whose scored cards are stored cards of the run: two read-outs per point"""
    src = os.path.join(ROOT, "runs", "e312_p_s0")
    for pid, m in STORED_E312.items():
        d = os.path.join(tmp, PT.point_run("e312_p_s0", pid))
        os.makedirs(d)
        for ro, stem in m.items():
            os.symlink(os.path.join(src, stem + ".counts.pkl"), os.path.join(d, PT.card_stem(pid, PT.READOUTS["e312"][ro]) + ".counts.pkl"))
    c = pickle.load(open(os.path.join(src, "ends_card_snap_thr0.1.counts.pkl"), "rb"))
    sc = np.asarray(c["scenes"])
    uniq = sorted(set(sc.tolist()))
    r = PT.RunData.__new__(PT.RunData)
    r.half, r.family, r.run, r.runs_dir = np.array([uniq.index(s) % 2 for s in sc]), "e312", "e312_p_s0", tmp
    return r


def test_full_loss_on_stored_cards():
    """the pooled whole is the stored card; the halves are the 438 / 435 frames; `full_loss` is the null minus the point on the WORSE of the two judged read-outs (max, with both losses of different size and sign), for dup_frac the point minus the null"""
    import qgraph_eval as QE
    src = os.path.join(ROOT, "runs", "e312_p_s0")
    heads = {}

    def num(stem, key="topo_F1", idx=None):
        heads.setdefault(stem, pickle.load(open(os.path.join(src, stem + ".counts.pkl"), "rb"))["head"])
        return QE.graph_numbers([heads[stem][i] for i in (range(len(heads[stem])) if idx is None else idx)])[key]
    assert abs(num("ends_card_snap_thr0.1") - json.load(open(os.path.join(src, "ends_card_snap_thr0.1.json")))["numbers"]["head"]["topo_F1"]) < 1e-9, "the pooled per-frame counts reproduce the stored card"
    with tempfile.TemporaryDirectory() as tmp:
        r = _stored_folders(tmp)
        s = STORED_E312
        assert int((r.half == 0).sum()) == 438 and r.full(7, "snap", "sel") is None and r.full_loss(7, "sel") is None
        for half, h in (("sel", 0), ("jud", 1), ("all", None)):
            idx = None if h is None else np.where(r.half == h)[0]
            l1, l2 = num(s[0]["snap"], idx=idx) - num(s[5]["snap"], idx=idx), num(s[0]["unsnap"], idx=idx) - num(s[5]["unsnap"], idx=idx)
            by = r.full_loss_by_readout(5, half)
            assert abs(by["snap"] - l1) < 1e-12 and abs(by["unsnap"] - l2) < 1e-12 and abs(l1 - l2) > 1e-3
            assert abs(r.full_loss(5, half) - max(l1, l2)) < 1e-12, "the WORSE read-out is the larger loss"
            assert abs(r.full(0, "snap", half) - num(s[0]["snap"], idx=idx)) < 1e-12
        l1, l2 = num(s[0]["snap"]) - num(s[5]["snap"]), num(s[0]["unsnap"]) - num(s[5]["unsnap"])
        assert l1 > 0.01 > 0 > l2 and abs(r.full_loss(5, "all") - l1) < 1e-12, (l1, l2)                              # snapped +0.0119, unsnapped better by 0.0187: min would report a gain
        d1, d2 = num(s[5]["snap"], "dup_frac") - num(s[0]["snap"], "dup_frac"), num(s[5]["unsnap"], "dup_frac") - num(s[0]["unsnap"], "dup_frac")
        got = r.full_loss(5, "all", "dup_frac")
        assert abs(got - max(d1, d2)) < 1e-12 and got > 0 and abs(got - 0.0061) < 2e-4 and abs(d1 - d2) > 1e-3, (got, d1, d2)     # dup_frac: lower is better, so the loss is ablated minus null (+0.0061 on the snapped read-out), the worse read-out the larger
        j = r.full_loss(5, "all", "junction_F1")
        assert abs(j - max(num(s[0][x], "junction_F1") - num(s[5][x], "junction_F1") for x in ("snap", "unsnap"))) < 1e-12


def test_boot_on_stored_cards():
    """`boot_point` calls jdiag.boot with A = the point, B = the null; the loss interval is the negative of jdiag's A - B with its ends swapped (dup_frac: unchanged); written next to the card, kept when it exists; `loss_upper` is the larger upper end"""
    import contextlib
    import io
    import jdiag
    with tempfile.TemporaryDirectory() as tmp:
        r = _stored_folders(tmp)
        PT.boot_point("e312_p_s0", "e312", 5, tmp, n=20)
        d = os.path.join(tmp, PT.point_run("e312_p_s0", 5))
        files = sorted(x for x in os.listdir(d) if x.endswith(".boot.json"))
        assert files == ["ends_card_e345005_snap_thr0.1.boot.json", "ends_card_e345005_thr0.1.boot.json"], files
        stem0, stem = PT.card_stem(0, PT.READOUTS["e312"]["snap"]), PT.card_stem(5, PT.READOUTS["e312"]["snap"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):                                                    # the same bootstrap again (seed 0): the printed A - B numbers
            jdiag.boot(d, stem, os.path.join(tmp, PT.point_run("e312_p_s0", 0)), stem0, n=20)
        ab = {m.group(1): tuple(float(m.group(i)) for i in range(2, 6)) for m in map(PT.BOOT_LINE.match, buf.getvalue().splitlines()) if m}
        b = r.boot(5, "snap")["loss"]
        pt, mn, lo, hi = ab["topo_F1"]
        assert (b["topo_F1"]["point"], b["topo_F1"]["mean"], b["topo_F1"]["lo"], b["topo_F1"]["hi"]) == (-pt, -mn, -hi, -lo) and b["topo_F1"]["lo"] <= b["topo_F1"]["hi"]
        pt, mn, lo, hi = ab["dup_frac"]
        assert (b["dup_frac"]["point"], b["dup_frac"]["lo"], b["dup_frac"]["hi"]) == (pt, lo, hi) and abs(b["dup_frac"]["point"] - 0.0062) < 2e-4 and abs(b["topo_F1"]["point"] - r.full_loss_by_readout(5, "all")["snap"]) < 1e-4
        assert r.boot(5, "unsnap")["loss"]["topo_F1"]["point"] < 0 < b["topo_F1"]["point"]
        assert r.loss_upper(5) == max(r.boot(5, x)["loss"]["topo_F1"]["hi"] for x in ("snap", "unsnap")) and r.loss_upper(0) is None
        before = os.path.getmtime(os.path.join(d, files[0]))
        PT.boot_point("e312_p_s0", "e312", 5, tmp, n=20)                                        # an existing file is kept
        assert os.path.getmtime(os.path.join(d, files[0])) == before


def test_gpu_gate_wrapper_and_deterministic_setting():
    import types
    waited = []
    fake = types.SimpleNamespace(DEFAULT_MB=9000, wait_for_room=lambda who="", timeout_s=None: waited.append((who, timeout_s, os.environ.get("GPU_GATE_TIMEOUT_S"))) or 0.0, used_mb=lambda: 12000.0)
    old_mod, saved_env = sys.modules.get("gpu_gate"), {k: os.environ.pop(k, None) for k in ("E345_GATE_TIMEOUT_S", "GPU_GATE_MB", "GPU_GATE_TIMEOUT_S")}
    sys.modules["gpu_gate"] = fake
    try:
        PP.wait_for_gpu("x", "cpu")
        assert not waited, "no gate on the CPU"
        os.environ["GPU_GATE_TIMEOUT_S"] = "60"                                                     # the chains' variable: not this job's
        try:
            PP.wait_for_gpu("jprune_probe sweep", "cuda")
            raise RuntimeError("a full GPU after the wait must stop the job")
        except SystemExit as ex:
            assert ex.code == 75
        assert waited == [("jprune_probe sweep", 12 * 3600.0, None)] and os.environ["GPU_GATE_TIMEOUT_S"] == "60", waited
        os.environ["E345_GATE_TIMEOUT_S"] = "100"
        fake.used_mb = lambda: 5000.0
        PP.wait_for_gpu("again", "cuda")                                                            # room: goes on
        assert waited[-1][:2] == ("again", 100.0)
        fake.used_mb = lambda: None
        PP.wait_for_gpu("unreadable", "cuda")                                                       # nvidia-smi unreadable: the gate's own rule is to go on
        fake.used_mb = lambda: 12000.0
        os.environ["GPU_GATE_MB"] = "0"                                                             # the gate switched off
        PP.wait_for_gpu("off", "cuda")
    finally:
        for k, v in saved_env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        if old_mod is None:
            sys.modules.pop("gpu_gate", None)
        else:
            sys.modules["gpu_gate"] = old_mod
    assert PP.gpu_peak_mb("cpu") is None
    # the deterministic setting is part of sweep.json: a resumed job takes it, `--deterministic` against passes made without it is refused
    with tempfile.TemporaryDirectory() as tmp:
        assert PP.resolve_deterministic(tmp, False) is False and PP.resolve_deterministic(tmp, True) is True
        json.dump(dict(deterministic=True, passes={"null": {}}, rows=[1]), open(os.path.join(tmp, "sweep.json"), "w"))
        assert PP.resolve_deterministic(tmp, False) is True and PP.resolve_deterministic(tmp, True) is True
        json.dump(dict(deterministic=False, passes={"null": {}}, rows=[1]), open(os.path.join(tmp, "sweep.json"), "w"))
        assert PP.resolve_deterministic(tmp, False) is False
        try:
            PP.resolve_deterministic(tmp, True)
            raise RuntimeError("mixed settings must be refused")
        except SystemExit as ex:
            assert "WITHOUT --deterministic" in str(ex)
        json.dump(dict(deterministic=False, passes={}, rows=[1]), open(os.path.join(tmp, "sweep.json"), "w"))
        assert PP.resolve_deterministic(tmp, True) is True                                          # nothing was run yet: the setting may change
    PP.check_resume(dict(rows=[1], passes={"a": {}}, deterministic=True), [1], True)
    for rec, rows, det in ((dict(rows=[1], passes={"a": {}}, deterministic=True), [1], False), (dict(rows=[2], passes={}), [1], False)):
        try:
            PP.check_resume(rec, rows, det)
            raise RuntimeError("a resume on other frames / another setting must be refused")
        except AssertionError:
            pass
    was, env = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic), os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        PP.apply_deterministic()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and torch.are_deterministic_algorithms_enabled() and torch.backends.cudnn.deterministic
    finally:
        torch.use_deterministic_algorithms(was[0])
        torch.backends.cudnn.deterministic = was[1]
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        if env is not None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = env


