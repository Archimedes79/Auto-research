p = "evaluation/tests/test_jprune_b.py"
s = open(p, encoding="utf-8").read()
a = s.index("def test_scoring_lines_pass_the_stored_checks():")
b = s.index("def test_partial_dump_cannot_land_in_runs():")
new = '''def parse_line(line, root):
    """a printed scoring line as the namespace `jends_eval.py score` builds from it (`$R` = root): `--key value` pairs and bare flags"""
    toks = line.replace("$R", root).split()
    toks = toks[toks.index("score") + 1:]
    ns, i = {}, 0
    while i < len(toks):
        k = toks[i][2:].replace("-", "_")
        if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            ns[k], i = toks[i + 1], i + 2
        else:
            ns[k], i = True, i + 1
    ns["seed"], ns["thr"] = int(ns["seed"]), float(ns["thr"])
    return SimpleNamespace(**ns)


def test_scoring_lines_pass_the_stored_checks():
    t1 = one_pass("t1")
    lines = PB.score_lines(0, "t1")
    assert list(lines) == ["snap", "primary", "attach", "unsnapped"] and PB.score_lines(0, "t1", groups=True)["snap"].count("e352g_t1_s0") == 2
    flags = {"snap": {"snap"}, "primary": {"snap", "snap_nogeom", "attach"}, "attach": {"attach"}, "unsnapped": set()}
    with tempfile.TemporaryDirectory() as tmp:
        folder = os.path.join(tmp, "runs", "e352_t1_s0")
        PB.write_point(folder, 0, "t1", t1)
        os.symlink(os.path.join(ROOT, "runs", "e273_p_s0"), os.path.join(tmp, "runs", "e273_p_s0"))
        files = PB.card_files("t1")
        for k, line in lines.items():
            ns = parse_line(line, tmp)                                                                   # the printed line itself goes through the stored checks
            assert {f for f in ("snap", "snap_nogeom", "attach") if getattr(ns, f, False)} == flags[k] and ns.run == "e352_t1_s0" and ns.tag == "t1" and ns.spec == QE.DEFAULT_SPEC == "close:1.5@0.5", (k, ns)
            assert os.path.isfile(ns.full) and os.path.isfile(ns.base) and ns.full.endswith("predfull_t1_s0.pkl"), (k, ns.full, ns.base)
            name, spec, tag = QE.score_target(ns, folder)
            assert (name, spec, tag) == (files[k], "close:1.5@0.5", "t1"), (k, name)
        old, JEV.ROOT = JEV.ROOT, tmp                                                                  # `jends_eval.score`: the seed / pyramid checks pass on the point folder, the scorer gets the dump of B'
        seen, saved = [], QE.cmd_score
        QE.cmd_score = lambda ns: seen.append(ns)
        try:
            JEV.score(parse_line(lines["snap"], tmp))
        finally:
            QE.cmd_score, JEV.ROOT = saved, old
        assert len(seen) == 1 and seen[0].full.endswith("predfull_t1_s0.pkl") and seen[0].tag == "t1" and seen[0].skip_base_check


'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
