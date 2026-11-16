p = "evaluation/tests/test_jprune_b.py"
s = open(p, encoding="utf-8").read()
a = s.index("    ids = [PB.run_id(s, n, g, k)")
b = s.index("    assert PB.point_run(0, \"null\", True)")
s = s[:a] + '''    ids = {(s, n, g if n != "null" else None, k): PB.run_id(s, n, g, k) for s in (0, 1) for n in PB.POINTS for g in (True, False) for k in (None, *PB.READOUTS)}          # the null is the same pass in both variants
    assert len(set(ids.values())) == len(ids), "run ids collide"
''' + s[b:]
a = s.index("        put_cards(runs, \"e352g_t1_s0\", \"gt1\", vec(0.58), [vec(0.55)] * 3)")
b = s.index("TESTS = [")
s = s[:a] + '''        put_cards(runs, "e352g_t1_s0", "gt1", vec(0.58), [vec(0.55)] * 3)                               # a T1 that loses 0.0459 on B plain while the random mean (0.6170) loses 0.0089: the ranking is worse than random: KILL
        assert any("KILL" in x and "YES" in x for x in PB.checks(PB.table_data((0,), runs_dir=runs)))
        for d in range(3):
            put_cards(runs, f"e352g_r1d{d}_s0", f"gr1d{d}", vec(0.55), [vec(0.50)] * 3)                  # the random draws lose 0.0759: more than T1: no kill
        assert any("KILL" in x and x.endswith("no") for x in PB.checks(PB.table_data((0,), runs_dir=runs)))
        put_cards(runs, "e352g_t1_s0", "gt1", vec(0.60), [vec(0.55)] * 3)                               # T1 loses 0.0259 (not more than 0.03): no kill whatever the draws do
        for d in range(3):
            put_cards(runs, f"e352g_r1d{d}_s0", f"gr1d{d}", vec(0.62), [vec(0.50)] * 3)
        assert any("KILL" in x and x.endswith("no") for x in PB.checks(PB.table_data((0,), runs_dir=runs)))


''' + s[b:]
open(p, "w", encoding="utf-8").write(s)
