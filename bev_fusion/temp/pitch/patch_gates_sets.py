import sys

mode = sys.argv[1]            # gates | testsreadme
path = sys.argv[2]
s = open(path, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


if mode == "gates":
    rep('''run "module timing"     "$PY" evaluation/bev_timing.py --selftest       # toy modules with known sleeps: per call / per frame, calls, hooks removed, training split, idle rule; CPU
''', '''run "module timing"     "$PY" evaluation/bev_timing.py --selftest       # toy modules with known sleeps: per call / per frame, calls, hooks removed, training split, idle rule; CPU
run "eval sets"         "$PY" evaluation/bev_sets.py --selftest         # the sets of the evaluation = the splits of splits/ (grammar, TEST refused, tags, lists, a toy folder); no data
''')
    rep('''    run "evaluate door, golden" "$PY" tests/test_evaluate_door.py --real   # 4 val frames on the CPU, number for number; about 3 min, 2 x 6 GB
''', '''    run "evaluate door, golden" "$PY" tests/test_evaluate_door.py --real   # 4 val frames on the CPU, number for number; about 3 min, 2 x 6 GB
    run "evaluate door, sets"   "$PY" tests/test_evaluate_door.py --sets   # large.val / large.train against val / train, small.val's own scenes; 4 frames on the CPU; about 6 min, 2 x 6 GB
''')
else:
    rep("`--real [RUN]` is the golden check: `evaluate.py RUN --quant` and `bev_suite.py RUN eval` give the same metrics.json, card.json and table.txt (CPU, 4 frames, about 2 minutes, by hand; skips without a checkpoint)",
        "`--real [RUN]` is the golden check: `evaluate.py RUN --quant` and `bev_suite.py RUN eval` give the same metrics.json, card.json and table.txt (CPU, 4 frames, about 2 minutes, by hand; skips without a checkpoint); the sets of `evaluation/bev_sets.py` (`--split large.val`, a name of `splits/`; a malformed name, TEST and `--all` with a set are refused; `--real` has its own gate: `--sets [RUN]`, CPU, about 6 minutes: `large.val` / `large.train` give the numbers of `val` / `train` number for number under files named by the set, and `small.val` evaluates exactly its own scenes)")
    rep("| `test_evaluate_door.py` |", "| `test_evaluate_door.py` |")
open(path, "w", encoding="utf-8").write(s)
print("ok")
