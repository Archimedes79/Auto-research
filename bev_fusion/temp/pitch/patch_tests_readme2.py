import sys
path = sys.argv[1]
s = open(path, encoding="utf-8").read()
old = "`--real [RUN]` is the golden check: `evaluate.py RUN --quant` and `bev_suite.py RUN eval` give the same metrics.json, card.json and table.txt (CPU, 4 frames, about 2 minutes, by hand; skips without a checkpoint)"
new = old + "; `--sets [RUN]` is the check of the named sets of `evaluation/bev_sets.py` (`--split large.val`, a name of `splits/`; CPU, 4 frames, about 6 minutes): `large.val` / `large.train` give the numbers of `val` / `train` number for number under files named by the set, and `small.val` evaluates exactly its own scenes (without option: a malformed name, TEST, a reserved split name, `--all` or `--time` with a set are refused)"
assert s.count(old) == 1, s.count(old)
open(path, "w", encoding="utf-8").write(s.replace(old, new))
print("ok")
