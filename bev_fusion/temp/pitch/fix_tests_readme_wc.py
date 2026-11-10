p = "tests/README.md"
s = open(p, encoding="utf-8").read()
a = "; the sets of `evaluation/bev_sets.py` (`--split large.val`, a name of `splits/`; a malformed name, TEST and `--all` with a set are refused; `--real` has its own gate: `--sets [RUN]`, CPU, about 6 minutes: `large.val` / `large.train` give the numbers of `val` / `train` number for number under files named by the set, and `small.val` evaluates exactly its own scenes)"
assert s.count(a) == 1, s.count(a)
open(p, "w", encoding="utf-8").write(s.replace(a, ""))
print("fixed")
