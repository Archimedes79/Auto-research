p = "splits/README.md"
s = open(p, encoding="utf-8").read()
old = "It has no list of its own: a new set for the evaluation is a new file here."
assert s.count(old) == 1, s.count(old)
open(p, "w", encoding="utf-8").write(s.replace(old, "It has no list of its own: a new set for the evaluation is a new file here. A split is not named `val`, `train` or `test` and does not start with `warm` or `ctl` (`evaluation/bev_sets.py` refuses such a file: the names would read like the bare sets, TEST and the output tags)."))
print("ok")
