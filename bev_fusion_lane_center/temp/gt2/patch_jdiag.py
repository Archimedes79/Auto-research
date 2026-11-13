p = "evaluation/jdiag.py"
s = open(p, encoding="utf-8").read()
old = "    assert abs(auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) - 0.5) < 1e-9 or True       # ties: rank formula, any value in [0, 1], not pinned\n"
assert old in s
s = s.replace(old, "", 1)
open(p, "w", encoding="utf-8").write(s)
