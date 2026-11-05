p = "evaluation/README.md"
t = open(p, encoding="utf-8").read()
a = "; `temp`-free mutation check: 12 planted bugs, 12 caught |"
assert t.count(a) == 1
t = t.replace(a, "; `bev_mutants.py` checks the test itself |")
b = "| `bev_response.py` |"
row = "| `bev_mutants.py` | mutation check of `bev_response.py`: plants one deliberate bug at a time into the metrics (12 monkeypatches: AUC inverted, ECE unweighted, Brier without + P, RMSE without the root, ASE with a square root, AVE as L1, ATE x 1.1, window to 70 m, TP threshold 1 m, AOE without the wrap, moving cells = all cells, camera mask = every cell) and counts the rows that go off; a mutant with 0 rows off ESCAPED, i.e. the test could not have found that bug (about 8 min CPU, on demand) | all 12 caught |\n"
assert t.count(b) == 1
t = t.replace(b, row + b)
open(p, "w", encoding="utf-8", newline="\n").write(t)
print("ok")
