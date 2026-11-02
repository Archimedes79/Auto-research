p = "evaluation/README.md"
t = open(p, encoding="utf-8").read()
def rep(a, b):
    global t
    assert t.count(a) == 1, (a[:60], t.count(a))
    t = t.replace(a, b)
rep("| `bev_gtcheck.py --selftest` (ceilings, null = prevalence and AUC 0.5, noise lowers AUC) |", "| `bev_curves.py --selftest` (AP / AUC against the exact curves of sorted scores for prevalence 30 / 1 / 0.3 %, Brier, additivity, one-class / NaN / float16) and `bev_response.py --selftest` |")
rep("`--selftest`: 86 numbers, in `tests/run_gates.sh` |", "`--selftest`: 144 numbers incl. the role-swapped rows (damaged set as GT, original GT as prediction), in `tests/run_gates.sh`; `temp`-free mutation check: 12 planted bugs, 12 caught |")
rep("then the KNOWN GAP rows (detections on unobserved boxes still count as false, D38)", "then the ROLE-SWAPPED rows (symmetric numbers stay, precision and recall trade places) and the KNOWN GAP rows (detections on unobserved boxes still count as false, D38)")
open(p, "w", encoding="utf-8", newline="\n").write(t)
print("readme patched")
