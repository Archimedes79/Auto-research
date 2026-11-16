p = "HANDOVER.md"
s = open(p, encoding="utf-8").read()
a = "5. Simplification: `day13` runs the removal arms E331 (evidence off, resumes), E332 a, E333, E334 o3 (rule `evaluation/jrule_removal.py`, adopt = loss <= 0.01 both seeds);"
assert s.count(a) == 1
b = ("5. Simplification: `day14` runs the removal arms (rule `evaluation/jrule_removal.py`, adopt = loss <= 0.01 on both seeds and both read-outs): **E331 (evidence off) NOT ADOPTED 2026-11-14** (unsnapped, differentiable read-out -0.0232 / -0.0124 topo F1; snapped within bounds; pair precision at E312 recall +0.028 / +0.022: the loss is at the p >= 0.1 operating point, not in the ranking; evidence stays), E332 a running, then E333, E334 o3;")
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(a, b))
