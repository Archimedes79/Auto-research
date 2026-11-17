p = "experiments/E375_champion-per-class-decode/experiment.md"
s = open(p, encoding="utf-8").read()
a = "| VAL | the chosen cell once on VAL: the E367 heads (`E375.val_s<S>`), the E373 fresh heads (`E375.cf_s<S>`) | run.py (CPU) | cards |"
assert s.count(a) == 1
s = s.replace(a, "| VAL | the chosen cell once on VAL with `--ref-cell 0.2,0.2,0.6,off` on its FIRST score (the reference cell is not in the card name and cards are never overwritten: the paired interval exists only if the first score has it): the E367 heads (`E375.val_s<S>`), the E373 fresh heads (`E375.cf_s<S>`); before choosing, the 9 diagonal cells of each per-class table must equal the gate-off cells of the stored 24-cell tables (code critic of 0668d3f308: a cheap consistency check) | run.py (CPU) | cards |")
open(p, "w", encoding="utf-8").write(s)
print("ok")
