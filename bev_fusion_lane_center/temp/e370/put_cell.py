p = "experiments/E370_e365-rule-distilled-init/experiment.md"
s = open(p, encoding="utf-8").read()
a = "its VAL read-out follows below as a diagnosis."
assert s.count(a) == 1
s = s.replace(a, "its VAL read-out (runs `E370.cell_b_s0 / s1`, a diagnosis for the cross-card comparison, never this card's verdict): topo F1 0.6585 / 0.6627, F1@1.0 0.835 / 0.838, lanes found 0.681 / 0.682, junction F1 0.218 / 0.213, dup_frac 0.056 / 0.063: at the S1 champion's level (0.6617 / 0.6767 at the same cell), i.e. the rule-distilled start ends where the random start ends once the operating point is set by the decode; no advantage over the champion (b_bal25: 0.6698 / 0.6769 on its fresh heads).")
open(p, "w", encoding="utf-8").write(s)
print("ok")
