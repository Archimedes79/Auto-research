p = "experiments/E322_soft-wiring/experiment.md"
s = open(p, encoding="utf-8").read()
old = "and E320's tight-radius read-outs (1.5 m / 1.0 m) test whether a learned end makes the wiring radius irrelevant. If they reach E312 snapped minus 0.01 the wiring is already learned by E320; if not, this card is the next end-to-end step."
new = "and E320's tight-radius read-outs (1.5 m / 1.0 m) test whether a contact wiring (a threshold around the attached end) can replace the choice among several pieces. That is still a distance rule on the output, not a learned wiring: if a tight read-out reaches E312 snapped minus 0.01 (with the guards) a contact threshold suffices at the read-out, if not the one-piece choice is needed; either way a learned, soft wiring is what this card is about."
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("e322 ok")
