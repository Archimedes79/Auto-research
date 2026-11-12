p = "experiments/E285_junction-gt-layer/experiment.md"
s = open(p, encoding="utf-8").read()
old = "An end is PAINTED when a stop line of a type other than the turn stop lies within 3 m, or a pedestrian crossing within 4 m (turn stops are waiting points inside the junction, not entries). Entries painted 38.5 % (3284 of 8535; stop line alone 34.9 %, crossing alone 21.2 %), exits painted 28.3 % (2297 of 8114): **about 61 % of the entries and 72 % of the exits have no paint to see and must be inferred from context**"
new = "An ENTRY is PAINTED when a stop line of a type other than the turn stop lies within 3 m, or a pedestrian crossing within 4 m (turn stops are waiting points inside the junction, not entries); an EXIT when a pedestrian crossing lies within 4 m (corrected 2026-11-12 after the critic's finding: the stop line of the arm's own entry side lies next to its exits and made them look painted; first version 28.3 % of the exits). Entries painted 38.5 % (3284 of 8535; stop line alone 34.9 %, crossing alone 21.2 %), exits painted 18.5 % (1503 of 8114, crossing only): **about 61 % of the entries and 81 % of the exits have no paint to see and must be inferred from context**"
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("ok")
