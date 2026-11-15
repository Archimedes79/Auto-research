p = "experiments/E324_e320-duplicates-refined/experiment.md"
s = open(p, encoding="utf-8").read()
assert "s is predicted +0.012 / +0.015 above the bound" in s
s = s.replace("s is predicted +0.012 / +0.015 above the bound", "s is predicted +0.012 / +0.017 above the bound", 1)
open(p, "w", encoding="utf-8").write(s)
# E320 section 3 reading 3 and E323: seed-1 labels
for q, old, new in (
 ("experiments/E320_soft-attach-replaces-snap/experiment.md", "seed 1: 10694 0.1532, 11849 0.1557, 12208 0.1589, 12503 0.1629", "seed 1: E316 10694 0.1532, E312 11849 0.1557, E320 12208 0.1589, E315 12503 0.1629"),
 ("experiments/E323_duplicates-operating-point/experiment.md", "seed 1 10694 0.1532, 11849 0.1557, 12208 0.1589, 12503 0.1629", "seed 1 E316 10694 0.1532, E312 11849 0.1557, E320 12208 0.1589, E315 12503 0.1629")):
    t = open(q, encoding="utf-8").read(); c = t.count(old); print(q, c)
    if c == 1:
        open(q, "w", encoding="utf-8").write(t.replace(old, new, 1))
