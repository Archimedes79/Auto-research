p = "experiments/E324_e320-duplicates-refined/experiment.md"
s = open(p, encoding="utf-8").read()
for a, b in (("(bootstrap [+0.0084, +0.0220] excludes 0 and the bound's side only just: the bound 0.01 is inside the interval)", "(bootstrap [+0.0084, +0.0220]: it excludes 0, and the bound 0.01 lies inside it)"),
             ("the learned attachment recovers about a quarter of what the snap's geometry gives.", "the learned attachment recovers a quarter to a third of what the snap's geometry gives (26 % / 34 %).")):
    assert s.count(a) == 1, a[:40]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
