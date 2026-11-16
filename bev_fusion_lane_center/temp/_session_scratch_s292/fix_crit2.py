import io
p = "experiments/E343_test-e324s/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
a = " The paired intervals of E312 on these fresh dumps are the stored VAL ones to the digits printed (snapped minus plain +0.0564 [+0.0423, +0.0694] / +0.0615 [+0.0445, +0.0795])."
assert s.count(a) == 1
s = s.replace(a, "")
io.open(p, "w", encoding="utf-8", newline="").write(s)
