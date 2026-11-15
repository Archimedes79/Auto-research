import io
p = "experiments/E291_sd-road-level-lane-counts/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
for a, b in (("in 35 - 40 % of the arm sides (deficit + surplus)", "in 32 - 40 % of the arm sides (deficit + surplus)"), ("(19 - 22 % of the found keypoints at the loose setting)", "(18 - 22 % of the found keypoints at the loose setting)")):
    assert s.count(a) == 1, a
    s = s.replace(a, b)
io.open(p, "w", encoding="utf-8", newline="").write(s)
