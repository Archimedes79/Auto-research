import io
p = "experiments/E368_s2-cross-attention-to-the-past/experiment.md"
s = io.open(p, encoding="utf-8").read()
a = "capacity control `c` not run (r - z stayed below +0.005 on seed 1 and reached only +0.0042 on seed 0: the trigger needs +0.005 on one seed: not met)."
b = "capacity control `c` not run (r - z was +0.0042 / +0.0023, below the +0.005 trigger on both seeds). **A noise remark:** the seed twin differs from z by -0.0279 in lanes found (and +0.0139 in junction F1), more than r - z shows there (+0.0004 / +0.0131): the lanes-found gain of seed 1 is inside the head-seed noise; only topo F1 and F1@1.0 are tight (-0.0033 / -0.0056)."
assert s.count(a) == 1
io.open(p, "w", encoding="utf-8").write(s.replace(a, b))
