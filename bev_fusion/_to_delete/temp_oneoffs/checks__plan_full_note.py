p = "plan.md"; s = open(p, encoding="utf-8").read()
old = "Full budget (1500 sequences, 3000 steps) queued for '< 1 m/s'\n"
new = ("Full budget (1500 sequences, 3000 steps, 48 held-out): only 2 / 5 - velocity error 1.55 m/s (a floor: 5x the training does not lower it), raw 0.758 (GT 0.681, "
       "gaming persists), extrapolation 0.443 = 73 % of GT 0.606 (FAIL), prediction - raw 0.12 (FAIL: the stronger decoder carries motion again). The remaining "
       "gap is the method, not the budget; next: locate the 1.55 m/s (box edges vs interiors, hypothesis mixing, cells without Doppler), keep the decoder from "
       "carrying motion as training goes on (raw-warp weight, or the decoder on the raw move)\n")
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
