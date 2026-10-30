"""one-off: record raw-warp 3 + prior 0.03 (2 seeds) in plan.md 7.3 and tests/README.md"""
p = "plan.md"
s = open(p, encoding="utf-8").read()
old = "prior 0.03 on 2 seeds and raw-warp 3 on seed 1 running\n"
new = old + ("    raw-warp 3 + prior 0.03 (seed 0 | seed 1): vehicles 1.41 / 0.94 | 1.59 / 1.05 per cell / per box, pedestrians 0.77 | 0.59, lanes 3.13 | 1.87 "
             "(baseline 1.69), raw 0.788 | 0.765, extrapolation 0.565 (88 %) | 0.580 (90.5 %), 3 / 5 on both (seed 1 misses the decoder-share claim by 0.006) - "
             "the best balance so far: the vehicle gain of raw-warp 3 with the lane gaming largely removed. Remaining gaps: per-cell scatter (1.4-1.6 m/s while the "
             "box means are ~1 m/s), lanes on seed 0, the raw map gamed (0.77-0.79 > GT 0.68)\n")
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))

p = "tests/README.md"
s = open(p, encoding="utf-8").read()
old = "rerun on the corrected data |"
new = ("rerun on the corrected data: best balance raw-warp 3 + zero-motion prior 0.03 (`--sharp --pyr --doppler --rawwarp 3.0 --prior 0.03`), 3 / 5 on both seeds - "
       "vehicles 1.41 / 1.59 m/s per cell, 0.94 / 1.05 per box mean, pedestrians 0.77 / 0.59, lanes 3.13 / 1.87, extrapolation 88 / 90.5 % of GT; baseline 0 / 5 "
       "(vehicles 5.70, lanes 1.69) |")
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("ok")
