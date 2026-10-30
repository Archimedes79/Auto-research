"""one-off: record the best box-test configuration (2 seeds) in plan.md 7.3 and tests/README.md"""
p = "plan.md"
s = open(p, encoding="utf-8").read()
old = "box means are ~1 m/s), lanes on seed 0, the raw map gamed (0.77-0.79 > GT 0.68)\n"
new = old + ("    BEST (2026-10-26): + edge-aware smoothness 0.05 (`--sharp --pyr --doppler --rawwarp 3.0 --prior 0.03 --smooth 0.05`), seed 0 | seed 1: vehicles "
             "1.39 / 0.94 | 1.55 / 1.05 m/s per cell / per box, pedestrians 0.44 | 0.43, lanes 2.06 | 2.27 (baseline 1.69), raw move 0.785 | 0.775, extrapolation "
             "0.680 | 0.660 (GT velocity 0.641: above the bilinear-blurred reference), 4 / 5 claims on BOTH seeds - only '< 1 m/s per cell' fails (the box means "
             "are at ~1 m/s; what is left is scatter inside the boxes). Baseline on the same data: 0 / 5, vehicles 5.70, extrapolation 0.296. No velocity labels "
             "anywhere; the anchors are the per-iteration Doppler update, the raw-warp explanation term, the zero-motion prior and the smoothness\n")
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))

p = "tests/README.md"
s = open(p, encoding="utf-8").read()
old = "baseline 0 / 5 (vehicles 5.70, lanes 1.69) |"
new = ("baseline 0 / 5 (vehicles 5.70, lanes 1.69); BEST + smoothness 0.05 (`--smooth 0.05`): 4 / 5 on both seeds - vehicles 1.39 / 1.55 m/s per cell, "
       "0.94 / 1.05 per box, pedestrians 0.44 / 0.43, lanes 2.06 / 2.27, extrapolation 0.68 / 0.66 (GT 0.64); only '< 1 m/s per cell' fails |")
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("ok")
