"""one-off: record seed 1 and the baseline of the box test on the corrected data in plan.md 7.3"""
p = "plan.md"
s = open(p, encoding="utf-8").read()
old = "decoder share 0.145 (lanes make it harder). Queued: zero-motion prior 0.03 (lanes), raw-warp 3.0 (decoder share)\n"
new = old + ("    seed 1: 2 / 5 as well - vehicles 2.02 per cell, 1.28 per box, interiors 2.05 / edges 2.01, pedestrians 0.99, lanes 4.73, raw 0.705, extrapolation "
             "0.496. Baseline on the corrected data (seed 0): vehicles 5.70 per cell, 4.45 per box, pedestrians 1.62, lanes 1.69, raw 0.389, extrapolation 0.296. "
             "The best setup is far better on everything that moves but WORSE on the lanes (3.4 / 4.7 vs 1.7 m/s): its refinement and raw-warp fit the picture "
             "by sliding lane cells along themselves (aperture gaming) - the zero-motion prior is the next test\n")
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("ok")
