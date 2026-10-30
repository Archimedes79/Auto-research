"""one-off: record the prior 0.03 and raw-warp 3 box-test runs on the corrected data in plan.md 7.3"""
p = "plan.md"
s = open(p, encoding="utf-8").read()
old = "by sliding lane cells along themselves (aperture gaming) - the zero-motion prior is the next test\n"
new = old + ("    seed 0, corrected data (vehicles per cell / per box, pedestrians, lanes, raw (GT 0.68), extrapolation (GT 0.64), claims): raw-warp 1 = 1.64 / 0.94, "
             "0.95, 3.37, 0.725, 0.524, 2 / 5; + prior 0.03 = 1.71 / 1.15, 0.58, 2.01, 0.730, 0.523, 2 / 5 (the prior cuts the lane gaming 40 % and helps slow "
             "pedestrians); raw-warp 3 = 1.47 / 0.95, 0.60, 4.99, 0.788, 0.590 (92 %), 4 / 5 - only '< 1 m/s per cell' fails, decoder share 0.08 - but the lanes "
             "are gamed more (5.0) and the raw map too (0.79 > GT). Trade-off: raw-warp helps vehicles and the extrapolation, the prior the lanes; raw-warp 3 + "
             "prior 0.03 on 2 seeds and raw-warp 3 on seed 1 running\n")
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("ok")
