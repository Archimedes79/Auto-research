"""one-off: record the learned-gate results in plan.md 7.4 and tests/README.md"""
p = "plan.md"
s = open(p, encoding="utf-8").read()
anchor = "(Mirko's constraint above: DOGMa / Kalman only as ideas mapped onto learned grid blocks).\n"
assert s.count(anchor) == 1, s.count(anchor)
new = anchor + (
    "- Learned soft gate built (2026-10-30, tests/test_flow_explains.py --gate): per cell softmax(conv([features, fresh v, carried v, their image "
    "residuals |F0 - F1(landing)|])) -> start = w_fresh v_fresh + w_carried v_carried, zero start 0.5 / 0.5; recursive training on keyframe triples "
    "(pair 1, carried detached, pair 2, same loss terms); the comparison is within one model. Pair test (gated model): 4 / 5 on both seeds, vehicles "
    "1.10 / 1.19 m/s, extrapolation 0.645 / 0.653 (recursive training also lowered the fresh error: moving at F1 2.04 vs 2.44 pair-trained). 24 "
    "sequences x 5 keyframes, t = 1..3, seed 0 | seed 1: ALL vehicles (mostly parked) gated best on both seeds (0.90 0.98 1.10 | 0.84 0.90 1.11; "
    "fresh 0.91 0.97 1.12 | 0.85 0.93 1.16; recursive 0.94 1.08 1.21 | 0.88 0.96 1.15) - the gate removes the parked-car penalty; MOVING vehicles "
    "the hard recursive start stays best (1.40 1.48 1.65 | 1.23 1.26 1.61; fresh 1.76 1.86 2.13 | 1.68 1.81 2.30), the gate keeps none (seed 0: "
    "1.79 1.90 2.11) to half (seed 1: 1.44 1.45 1.82) of that gain. Reading: the gate's only evidence is the image residual, which the warp-loss "
    "gaming distorts. Next: the Doppler residual of each candidate (wrap(v . u - d) per cell, closed form, not gameable) as gate input.\n")
s = s.replace(anchor, new, 1)
open(p, "w", encoding="utf-8").write(s)

p = "tests/README.md"
s = open(p, encoding="utf-8").read()
old = "recursion (`--recursive`, `--sequences`): on moving vehicles 19-27 % below the fresh estimate, on parked cars slightly worse;"
assert s.count(old) == 1
new = (old + " learned soft gate carried / fresh (`--gate`, recursive training): best on all vehicles over 5-keyframe sequences on both seeds, "
       "but on moving vehicles below the hard recursive start;")
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("ok")
