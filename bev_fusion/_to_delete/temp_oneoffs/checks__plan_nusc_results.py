"""one-off: record the nuScenes box-test and recursion results in plan.md 7.4 and tests/README.md"""
p = "plan.md"
s = open(p, encoding="utf-8").read()
old = "refinement (recursive speed refinement by comparing the next frame). Runs: baseline and the best synthetic setup, 2 seeds.\n"
assert s.count(old) == 1
new = old + (
    "- Results on real nuScenes GT (2026-10-26 evening, quick budget, held-out scenes, 16 three-keyframe sequences, VAL-like scene split; no velocity "
    "labels in training): baseline 0 / 5 claims - vehicles 3.85 m/s per cell / 2.95 per box, moving vehicles 8.33, raw move 0.51 (worse than v = 0: "
    "0.53; the decoder carries the motion); best synthetic setup (--sharp --pyr --doppler --rawwarp 3 --prior 0.03 --smooth 0.05) 4 / 5 - vehicles "
    "1.17 / 0.91, moving 2.29, static 0.50, pedestrians 0.45, lanes 0.41, raw move 0.73 (GT velocity 0.71), extrapolation 0.63 (96 % of 0.65), "
    "prediction - raw 0.05; only '< 1 m/s per cell' fails. The synthetic findings transfer to real boxes.\n"
    "- Recursive speed refinement (carried estimate of the previous pair as the START of the next pair's refinement), moving vehicles at F1: "
    "recursive 2.00 vs fresh 2.44 vs carried only 3.21 (baseline: 6.86 / 8.56 / 6.60). Over 24 held-out sequences of 5 keyframes: on moving "
    "vehicles recursive beats fresh at every step by 19-27 % (t = 1..3: 1.37 / 1.48 / 1.72 vs 1.69 / 1.80 / 2.34); the error does not keep "
    "falling with t (the fresh curve rises too: later keyframes are harder); on ALL vehicles (mostly parked) recursion is slightly worse (+0.08 to "
    "+0.15 m/s): small wrong speeds of static cars are carried and compound. Next: do not use the carried estimate as a hard start - weight carried "
    "vs fresh per cell by how well each explains the new frame (DOGMa-style likelihood weighting / a Kalman prior with uncertainty).\n")
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)

p = "tests/README.md"
s = open(p, encoding="utf-8").read()
old = "baseline 0 / 5 (vehicles 5.70, lanes 1.69); BEST + smoothness 0.05"
assert s.count(old) == 1
new = ("REAL nuScenes GT (`--data nuscenes`, held-out scenes): baseline 0 / 5 (vehicles 3.85, moving 8.33 m/s), best setup 4 / 5 (vehicles 1.17 per "
       "cell / 0.91 per box, moving 2.29, extrapolation 96 % of GT); recursion (`--recursive`, `--sequences`): on moving vehicles 19-27 % below the "
       "fresh estimate, on parked cars slightly worse; synthetic: baseline 0 / 5 (vehicles 5.70, lanes 1.69); BEST + smoothness 0.05")
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("ok")
