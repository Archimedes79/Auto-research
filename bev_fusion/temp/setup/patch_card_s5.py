"""One-off: write the synthetic-gate result (first full run of tests/test_s1_pipeline.py, 2026-10-31) into the S1a card, section 5, and the G4 -> G4b change into section 3."""
p = "experiments/s1a_lift_pd_vs_p/experiment.md"
s = open(p, encoding="utf-8").read()
a = "## 5. Summary\n(after the runs)"
assert a in s
new = """## 5. Summary
### 5.1 Synthetic gate, first full run (2026-10-31, `tests/test_s1_pipeline.py`, result file `temp/checks/s1_pipeline_result.json`, log `temp/checks/s1_pipeline_log.txt`)
Setup as in section 3: 300 train / 60 held-out synthetic scenes, 1,200 steps, batch 8, IoU at 0.5 m on camera-visible cells, run on the shared GPU (about 12 min per arm; the DualBEV-width preset 27 M parameters). Null (per-cell prior map, best threshold on train): lanes 0.318, vehicles 0.032, pedestrians 0.000, class mean 0.116.

| arm | lanes (0-20 / 20-40 / 40-60 m) | vehicles | pedestrians | class mean |
|---|---|---|---|---|
| ours PD, oracle depth, seed 0 / 1 | 0.700 (0.99 / 0.82 / 0.44) / 0.696 | 0.824 / 0.832 | 0.832 / 0.890 | 0.785 / 0.806 |
| ours P (no depth), seed 0 / 1 | 0.477 / 0.547 | 0.280 / 0.293 | 0.000 / 0.000 | 0.252 / 0.280 |
| ours PD, blurred depth (sigma 0.01 in 1/z) | 0.619 | 0.435 | 0.000 | 0.351 |
| ours PD, next sample's depth ("shuffled") | 0.685 | 0.477 | 0.000 | 0.388 |
| Simple-BEV preset (P, 8 slices) | 0.522 | 0.288 | 0.000 | 0.270 |
| LSS preset (S, heights summed) | 0.690 | 0.796 | 0.860 | 0.782 |
| DualBEV-style preset (13 heights summed, width 1.0) | 0.699 | 0.814 | 0.846 | 0.787 |

Gates (pre-registered): G1 wiring PASS (1.000 on every class); G2 learning PASS; G3 depth PASS (vehicles PD - P +0.54; paired scene bootstrap, seed 0, 60 scenes: lanes +0.223 [0.185, 0.260], vehicles +0.544 [0.479, 0.604], pedestrians +0.832 [0.731, 0.919]); G4 control FAIL (vehicles with the next sample's depth 0.477 against P 0.286, limit P + 0.03); G5a PASS (all three published presets learn); G5b PASS (ours 0.796 against the best preset 0.787, limit -0.02; the seed spread is sd 0.010, so ours is level with them, not ahead). Gate verdict: FAIL, by G4 alone.
Readings:
- With exact depth the depth weight is worth a lot (vehicles 0.28 -> 0.82, pedestrians 0 -> 0.83, lanes 0.51 -> 0.70); with a depth that is blurred like the frozen head's far range (blurred) it keeps about a quarter of it for vehicles (0.435): the quality of the depth head decides how much of this carries over to real images.
- Our design is anchored to the published ones: LSS splat, DualBEV-style summed heights with its full-width trunk and ours are level (0.782 / 0.787 / 0.785), Simple-BEV equals our P. The details of our lift (slice at z = 0, 4 sub-heights, heights as channels) show no measurable advantage on this task; the depth weighting is what matters.
- Why G4 failed (`temp/checks/check_shuffle_control.py`): the 'shuffled' depth is the next sample's histogram, and in the synthetic world the camera, the road and so the GROUND depth of a pixel group are the same in every sample: 89 % of the ground groups have an expected depth within 1 m of the truth (median error 0.00 m), but only 7 % of the vehicle groups (median error 10 m). So it is not a null: it still contains the flat-ground depth, and lanes reach 0.685 against 0.700 with exact depth. The control was wrong, the lift is not broken. G4 is replaced by G4b, a pixel-permuted control (the sample's own histograms at random other pixel groups: marginal kept, depth-pixel relation destroyed), added to the test with its arm; it has not run yet (needs a GPU slot; `python tests/test_s1_pipeline.py --arms permuted,ours P` runs it).
- A consequence worth testing on real data: a depth prior computed from the calibration alone (the flat-ground intersection of every pixel's ray with the road plane, no learned part, in line with R-GEO) would give the lane gain without a depth head; it breaks with slopes and pitch. Candidate extra arm of the real run: PD-flat against PD and P.
Not done yet: the real-data run (section 2), the idle-GPU timing (the retries of 2026-10-31 00:17 found the GPU busy), the re-run with the permuted control.
"""
s = s.replace(a, new.rstrip("\n"))
# section 3 gates table: G4 row
old = "| G4 control | PD with shuffled depth: vehicles <= P + 0.03 (the depth content carries the gain, not the extra weight) |"
assert old in s
s = s.replace(old, "| G4 control (FAILED, replaced) | PD with the next sample's depth: vehicles <= P + 0.03; failed 0.477 vs 0.286 because that depth keeps the scene-independent flat-ground depth (section 5.1) |\n| G4b control | PD with pixel-permuted depth (the sample's own histograms at random other pixel groups): vehicles <= P + 0.03 (added after G4 failed; not run yet) |")
open(p, "w", encoding="utf-8").write(s)
print("card patched")

p = "tests/README.md"
s = open(p, encoding="utf-8").read()
s = s.rstrip("\n") + "\n| `test_s1_pipeline.py` (+ `synthetic_s1_data.py`) | the camera-only S1 pipeline (`training/s1_camera.py` + `train_s1.py`) trained and tested on synthetic scenes in the real cache format at stride 16: arms ours PD (oracle / blurred / shuffled / permuted depth), ours P, and the published presets (Simple-BEV, LSS, DualBEV-style); pre-registered gates G1-G5 (card `../experiments/s1a_lift_pd_vs_p/experiment.md`) | G1 overfit, G2 above null, G3 PD - P >= 0.05, G4b permuted depth <= P + 0.03, G5 presets learn and ours >= best - 0.02 | first full run 2026-10-31: G1 / G2 / G3 / G5 PASS (vehicles PD 0.82 vs P 0.28; presets level with ours), G4 (next sample's depth) FAILED because that depth keeps the flat-ground depth; replaced by G4b, not run yet |\n"
open(p, "w", encoding="utf-8").write(s)
print("readme patched")
