# Lane-marking diagnostics, 2026-11-05 / 06 (paint vs label, overfit on 8 scenes)

Mirko's order (2026-11-05): run both tests after the lane (Spurmitte) work, then shut down. Diagnostics only: no verdict, no TEST, rules "by hand: diagnostic" with the reading bands written before the run (in the ledger rows).
Question: why is the lane-marking IoU weak (val 0.43 at 0.5 m for lane_divider)? Hypotheses: (a) the labels do not lie on the visible paint (noise / registration), (b) loss or representation cannot carry a 1.3 % thin class.

## 1. Paint vs label (diag.paint_val, CPU, 16 s, tag bf/diag.paint_val)
Ground-plane projection of the stored GT lines (GT 6a7aa7864a) into the front camera image, rectified lateral brightness profiles within 1 m, stripe = 25 % contrast over the asphalt next to it; control = labels moved 1.75 m sideways.
249 val frames (3 per scene, 83 scenes). Command: `python temp/lane_tests/paint_vs_label.py --out temp/lane_tests/paint_out`; numbers in `paint_table.txt` / `paint_result.json` of that folder (scratch, kept by the run's tag).
A 12-frame functional check of the script preceded the run (not a result).

| layer, range | points | hit within +-0.5 m | control | hit within +-0.2 m | control |
|---|---|---|---|---|---|
| lane_divider 5-12 m | 1100 | 0.46 | 0.11 | 0.35 | 0.07 |
| lane_divider 12-20 m | 3190 | 0.41 | 0.23 | 0.34 | 0.15 |
| lane_divider 20-30 m | 5533 | 0.30 | 0.24 | 0.21 | 0.17 |
| road_divider 5-12 m | 790 | 0.74 | 0.13 | 0.61 | 0.07 |
| road_divider 12-20 m | 1239 | 0.59 | 0.12 | 0.44 | 0.09 |
| road_divider 20-30 m | 1408 | 0.42 | 0.14 | 0.32 | 0.10 |

Best lateral offset per frame (lane_divider, 154 frames): mean 0.075 m, SD 0.60 m, |d| > 0.5 m in 52 % of the frames; control SD 0.72 m. Road_divider (104 frames): SD 0.47 m, control 0.73 m.
Reading with the written rule (prediction: hit rate >= 0.6 and SD <= 0.25 m at 5-20 m for lane_divider): not met. The labels lie on paint more often than chance (0.41-0.46 vs 0.11-0.23), but with a lateral scatter of the order of half a metre, and beyond 20 m the lane_divider hit rate is near the control. Road dividers fit better. So label / registration noise at the 0.5 m scale is a real part of the picture, but the check cannot separate "wrong position" from "no visible paint" (worn, occluded, dark frames) and a 25 % contrast stripe detector is crude. By hand, no verdict.

## 2. Overfit on 8 training scenes (diag.overfit8_before / _s0 / _after)
Scenes (seeded, from the scenes the baseline trained on): scene-0410, 0242, 0069, 0446, 0019, 0534, 0259, 0467; 306 samples. Warm start from the t20 baseline (runs/t20_base_s0, step 6000), training on these scenes only, 600 steps, no flip augmentation, lr 1e-4 (warm-up 50), clips of 3 keyframes, batch 3, accumulation 2, EMA 0.95; evaluation of the same 306 frames before and after with `bev_suite.py ... --split train eval --scenes ...` (`temp/lane_tests/eval8.sh`; the "before" run is a pseudo run folder so that the original eval tables stay untouched).
Config `temp/compose/overfit8_s0.json` (from `overfit_prepare.py`). Numbers read from `runs/overfit8_*/eval/card_train.json` with `temp/lane_tests/read_overfit.py`.

| metric (seg_fov, dt 0, 306 train frames) | before | after 600 steps |
|---|---|---|
| lane_divider IoU@0.5 m | 0.463 | 0.791 |
| lane_divider IoU@1 m | 0.577 | 0.869 |
| lane_divider zero-crossing F1@0.5 m | 0.116 | 0.605 |
| lane_divider zero-crossing precision / recall @0.5 m | 0.894 / 0.062 | 0.990 / 0.436 |
| road_divider IoU@1 m | 0.719 | 0.876 |
| centreline IoU@0.5 m | 0.428 | 0.491 |
| drivable area IoU@0.5 m | 0.856 | 0.926 |

Reading with the written rule (>= 0.85 at 0.5 m: capacity carries the labels; <= 0.65: even 8 scenes not fitted; between: inconclusive, a longer run is the next step): 0.791 is in the middle band, inconclusive. The network clearly can fit these labels (+0.33 IoU@0.5 in 600 steps, precision ~0.99), the schedule ended (final lr 1e-7) before the fit reached the 0.85 target (whether it was still improving was not checked), recall of the thin line is the weak point (0.44 after, 0.06 before at the zero crossing).
Together with section 1: the train IoU ceiling is not obviously capped by label noise at 0.5 m (0.79 reached, 0.87 at 1 m), while the paint check says the labels scatter by ~0.5 m on the paint: consistent with the network memorising the label positions on these scenes. A longer overfit run (2000+ steps) would show whether 0.85 at 0.5 m is reachable; not run.

## Deviations
- The overfit training was planned pinned to the code of bf/t21.ctl_s0; `run.py` refuses `--pin` for a first attempt of an id, so it ran unpinned on the working tree (snapshot tag bf/diag.overfit8_s0), which holds the t22 stage-B change (commit eebd9640: `camera_student`, off by default, not set in this config). The first chain pass (chain2.sh) lost the training and the evaluation after it to that refusal; chain3.sh repeated both (ledger rows of this record are the second pass of the "after" id).
- The GPU part waited for the lane chat's E275 scoring (flag SCORED); its first scoring failed (E275.n_s0), the flag never came, and the fallback after 01:30 started this chain at 02:54 on a free GPU.
- The shutdown after the tests was Mirko's explicit order.
