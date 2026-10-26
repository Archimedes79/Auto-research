# t1 - the combined camera + radar BEV system, trained once on nuScenes (report of 2026-10-26)

Everything below is measured on the nuScenes VAL split (82 scenes, 3109 keyframes: the parts downloaded so far), EMA weights, front
camera + five radars, 0.5 m grid, 60 m range. Numbers come from `results/` (copies of `runs/<run>/eval/*.json`, built by
`temp/train/build_tables.py`); nothing is typed by hand except this text. The TEST split was not used. One seed per run.

## 1 Summary

- **One combined system was built and trained once** (6000 optimiser steps): frozen DualBEV image heads
  -> lift PD -> camera temporal fusion with a carried world memory -> radar BEV -> concat fusion -> ResNet/FPN trunk -> heads
  decoded at t - 0.5 s, t, t + 0.5 s through the splat -> object head with a direction class and the S5c box refiner.
  It is compared with the window-only baseline `t1_r0` (5000 steps) that was trained before the fixes.
  (Training took about 5.8 h wall time on the RTX 3060, with other jobs on the machine for part of it.)
- **It is not a better system overall.** The vehicle heading error fell from 1.78 to 0.59 rad (the fix works), and the carried
  memory helps outside the camera view (drivable area +0.021 IoU, lane-divider masks +0.036, same frames, memory vs empty
  memory). But the map layers are 0.00-0.04 lower than the baseline and the box AP is 0.01-0.07 lower (paired-scene 95 %
  intervals exclude 0 for walkway, lane dividers and vehicle / pedestrian AP, drivable area is a tie). By the decision rule written
  in the card (no significant loss and a gain above the 0.010 band) the combination is **not adopted as an improvement**; its
  two working parts (direction head, memory) are worth keeping and testing alone.
- The refiner (S5c) lowers the nuScenes-style AP (vehicle 0.632 -> 0.596, pedestrian 0.199 -> 0.138, because its quality
  re-scoring reorders the detections) while it improves the box size (ASE 0.184 -> 0.161) and slightly the centre (ATE 0.59 ->
  0.58 m): use it without re-scoring, or not.
- Speed is unchanged: 43.4 ms (baseline) vs 43.9 ms (combined) per frame for the chain at dt 0 (62.8 ms for three time offsets),
  +3.6 ms for decoding and refining the boxes, +17.3 ms for the camera heads, idle RTX 3060, batch 1.
- Caveat: one training seed per run. The bootstrap resamples validation scenes, not training seeds; differences of about 0.02
  could be seed noise. The causes of the small losses are not isolated (section 6).

## 2 What was built, and the defects fixed first

Architecture: `images/architecture.png` (blue = trained, grey = frozen / cached, green = outputs, sand = training only).

| defect found (evidence) | what the combined run does about it |
|---|---|
| vehicle heading not learned: error 1.77 rad, loss at the level of random directions, cos sign agreement 0.45-0.52 | object head with sign-free axis loss + front / back class (SECOND); self-test; **fixed: 0.59 rad** (median error of matched vehicles at step 1000: 0.07 rad against 2.99) |
| the world memory was never used in training (every earlier run saw only the 3-frame window) | sequence training on clips of 3 consecutive keyframes, memory carried with the ego step, gradient through it (BPTT); memory written only where the camera sees (bug of the first memory code: the empty BEV outside the view overwrote what was seen); memory confidence cap 8 -> 2 (the fuse conv saw confidence values that never occur in a 3-keyframe clip); **works: +0.02 outside the view** |
| dataset loader limited training to 0.35-0.6 s per sample (radar sweeps read from 35 files) | exact per-sample radar cache (identical values, self-tested), 1 ms per sample |
| a bug of mine on the way: the cached dataset lost its feature attachment after the cache edit (crash on the first batch) | fixed, the self-test now loads a whole sample |
| box quality | S5c refinement trained in the chain (see the caveat above: not a gain in AP) |
| not fixed, reported: line zero-crossing decoding (baseline at step 2000: the predicted distance on GT lines has a median of 1.36 m and never reaches the 0.25 m of the decoder; final F1 0.068 baseline / 0.029 combined), the motion head (its gradient into the trunk stays near 1e-6 for the first 2000 steps of the baseline and 3000 of the combined run), the radar smear of fast movers inside the 0.5 s window | no code defect found in the loss; these need more training or a different design |

## 3 Setup

- Data: nuScenes trainval parts 1-6 (428 train scenes = 16,289 keyframes, 82 val scenes = 3,109 keyframes), front camera
  with two past frames (stride 2 at 12 Hz), five radars with all sweeps of the last 0.5 s, GT = HD-map layers and annotated boxes
  of the keyframes k - 1, k, k + 1 (decoded dt = -0.5 / 0 / +0.5 s with the true keyframe gap and the gap-mean ego motion).
- Baseline `t1_r0`: 5000 steps x 16 samples (4 x 4), Kendall task weights, AdamW 3e-4 cosine, EMA 0.999, window-only.
  Combined `t1_r3_combined`: 6000 steps x 18 samples (2 clips x 3 keyframes, 3 accumulation steps), otherwise identical; code
  pinned at commit 04a7de9 (`runs/t1_r3_combined/code`). Both: 4.9 / 6.6 passes over the training keyframes.
- The run was shortened from 8000 to 6000 steps at the step-1000 checkpoint (3.4 s per step instead of 2.6 s).
- Baselines of the comparison: prior map (per-cell mean of the GT over 500 training keyframes) and the null prediction (IoU 0).

## 4 Results (VAL, dt = 0 unless stated)

| metric | prior map | baseline t1_r0 | combined | combined - baseline, paired-scene 95 % interval |
|---|---|---|---|---|
| drivable area IoU | 0.429 | 0.699 | 0.696 | -0.003 [-0.009, +0.003] (tie) |
| walkway IoU | 0 | 0.278 | 0.261 | -0.017 [-0.025, -0.010] |
| lane-divider mask IoU @ 0.5 m | 0 | 0.327 | 0.306 | -0.021 [-0.031, -0.012] |
| lane-divider zero-crossing F1 @ 0.5 m | - | 0.068 | 0.029 | -0.040 [-0.054, -0.026] |
| vehicle coverage IoU | 0 | 0.435 | 0.440 | +0.005 [+0.001, +0.011] |
| vehicle AP @ 2 m (plain boxes) | 0 | 0.643 | 0.632 | -0.011 [-0.018, -0.004] |
| pedestrian AP @ 2 m | 0 | 0.214 | 0.199 | -0.015 [-0.027, -0.003] |
| two-wheeler AP @ 2 m | 0 | 0.258 | 0.190 | (not in the bootstrap set) |
| mAP (3 classes) | 0 | 0.307 | 0.282 | - |
| vehicle heading error AOE (rad) | - | 1.78 | 0.59 | - |
| vehicle size error ASE / centre error ATE (m) / velocity error AVE (m/s) | - | 0.168 / 0.60 / 1.92 | 0.184 / 0.59 / 2.32 (refined boxes: 0.161 / 0.58 / 2.33) | - |
| vehicle AP @ 2 m at t - 0.5 s / t + 0.5 s | - | 0.479 / 0.411 | 0.451 / 0.387 | - |
| drivable area IoU at t - 0.5 s / t + 0.5 s | - | 0.724 / 0.683 | 0.711 / 0.680 | - |

Pre-registered predictions of the card (written before the run): drivable IoU >= 0.60 and above the prior + 0.15, vehicle AP@2 m >= 0.25,
t + 0.5 s within 0.05 IoU of t - all three hold for both runs.

Does the carried memory help? (600 val clips of 3 keyframes, last keyframe, same frame with the memory carried vs from an empty
memory; `results/memory.json`):

| at dt = 0 | with memory | empty memory | difference |
|---|---|---|---|
| drivable area IoU, cells **outside** the current camera view | 0.567 | 0.546 | +0.021 |
| walkway IoU, outside the view | 0.255 | 0.240 | +0.015 |
| lane-divider mask IoU @ 1 m, outside the view | 0.358 | 0.322 | +0.036 |
| drivable area IoU, inside the view | 0.696 | 0.695 | +0.001 |
| vehicle AP @ 2 m | 0.628 | 0.626 | +0.002 |

At t - 0.5 s the memory also gives vehicle AP@2 m +0.031 and pedestrian AP@2 m +0.026. The training-time paired loss agrees: the last
keyframe scores 7.595 with and 7.689 without the memory at step 6000. The loss totals of the two runs themselves are not comparable
(the combined loss contains the refinement terms and is measured on clips); only the paired pairs inside one run are.

All tables, with the paired bootstrap, the time offsets and the latency: `results/tables.md`.
Training curves (loss groups, task weights, gradient norms): `images/curves.png`.

## 5 What the images show (`images/compare_*.png`: baseline on top, combined below, same validation frame)

- Both systems find the lead vehicle and the two-wheeler in view at the right place and size; both paint the road, the walkway
  and some crossings, but with too much drivable area on grass and parked-car zones (the red false positives in the error map are
  nearly identical in both).
- The road-divider and lane-divider lines are fainter in the combined system (frame 1332: the baseline's red divider dashes
  are clearly visible, the combined ones barely), which matches the lower line IoU.
- With the refiner the combined system sometimes draws several overlapping candidate boxes at t + 0.5 s next to a vehicle at the
  edge of the view (frame 444) - the cause of its lower AP; a nearby car at the lower right of that frame is boxed by the baseline and
  only blobbed by the combined system.
- The splat mass at t + 0.5 s marks the unknown cells at the far edge (and behind moving objects) in both.

## 6 What it says, weaknesses, next steps

- The two intended fixes work in isolation: the heading (AOE 1.78 -> 0.59 rad) and the memory (+0.02 outside the view, +0.09 in the
  paired loss). The package as a whole lost 0.01-0.04 on lines, walkway and AP. Candidate causes, **not tested**: (1) a clip batch
  has 2 scenes x 3 correlated keyframes, so BatchNorm and the gradient see less diversity than 4 independent samples; (2) the refiner
  terms enlarge the objects group (its Kendall weight ended lower: s 0.91 against 0.77) and its gradient into the trunk
  (about 3 x larger at step 500: 0.18 against 0.053) takes more of the clipped step; (3) one seed each: the run-to-run
  spread was not measured.
- Weaknesses of this experiment: one seed per run; not an ablation (all changes at once, as requested); the combined run saw
  35 % more samples (108,000 against 80,000); the bootstrap resamples scenes only; the line zero-crossing F1 is low in both (0.07 / 0.03), so the line
  numbers should be read from the mask IoU; no TEST split use; the camera heads are frozen and cached (their ms are measured
  separately).
- Next experiments (3-5 h each on this machine, same pipeline): (A) baseline recipe + direction head only (expect the heading fix with
  baseline maps); (B) baseline recipe + sequence memory only (isolates the clip-batch cost); (C) the combined run with the refiner
  switched off or without re-scoring; (D) a second baseline seed to measure the noise; then the line decoder and the motion head.
- Open from the review of the temporal code: the memory's static weight comes from its own velocity head that no loss supervises
  (BPTT trains it indirectly; a direct supervision is not built).

## 7 Files

`experiment.md` (card, decisions, log), `results/` (tables.md, metrics.json of both runs, memory.json, latency.json), `images/`
(architecture.png, curves.png, compare_*.png; not in git), code: `evaluation/evaluate_run.py`, `evaluation/eval_memory.py`,
`training/configs/t1_r0.json`, `training/configs/t1_r3_combined.json`; runs and checkpoints: `runs/t1_r0`, `runs/t1_r3_combined`
(not in git; the combined run's code snapshot is in `runs/t1_r3_combined/code`).
