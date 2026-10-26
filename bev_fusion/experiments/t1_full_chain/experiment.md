# T1 First whole-chain training on nuScenes (camera + radar BEV, three time offsets) and two refinement rounds

Loop task of Mirko (2026-10-26): "build the network together and train on nuScenes; try 2 additional refinement rounds, build
result images, summarise results qualitatively and quantitatively and generate a report". The chain is `training/model.py`
(plan.md 1-8) trained by `training/train.py` with `training/losses.py`; numbers from `evaluation/evaluate_run.py`
(`evaluation/metrics.py`). Runs: `runs/<name>/` (not in git), metric cards: `runs/<name>/eval/metrics.json`, copied
into this folder as `results/<name>_metrics.json` (text, in git).

## 1. Card
| | |
|---|---|
| lens | system review (first real-data run of the whole chain) |
| sources | plan.md 8.4 (losses, open decisions), training/configs/README.md (arms), tests/test_end_to_end.py (mini overfit gate) |
| status | running |

- **Statement**: the assembled chain (frozen DualBEV image heads -> lift PD -> 3-frame window memory; radar BEV; concat
  fusion; ResNet / FPN trunk; heads decoded at dt = -0.5 / 0 / +0.5 s through the splat) learns the map layers and objects
  of the front field of view from the 510 downloaded nuScenes scenes better than the position prior, at all three times.
- **Mechanism**: every block passed its synthetic gate and the mini overfit gate; the camera heads are pre-trained.
- **Prediction** (VAL, all 3109 keyframes of the downloaded val scenes, camera cells): drivable area IoU@0 at dt 0
  >= 0.60 and >= the prior-map baseline + 0.15; vehicle AP@2 (nuScenes-style, front field of view, <= 60 m) >= 0.25;
  dt +0.5 s within 0.05 IoU of dt 0.
- **Decision rule**: below the prior + 0.05 on the drivable area -> the chain is broken (debug before any arm). A
  refinement arm is adopted when the paired scene bootstrap of its headline numbers (drivable IoU, lane-divider F1@0.5,
  vehicle AP@2) shows no significant loss and at least one gain whose 95 % interval excludes 0 and exceeds 0.010; inside
  the band it is a tie (keep the simpler / cheaper one).
- **Test design**: round 0 = `t1_base` (configs/t1_base.json, shortened to the measured budget). Rounds 1 and 2 = one
  or two arms each from configs/README.md, chosen from round 0's diagnostics (loss groups, gradient norms per group,
  per-class numbers), each at the same step budget as its control. Controls: null (IoU 0, AP 0), prior-map baseline (per-cell
  mean of the GT over 1000 train keyframes), the matched control run of the round. One seed per arm (budget); a
  difference inside 0.03 on detection is reported as unresolved (seed spread unknown).
- **Cost**: set after the speed test (section 2).
- **Inputs**: ready (feature cache 58,101 frames; GT cache 16,289 train + 3,109 val keyframes).

## 2. Detail plan and log
- 2026-10-26 04:42-04:47, speed (30 optimiser steps, idle GPU, 4 loader workers; temp/train/speed_b2.json, speed_b4.json):
  micro-batch 2 x 8 = 3.7 s per step (2.4 GB), 4 x 4 = 2.26 s (4.7 GB). Loader alone (temp/train/loader_speed.py): 10.4 /
  14.3 / 18.6 samples per s at 4 / 8 / 10 workers, so data loading is part of the step time. Gradient diagnostic at the
  BEV feature (step 30): objects 0.235, lines 0.0014, areas 0.0007, motion 7e-6 (objects about 300 x areas).
- Budget: 5000 optimiser steps per run (effective batch 16 = 4 x 4; 80,000 samples = 4.9 epochs of the 16,289 train
  keyframes), warm-up 300, 8 workers: `training/configs/t1_r0.json` (= t1_base with those three fields). Measured 1.71-1.74
  s per step: 2.4 h per run.
- Round 0 `t1_r0` started 04:49. Step 500: total 7.8 (from 20.5); Kendall s: areas -0.16, lines -0.03, objects +0.13,
  motion -0.10; gradient at the BEV feature objects 5.3e-2, lines 7.0e-3, areas 1.0e-3, motion 1.2e-6; group gradient
  cosines all |cos| <= 0.02 (no conflict, nearly orthogonal).
- Step 1000 checkpoint, VAL 200 samples spread over all 82 val scenes (evaluation/evaluate_run.py --n 200 --prior 200):
  drivable IoU@0 dt 0 0.642 (prior 0.424), dt +0.5 0.613, dt -0.5 0.645; walkway 0.160, carpark 0.116, ped crossing 0.000;
  lane divider F1@0.5 0.009 (not learned yet), road divider 0.090; vehicle coverage IoU 0.29 (dt +0.5); vehicle AP@2 dt 0
  0.590, dt +0.5 0.219, dt -0.5 0.284; pedestrian AP@2 0.173; two-wheeler 0.065.
- Heading check (temp/train/yaw_check.py, rot_check.py): vehicle AOE 1.76 rad looked like a convention bug; it is not.
  Target, loss and decoding agree; the GT heading agrees with the GT velocity direction (median 0.01 rad on moving
  vehicles). The head has not learned the heading at step 1000: rot loss 1.35 on val and train (random unit vectors give
  about 1.27), cos sign agreement 0.45-0.52 (chance), the prediction is a nearly constant direction (mean (sin, cos) =
  (+0.4, -0.6)) while the targets average (0, 0) (half the vehicles face the ego car). The small axis error (median 0.04
  rad) is an artefact of most boxes lying along x. To re-check at later checkpoints.
- Step 2000 (VAL 200): drivable 0.673, walkway 0.230, carpark 0.165, ped crossing 0.037; vehicle coverage IoU 0.412;
  vehicle AP@2 dt 0 0.632, +0.5 0.328, -0.5 0.419; pedestrian AP@2 0.204. Heading still not learned (rot loss 1.25-1.36,
  identical mean prediction); the rot branch does train (weights change like size / offset, its loss gradient reaches it:
  temp/train/rot_grad.py), so it is not a bug but the front / back ambiguity -> round-1 arm a.
- Lines: the zero-crossing F1 is about 0 because the regressed distance field is too smooth: on GT lane-divider cells the
  predicted distance has median 1.36 m, p10 0.47 m, and is never <= 0.25 m (the decoder's rule), so no line cell is
  decoded (temp/train/line_check.py). The line MASK head does find lines (30 % of the lane-divider cells > 0.5, 2 % of
  other cells): line masks are now evaluated as tolerant IoU next to the zero-crossing F1.
- Round 1 = two arms against t1_r0 at the same 5000 steps: a = object heading as sign-free axis + front / back class
  (`model.object_direction`, SECOND / PointPillars; code in object_head.py with a known-answer self-test), b = UW-SO task
  balancing (T 5; the objects gradient at the BEV feature is still 50 x the areas gradient at step 1000 under Kendall).
- Loader: two runs in parallel gave no extra throughput. Profile (temp/train/profile_sample.py): radar sweeps 558 ms
  (cold) of a sample, GT 33 ms, features 30 ms, camera index 8 ms. Built an exact per-sample radar cache
  (data/cache/radar, 19,398 samples, 1.0 GB, 40 min on 5 CPU workers; cached = computed exactly, self-tested). A
  placement bug in that edit (CachedFrontRadarDataset lost its __getitem__) crashed the first cached start (KeyError
  'prob'); fixed in 2ddf511, the self-test now loads a whole item. Runs of round 1 use the cache; t1_r0 does not (same data).
- With two runs on the GPU the GPU became the shared limit (83 % busy, 0.53 steps/s together vs 0.58 alone) and free RAM
  fell to 2-3 GB (6 workers per run: 10 GB per run). Round-1 arm a was stopped twice (step 20, step 140, step 100;
  folders in _to_delete/) - every restart begins at step 0, so no partial run enters a comparison.
- Other chats edit the shared training code (an uncommitted sequence-training change in train.py appeared at 07:30): the
  launcher (temp/train/run_cfg.sh) now trains from a snapshot of the COMMITTED training/ code (git archive HEAD into
  runs/<name>/code/, commit in SNAPSHOT.txt).

## 3. Summary
(after round 2)
