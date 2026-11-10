# What the observability report of t1_r4_soa showed (2026-10-30, main BEV session, at Mirko's request)

Tool: `evaluation/observe_run.py` + `evaluation/video_encode.py` (see evaluation/README.md). Frames: middle keyframe of 30 val scenes
(frozen parts 1-6 scene lists), run cold-to-warm through the clip, EMA weights of step 7000, CPU. Probes in this folder
(`wsl ~/miniconda3/envs/gcar/bin/python temp/review/observe_2026-10-30/<script> runs/t1_r4_soa`; they import `evaluation/observe_run.py`).

## Tool faults found by looking at the pictures (fixed)
| # | fault | fix |
|---|---|---|
| T1 | scene video did not play: cv2 `mp4v` (MPEG-4 part 2) | H.264 through an installed ffmpeg with libx264, read back and checked; plays in the built-in browser |
| T2 | panel "what the temporal fusion added" = fused - current frame: two different feature spaces, the difference is meaningless (ratio 1.27 in every frame) | panel and check now show what the past supplies (window frames moved to now, memory confidence) from the block's aux output |
| T3 | box check counted GT boxes of the whole grid, not the front field of view DetMeter scores | `in_front` filter |
| T4 | old t1 configs without scene lists silently mean 700 / 150 scenes since parts 7-10; 2610 of 5719 val samples had no cached camera features (KeyError) | frozen scenes `data/scenes_parts1_6.json` when a config has no list |

## System findings (all on t1_r4_soa)
1. **Velocity of moving objects is barely recovered, by both velocity paths.** Dense motion head, 23 val frames with moving vehicles:
   error 3.05 m/s per component against 3.69 for the zero predictor, 70 % of the moving vehicle cells predicted < 0.5 m/s; on train
   frames 3.41 vs 4.54 and 59 % (so it is under-fitting, not over-fitting). Box head velocity on matched vehicles (60 frames, 54 moving):
   predicted 2.74 m/s against 7.60 GT, error 5.44 against 7.60 for zero; static vehicles are right (0.15 m/s error). Consequence: at
   dt = +-0.5 s moving vehicles stay where they were (AP@2 vehicle 0.627 at dt 0 -> 0.399 at +0.5 s, ATE 0.62 -> 0.77).
   `velocity_probe.py`, `score_probe.py`.
2. **Not a loss or target bug**: a fresh MotionHead trained on the frozen trunk features with the real loss reaches the same level as the
   trained one (val 0.80 vs 0.77, zero predictor 0.95); the target and the features are consistent, the features carry little motion
   (`velocity_overfit.py`).
3. **The information is there**: 94 % of the 285 moving observed vehicles in the front view have radar detections in the 0.5 s window
   (median 10), and the median Doppler of the detections inside the box reproduces the GT radial speed with correlation 0.989, error
   0.45 m/s (7 %) (`doppler_info.py`). The tangential component (mean 2.8 m/s against 5.1 radial) needs the temporal frames.
4. **Adding the Doppler vector as an input map to the head is not enough**: S + (vr x line of sight, occupancy) gives val 0.775 vs 0.804
   (600 steps, 48 train frames) - a 3 x 3 conv head cannot gather the 10 sparse detections over a 9 x 4 cell vehicle. What is missing is
   an aggregation over the object (box-level or instance-level pooling of the Doppler), which is what the box-test branch
   (`--gate-doppler`, Doppler residual per candidate) explores.
5. Box scores are calibrated: F1-optimal threshold 0.30 (precision 0.81, recall 0.57) on 60 val frames; recall is the limit (far / small
   vehicles), not the threshold. Drivable area: the prediction is pale and too small in 7 of 30 frames (share 0.25 vs GT 0.40).
6. Fine: no NaN or dead channels in any stage, camera BEV energy only inside the camera cone (0.0 % outside), depth head median
   10-14 m, GT boxes sit on camera-BEV blobs and on radar returns (alignment of both branches), world memory covers 62 % of the cells.

## Weighting (asked 2026-10-30 evening; t1_r4_soa, `grad_share.py`, output in `grad_share_t1_r4_soa.txt`)
- Kendall at the end: objects s = 1.0 (the upper clamp, weight 0.37), areas 0.94, lines 0.84, motion 1.44.
- Gradient norm of the task groups at the BEV feature (log.jsonl diag rows): motion 0.000 (step 500) .. 0.006 (step 7000) against
  objects 0.08 .. 0.73 and areas 0.001 .. 0.005: the dense motion head hardly shapes the shared features (18x below objects at the end).
  The box head's velocity terms (3 dt) carry about 3 % of the summed per-term gradient at S (hm, size, rad, rot ~2-3 % each per dt).
- Composition: moving vehicles are 19 % of the vehicle cells and 20 % of the observed vehicle boxes (692 / 3023 cells, 37 / 147 boxes in
  12 train frames). Both velocity losses are L1 over all objects, whose optimum on objects the features cannot separate is the median = 0.
- `t3_vel10` (velocity weight x10, 3000 steps) did not help: vehicle AVE at dt 0 2.38 m/s (t1_r4_soa 1.82), so plain re-scaling is not the lever.
- NOT done: the test whether moving-cell balancing / MSE / Huber changes what a head can learn on the frozen features
  (`velocity_weighting.py`): the run died with "Cannot allocate memory" (the box is full of other chats' jobs). Rerun when memory is free,
  or better as a real training arm. Candidates for the arm: moving cells weighted by 1 / moving share, per-object normalisation of the
  dense loss (as the focal loss does), Huber / L2 instead of L1, a Doppler-aggregating branch per object.
- The velocity GT itself is not the suspect: autogt (GT chat) finds the annotation tracks agree with the radar to ~0.2 m/s.
