# Eight tries to improve the camera + radar BEV system: what works, what does not, why, and where it stands against SOA

Status 2026-10-30 14:50. Eight tries done (plus one running). Every number comes from `runs/*/eval/metrics*.json` through `temp/study/collect.py`
(`../t2_block_swaps/results/arms.md`); no number is typed by hand. All tables: cold VAL, 1000 keyframes spread over the 82 validation scenes (parts 1-6 of
nuScenes, 150 scenes exist), EMA weights, 3000 optimiser steps from the same 428 training scenes, seed 0 unless stated. This is a short-schedule
screening study of blocks, not a final system.

## 1 Bottom line

1. **The reference system is robust against block swaps.** Of eleven exchanges, none is BETTER than the reference outside the noise band. Seven tie (gate fusion, CMX fusion,
   ConvGRU, pillars radar encoder, no temporal block, and the two camera-side fixes), four are worse (no depth weight in the lift, random-initialised BEV encoder, radar removed,
   camera removed). The best standard replacement is the pillars radar encoder (mAP2 0.349 against 0.339 and 0.324 for the two reference seeds: a candidate, not a result).
2. **Both sensors are strong blocks and they complement each other.** radar only 0.062, camera only 0.210, camera + radar 0.339 / 0.324 (mAP2 = vehicle + pedestrian,
   four matching distances). The radar makes the camera precise (vehicle AP@0.5 0.045 -> 0.228), the camera makes the radar usable.
3. **Weak spot 1: camera localisation.** Camera-only AP@0.5 is 0.045 while AP@2 is 0.448. Two camera-side fixes (trainable adapter on the frozen features, standard
   height-sum collapse of the lift) change nothing (0.211 and 0.205 against 0.210). Probes show the information is in the features and survives to the head input, so
   the limit is what stride-16 frozen image features with depth uncertainty can localise, not a broken path.
4. **Weak spot 2: the velocity head has not learned in 3000 steps.** Every 3000-step model predicts 0.00 m/s for every vehicle; the velocity error equals the zero-velocity
   baseline exactly (2.39 m/s). After 7000 x 2 steps it predicts 0.85 m/s on average (error 1.85 against the zero baseline 2.40). Doppler cannot show its value yet.
5. **Training length and data, not blocks, are the visible lever**: the same architecture trained 2.6 epochs (`t1_r4_soa`) reaches mAP2 0.364 (1000 frames) with lane dividers
   0.273 (0.181 at 3000 steps) and walkway 0.246 (0.191). Their runs are queued here (`t3_ref_long`, 700 scenes) but not finished.
6. **Not comparable to SOA, and far from it**: ours is the front camera, 60 m, three class groups, short schedule; published R50 camera + radar systems reach 45-52 mAP
   (10 classes, six cameras, val). Section 8.
7. **Errors made and corrected during the work** (section 9): the radar-only arm of the first study ran with the camera on; I built a wrong "no-sensor floor" on it and
   retracted that; the 3-class mAP is dominated by seed noise of a rare class.

## 2 Setup

Architecture and the outcome of every exchange: ![architecture](images/architecture_and_verdicts.png)

- Chain: frozen DualBEV image heads (ResNet-50 + FPN + depth net, cached features: depth 118 bins + context 80 channels at 16 x 44) -> own lift PD (depth-weighted pull,
  8 heights) -> 1x1 collapse -> camera temporal fusion (3-frame window + carried world memory) -> concat fusion with the radar BEV block (5 radars, 0.5 s window, Doppler shift,
  modality dropout 0.2) -> pre-trained DualBEV BEV encoder (stride-1 detail path, 96-ch head reduce) -> light heads (semantic maps, CenterPoint objects, velocity) at t - 0.5 / t / t + 0.5 s.
- Each arm changes ONE block (configs `training/configs/t2_*.json`, `t3_*.json`; blocks `training/blocks_new.py`, `training/model.py`, flags checked by self-tests).
  Same data, schedule, seed; two seeds of the reference give the noise band.
- Metrics (decided on the data, `experiment.md` section 4): **mAP2** = mean of vehicle and pedestrian AP over matching distances 0.5 / 1 / 2 / 4 m; vehicle and pedestrian
  AP@1 and AP@2; drivable, walkway IoU, lane-divider IoU at 0.5 m tolerance. The 3-class mAP is not used for verdicts: the rare `two_wheeler` class (174 GT in 1000 frames) swings
  its AP@2 between 0.196 and 0.047 across the two seeds (mAP3 0.277 / 0.228) while vehicle AP@1 is 0.468 / 0.465.
- Verdict rule (written before the runs): WORSE / BETTER = difference beyond the band (2 x seed spread of the stable metrics = 0.054) AND the paired scene bootstrap interval excludes 0;
  otherwise TIE (then the cheaper block). One seed per arm: a TIE means "not distinguishable from the reference here", not "equal".
- Sanity checks that passed: calibration (geometry vs the devkit to 1e-10 px; depth-head alignment on val and train scenes peaks at zero shift; radar footprint check peaks at zero
  shift / zero time offset), cache alignment, a no-sensor prior detector scores mAP2 0.000 (the metric cannot be gamed by a position prior), self-tests of every new block.

## 3 Results of all arms (cold VAL-1000, 3000 steps; difference to `ref_s0`; `*` = bootstrap interval excludes 0)

| block | arm | mAP2 | veh AP@1 | ped AP@1 | drivable | walkway | divider | verdict |
|---|---|---|---|---|---|---|---|---|
| reference | `ref_s0` / `ref_s1` | 0.339 / 0.324 | 0.468 / 0.465 | 0.141 / 0.122 | 0.666 / 0.663 | 0.191 / 0.164 | 0.181 / 0.179 | noise floor (seed spread of mAP2 0.015) |
| sensors | radar only (`t3_radar_only`) | 0.062 | 0.079 | 0.000 | 0.586 | 0.112 | 0.098 | WORSE on everything |
| sensors | camera only (`radar_off`) | 0.210 | 0.209 | 0.033 | 0.637 | 0.107 | 0.164 | WORSE on detection and walkway |
| lift | no depth weight (`lift_p`) | 0.290 | 0.426 | 0.094 | 0.677 | 0.170 | 0.167 | WORSE (vehicle AP@2, pedestrian AP@2) |
| BEV encoder | random init (`trunk_rand`) | 0.318 | 0.457 | 0.117 | 0.653 | 0.136 | 0.149 | WORSE (walkway -0.055*, divider -0.032*) |
| temporal | none (`camtemp_none`) | 0.321 | 0.464 | 0.113 | 0.659 | 0.182 | 0.173 | TIE |
| temporal | ego warp + ConvGRU (`camtemp_gru`) | 0.327 | 0.465 | 0.118 | 0.675 | 0.200 | 0.192 | TIE |
| fusion | gates (`fusion_gate`) | 0.330 | 0.470 | 0.126 | 0.668 | 0.185 | 0.190 | TIE |
| fusion | CMX (`fusion_cmx`) | 0.332 | 0.471 | 0.127 | 0.659 | 0.157 | 0.173 | TIE (walkway -0.034*, inside the band) |
| radar encoder | pillars (`radar_pillars`) | 0.349 | 0.475 | 0.164 | 0.664 | 0.185 | 0.200 | TIE, best (+0.011, pedestrian AP@1 +0.023*, divider +0.019*) |

Camera-only fixes are compared with camera-only `radar_off` (0.210), not with the reference: adapter `t3_cam_adapter` mAP2 0.211, vehicle AP@0.5 0.046 (against 0.045), divider 0.185 (0.164);
height-sum collapse `t3_cam_zsum` 0.205, vehicle AP@0.5 0.037, walkway 0.089 (0.107): both TIE. Not run yet (queue): SECOND + SECONDFPN encoder, width 0.5, forward splat (`trunk_second`,
`lift_s`, `trunk_small`), 6000 steps (`t3_ref_long`), velocity weight x10 (`t3_vel10`, running).

## 4 The eight tries

| # | question | change | prediction (written before) | result | decision |
|---|---|---|---|---|---|
| 1 | does the model use each input? | evaluate trained models with zeroed / shuffled radar and camera (`--control`) | zero camera: mAP3 0.20-0.26 (wrong) | zero / shuffled camera collapses `ref_s0` to 0.013 / 0.028 (mAP3), shuffled radar gives 0.151 (mAP3); the long-trained system 0.031 / 0.036 and 0.156 (mAP3) | both inputs are used; (its radar-only arm was invalid, section 9) |
| 2 | camera fix: standard height collapse | `t3_cam_zsum`: sum over heights, 1x1 (10 x fewer parameters) | not written | mAP2 0.205 vs 0.210 camera-only | TIE, not adopted |
| 3 | camera fix: are frozen features the bottleneck? | `t3_cam_adapter`: zero-init residual adapter on the cached context | maps +0.02, mAP +0.01 | 0.211 vs 0.210; vehicle AP@0.5 0.046 vs 0.045 | TIE, not adopted |
| 4 | radar-only / camera-only value | `t3_radar_only` (camera off, fixed code) | - | radar only 0.062 | both sensors STRONG |
| 5 | standard fusion 1 | gates | tie | 0.330 vs 0.339 | TIE |
| 6 | tracking / time | ConvGRU, no block | GRU tie, none -0.01 | 0.327, 0.321 | TIE |
| 7 | standard fusion 2 | CMX | -0.02 | 0.332 | TIE (better than predicted) |
| 8 | standard radar encoder | pillars | tie | 0.349 | TIE, candidate |
| 9 (extra) | velocity head | `t3_vel10`: velocity loss weight 10 instead of 1 | - | running | pending |

## 5 Why the ideas that did not work do not work

- **Frozen-feature adapter and height-sum collapse (camera-only localisation).** The information is there: a linear probe on the lifted features, fitted on 30 train scenes and tested on 20 unseen
  val scenes, reaches vehicle IoU 0.249 (per-cell prior map 0.021), drivable 0.577 (0.459), walkway 0.200 (0.121); in the trained camera-only model the probe stays at 0.26-0.29 vehicle IoU
  through camera BEV, fused BEV, trunk and head input, and the model's own dense vehicle output reaches 0.279 (`temp/calib/probe_*.log`). Camera-only AP rises from 0.045 at 0.5 m to 0.448 at 2 m:
  the camera finds vehicles but not their exact position, as expected from stride-16 features and a depth distribution of 0.5 m bins. Untested next steps: stride-8 image features, a depth-refinement
  step, and what the radar fusion already does (AP@0.5 0.228).
- **Standard fusion, temporal and radar blocks tie.** In a 3000-step screening (about half an epoch of 428 scenes) the model is limited by training, not by the block design; every block can
  reach the same level. The simplest block (concat, window memory, mean raster) is not beaten, so it stays (cheapest). The cold protocol (three frames, empty memory) also cannot show
  the value of a carried memory; the warm whole-scene protocol was not compared. There is no explicit tracker (association, ids) in the system, only temporal fusion.
- **Velocity.** The velocity loss weight (1.0) and the schedule leave the head at zero output; the zero-velocity baseline equals the measured error for `ref_s0` (2.39), `radar_off` (2.53),
  `t3_radar_only` (2.59). Whether weight 10 fixes it is running.
- **Encoder pre-training.** Random initialisation only hurts maps (walkway, divider), and CMX with a random-initialised encoder reaches the reference level: the pre-trained DualBEV encoder is less important
  for detection than assumed.

## 6 Output images

Same four validation frames, four systems (camera | ground truth | camera only | radar only | camera + radar 3000 steps | camera + radar 7000 x 2 steps; boxes score >= 0.3, GT dashed):
![systems](images/compare_systems.png)
Reading: radar alone gives speckled clutter and no pedestrians; camera alone finds vehicles at about the right place but with range errors and misses; the fusion gives clean boxes; the long-trained
system adds pedestrians and sharper maps. More frames: `runs/t3_img_*/eval/images/`.

## 7 Speed

Only one idle-GPU measurement exists (`runs/t1_r4_soa/eval/latency.json`, RTX 3060, batch 1): BEV network 50.7 ms per frame at one target time, 68.2 ms with the three target times of the time mover,
plus 17.2 ms for the image heads (ResNet-50). Per-block timings (gate, CMX, pillars, SECOND) were NOT measured: the GPU is occupied by the training queue; they are the first job for an idle window.
By the rule "TIE -> the cheaper block" no standard block replaces the simple reference for now.

## 8 Compatibility with SOA

nuScenes val, ResNet-50, 256 x 704, six cameras, 10 classes (secondary sources, to be verified against the papers; `soa_reference.md`): CRN camera + radar mAP 0.490 / NDS 0.560, RCBEVDet 0.453 / 0.568,
RCBEVDet++ 0.519 / 0.604 ([RCBEVDet++ paper](https://arxiv.org/pdf/2409.04979)); camera-only BEVDepth 0.351 / 0.475 (quoted table, unverified). DualBEV's own R50 val number was not found
(test set, V2-99: 0.552 / 0.634, [arXiv 2403.05402](https://arxiv.org/pdf/2403.05402)).

Not comparable with ours, and where the gap is: front camera only (one of six), 60 m x 52 m front sector, three class groups, no NDS, mAP2 over four distances instead of the 10-class mAP, 82 of 150 val scenes,
3000 optimiser steps on 428 scenes (half an epoch; the published systems train 20-24 epochs on 850 scenes), frozen image heads, no velocity (error equals zero baseline). What carries over: radar adds to the camera
(published 5-10 mAP points on the same backbone; here camera 0.210 -> camera + radar 0.339 mAP2), a camera-only head localises coarsely, the standard fusion blocks are interchangeable. What is missing for a SOA claim:
six cameras, the full class set and NDS, 20 epochs, velocity learned, the full 850 scenes. No claim of SOA performance is made.

## 9 Errors made and corrected during this work

1. `CachedBEVFusionModel` ignored `camera_off`: the first radar-only arm (`t2_cam_off`) ran with the camera and equalled the reference. I derived a "no-sensor floor" and "the camera path does not work" from it, and
   retracted both (`experiment.md` section 5). Found by a stage probe of the supposed floor model; fixed (commit d57bcc2, self-test `camera-off-selftest`); the real radar-only arm is `t3_radar_only`.
2. The 3-class mAP led to a seed spread that looked like a kill criterion (0.049); it is one rare class. Metrics changed as described in section 2.
3. A full C: disk (orphaned WSL swap files, cleaned with Mirko's OK) crashed `fusion_gate` at its last checkpoint and killed the queue for 4.5 h; `fusion_gate` was resumed from step 2500 (500 of 3000 steps run after the restart).
4. My first predictions for the sensor controls were wrong in both directions (zero camera mAP3 0.013 instead of 0.20-0.26).

## 10 Weak spots and what to do next (ranked)

1. Train longer and on more data (6000 steps queued; 700 scenes need the parts 7-10 caches, built, plus 11 GB of feature shards moved back from D:). Expected from the long run: dividers, walkway, velocity.
2. Velocity head: weight x10 (running); if it stays at zero, a Doppler-aware velocity branch.
3. Camera localisation: stride-8 features, depth refinement; the frozen detection features are not the limit (adapter TIE).
4. Per-block latency on an idle GPU; second seeds for the candidates (pillars) before adoption.
5. Compare the warm whole-scene protocol for the temporal blocks and build a real tracker if tracking is the goal.
6. Open: the full SOA protocol (six cameras, 10 classes) is a different project step.

## 11 Reproduce

Code at commits listed in `experiment.md` (`0a3d73c` swap arms, `1a6b8d8` camera fixes, `d57bcc2` camera-off fix); queue and lanes `temp/study/lane2.sh`, `lane3.sh`; tables `temp/study/collect.py`;
controls `evaluate_run.py --control`; probes `temp/calib/probe_*.py`; prior detector `temp/calib/prior_detector.py`; checkpoints of finished runs moved to `D:\Auto-research\bev_fusion_ckpts\` (copy back to re-evaluate).
