# Eight tries to improve the camera + radar BEV system: what works, what does not, why, and how it relates to SOA

Status 2026-10-30, reviewed once by a fresh reviewer against the primary files (corrections applied). Eight tries done (plus one running). The tables come from
`runs/*/eval/metrics*.json` through `temp/study/collect.py` (`../t2_block_swaps/results/arms.md`); other numbers come from the named scripts and logs (probes `temp/calib/probe_*.log`,
velocity `temp/calib/velocity_baseline.py`, controls `try1_controls.md`, latency `runs/t1_r4_soa/eval/latency.json`); no number is typed by hand. All tables: cold VAL, 1000 keyframes spread over the 82 validation scenes (parts 1-6 of
nuScenes, 150 scenes exist), EMA weights, 3000 optimiser steps from the same 428 training scenes, seed 0 unless stated. This is a short-schedule
screening study of blocks, not a final system.

## Update 2026-11-01 (read this first; sections 1-11 are the state of 2026-10-30 and stay as written)

Added since: tries 9 and 10 (`t3_vel10`, `t4_fpn_order`) and six follow-up steps (`../t4_soa_front`, `../t6_soa_gap`, `../t7_lane_branch`, `../t8_bevfeat`,
`../t10_dual_features`, `../t11_vel_cause`). Numbers: cold VAL, the same 1000 keyframes of the 82 validation scenes unless a row says otherwise; maps from
`runs/*/eval/metrics.json` (`temp/pitch/ladder.py`), objects in the front sector (0-50 m, +-32.3 deg) from `../t4_soa_front/results_*.md` and
`../t10_dual_features/results_x50.md` (`temp/t4/compare_front.py`). Seeds: most arms have one seed; where a seed twin exists the row says so.

### What changes in the bottom line

1. **The lever is the image features, not the blocks.** Eleven block exchanges tied; swapping the frozen detection-trained stride-16 image features for map-trained
   stride-8 BEVFusion features nearly doubled the maps (lane divider 0.191 -> 0.351 / 0.345, walkway 0.178 -> 0.339 / 0.342, crossing 0.002 -> 0.145 / 0.165) and halved
   the objects (front mAP2 0.367 -> 0.183 / 0.180). Both feature sets in one chain keep both: lane divider 0.335, walkway 0.340, drivable 0.806, front mAP2 0.361 (one
   seed; the seed twin is training). The 3000-step dual chain beats the start recipe trained 4.7 x longer on five of six map layers (ladder below).
2. **The velocity head is dead, not slow.** The output of the branch is its bias; 0.011 % of its hidden units are active at the supervised centres (13 - 50 % for the other
   branches); 24 of the 29 classified runs are dead. Try 9 (loss weight x10) could not help a dead ReLU layer. The radar Doppler read-out at the GT boxes holds 0.79 of the
   velocity of moving vehicles (upper bound, GT heading), our best learned heads 0.22 - 0.25.
3. **The SOA gap is measured like for like where a published system can be run on our frames**: DualBEV (front camera only) 0.250 front mAP2 against our camera-only 0.242 and
   camera + radar 0.359 - 0.400; GaussianCaR on the maps (section 8 of the update below). Still not comparable with published leaderboards.
4. **Three errors found since** (they continue the list of section 9): the FPN-LSS inputs of our trunk were concatenated in the wrong order for the pre-trained weights
   (`t4_fpn_order`: a tie); the GT cache is built from live code, so runs of different days had different object targets (same-code controls, decision D29);
   the velocity branch was dead (above).

### Tries 9 - 16

| # | question | change | prediction (written before) | result | decision |
|---|---|---|---|---|---|
| 9 | does a larger velocity loss weight make the head learn? | `t3_vel10`: weight 10 | - | front mAP2 0.360 (reference 0.375 / 0.359); the velocity output stays zero (largest predicted speed 0.031 m/s) | no effect, not adopted; cause found in 16 |
| 10 | does the wrong FPN-LSS input order of our trunk matter? | `t4_fpn_order`: original order (flag) | - | front mAP2 0.367 against 0.375 / 0.359 | TIE, flag stays default off |
| 11 | how far is the system from a SOA camera detector in the front area? | DualBEV run with CAM_FRONT only, unchanged weights, same 1000 frames (`t4_soa_front`) | DualBEV 0.40 - 0.55 (miss) | DualBEV 0.250; ours camera-only 0.242; camera + radar 0.359 / 0.375 (3000 steps), 0.400 (7000 x 2) | camera path = SOA camera path; radar +0.11 - 0.15 (intervals exclude 0) |
| 12 | why are the maps far from the SOA? | GaussianCaR scored by our metric, oracles, probes, a maps-only run (`t6_soa_gap`) | - | ours at 17 - 55 % of GaussianCaR except drivable (85 %); not range, threshold, over-fitting, BEV network or labels; task competition small (+0.015 - 0.051) | main candidate: frozen detection-trained stride-16 features (M6) plus the training budget |
| 13 | do stride-4 image details help the thin map classes? | `t7_res4` / `t7_rgb`: a frozen ResNet stride-4 branch lifted into the BEV | lane divider +0.04 | res4: lane divider +0.065 / +0.042 (two seeds), road divider IoU@1 +0.119 / +0.096, walkway and crossing unchanged; rgb +0.015 | helps the thin lines, not the area classes; verdict waits for the equal-GT control |
| 14 | are frozen detection-trained features the limit of the maps? | `t8_bevfeat`: cached BEVFusion map-trained stride-8 features, same chain, same-code control `t8_ctl` | maps up | maps x1.8 - 1.9 (two seeds agree), front mAP2 0.367 -> 0.183 / 0.180, pedestrians gone | supports M6; neither feature set alone gives both |
| 15 | can both be had? | `t10_dual`: both caches, two lifts, merged by a 1x1 convolution | lane divider 0.34, drivable 0.80, walkway 0.33, front mAP2 0.33 (0.25 - 0.38), pedestrian AP@2 0.15 | lane divider 0.335, drivable 0.806, walkway 0.340, front mAP2 0.361, pedestrian AP@2 0.199 | rule E1 holds (one seed); seed twin training |
| 16 | why is the box velocity zero in every run? | diagnosis from stored checkpoints, detections and a probe (`t11_vel_cause`) | active share < 1 % at the centres | 0.011 % (other branches 13 - 50 %); 24 of 29 runs dead | fresh branch, BatchNorm, moving / static gate queued as arms |

### Maps: the ladder (same metric, VAL, `seg_fov`, dt 0)

| run | what | step | frames | drivable | walkway | carpark | crossing | lane div. @0.5 m | road div. @0.5 m |
|---|---|---|---|---|---|---|---|---|---|
| t2_ref_s0 | start, 3000 steps | 3000 | 1000 | 0.666 | 0.191 | 0.106 | 0.002 | 0.181 | 0.220 |
| t3_ref_long | start, 6000 steps | 6000 | 1000 | 0.689 | 0.225 | 0.155 | 0.057 | 0.234 | 0.297 |
| t1_r4_soa | start, 7000 x 2 steps (4.7 x the training) | 7000 | 3109 | 0.698 | 0.244 | 0.162 | 0.093 | 0.274 | 0.322 |
| t8_ctl_s0 | same-code control of t8 / t10 | 3000 | 1000 | 0.668 | 0.178 | 0.115 | 0.002 | 0.191 | 0.224 |
| t8_bevfeat_s0 | map-trained features | 3000 | 1000 | 0.754 | 0.339 | 0.173 | 0.145 | 0.351 | 0.437 |
| t8_bevfeat_s1 | map-trained features, seed 1 | 3000 | 1000 | 0.759 | 0.342 | 0.168 | 0.165 | 0.345 | 0.439 |
| t10_dual_s0 | both feature sets | 3000 | 1000 | 0.756 | 0.340 | 0.144 | 0.154 | 0.335 | 0.442 |

The 3000-step dual chain is above the 7000 x 2-step start on drivable, walkway, crossing, lane divider and road divider and below it on carpark (0.144 against 0.162); the
frame sets differ (3109 against 1000; the same model scores 0.719 on 400 frames against 0.698 on 3109, so the subset effect is about 0.02).

[[LIKE_FOR_LIKE_MAPS]]

### Weak spots, ranked (replaces section 10)

1. **Velocity**: the branch is dead; fresh branch, BatchNorm, gate and a Doppler read-out are queued (`t11_vel_hurdle`, `t11_motion_field`).
2. **Thin map classes and the training budget**: lane divider 0.335 against 0.691 for GaussianCaR; 3000 steps are 0.55 epoch of 428 scenes. Dual features trained longer, the
   stride-4 branch on top, all 700 scenes.
3. **Objects at range**: precision of the confident detections falls from 0.89 (0 - 20 m) to 0.64 (40 - 60 m); far objects are mostly detected with low confidence (t6).
4. **No SOA camera + radar detector was run on our frames** (CRN checkpoint on disk).
5. **Per-block speed on an idle GPU** is still unmeasured.
6. **Infrastructure**: C: at 4.1 GB free (a full disk crashed a run once); one GT version (D29) is open.

### Errors made and corrected since (continues section 9)

5. The FPN-LSS inputs of our trunk were concatenated in the wrong order for the pre-trained weights since the first run; found by the DualBEV port; a tie in effect.
6. Runs of different days used different object targets because the GT is built from live code; found 2026-10-31; same-code controls added.
7. "The head has not learned in 3000 steps" (weak spot 2 of section 1) was a diagnosis of the wrong kind: the branch was dead, and the loss-weight try could not work.
8. My predictions missed twice more (DualBEV front-only 0.40 - 0.55, measured 0.250); recorded as misses.

## 1 Bottom line

1. **No block swap shows an effect larger than the noise (about 0.05 mAP2 with one seed and 3000 steps).** Of eleven exchanges, none is BETTER than the reference. Seven are
   not distinguishable from their control (gate fusion, CMX fusion, ConvGRU, pillars radar encoder, no temporal block against the reference; the two camera-side fixes against
   camera-only), four are worse (no depth weight in the lift, random-initialised BEV encoder - both marginal, one metric just over the band -, radar removed, camera removed).
   The pillars radar encoder has the highest mAP2 (0.349 against 0.339 and 0.324 for the two reference seeds, a difference below the seed spread): worth a second seed, not a result.
2. **The two sensors complement each other; the camera is the stronger one.** radar only 0.062 (pedestrian AP 0.000, vehicle AP@1 0.079), camera only 0.210, camera + radar 0.339 / 0.324
   (mAP2 = vehicle + pedestrian, four matching distances; 0.062 + 0.210 < 0.339). The radar makes the camera precise (vehicle AP@0.5 0.045 -> 0.228), the camera makes the radar usable.
3. **Weak spot 1: camera localisation.** Camera-only vehicle AP@0.5 is 0.045 while vehicle AP@2 is 0.448. Two camera-side fixes (trainable adapter on the frozen features, standard
   height-sum collapse of the lift) do not change mAP2 beyond the band (0.211 and 0.205 against 0.210; individual metrics move by at most 0.02, section 3). Linear probes show that the
   vehicle / drivable / walkway information is in the lifted features and still linearly readable at the head input of the trained model (cell IoU, not 0.5 m localisation), which argues against a broken
   path; that the limit is stride-16 frozen features with depth uncertainty is the working hypothesis, not shown.
4. **Weak spot 2: the velocity head has not learned in 3000 steps.** Every 3000-step model predicts 0.00 m/s for every vehicle; the velocity error equals the zero-velocity
   baseline of its own matched vehicles exactly (2.39 m/s for the reference, 2.36-2.60 across arms). After 7000 x 2 steps it predicts 0.85 m/s on average (error 1.85 against the zero baseline 2.40). Doppler cannot show its value yet.
5. **Longer training with a larger batch is the visible lever, not blocks**: the same architecture trained 7000 steps with gradient accumulation 2 (2.6 epochs, `t1_r4_soa`, a different
   schedule and config) reaches mAP2 0.364 on the same 1000 frames with lane dividers 0.273 (0.181 at 3000 steps) and walkway 0.246 (0.191); it also overfits more (train subset mAP2 0.462
   vs 0.361 on its full 3109-frame val, `metrics_train.json`). The amount of data was not varied; the clean tests (`t3_ref_long`, 700 scenes) are queued, not finished.
6. **Not comparable to SOA, and far from it**: ours is the front camera, 60 m, three class groups, short schedule; the published R50 camera + radar systems quoted in section 8
   report 45-52 mAP (10 classes, six cameras, val; secondary sources, unverified).
7. **Errors made and corrected during the work** (section 9): the radar-only arm of the first study ran with the camera on; I built a wrong "no-sensor floor" on it and
   retracted that; the 3-class mAP is dominated by seed noise of a rare class.

## 2 Setup

Architecture and the outcome of every exchange: ![architecture](images/architecture_and_verdicts.png)

- Chain: frozen DualBEV image heads (ResNet-50 + FPN + depth net, cached features: depth 118 bins + context 80 channels at 16 x 44) -> own lift PD (depth-weighted pull,
  8 heights) -> 1x1 collapse -> camera temporal fusion (3-frame window + carried world memory) -> concat fusion with the radar BEV block (5 radars, 0.5 s window, Doppler shift,
  modality dropout 0.2) -> pre-trained DualBEV BEV encoder (stride-1 detail path, 96-ch head reduce) -> light heads (semantic maps, CenterPoint objects, velocity) at t - 0.5 / t / t + 0.5 s.
- Each arm changes ONE block, except `fusion_cmx` which replaces the fusion AND the BEV encoder (CMX is its own pyramid + decoder) (configs `training/configs/t2_*.json`, `t3_*.json`; blocks `training/blocks_new.py`, `training/model.py`, flags checked by self-tests).
  Same data, schedule, seed; two seeds of the reference give the noise band.
- Metrics (decided on the data, `experiment.md` section 4): **mAP2** = mean of vehicle and pedestrian AP over matching distances 0.5 / 1 / 2 / 4 m; vehicle and pedestrian
  AP@1 and AP@2; drivable and walkway IoU, lane-divider IoU at 0.5 m tolerance (the other maps at tolerance 0). The 3-class mAP is not used for verdicts: the rare `two_wheeler` class (174 GT in 1000 frames) swings
  its AP@2 between 0.196 and 0.047 across the two seeds (mAP3 0.277 / 0.228) while vehicle AP@1 is 0.468 / 0.465.
- Verdict rule (written before the runs): WORSE / BETTER = difference beyond the band (2 x seed spread of the stable metrics = 0.054, set by walkway; the mAP2 seed spread is 0.015) AND the
  paired scene bootstrap interval excludes 0; otherwise TIE (then the cheaper block). One seed per arm and n = 2 seeds for the band: a TIE means "not distinguishable from the reference here",
  not "equal", and a WORSE based on one metric just over the band is marginal. The bootstrap resamples scenes, not training seeds: the two reference seeds themselves differ with `*` on walkway
  (-0.027*), pedestrian AP@1 (-0.020*) and AP@2 (-0.016*). Only effects above about 0.05 can be called with this design.
- Sanity checks that passed: calibration (geometry vs the devkit to 1e-10 px; depth-head alignment on val and train scenes peaks at zero shift; radar footprint check peaks at zero
  shift / zero time offset), cache alignment, a no-sensor prior detector scores mAP2 0.000 (the metric cannot be gamed by a position prior), self-tests of every new block.

## 3 Results of all arms (cold VAL-1000, 3000 steps; difference to `ref_s0`; `*` = bootstrap interval excludes 0)

| block | arm | mAP2 | veh AP@1 | ped AP@1 | drivable | walkway | divider | verdict |
|---|---|---|---|---|---|---|---|---|
| reference | `ref_s0` / `ref_s1` | 0.339 / 0.324 | 0.468 / 0.465 | 0.141 / 0.122 | 0.666 / 0.663 | 0.191 / 0.164 | 0.181 / 0.179 | noise floor (seed spread of mAP2 0.015) |
| sensors | radar only (`t3_radar_only`) | 0.062 | 0.079 | 0.000 | 0.586 | 0.112 | 0.098 | WORSE on every detection metric, drivable, walkway, divider |
| sensors | camera only (`radar_off`) | 0.210 | 0.209 | 0.033 | 0.637 | 0.107 | 0.164 | WORSE on detection and walkway |
| lift | no depth weight (`lift_p`) | 0.290 | 0.426 | 0.094 | 0.677 | 0.170 | 0.167 | WORSE, marginal (vehicle AP@2 -0.057*, pedestrian AP@2 -0.056*; mAP2 -0.048 is inside the band; pedestrian AP@2 vs `ref_s1` -0.039 would be inside it) |
| BEV encoder | random init (`trunk_rand`) | 0.318 | 0.457 | 0.117 | 0.653 | 0.136 | 0.149 | WORSE, marginal (walkway -0.055*, one metric; vs `ref_s1` it would be -0.028 = the seed spread; also pedestrian AP@1 -0.024*, mAP2 -0.021 inside the band) |
| temporal | none (`camtemp_none`) | 0.321 | 0.464 | 0.113 | 0.659 | 0.182 | 0.173 | TIE |
| temporal | ego warp + ConvGRU (`camtemp_gru`) | 0.327 | 0.465 | 0.118 | 0.675 | 0.200 | 0.192 | TIE |
| fusion | gates (`fusion_gate`) | 0.330 | 0.470 | 0.126 | 0.668 | 0.185 | 0.190 | TIE |
| fusion | CMX (`fusion_cmx`) | 0.332 | 0.471 | 0.127 | 0.659 | 0.157 | 0.173 | TIE (walkway -0.034*, inside the band) |
| radar encoder | pillars (`radar_pillars`) | 0.349 | 0.475 | 0.164 | 0.664 | 0.185 | 0.200 | TIE (highest mAP2, +0.011 is below the seed spread 0.015; pedestrian AP@1 +0.023* is as large as the seed pair's -0.020*) |

Camera-only fixes are compared with camera-only `radar_off` (0.210), not with the reference (a comparison made with the same bootstrap tool outside `collect.py`; `collect.py` lists them against
`ref_s0`, where they are WORSE as camera-only systems): adapter `t3_cam_adapter` mAP2 0.211, vehicle AP@0.5 0.046 (against 0.045), divider 0.185 (0.164, +0.021* but inside the band);
height-sum collapse `t3_cam_zsum` 0.205, vehicle AP@0.5 0.037, walkway 0.089 (0.107, -0.018*), drivable 0.624 (0.637, -0.014*): both TIE by the rule, with small significant moves on single maps. Not run yet (queue): SECOND + SECONDFPN encoder, width 0.5, forward splat (`trunk_second`,
`lift_s`, `trunk_small`), 6000 steps (`t3_ref_long`), velocity weight x10 (`t3_vel10`, running).

## 4 The eight tries

| # | question | change | prediction (written before) | result | decision |
|---|---|---|---|---|---|
| 1 | does the model use each input? | evaluate trained models with zeroed / shuffled radar and camera (`--control`) | zero camera: mAP3 0.20-0.26 (wrong) | zero / shuffled camera collapses `ref_s0` to 0.013 / 0.028 (mAP3), shuffled radar 0.151, empty radar 0.177 (all mAP3); the long-trained system 0.031 / 0.036 and 0.156 (mAP3) | both inputs are used; (its radar-only arm was invalid, section 9) |
| 2 | camera fix: standard height collapse | `t3_cam_zsum`: sum over heights, then 1x1 (5,120 instead of 40,960 weights in that 1x1) | not written | mAP2 0.205 vs 0.210 camera-only | TIE, not adopted |
| 3 | camera fix: are frozen features the bottleneck? | `t3_cam_adapter`: zero-init residual adapter on the cached context | maps +0.02, mAP +0.01 | mAP2 0.211 vs 0.210 (miss); vehicle AP@0.5 0.046 vs 0.045; divider +0.021 (hit, one map) | TIE, not adopted |
| 4 | radar-only / camera-only value | `t3_radar_only` (camera off, fixed code) | - | radar only 0.062, camera only 0.210, both 0.339 | radar alone is weak, each sensor is needed, the two complement each other |
| 5 | standard fusion 1 | gates | tie | 0.330 vs 0.339 | TIE |
| 6 | tracking / time | ConvGRU, no block | GRU tie, none -0.01 | 0.327, 0.321 | TIE |
| 7 | standard fusion 2 | CMX | -0.02 | 0.332 | TIE (better than predicted) |
| 8 | standard radar encoder | pillars | tie | 0.349 | TIE (highest mAP2, below the seed spread): needs a second seed |
| 9 (extra) | velocity head | `t3_vel10`: velocity loss weight 10 instead of 1 | - | no effect: front mAP2 0.360, velocity output still zero (update above, try 9) | not adopted |

## 5 Why the ideas that did not work do not work

- **Frozen-feature adapter and height-sum collapse (camera-only localisation).** The information is there: a linear probe on the lifted features, fitted on 30 train scenes and tested on 20 unseen
  val scenes, reaches vehicle IoU 0.249 (per-cell prior map 0.021), drivable 0.577 (0.459), walkway 0.200 (0.121); in the trained camera-only model the probe stays at 0.26-0.29 vehicle IoU
  through camera BEV, fused BEV, trunk and head input, and the model's own dense vehicle output reaches 0.279 (`temp/calib/probe_*.log`). Camera-only vehicle AP rises from 0.045 at 0.5 m to 0.448 at 2 m (pedestrian 0.002 -> 0.104):
  the camera finds vehicles but not their exact position. The probes are cell IoU of vehicle / drivable / walkway masks (lane dividers: 0.031 against a prior of 0.018, not readable), so they show
  that coarse information survives, not 0.5 m localisation; stride-16 features and a depth distribution in 0.5 m bins are the working hypothesis for the precision limit. Untested next steps: stride-8 image features, a depth-refinement
  step, and what the radar fusion already does (AP@0.5 0.228).
- **Standard fusion, temporal and radar blocks tie.** In a 3000-step screening (about half an epoch: 9,000 clips of the 16,289 training samples of 428 scenes) these blocks reach the same level within the noise; I read that as training-limited,
  but the long runs that would show it are not finished. The simplest block (concat, window memory, mean raster) is not beaten, so it stays (cheapest). The cold protocol (three frames, empty memory) also cannot show
  the value of a carried memory; the warm whole-scene protocol was not compared. There is no explicit tracker (association, ids) in the system, only temporal fusion.
- **Velocity.** The velocity loss weight (1.0) and the schedule leave the head at zero output: predicted speed 0.00 m/s, so the measured error equals the zero-velocity baseline of each model's matched vehicles
  (`ref_s0` 2.39, `radar_off` 2.53, `t3_radar_only` 2.59; 12 of the 13 3000-step models checked). Whether weight 10 fixes it is running.
- **Encoder pre-training.** Random initialisation (`trunk_rand`) is worse on walkway (-0.055*), with smaller same-sign moves on pedestrian AP (-0.024*, -0.026*) and vehicle AP@2 (-0.014*); mAP2 -0.021 is
  inside the band. No arm isolates the encoder inside CMX (it changes fusion and encoder together), so nothing is concluded about the value of the pre-trained encoder for detection.

## 6 Output images

Same four validation frames, four systems (camera | ground truth | camera only | radar only | camera + radar 3000 steps | camera + radar 7000 x 2 steps; boxes score >= 0.3, GT dashed):
![systems](images/compare_systems.png)
Reading: radar alone gives speckled clutter and no pedestrians; camera alone finds vehicles at about the right place but with range errors and misses; the fusion gives clean boxes; the long-trained
system detects more pedestrians (pedestrian AP@1 0.172 vs 0.141 on the same frames; one seed) and has sharper maps. More frames: `runs/t3_img_*/eval/images/`.

## 7 Speed

Only one idle-GPU measurement exists (`runs/t1_r4_soa/eval/latency.json`, RTX 3060, batch 1): BEV network 50.7 ms per frame at one target time, 68.2 ms with the three target times of the time mover,
plus 17.2 ms for the image heads (ResNet-50). Per-block timings (gate, CMX, pillars, SECOND) were NOT measured: the GPU is occupied by the training queue; they are the first job for an idle window.
By the rule "TIE -> the cheaper block" no standard block replaces the simple reference for now.

## 8 Compatibility with SOA

nuScenes val, ResNet-50, 256 x 704, six cameras, 10 classes (secondary sources, to be verified against the papers; `soa_reference.md`): CRN camera + radar mAP 0.490 / NDS 0.560, RCBEVDet 0.453 / 0.568,
RCBEVDet++ 0.519 / 0.604 ([RCBEVDet++ paper](https://arxiv.org/pdf/2409.04979)); camera-only BEVDepth 0.351 / 0.475 (quoted table, unverified). DualBEV's own R50 val number was not found
(test set, V2-99: 0.552 / 0.634, [arXiv 2403.05402](https://arxiv.org/pdf/2403.05402)).

Not comparable with ours, and where the gap is: front camera only (one of six), 60 m x 52 m front sector, three class groups, no NDS, mAP2 over four distances instead of the 10-class mAP, 82 of 150 val scenes,
3000 optimiser steps on 428 scenes (half an epoch; the published systems train about 20-24 epochs on the 700 training scenes of the official split, six cameras), frozen image heads, no velocity (error equals zero baseline). What carries over qualitatively: radar adds to the camera
(here camera 0.210 -> camera + radar 0.339 mAP2; the published camera-only versus camera + radar gaps were not verified for this report), a camera-only head localises coarsely, the standard fusion blocks are
interchangeable here. What is missing for a SOA claim: six cameras, the full class set and NDS, about 20 epochs, velocity learned, all 700 training scenes of the official split. No claim of SOA performance is made.

## 9 Errors made and corrected during this work

1. `CachedBEVFusionModel` ignored `camera_off`: the first radar-only arm (`t2_cam_off`) ran with the camera and equalled the reference. I derived a "no-sensor floor" and "the camera path does not work" from it, and
   retracted both (`experiment.md` section 5). Found by a stage probe of the supposed floor model; fixed (commit d57bcc2, self-test `camera-off-selftest`); the real radar-only arm is `t3_radar_only`.
2. The 3-class mAP led to a seed spread that looked like a kill criterion (0.049); it is one rare class. Metrics changed as described in section 2.
3. A full C: disk (orphaned WSL swap files, cleaned with Mirko's OK) crashed `fusion_gate` at the write of its step-3000 checkpoint (last good checkpoint 2500) and killed the queue (about 4.5 h of idle GPU, per the experiment log); `fusion_gate` was resumed from step 2500 (500 of 3000 steps run after the restart).
4. My first predictions for the sensor controls were wrong in both directions (zero camera mAP3 0.013 instead of 0.20-0.26).

## 10 Weak spots and what to do next (ranked)

1. Train longer and on more data (6000 steps queued; 700 scenes need the parts 7-10 caches, built; 11 GB of feature shards must move back from D:). Hypothesis to test: dividers, walkway, velocity improve.
2. Velocity head: weight x10 (running); if it stays at zero, a Doppler-aware velocity branch.
3. Camera localisation: stride-8 features, depth refinement; the adapter on the frozen features did not help beyond the band.
4. Per-block latency on an idle GPU; second seeds for the candidates (pillars) before adoption.
5. Compare the warm whole-scene protocol for the temporal blocks and build a real tracker if tracking is the goal.
6. Open: the full SOA protocol (six cameras, 10 classes) is a different project step.

## 11 Reproduce

Code at commits listed in `experiment.md` (`0a3d73c` swap arms, `1a6b8d8` camera fixes, `d57bcc2` camera-off fix); queue and lanes `temp/study/lane2.sh`, `lane3.sh`; tables `temp/study/collect.py`;
controls `evaluate_run.py --control`; probes `temp/calib/probe_*.py`; prior detector `temp/calib/prior_detector.py`; checkpoints of finished runs moved to `D:\Auto-research\bev_fusion_ckpts\` (copy back to re-evaluate).
