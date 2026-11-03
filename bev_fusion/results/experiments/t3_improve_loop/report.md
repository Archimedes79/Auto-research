# Eight tries to improve the camera + radar BEV system: what works, what does not, why, and how it relates to SOA

Status 2026-10-30, reviewed once by a fresh reviewer against the primary files (corrections applied). Eight tries done (plus one running). The tables come from
`runs/*/eval/metrics*.json` through `temp/study/collect.py` (`../t2_block_swaps/results/arms.md`); other numbers come from the named scripts and logs (probes `temp/calib/probe_*.log`,
velocity `temp/calib/velocity_baseline.py`, controls `try1_controls.md`, latency `runs/t1_r4_soa/eval/latency.json`); no number is typed by hand. All tables: cold VAL, 1000 keyframes spread over the 82 validation scenes (parts 1-6 of
nuScenes, 150 scenes exist), EMA weights, 3000 optimiser steps from the same 428 training scenes, seed 0 unless stated. This is a short-schedule
screening study of blocks, not a final system.

## Update 2026-11-01 (read this first; sections 1-11 are the state of 2026-10-30 and stay as written)

Added since: tries 9 and 10 (`t3_vel10`, `t4_fpn_order`) and six follow-up steps (`../t4_soa_front`, `../t6_soa_gap`, `../../../experiments/t7_lane_branch`, `../t8_bevfeat`,
`../../../experiments/t10_dual_features`, `../t11_vel_cause`). Numbers: cold VAL, the same 1000 keyframes of the 82 validation scenes unless a row says otherwise; maps from
`runs/*/eval/metrics.json` (`temp/pitch/ladder.py`), objects in the front sector (0-50 m, +-32.3 deg) from `../t4_soa_front/results_*.md` and
`../../../experiments/t10_dual_features/results_x50.md` and `results_x50_2seeds.md` (`temp/t4/compare_front.py`). Seeds: most arms have one seed; where a seed twin exists the row says so.
Counts and dates as of 2026-11-01 04:30. Reviewed against the primary files by a fresh reviewer at 04:20 (its corrections are applied). Updated 08:40 with the velocity fix (try 17) and
the like-for-like map scores (try 18, replacing the indicative shares); these two additions have not been re-reviewed.

### What changes in the bottom line

1. **The lever is the image features, not the blocks.** Eight swaps of the trunk, fusion, temporal and radar blocks tied (the study's eleven exchanges: 7 ties, 4 worse); swapping the frozen
   detection-trained stride-16 image features for map-trained stride-8 BEVFusion features raised lane divider, walkway and road divider x1.8 - 2.0 (lane divider 0.191 -> 0.351 / 0.345, walkway
   0.178 -> 0.339 / 0.342, crossing 0.002 -> 0.145 / 0.165; drivable x1.1, carpark x1.5) and halved the objects (front mAP2 0.367 / 0.368 -> 0.183 / 0.180). Both feature sets in one chain keep
   the maps and nearly keep the objects, on two seeds: lane divider 0.335 / 0.350, walkway 0.340 / 0.354, drivable 0.806 / 0.807 (drivable@0.5 m), front mAP2 0.361 / 0.364 against the control's
   0.367 / 0.368 (0.003 - 0.007 lower in every pairing, inside the rule's 0.054 band, bootstrap intervals touching zero; both second seeds were resumed after a crash on a later code snapshot).
   The 3000-step dual chain beats the start recipe trained with 4.7 x the training samples (7000 steps at accumulation 2 = 2.3 x the steps at double the batch; a different config and frame set)
   on five of six map layers (ladder below). The better features are a published, map-trained model's: the saving is partly borrowed compute. The swap is a bundle (backbone,
   stride, objective and depth head differ), planned in plan 2.8 with a qualitative rule, not a numeric card. Not claimed: the rare two-wheeler class and the 3-class mAP
   (two-wheeler AP@2 control 0.121 / 0.051, dual 0.016 / 0.061; 3-class mAP control 0.251 / 0.235, dual 0.221 / 0.234: differences of the order of the control's own seed spread, but below its range except one value); an earlier one-seed gain of
   the pedestrian segmentation (0.000 -> 0.091) was noise, the control's second seed scores 0.090.
2. **The velocity head was dead, not slow, and a fix revived it.** The output of the branch was its bias; 0.011 % of its hidden units are active at the supervised centres (13 - 50 % for the other
   branches); 28 of the 30 runs with an effective batch of 3 are dead (`../../../temp/box_vel/out/alive_table.txt`). Try 9 (loss weight x10) could not help a dead ReLU layer. A moving / static
   logit with the regression on moving centres only (`../t11_vel_hurdle`, card written before any code) lifted the vehicle velocity skill from 0.000 to 0.470 (batch 3) and 0.566 (accumulation 2)
   in 1000 warm-start steps from the dual chain, one seed: vehicle AVE 2.39 -> 1.19 m/s, at a price of front mAP2 -0.014 and an error of 0.28 m/s at static cars. The radar Doppler read-out at
   the GT boxes holds 0.79 of the velocity of moving vehicles (upper bound, GT heading); the older learned heads reached 0.22 - 0.25 only after 7000 x 2 steps.
3. **The SOA gap is measured like for like where a published system can be run on our frames**: DualBEV (front camera only) 0.250 front mAP2 against our camera-only 0.242 and
   camera + radar 0.359 - 0.400; GaussianCaR on the maps (tables below). Still not comparable with published leaderboards.
4. **Three errors found since** (they continue the list of section 9): the FPN-LSS inputs of our trunk were concatenated in the wrong order for the pre-trained weights
   (`t4_fpn_order`: a tie); the GT cache is built from live code, so runs of different days had different object targets (same-code controls, decision D29);
   the velocity branch was dead (above).

### Tries 9 - 16

| # | question | change | prediction (written before) | result | decision |
|---|---|---|---|---|---|
| 9 | does a larger velocity loss weight make the head learn? | `t3_vel10`: weight 10 | - | front mAP2 0.360 (reference 0.375 / 0.359); the velocity output stays zero (largest predicted speed 0.031 m/s) | no effect, not adopted; cause found in 16 |
| 10 | does the wrong FPN-LSS input order of our trunk matter? | `t4_fpn_order`: original order (flag) | - | front mAP2 0.367 against 0.375 / 0.359 | TIE, flag stays default off |
| 11 | how far is the system from a SOA camera detector in the front area? | DualBEV run with CAM_FRONT only, unchanged weights, same 1000 frames (`t4_soa_front`) | DualBEV 0.40 - 0.55 (miss) | DualBEV 0.250; ours camera-only 0.242; camera + radar 0.359 / 0.375 (3000 steps), 0.400 (7000 x 2) | camera path within 0.01 of the SOA port (-0.009, interval [-0.021, -0.002]); radar +0.11 - 0.15 above it (intervals [+0.093, +0.169]) |
| 12 | why are the maps far from the SOA? | GaussianCaR scored by our metric, oracles, probes, a maps-only run (`t6_soa_gap`) | - | our 7000 x 2-step chain at 17 - 55 % of GaussianCaR on 400 frames except drivable (85 %); not range, threshold, over-fitting (big areas), BEV blocks (eight swaps tied) or labels; task competition small (+0.015 - 0.051) | main candidate: frozen detection-trained stride-16 features (M6) plus the training budget |
| 13 | do stride-4 image details help the thin map classes? | `t7_res4` / `t7_rgb`: a frozen ResNet stride-4 branch lifted into the BEV | lane divider +0.04 | res4: lane divider +0.065 / +0.042 (two seeds), road divider IoU@1 +0.119 / +0.096, walkway and crossing unchanged; rgb +0.015 | helps the thin lines, not the area classes; verdict waits for the equal-GT control |
| 14 | are frozen detection-trained features the limit of the maps? | `t8_bevfeat`: cached BEVFusion map-trained stride-8 features, same chain, same-code control `t8_ctl` | maps up | lane divider, walkway, road divider x1.8 - 2.0, drivable x1.1, carpark x1.5 (two seeds agree); front mAP2 0.367 -> 0.183 / 0.180, pedestrians gone | supports M6; neither feature set alone gives both |
| 15 | can both be had? | `t10_dual`: both caches, two lifts, merged by a 1x1 convolution | lane divider 0.34, drivable 0.80, walkway 0.33, front mAP2 0.33 (0.25 - 0.38), pedestrian AP@2 0.15 | seed 0 / seed 1: lane divider 0.335 / 0.350, drivable 0.806 / 0.807, walkway 0.340 / 0.354, front mAP2 0.361 / 0.364, pedestrian AP@2 0.199 / 0.202 | rule E1 holds on both seeds; the dual chain is the proposed new base (promotion is Mirko's) |
| 16 | why is the box velocity zero in every 3000-step run? | diagnosis from stored checkpoints, detections and a probe (`t11_vel_cause`) | active share < 1 % at the centres; linear read-out skill >= 0.15 | 0.011 % (other branches 13 - 50 %); 28 of 30 batch-3 runs dead (alive table of 36 checkpoints; every row with an effective batch of 6 or more is alive); the linear read-out gives 0.019 (that prediction failed) | a diagnosis; the fix is try 17 |
| 17 | does a moving / static split of the velocity loss revive the branch? | `t11_vel_hurdle` (B', batch 3) and `t11_vel_hurdle_acc2` (B2, accumulation 2): a moving / static logit, the regression on moving centres only, a decode gate; 1000 steps from `t10_dual_s0`, fresh velocity branch | vehicle velocity skill 0.25 (0.12 - 0.40), error at static cars 0.20 (<= 0.30 m/s), AVE 1.95 | B': skill 0.470, static 0.336, AVE 1.445, front mAP2 0.346; B2: skill 0.566, static 0.284, AVE 1.189, front mAP2 0.347 (-0.014 [-0.020, -0.006]); drivable, walkway and lane divider within 0.01 of `t10_dual_s0`, carpark +0.03 (more samples or the hurdle, not separable) | V1 (learns) for B2 by the card's rule; B' misses the static clause; the prediction missed on the good side; one seed; the 6000-step run tests it from scratch |
| 18 | how far are the maps from GaussianCaR on identical frames? | the same 400 frames scored for GaussianCaR and four of our chains (`pitch.maps_lfl`, GPU, checked against the stored CPU run) | dual chain 0.90 - 0.95 / 0.57 - 0.65 / 0.25 - 0.40 / 0.28 - 0.35 / 0.48 - 0.56 / 0.70 - 0.80 of GaussianCaR (drivable, walkway, carpark, crossing, lane divider, road divider) | dual chain, two seeds: 94 / 57 - 62 / 31 - 34 / 26 - 31 / 50 - 52 / 72 - 76 % | seed 1 inside all six predicted ranges, seed 0 inside four (walkway 0.567 against 0.57, crossing 0.26 against 0.28); replaces the indicative shares |

### Maps: the ladder (same metric, VAL, `seg_fov`, dt 0)

| run | what | step | frames | drivable | walkway | carpark | crossing | lane div. @0.5 m | road div. @0.5 m |
|---|---|---|---|---|---|---|---|---|---|
| t2_ref_s0 | start, 3000 steps | 3000 | 1000 | 0.666 | 0.191 | 0.106 | 0.002 | 0.181 | 0.220 |
| t3_ref_long | start, 6000 steps | 6000 | 1000 | 0.689 | 0.225 | 0.155 | 0.057 | 0.234 | 0.297 |
| t1_r4_soa | start, 7000 x 2 steps (4.7 x the training samples) | 7000 | 3109 | 0.698 | 0.244 | 0.162 | 0.093 | 0.274 | 0.322 |
| t8_ctl_s0 | same-code control of t8 / t10 | 3000 | 1000 | 0.668 | 0.178 | 0.115 | 0.002 | 0.191 | 0.224 |
| t8_bevfeat_s0 | map-trained features | 3000 | 1000 | 0.754 | 0.339 | 0.173 | 0.145 | 0.351 | 0.437 |
| t8_bevfeat_s1 | map-trained features, seed 1 | 3000 | 1000 | 0.759 | 0.342 | 0.168 | 0.165 | 0.345 | 0.439 |
| t10_dual_s0 | both feature sets | 3000 | 1000 | 0.756 | 0.340 | 0.144 | 0.154 | 0.335 | 0.442 |
| t10_dual_s1 | both feature sets, seed 1 | 3000 | 1000 | 0.757 | 0.354 | 0.139 | 0.173 | 0.350 | 0.442 |
| t8_ctl_s1 | same-code control, seed 1 | 3000 | 1000 | 0.668 | 0.176 | 0.106 | 0.000 | 0.180 | 0.227 |

The 3000-step dual chain is above the 7000 x 2-step start on drivable, walkway, crossing, lane divider and road divider and below it on carpark (0.144 against 0.162); the
frame sets differ (3109 against 1000 for the 7000 x 2 run, which differ by at most 0.003; the 400 GaussianCaR frames are easier, see below).

### Maps against the SOA (GaussianCaR), on identical frames (measured 2026-11-01)

Run `pitch.maps_lfl` (GPU, `temp/soa_gap/map_ref_dev.py`): GaussianCaR and four of our chains scored on the SAME 400 frames of 43 validation scenes (the frames where GaussianCaR's cached logits
exist), with the same camera-cell mask, tolerances and threshold (0.5), EMA weights, dt 0, cold. The GPU path was checked against the stored CPU run of another model (`pitch.maps_lfl_check`:
largest difference over all numbers 0.0002, the GaussianCaR block identical). Tables: `temp/pitch/maps_lfl.md` (made by `temp/pitch/maps_shares.py`). One run per model except the dual chain
(two seeds). The earlier indicative shares (our VAL-1000 numbers over the 400-frame SOA: 89 / 57 / 22 / 29 / 48 / 70 % for the dual chain, seed 0) and the frame-set estimate made from another
model's ratios (92 / 59 - 62 / 36 - 37 / 28 - 32 / 52 - 54 / 76 %) are replaced by the measured values below; they lie within 0.06 of that estimate on every layer.

IoU on the 400 frames, share of GaussianCaR in brackets (both systems at the 0.5 threshold):

| layer | GaussianCaR | image features only (control `t8_ctl_s0`) | BEVFusion features only (`t8_bevfeat_s0`) | dual chain, seed 0 | dual chain, seed 1 |
|---|---|---|---|---|---|
| drivable (0 m) | 0.845 | 0.693 (0.82) | 0.788 (0.93) | 0.790 (0.94) | 0.796 (0.94) |
| walkway (0 m) | 0.594 | 0.196 (0.33) | 0.348 (0.59) | 0.337 (0.57) | 0.366 (0.62) |
| carpark (0 m) | 0.659 | 0.181 (0.27) | 0.303 (0.46) | 0.222 (0.34) | 0.207 (0.31) |
| crossing (0 m) | 0.536 | 0.001 (0.00) | 0.147 (0.27) | 0.140 (0.26) | 0.164 (0.31) |
| lane divider (0.5 m) | 0.691 | 0.191 (0.28) | 0.366 (0.53) | 0.347 (0.50) | 0.362 (0.52) |
| road divider (0.5 m) | 0.634 | 0.238 (0.38) | 0.471 (0.74) | 0.455 (0.72) | 0.479 (0.76) |

Reading: four of six layers are at or above 50 % of GaussianCaR for both dual seeds (drivable 94, road divider 72 - 76, walkway 57 - 62, lane divider 50 - 52 %); carpark (31 - 34 %) and crossing
(26 - 31 %) remain the farthest. With each system at its best of three thresholds (in-sample for both) the lane divider falls to 46 / 48 % and the road divider to 66 / 70 % (GaussianCaR gains
more: lane divider 0.691 -> 0.761, road divider 0.634 -> 0.697); the other layers move by at most 0.03. BEVFusion's features alone are level with the dual chain on five layers and ahead on the
carpark (0.46 against 0.31 - 0.34 on these frames, 0.17 against 0.14 on VAL-1000): the dual chain was built to bring the objects back, not to improve the maps. Prediction written before the run
(ledger row `pitch.maps_lfl`): seed 1 inside all six written ranges, seed 0 inside four (its walkway 0.567 just below 0.57, its crossing 0.26 below 0.28 - 0.35). Limit of the harness (`../t6_soa_gap/report.md` section 1):
GaussianCaR's grid was aligned to ours by an orientation search on 40 frames and a forward offset of +-0.5 m is not resolved (drivable 0.840 at 0, 0.842 at +0.94; +0.5 m used); the thin layers at 0.5 m tolerance could move by about
one cell, so every share carries that uncertainty; it was not re-scored.

### Weak spots, ranked (replaces section 10)

1. **Velocity**: the branch was dead; the gate with accumulation 2 revives it (skill 0.566, one seed, front mAP2 -0.014); the seed twin and the from-scratch run (the 6000-step run) are next; a Doppler read-out for the moving error is a ranked candidate (`t11_vel_hurdle`, `t11_motion_field`).
2. **Thin map classes and the training budget**: lane divider 0.347 against 0.691 for GaussianCaR on the same 400 frames (0.335 on VAL-1000); 3000 steps are 0.55 epoch of 428 scenes. Dual features trained longer, the
   stride-4 branch on top, all 700 scenes.
3. **Objects at range**: precision of the confident detections falls from 0.89 (0 - 20 m) to 0.64 (40 - 60 m); far objects are mostly detected with low confidence (t6).
4. **No SOA camera + radar detector was run on our frames** (CRN checkpoint on disk).
5. **Per-block speed on an idle GPU** is still unmeasured.
6. **Infrastructure**: C: was at 4.1 GB free overnight (67 GB after the probe dumps were archived; a full disk crashed a run once); the dual seed twin was killed once by the out-of-memory
   killer and the control's second seed resumed after a CUDA crash, both on later code snapshots; one GT version (D29) is open.

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
