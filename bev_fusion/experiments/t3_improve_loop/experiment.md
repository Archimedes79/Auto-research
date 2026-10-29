# t3_improve_loop - eight documented tries to improve the camera + radar BEV system (card + living log, started 2026-10-29)

Request (Mirko, 2026-10-29): improve performance step by step; start by using what already performs (port weak blocks into our system, or our
blocks into a standard network); test components and training on the way to SOA level; look into the components that do NOT work and say why;
document every step; after 8 tries a report with output images, performance, weak spots and compatibility with SOA.
Earlier guidance (2026-10-29): start from a good vision BEV network, one step at a time, calibration first, fall back to fast standard blocks.

## 0 Where we start (from `t2_block_swaps`, cold VAL-1000, 82 scenes, 3000 steps, seed 0, numbers from `runs/t2_*/eval/metrics.json`)

| system | mAP | veh AP@2 | ped AP@2 | drivable | walkway | lane div@0.5 |
|---|---|---|---|---|---|---|
| `ref_s0` (champion: vision net + radar + window memory + concat fusion) | 0.277 | 0.606 | 0.204 | 0.666 | 0.191 | 0.181 |
| `radar_off` (rung 0, camera only) | 0.147 | 0.448 | 0.104 | 0.637 | 0.107 | 0.164 |
| `cam_off` (radar only) | 0.247 | 0.610 | 0.195 | 0.653 | 0.174 | 0.171 |

Calibration is checked and right (`t2_block_swaps/results/calibration.md`). Open puzzles: (a) radar-only keeps almost all detection AP,
including pedestrians (0.195): is that real or an artefact? (b) the camera adds little on top of radar; (c) maps are blurry, detection overfits.

## 1 The eight tries (one change each against the champion `ref_s0`, same data / seed / schedule unless the try says so; ~1.3 h GPU each)

| try | question | what is done | prediction (written before the run) | decision rule |
|---|---|---|---|---|
| 1 | does the model really use each input? is radar-only real? | sanity controls on trained checkpoints: shuffled / zeroed radar, shuffled / zeroed camera features (`evaluate_run.py --control`), no training | `cam_off` + shuffled radar: mAP < 0.03; `ref_s0` + zero camera: mAP 0.20-0.26; `ref_s0` + shuffled camera: below zero camera | shuffled radar mAP > 0.10 for `cam_off` = artefact (prior / leak) -> stop and find it first |
| 2 | standard fusion 1: rectify gates (GaussianCaR style) | `fusion_gate` | tie (+-0.01 mAP) | band = max(0.010, 2 x seed spread), paired bootstrap |
| 3 | tracking / time vs a standard element: ego warp + ConvGRU (FIERY style); null = current frame only | `camtemp_gru`, `camtemp_none` | GRU -0.01 .. tie; none -0.01 | same |
| 4 | standard fusion 2: CMX cross-modal fuser | `fusion_cmx` | -0.02 | same |
| 5 | standard radar encoder: point MLP + scatter-max pillars | `radar_pillars` | tie | same |
| 6 | are the frozen detection-trained image features the bottleneck (blurry maps, little camera gain)? | trainable zero-initialised residual adapter on the cached context features before the lift (`cam_adapter`) | maps +0.02 IoU, mAP +0.01 | BETTER on a headline metric and no WORSE -> adopt |
| 7 | is 3000 steps too short? | champion config with 6000 steps (`ref_long`) | mAP +0.03 | gain > band -> schedule matters, use the long schedule from here |
| 8 | does more data help? | best adopted blocks, 6000 steps, all 700 train scenes (parts 1-10), evaluated on the same 82 val scenes and on all 150 | +0.02 mAP over try 7 | gain > band -> data-limited |

Controls of every try: `ref_s0` and `ref_s1` (noise floor); a try is WORSE / BETTER only outside the band AND with a paired-bootstrap interval that
excludes 0; TIE -> the cheaper variant. Order in the GPU queue: 1, 2, 3, 4, 5 (queue `temp/study/queue.txt`, lane `temp/study/lane2.sh`), then 6, 7, 8
(they need new code; lane 3). Not in the eight (kept in the queue behind them): `trunk_second`, `lift_s`, `trunk_small` of the swap study.

## 2 Step log (one entry per step: what was done, the numbers, what it means)

| time | step | result |
|---|---|---|
| 2026-10-29 20:43 | queue lane started (detached Windows process); `camtemp_none` resumes from step 2000 | running |
| 2026-10-29 21:55 | **Try 1 done: sanity controls** (table `try1_controls.md`; raw `runs/*/eval/metrics_ctl_*.json`; `temp/study/collect_ctl.py`) | see section 3 |

## 3 Try 1 - what the model really uses (cold VAL-1000, 82 scenes; controls at evaluation only)

| system | normal | no / shuffled radar | no / shuffled camera |
|---|---|---|---|
| `ref_s0` (3000 steps) mAP | 0.277 | 0.177 / 0.151 | 0.013 / 0.028 |
| `t1_r4_soa` (7000 steps x accum 2) mAP | 0.307 | - / 0.156 | 0.031 / 0.036 |
| `cam_off` (trained without camera) mAP | 0.247 | 0.156 / 0.136 | - |
| `radar_off` (trained without radar) mAP | 0.147 | - | 0.000 / 0.009 |

Findings (numbers: `try1_controls.md`):
1. **The metric has a high floor.** `cam_off` with its radar removed has NO sensor input at all, and still scores mAP 0.156, vehicle AP@2 0.445, drivable
   0.629, lane divider 0.162 (a learned mean-position prior + ego motion; a likely reason for the height of the floor, NOT measured: up to k = 100 detections per frame at score 0.05 with
   2 m / 4 m matching circles cover a large part of the 60 x 52 m area). My prediction "shuffled radar < 0.03" was wrong for that reason. From now on every table carries the floor row and the gain over the floor:
   `ref_s0` mAP gain +0.121 (of 0.277), vehicle AP@2 +0.161, drivable +0.037, walkway +0.117, lane divider +0.019.
2. **Radar carries the detection signal.** Radar-only (`cam_off`) gains +0.091 over the floor, with real radar replaced by another sample's the model drops to the floor.
   Radar-only is NOT a leak: shuffled radar gives the floor, real radar +0.09.
3. **THE CAMERA PATH DOES NOT WORK ON ITS OWN.** Camera-only (`radar_off`) scores mAP 0.147 = the floor (0.156); vehicle AP@2 0.448 vs floor 0.445; AP@0.5 0.045 vs
   0.047; AP@4 0.617 vs 0.610. On TRAIN scenes the same model gets mAP 0.213 (vehicle AP@2 0.566, `metrics_train.json`): it fits training scenes but learns nothing that
   transfers. The long-trained `t1_r4_soa` behaves the same (shuffled radar = floor 0.156): the camera adds +0.03 over radar-only in `ref_s0` and nothing without radar.
4. **The joint model depends on the camera input without gaining from it alone**: zero / shuffled camera collapses `ref_s0` to 0.01-0.03 (below the floor: out-of-distribution
   fused features), though `cam_off` trained without it keeps 0.247.
5. **Not a data / calibration bug**: depth-head alignment on TRAIN scenes peaks at (0, 0) exactly as on VAL (mass 0.691, `temp/calib/check_calibration_train.log`);
   geometry matches the devkit to 1e-10. Camera-only training scenes are correctly paired with their features.
6. Maps: the camera adds +0.03 walkway over the floor in camera-only; lane dividers are at the floor in every system (0.162-0.181): the divider head learns almost nothing
   beyond the prior, in the long-trained system too (0.273 vs floor 0.162, the only strong gain, from 2.6 epochs).

Decision (card rule: "mAP > 0.10 for shuffled radar = find the cause first"): the cause is the metric floor, not a leak; radar-only is genuine. The weak spot of the
system is the camera path (information gain ~0 on unseen scenes). Tries 6 and 7 are re-aimed at it and run as CAMERA-ONLY arms (the radar hides camera gains in the full system):
`t3_cam_adapter` (frozen features are the bottleneck?) and `t3_cam_zsum` (the 640-channel height-into-channels 1x1 conv is learned from scratch and overfits; the standard LSS /
DualBEV collapse sums over heights and has 10 x fewer parameters). Success criterion for a camera-only arm: mAP above the floor 0.156 by more than the band, i.e. >= 0.17, and
AP@0.5 above the floor 0.047.

## 4 Metric hygiene decided on the data (2026-10-29 23:20, after the noise seed `ref_s1` came in)

`ref_s0` and `ref_s1` (same config, seed 0 / 1, `results/arms.md`) differ by 0.049 in the 3-class mAP, far above the 0.02 kill criterion of the swap card - but the whole
difference is the rare class `two_wheeler` (174 GT in 1000 frames; AP@2 0.196 vs 0.047). Vehicle AP@1 / AP@2 (0.468 / 0.465, 0.606 / 0.605), drivable (0.666 / 0.663)
and lane divider (0.181 / 0.179) agree; walkway differs by 0.027, pedestrian AP@1 by 0.020. Consequences, applied from here on (tool `temp/study/collect.py`, `evaluate_run.compare`):
- headline = `mAP2` (vehicle + pedestrian, mean over the four matching distances), vehicle / pedestrian AP@1 and AP@2, drivable / walkway / lane-divider IoU; `mAP3` and the
  `two_wheeler` AP are reported but never decide a verdict (one seed cannot resolve them);
- AP@1 added to the paired bootstrap (AP@2 has the high no-sensor floor, try 1); the noise band is 2 x the seed spread of the stable metrics;
- every table carries the no-sensor floor row (`cam_off` with zero radar: mAP2 / AP@1 / drivable / walkway / divider = see `try1_controls.md`).
Re-read verdicts of the swap study with these rules (`results/arms.md`, seed spread of mAP2 0.015): `radar_off` WORSE (veh AP@1 -0.26*, ped AP@1 -0.11*), `lift_p` WORSE
(veh AP@1 -0.042*, ped AP@1 -0.048*), `trunk_rand` WORSE (walkway -0.055*), `cam_off` TIE, `camtemp_none` TIE (ped AP@1 -0.028* is close to the band).

| 2026-10-29 23:20 | `ref_s1` evaluated; mAP3 noise recognised (two_wheeler); metric set changed (section 4) | `cam_off` (radar only) is a TIE with the full system on detection |
| 2026-10-29 23:50 | housekeeping: C: fell to 2.9 GB free (not from this study: its runs on C: total 3.6 GB); the checkpoint mover had stopped with lane 2; superseded checkpoints of `t1_r0`, `t1_r3_combined`, `t1_r4_soa` moved to `D:\Auto-research\bev_fusion_ckpts\` (size-checked, MOVED.txt), mover restarted and extended to `runs/t3_*` | C: 4.2 GB free; D: has the rest |
