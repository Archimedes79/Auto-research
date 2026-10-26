# S1a Lift PD against lift P, front camera only, nuScenes VAL IoU by range (+ idle-GPU timing)

First real-data run of the camera branch (plan.md section 11, stage S1; Mirko, 2026-10-26: "plan the first real-data camera-only run (lift PD against lift P on the same frames, VAL IoU by range) and the idle-GPU timing" and "plan really a unit test allowing to train and test artificially that it works"). Status: PLAN, nothing built or run yet. Order: section 3 (synthetic unit test of the same code) must pass before section 2's real run starts; section 4 (timing) runs in the first idle-GPU window, independent of the rest.

## 0. What the run is (the fields of a later `experiment.json`)
| field | value |
|---|---|
| id / title | S1a / lift PD vs lift P, front camera, frozen DualBEV camera heads |
| kind | block (new code: feature cache, training / evaluation driver, metric) |
| metric, direction, band | vehicle IoU on camera-visible cells, 0-60 m, VAL; higher; band 0.010 (checked against the seed spread, the larger one counts) |
| prediction | PD - P = +0.03 vehicle IoU (0-60 m); drivable area and dividers within the band |
| arms | P: LiftPD(use_depth=False, oversample=4, z = 0 slice), 2 seeds; PD: LiftPD(use_depth=True, same), 2 seeds; PD-shuffled (control): PD with the depth of another frame, 1 seed |
| control | P (the matched reference: same frames, cache, trunk, head, steps, seeds) |
| rule | see 1. Decision rule |
| budget, kill | about 5 GPU h for the five runs + 1 h feature cache; kill: see 1 |
| evaluate | VAL (official val scenes on disk), all keyframes, camera-visible cells; by class and range band 0-20 / 20-40 / 40-60 m |

## 1. Card
| | |
|---|---|
| lens | foundations (does the depth weight carry information on real images?) |
| sources | plan.md 2.4 (PD, DualBEV HeightTrans 2403.05402: depth-weighted sampling 30.6 mAP / 39.0 NDS at 0.32 ms vs unweighted 30.5 / 37.6 at 7.15 ms, detection, 6 cameras); Simple-BEV 2206.07959 (the depth-free pull is competitive on BEV segmentation); oracle-lift gate (tests/test_oracle_lift.py: with exact depth lanes 1.00 / 1.00 / 0.81 by range) |
| status | proposed |

- **Statement**: on real front-camera images with the frozen DualBEV heads, weighting the pulled features by the predicted depth probability (PD) gives a higher vehicle IoU at 0-60 m than the same pull without depth (P).
- **Mechanism**: P writes a pixel's feature into every voxel its ray crosses (only the height channels tell the trunk where along the ray it belongs); PD puts most of the mass where the depth head believes the surface is. Objects stand above the road, so for them the height cue of P is weakest and the depth weight helps most. Against it: Simple-BEV shows that a trunk can undo much of the smearing, and the frozen depth head is sharp near and blurred far (min-depth targets, synthetic finding of 10.1).
- **Prediction**: PD - P = +0.03 vehicle IoU (VAL, 0-60 m, camera-visible cells); drivable area and dividers within +-0.010. The range pattern is reported, not predicted.
- **Decision rule** (paired scene bootstrap over VAL scenes, 2 seeds per arm):
  - PD - P on vehicle IoU >= band with the 95 % interval above 0, and no class worse than -band -> PD confirmed as the camera block (as already fixed, plan 12)
  - |PD - P| < band on every class -> TIE: report to Mirko with the cost (P needs no DepthNet: 35.6 of 71.4 camera GFLOPs); the block decision is his (DECISION entry), not automatic
  - P better by more than the band -> PD refuted on real data: check the depth probe of 10.2 before any other change
  - PD-shuffled must fall to P or below (otherwise PD's gain is not from the depth content: the harness is broken)
  - seed spread larger than the gap -> INCONCLUSIVE: add a seed per arm, no adoption
- **Test design**: arms above; controls: null (per-cell, per-class training mean), matched reference P, PD-shuffled, seed twins, equal budget (same steps, batch, schedule, data order per seed). No real-data oracle (no lidar; plan 12 open 6).
- **Cost**: feature cache about 1 GPU h, GT cache CPU (measure on 50 keyframes first), 5 runs at about 1 GPU h each. **Kill**: the real overfit of one scene (3.4) does not reach IoU 0.9; or after 2,000 steps a run is below null on VAL; or loss NaN / divergence (lr <= 6e-4, clipping).
- **Inputs**: ready: camera_encoder.py (frozen heads, 0 missing keys), lift_pd.py, bev_trunk.py (T2), gt_bev.py, nuscenes_dataset.py (being written by another session: the run uses it read-only with n_cam = 1, no radar). `training/feature_cache.py`, `evaluation/metrics.py` (both of the main chain session, used read-only). To build: `training/train_s1.py`, the per-scene bootstrap on SegMeter counts, the synthetic data adapter of section 3. Loss weights follow plan 8.4: occluded map cells weight 0.5, dense object coverage stays in the semantic head.

## 2. Detail plan of the real run
What would make it fail, in our own words: (a) the frozen context features were trained WITH depth weighting, so P might be handicapped by features it was never trained on (it is the fair test of "does our pre-trained branch need its depth head", not of P as a family; stated in the report); (b) map registration offsets of 0.5-1 m (plan 9.5) blur the divider GT; (c) parts 1-6 only (510 of 850 scenes), so the numbers are not comparable to published full-val numbers (VAL: 82 scenes, a bootstrap over 82 scenes is coarse: expect intervals of a few IoU points); (d) the real encoder's features are at stride 16 (16 x 44), the oracle gate ran at stride 8: re-run at stride 16 (`temp/checks/check_oracle_lift_stride16.py`, 2026-10-26, exact depth, 6 scenes) the recommended lift (oversampling x4 + histogram) keeps road 1.00 and vehicles 0.98 but lane lines fall to 0.70 recall and lane recall by range to 0.99 / 0.40 / 0.17 (stride 8: 1.00 / 1.00 / 0.81) - with a perfect depth. So far-range thin lines are limited by the 16 x 16 pixel groups before any depth error: expect far-range divider IoU in S1a to be low for BOTH arms, and judge PD against P mainly on vehicles, drivable area and the near / mid bands. A stride-8 feature map (DualBEV's FPN has a stride-8 level it does not use) is the remedy to test next, not part of S1a.

| step | what exactly | where | done when | budget |
|---|---|---|---|---|
| data | train = official train scenes on disk, VAL = official val scenes on disk (counted 2026-10-26 with `temp/setup/count_split_scenes.py`: train 428 of 700 scenes, 17,145 CAM_FRONT keyframes (Boston-Seaport 252 scenes, Singapore One-North 122, Queenstown 35, Holland Village 19); VAL 82 of 150 scenes, 3,273 keyframes (Boston-Seaport 55, One-North 27 - VAL covers two of the four locations of train, so it holds no test on a location outside the training scenes)); no TEST use in S1a (a TEST split of held-out val scenes is cut before the first headline claim) | `nuscenes_dataset.available_scenes` | split file `data/splits_s1.json` with scene names | - |
| GT cache | keyframe GT at 0.5 m on the front grid: drivable_area, ped_crossing, walkway, carpark_area, divider (lane + road), vehicle, pedestrian; masks camera, occluded | `gt_bev.NuScenesGT` via the dataset's npz cache | every keyframe of both splits cached; 20 pictures checked (`temp/checks/view_training_samples.py`) | CPU, measure first |
| feature cache | REUSE `training/feature_cache.py` (written by another session, 2026-10-26 01:03): frozen DualBEV heads once per camera frame -> depth probability (118 bins) + context (80 ch) at 16 x 44, float16, shards per scene in `data/cache/` (not in git, resumable); `CachedFrontRadarDataset` / `collate_cached`; build with `feature_cache.py build <dataroot> <version> data/cache --split train|val --device cuda`. S1a needs only the keyframe frames (n_cam = 1): about 0.28 MB per frame | its own self-test (cached BEVs equal the standard model's within fp16); 10 frames of S1a recomputed and compared | about 1 GPU h for the keyframes of both splits |
| driver | one training / evaluation driver for both arms and both data sources (real cache or synthetic): lift table from each sample's calibration, LiftPD(oversample=4, z-slices incl. 0), out_conv -> BEVTrunk T2 -> semantic head (7 classes); BCE per class with pos weight, loss only on camera-visible cells; BEV flip augmentation (y mirrored on the lifted volume and the GT together); AdamW lr 4e-4, clipping 5, batch 8, 10,000 steps, cosine; checkpoint every 500 steps with resume; writes metrics.json | `training/train_s1.py` | the synthetic unit test (3) passes | - |
| metric | IoU per class at threshold 0.5 (GT coverage > 0.5) on camera-visible cells, by range band, per scene counts for the bootstrap; null baseline; divider also with a 1-cell tolerance (map offsets) as a secondary number | `evaluation/metrics.py` `SegMeter` (written by the main chain session): one SegMeter PER SCENE (its counts `meter.c[(dt, layer, subset, bin)]["cnt"]`: tp_pred, fp, tp_gt, fn per tolerance) so the bootstrap can resample scenes; range bins 0-20 / 20-40 / 40-60 m built in; tolerances (0.0, 0.5) give the strict IoU and the 1-cell divider IoU; mask = (gt["camera"] > 0) & (gt["undefined"] == 0); null baseline = the per-cell training-mean map fed through the same SegMeter as a prediction | its self-test passes (known-answer, run in `run_gates.sh`) | - |
| wiring on real data | overfit one real scene (40 keyframes) with each arm | driver, `--overfit` | train IoU >= 0.9 on vehicles and drivable area | 10 min |
| train | P s0, s1; PD s0, s1; PD-shuffled s0 | driver, background, at most 2 GPU jobs, GPU < 11 GB | five metrics.json | about 5 GPU h |
| evaluate | VAL, by class and range; paired scene bootstrap PD vs P; pictures of 6 VAL frames (image, GT, P, PD, depth-weight map) | `evaluation/metrics.py` | table + pictures to Mirko | - |
| decide | the rule of 1 | by hand from the bootstrap output until bev_fusion is an autoresearch problem folder | verdict written here (section 5) | - |

Prerequisite (Mirko to decide): bev_fusion is not yet a problem folder of `../autoresearch/` (no project.json, no ledger, no run.py), so the CLAUDE rule "never start a run outside run.py" cannot be followed as is. Either set up the minimal problem files first (project.json, PROBLEM.md, experiment_execution/ ledger; runs then get a commit tag and a ledger row), or run S1a as documented scripts with result files committed and a short run log in this file.

## 3. Synthetic unit test of the same pipeline (built 2026-10-26, before any real run)
Purpose: train and test the REAL module and the REAL driver end to end on artificial data where the answer is known, so a real-data failure can only come from data, calibration or the frozen heads, not from wiring, losses, the metric or the comparison. AND keep the design anchored to the state of the art (Mirko, 2026-10-26: "do not go away totally from SOA"): the module switches between our design and the published designs with ONE config preset, and the same test trains all of them.

| file | what |
|---|---|
| `training/s1_camera.py` | `CameraBEVSeg(S1Config)`: cached (prob, ctx) -> lift (PD / P / S) -> heights as channels or summed -> reduce -> BEVTrunk -> semantic head. Presets: `ours` (PD, 8 slices incl. z = 0, 4 sub-heights, channels, width 0.5), `dualbev` (HeightTrans heights: 13 fixed, summed to 80 channels, no reduce, trunk width 1.0 = DualBEV's BEV encoder; deviations listed in the file), `simplebev` (P, 8 default slices as channels), `lss` (S forward splat, heights summed). Self-test: shapes, gradients, presets differ in the named choices only, one-hot depth reaches only its own depth |
| `training/train_s1.py` | the driver for synthetic AND real data (it only sees batch dicts): loss (soft BCE + soft IoU on camera-visible cells), AdamW + warm-up + cosine, one SegMeter per scene, null baseline (prior map), paired scene bootstrap. Self-test with known answers (loss, meters, range bands, bootstrap, null, tiny training) |
| `tests/synthetic_s1_data.py` | synthetic samples in the real cache format at the real feature resolution (stride 16, 16 x 44): context = pixel-class fractions + 75 noise channels (perfect semantics, unknown position); depth = oracle histogram, `blurred` (Gaussian in 1/z, 0 m at 3 m, 9 m at 51 m), `shuffled` (another sample's), `uniform`; GT lanes (presence at a coverage of 0.1: the line is 15 cm wide), vehicles, pedestrians; the camera-visible mask; lane lines vary per sample (offset, spacing, yaw) so a per-cell prior cannot solve them |
| `tests/test_s1_pipeline.py` | the gate (below), exit 0 / 1, `run_gates.sh --all` |

Gates (written before the first full run; class mean = mean of IoU@0.5 m over lanes / vehicles / pedestrians, camera-visible cells, 60 held-out synthetic scenes, 300 train scenes, 1,200 steps, batch 8):
| gate | criterion |
|---|---|
| G1 wiring | overfit ONE scene (ours, oracle depth): every class IoU@0.5 m >= 0.9 on that scene |
| G2 learning | ours PD oracle: class mean >= null (prior map) + 0.15 on both seeds |
| G3 depth | vehicles: PD-oracle - P >= +0.05 (two seeds each) |
| G4 control | PD with shuffled depth: vehicles <= P + 0.03 (the depth content carries the gain, not the extra weight) |
| G5 anchors | the three published presets each learn (class mean >= null + 0.15), and ours >= the best of them - 0.02 (our design is not worse than the published designs on the same task, budget and data) |
Smoke run (300 steps, one seed): vehicles IoU@0.5 m PD-oracle 0.83 (0-20 / 20-40 / 40-60 m: 0.91 / 0.88 / 0.72) against P 0.29 (0.72 / 0.27 / 0.06) and simplebev 0.25; pedestrians were 0 and lanes not measured (the lane GT was a coverage below the 0.5 threshold - changed to a presence mask before the full run). The full result is in section 5.
Not tested here (needs real images): the frozen heads' own depth quality (probe 10.2 on real data) and the encoder path on rendered images (the heads were trained on nuScenes; a rendered image is out of their distribution).

## 4. Idle-GPU timing (plan 10, level 6; independent of 2 and 3)
- Script `temp/timing/time_s1_chain.py` (one-off, results to `temp/timing/result_<date>.txt`): batch 1, CUDA events, cudnn.benchmark on, 50 warm-up and 300 timed calls per block, median and 90 % quantile, fp32 and fp16 (autocast)
- Blocks: image normalisation + ResNet-50, CustomFPN, DepthNet, lift P, lift PD (oversample 1 and 4), out_conv, trunk T2 / T3 / T0 / T2 + detail path, semantic head; totals from the image and from the cached features (training speed)
- Idle check inside the script: no other compute process (`nvidia-smi --query-compute-apps`) and utilisation < 5 % over 10 s before and after; if another process appears the result is marked INVALID and the script exits 1; recorded: GPU, clocks, driver, torch / cuDNN versions
- When: the lane project's jobs and any bev_fusion training must be finished (nothing is stopped for it); a monitor waits for the first window with an empty GPU and starts the script (about 5 minutes)
- References to read the numbers against: DualBEV HeightTrans 0.32 ms (RTX 3090, view transform only), Fast-BEV gather 0.006 ms; our FLOPs (plan 6: camera heads 71.4 GFLOPs, lift + out_conv 2.1, T2 about 18.7); the gate is only "measured and reproducible (two calls within 5 %)", no target yet

## 5. Summary
(after the runs)
