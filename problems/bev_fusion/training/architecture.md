# bev_fusion: camera + radar BEV system in five layers

![bev_fusion: camera + radar BEV system in five layers](architecture.svg)

`archdraw.py check training/architecture.json` (part of `research tests`) confirms that the code every box names exists and that every `.py` of `training/` is in a box; it does not check the wiring between the boxes.

## Five layers
| layer | code | what it holds | its rule |
|---|---|---|---|
| 1 Precompute | `precompute.py`, the dataset and feature caches | calibration, ego steps and gaps, radar detections compensated to now, frozen image features, the **visibility maps** (camera now, radar now) | everything before the first learned weight: deterministic, no gradient; the camera map is cached per calibration (the same function runs live in a car) |
| 2 Input interface | `interface.py` (`INPUTS`, `CALL`, `check_inputs`) | the batch dict and the forward arguments, each named with shape, unit and frame | the only thing the network reads; the flip flag travels with it, the mirror is a BEV operation after the camera slot |
| 3 Network | `model.py` (builder tables, `check_config`) and the block files | slots: camera encoder, radar encoder, temporal fusion per branch, lane add-on (option), motion block (option), fusion, backbone + neck, heads (maps, 1 m centre-line head (option), objects, time mover), soft extraction | one kind per slot chosen in `ModelConfig`; no rule step between the interfaces; `tests/test_layers.py` checks the layers (v1's slot test `test_slots.py`, which built every kind, is not kept: tag v1) |
| 4 Output interface | `interface.py` (`OUTPUTS_PER_DT`, `check_outputs`), `extract.py` | per dt: maps, object maps, the dense box output (peak weight + box per cell), the soft line map; `v_hat`, features S, the temporal state | dense and differentiable: the loss, the post-processing and later the planner read only this |
| 5 Post-processing | `extract.select_boxes`, the line threshold (`evaluation/metrics.line_cells` stays the metric's definition) | the selection of boxes and line cells | for the metrics and the prototype only; with the hard gate it gives exactly the boxes of `object_head.decode` |

The **planner** is not part of the system: it comes after it and reads the output interface; because that interface is dense and differentiable, a later joint training lets its loss flow back into the fusion.

## The visibility maps: made once, used three times (D36)
| map | the network | the training | the evaluation |
|---|---|---|---|
| camera now (`vis_cam`) | write mask of the camera memory (mirrored with the camera BEV) | loss weight `w_cam` of the camera outputs (lines, areas) built in `losses.py`; `loss.fusion_area_only` (D31, option) restricts the fused heads to the fusion area | scoring mask of the camera outputs (the fusion-area rows are the headline since D31) |
| radar fan (`vis_radar`, D36: "obs_radar stays the fan") | - | loss weight `w_radar` of the radar outputs (velocity, radar aux) built in `losses.py` | coverage classes |
| fusion area (`vis_fusion` = camera x radar, D37) | - | `w_fusion`: the fused heads are masked to it when `loss.fusion_area_only` is on (D31) | the area fusion results are reported on (camera-only baselines on the same area); the cells hidden behind a GT box are removed on top |
| radar hits (`radar_hits`, cells with detections now) | write mask of the radar memory | - | - |

`vis_radar` is the real fan geometry: `Precompute.radar_fan` runs `GtGenerator.coverage.radar_cells` (+-60 deg / 70 m per radar, the definition the ground truth uses) on the five calibrated radar poses the dataset carries (`batch["radar_poses"]`), once per rig; no constant stands in for it and a batch without poses is refused. On real val frames it covers 99.6 % of this grid (4 frames), as the sensor-coverage record found, so today it restricts little: the point is the geometry, and the fusion area `vis_fusion` follows the camera.

## Network slots
| slot | contract | kinds (builder table) |
|---|---|---|
| camera encoder | `BEVInputs` -> BEVs (B, F, C_cam, X, Y); visibility from layer 1 | `camera_features` dualbev / bevfusion / dual (the `CAMERAS` table); `LIFTS` pd / p; `COLLAPSES` zconv / zsum; option `camera_student` (t22 stage B, off): a small trained image encoder (`STUDENTS` evit_b1 / resnet18) replaces the cached teacher outputs of the CURRENT frame of both sources from `lane_img`, older frames keep the cached ones |
| radar encoder | radar points -> (B, C_rad, X, Y); the raster is its first op, before any weight | `RADARS` raster / delta_slots (lossless slot raster + side loss, `radar_encoder.py`) / pillars / off |
| temporal fusion | (BEV now, past BEVs with their ego steps, state, step, visibility) -> (BEV, state); the memory re-centre is a whole-cell roll, the gradient passes | `TEMPORALS` window_memory / none, one block per branch (`cam_temporal`, `rad_temporal`) |
| lane add-on | (lane image, K, T, ida, the camera depth) -> a BEV added after the camera's temporal block | `lane_branch` res4 / rgb / off, with `camera_features` dualbev or dual (t20) |
| motion block | (two camera BEVs, ego steps, radar points) -> the velocity field in a 3-level pyramid; the time mover reads it instead of its 1 x 1 head | `motion_block` off / on (`motion_strides`, `box_vel_from_field`) |
| fusion | (camera BEV, radar BEV) -> (B, C_fuse, X, Y), modality dropout | `FUSIONS` concat / gate |
| backbone + neck | fused BEV -> S | `TRUNKS` dualbev / second (v1's fallback kinds lift s, ego_gru and cmx were removed in v2: tag v1) |
| heads | S moved to each dt by the shared time mover -> maps, object maps | maps (channels from the layer table `layers.py`), 1 m centre-line head (`centreline_1m`: presence, distance, direction at 1 m cells, targets pooled from the 0.5 m GT), objects (`object_direction`, `object_moving`, refiner option) |
| soft extraction | head maps -> dense boxes (peak weight, box, heading vector), soft line map (fixed math) | `dense_outputs` off (default; nothing reads the outputs yet; v1's `test_slots.py` switched it on) |

## Known gaps
- The refiner puts GT matching inside the network (`_refine_train`); it belongs in the loss (harmless while `refine` is off).
- Two box extractors: the evaluation adapter scores with `object_head.decode`, `model.boxes()` with `select_boxes` (equal box for box).
- The radar memory has no gradient in its window part and velocity head inside one call (the 0.5 s of sweeps are already points); the
  option `rad_static_gate` (off by default) makes it write only static detections; no run has used it.

<!-- archdraw:begin: generated from architecture.json by autoresearch/reporter/archdraw.py; edit the spec, then `archdraw.py render` -->
## File map: every box and the code it stands for

| box | kind | code | note |
|---|---|---|---|
| Cameras window, or one frame + state | sensor | `net/prep/nuscenes_dataset.py::FrontRadarDataset` | default: the window of n_cam frames of the batch (cam_buffer 'window'); options (t21, D91): ONE frame per step with the past carried as the state of the previous step at BEV-feature level (cam_buffer 'pair' / 'recurrent'); today the front camera; six cameras are one more entry of the interface, not a new chain |
| Radars detections: 0.5 s or one sweep | sensor | `net/prep/nuscenes_dataset.py::FrontRadarDataset._radar` | all five radars; default: the sweeps of the last 0.5 s accumulated at point level; option radar_per_frame (t25, D91): ONE sweep per camera frame (batch['radar_seq']) |
| Ego motion poses / CAN | sensor | `net/prep/batches.py::gap_ego_motion` | from the poses in nuScenes; a deployed system supplies its own odometry |
| Split lists small \| large \| xl | training only (data, GT, loss, trainer) | `net/prep/splits.py`, `splits/` | scene names only (small 60 / 10, large 428 / 82, xl 700 / 82), each with the GtGenerator version its caches were checked against; the base config carries the lists of large inline (the run's config.json holds the exact lists); net/prep/splits.py resolves a "split" name into the lists (build_image_cache.py), names the split of a run (manifest.json, split.json) and is the evaluation's set table; gt_scenes.py turns the lists into the GT a run needs |
| Image features frozen encoder, cached | precompute: deterministic, cached with an identity, no weights learned before it | `net/prep/images.py`, `net/prep/image_cache.py`, `net/prep/build_feature_cache.py`, `net/prep/cameras.py`, `net/model/camera_encoder.py::FrontCameraEncoder`, `net/model/bevfusion_camera.py`, `net/prep/feature_cache.py::FeatureStore`, `net/prep/cache_bevfusion_features.py` | frozen DualBEV / BEVFusion image heads: depth probabilities + context per pixel; while frozen they are a cache (with a meta.json identity), once unfrozen the image encoder becomes the first network slot (the optional student encoder of the camera slot, t22 / t26, is that step) |
| Radar points to now pose + Doppler, cached | precompute: deterministic, cached with an identity, no weights learned before it | `net/prep/nuscenes_dataset.py::FrontRadarDataset._radar`, `net/prep/feature_cache.py::build_radar_cache` | the sweeps of 0.5 s compensated to the current time and cached as points (not as a grid): the radar slot keeps the choice of raster or point encoder |
| Ego steps, calibration (dt, v, omega), K, T | precompute: deterministic, cached with an identity, no weights learned before it | `net/prep/batches.py::gap_ego_motion`, `net/prep/batches.py::batch_ego`, `training/train.py::clip_step`, `net/prep/nuscenes_dataset.py::FrontRadarDataset` | the ego step from every past frame to now (dataset, from the poses), calibration and crop, the radar poses, and the ego steps to the neighbouring keyframes (batch['gap_steps']: the true gap to the target time and the memory step between calls, from the poses; train.batch_ego / clip_step read them, so the network runs without GT) |
| Visibility maps camera, radar fan, fusion area | precompute: deterministic, cached with an identity, no weights learned before it | `net/prep/geometry.py`, `net/prep/precompute.py::Precompute`, `net/prep/precompute.py::radar_hits`, `net/prep/precompute.py::radar_static`, `net/prep/geometry.py::camera_visibility` | the precompute step (precompute.py): vis_cam from calibration + grid (cached per calibration, the same function live), vis_radar = the radar fans of the calibrated poses (D36; GtGenerator.coverage.radar_cells, cached per rig), vis_fusion = camera x radar (D37: fusion results are scored there), radar_hits = cells with detections now (the radar memory's write mask), radar_static = the same weighted by the measured Doppler (the write mask with rad_static_gate); read by the network, the training (D36 loss masks) and the evaluation (scoring masks) |
| Ground truth GtGenerator, cached | training only (data, GT, loss, trainer) | `../../GtGenerator/`, `net/prep/gt_cache_build.py`, `training/gt_scenes.py` | GT release v14 (front v13 + lane l6, merged under D83); the BEV trainer's cache is keyed by the front signature |
| INPUT INTERFACE BEVInputs (typed dict + check): image features, lane image, K, T, ida, radar points, ego steps, gaps, flip | interface: one typed contract (names, shapes, units, frames) | `net/model/interface.py::check_inputs`, `net/prep/nuscenes_dataset.py::collate`, `net/prep/feature_cache.py::collate_cached` | BEVInputs = the batch dict with its contract in interface.INPUTS (names, shapes, units, frames) and check_inputs(); the forward arguments dts, state, step, ego in interface.CALL; the per-sample flip flag travels with it (the mirror is a BEV operation after the camera slot) |
| Datasets, caches clips, mirror | training only (data, GT, loss, trainer) | `net/prep/feature_cache.py::CachedFrontRadarDataset`, `net/prep/batches.py::ClipDataset`, `net/prep/augment.py::flip_batch` |  |
| Camera BEV encoder lift + collapse; student (option) | network slot (learned, swapped by config) | `net/model/cached_model.py`, `net/model/lift_pd.py::LiftPD`, `net/model/model.py::BEVFusionModel.camera_bevs`, `net/model/model.py::BEVFusionModel.camera_visibility`, `net/model/blocks_new.py::ZSumReduce`, `net/model/student_encoder.py::StudentEncoder` | contract: BEVInputs -> (BEVs (B, F, C_cam, X, Y), visibility (B, 1, X, Y)), mirrored in BEV for flipped samples (cached perspective features cannot be mirrored); kinds dualbev, bevfusion, dual; option camera_student (t22 stage B, t26): a small trained image encoder (StudentEncoder, EfficientViT-B1 or ResNet-18) replaces the cached teacher outputs of the current frame, with student_frames 'all' (t26) of every frame of the window, and can be trained on the task (student_trainable) |
| Radar BEV encoder raster \| delta slots \| pillars \| off | network slot (learned, swapped by config) | `net/model/radar_bev.py::RadarBEV`, `net/model/radar_bev.py::rasterize`, `net/model/blocks_new.py::RadarPillars`, `net/model/radar_encoder.py::RadarEncoder` | contract: radar points -> (B, C_rad, X, Y); the raster is the first op, before any weight, so it costs no gradient; radar history is the sweeps inside the points (with radar_per_frame, t25: one sweep per camera frame, the past in the radar memory of the temporal block); kind delta_slots (radar_encoder.py, in the t16 / t20 chain): the lossless delta-coded slot raster (K = 3 exact slots per cell + remainder mean + count) and two convolutions, with a reconstruction side loss (radar_aux) whose weight decays to a floor |
| Temporal fusion per branch: camera, radar | network slot (learned, swapped by config) | `net/model/temporal_fusion.py::TemporalFusion`, `net/model/temporal_fusion.py::recentre`, `net/model/splat.py::splat_window` | the same block for both branches (TemporalFusion takes channels and grid); camera: window (default) or one buffered frame, pair / recurrent (cam_buffer, t21, D91), + world memory; radar: none by default (its 0.5 s history is in the points), the radar memory with the Doppler static gate (rad_temporal 'window_memory') and the per-frame recursion (radar_per_frame, t25) are options, not trained successfully so far; contract: (BEV now, past BEVs with their ego steps, state, step since the last call, coverage) -> (BEV, state); the re-centre is a whole-cell roll, the gradient passes; finding (v1's slot test): with no past radar frames the window part and the velocity head of the radar block get no gradient, so the radar memory needs its own static gate (the measured Doppler) or past radar BEVs |
| Lane add-on (option) stride 4, after temporal | network slot (learned, swapped by config) | `net/model/lane_branch.py::LaneBranch` | reads the lane image, K, T, ida and the DualBEV depth of the current frame; its BEV is added to the camera BEV after the temporal block; mirrored on its own; off for new from-scratch base runs (D65; its config t20_base_nol4.json is in the tag v1); since t20 it also works on the dual camera features (it reads the first source's depth, K and crop and adds to the merged camera BEV): at 20 m a stride-16 image pixel is 0.57 m wide (more than a 0.5 m cell), a stride-4 pixel 0.14 m |
| Motion block M velocity field, 3 levels | network slot (learned, swapped by config) | `net/model/motion_block.py::MotionBlock`, `net/model/motion_block.py::block_inputs` | t15 joint form (ModelConfig.motion_block, on in t16 / t20): the velocity field from the current and the oldest camera BEV (ego-aligned) and the fixed radar raster (count, Doppler, ray, age) in a small pyramid (2 / 1 / 0.5 m, motion_strides), a zero-initialised update per level, every level returned as v_levels for deep supervision; the time mover takes its field from here, box_vel_from_field makes it the object head's velocity; gradient clipping of its own (grad_clip_split) |
| Trainer + trainkit slot tables, check_config | training only (data, GT, loss, trainer) | `training/build_image_cache.py`, `training/train.py::train`, `../../autoresearch/trainkit/`, `training/freeze.py`, `net/model/model.py::check_config`, `net/prep/feature_cache.py::CachedBEVFusionModel` | ModelConfig picks one kind per slot from the builder tables of model.py (LIFTS, COLLAPSES, TEMPORALS, RADARS, FUSIONS, TRUNKS); check_config refuses every unknown kind and unfitting combination in one place; since D48 train() builds model, loss and data and hands them to the project-independent loop autoresearch/trainkit (checkpoints with resume, EMA, clipping, the stop request), train.py's ProbeHooks calls the probes of trainkit/probe.py |
| Fusion concat \| gate | network slot (learned, swapped by config) | `net/model/fusion.py::ConcatFusion`, `net/model/fusion.py::GateFusion`, `net/model/fusion.py::drop_modality` | contract: (camera BEV, radar BEV) -> (B, C_fuse, X, Y); modality dropout lives here (off when a sensor is off) |
| BEV backbone + neck ResNet + FPN \| SECOND | network slot (learned, swapped by config) | `net/model/bev_trunk.py::BEVTrunk`, `net/model/blocks_new.py::SecondTrunk` | contract: (B, C_fuse, X, Y) -> features S (B, C, X, Y), the optional 1x1 head reduction included |
| 1 m head (option) centre-line | network slot (learned, swapped by config) | `net/model/bev_trunk.py::SemanticHead`, `net/model/layers.py::layer_spec_1m` | centreline_1m (D48, t20): one pyramid level down: a SemanticHead on the map head's moved state, average-pooled 2 x 2 (1 m cells); presence, truncated distance field and direction (cos, sin) of the centre lines, one row (centreline, polyline, 1m) of the layer table; targets are pooled from the 0.5 m GT, no new GT version; output sem_1m (B, C, X/2, Y/2); the modules are built so that the option leaves the initialisation of all others unchanged |
| Map head (0.5 m) layers per dt | network slot (learned, swapped by config) | `net/model/bev_trunk.py::SemanticHead`, `net/model/timed_head.py::TimedHead` | contract: (S moved to dt) -> map probabilities (B, L, X, Y), the channels from the layer table (5 areas, 2 line masks + 2 distance fields, 3 dense objects); the aux heads read the branch BEVs, not S (training only) |
| Object head CenterPoint maps per dt | network slot (learned, swapped by config) | `net/model/object_head.py::ObjectHead`, `net/model/object_head.py::BoxRefiner` | contract: (S moved to dt) -> heatmap, offset, size, yaw, velocity, moving maps; the refiner (option, off in the base chain) decodes top-k and matches the GT inside the training path: a rule step when it is on |
| Time mover velocity + ego splat | network slot (learned, swapped by config) | `net/model/timed_head.py::TimeMover`, `net/model/splat.py::splat_bev` | the shared TimeMover: moves S to each target time dt (velocity head + ego splat) before both heads, gives v_hat at dt 0; with the motion block (t15, on in t16 / t20) the velocity field comes from M and the 1 x 1 velocity head is removed |
| Soft extraction peaks, box params, line map | fixed math, differentiable | `net/model/extract.py::dense_boxes`, `net/model/extract.py::soft_line_map` | built from today's decode and line extraction: per cell a peak weight (heatmap x exp(-beta (maxpool - heatmap))), the box parameters (centre with offset, size, yaw with a soft direction sign, velocity x sigmoid(moving)) and a soft line map (near-zero distance x soft local minimum); fixed math, differentiable |
| Loss BEVLoss + Kendall | training only (data, GT, loss, trainer) | `net/loss/object_loss.py`, `net/loss/radar_loss.py`, `net/loss/motion_loss.py`, `net/prep/targets.py`, `net/loss/losses.py::BEVLoss`, `net/loss/loss_extras.py` | one module: the Kendall weights and the cross-head terms (motion and time consistency, refine) stay together; heads declare their output keys; the terms per layer come from the layer table (area: BCE + overlap, polyline: dilated Dice + distance + direction, dense box); loss.hygiene_2026_11_02 switches the corrected terms of the 2026-11-02 findings |
| Layer table name, kind, level | interface: one typed contract (names, shapes, units, frames) | `net/model/layers.py::Layer`, `net/model/layers.py::layer_table`, `net/model/layers.py::pool_targets`, `net/model/layers.py::layer_spec` | the layers of the map head as rows (name, kind area \| polyline \| dense_box, level 0.5m \| 1m): the kind decides the channels, the target and the loss, so a new layer is a row, not new code; the map head, the 1 m head, the pooled targets and BEVLoss derive from it (losses.layer_spec); the old layouts are reproduced channel by channel (tests/test_layers.py) |
| OUTPUT INTERFACE maps, object maps per dt, v_hat, features S, state | delivered output | `net/model/interface.py::check_outputs`, `net/model/model.py::BEVFusionModel.forward` | BEVOutputs = per dt the maps, the object maps, the dense boxes and the soft line map (interface.OUTPUTS_PER_DT / OPTIONAL_PER_DT, check_outputs), plus v_hat, the features S and the temporal state |
| Line cells threshold | rule step (no gradient) | `evaluation/metrics.py::line_cells` | thresholds the soft line map for the line metric; the metric's reference line_cells stays the definition and a test checks the two agree |
| Select boxes local max, top-k, threshold | rule step (no gradient) | `net/model/extract.py::boxes_from_maps`, `net/model/extract.py::select_boxes` | ONE extractor: boxes_from_maps reads only the K peak cells of the head maps (vectorised, float64, the same _params as the dense output) and is what object_head.decode, model.boxes(), the evaluation adapter and the prototype call; select_boxes does the same on a dense output (planner side); extract.py's self-test checks both box by box against the plain per-box loop (6 mutants caught); 1.7 ms instead of 6.5 ms (decode) and 1.5 ms instead of 9.0 ms (model.boxes) per frame at 100 candidates, CPU |
| Boxes, lines metrics, prototype | delivered output | `evaluation/bev_adapter.py::predictions`, `evaluation/bev_eval.py`, `evaluation/bev_task.py::Board7Metric` | the numbers, pictures and timing of a run come through `Evaluation` (its adapter evaluation/bev_eval.py; the research step evaluate); the score board7 is Board7Metric (the six map rows and the object mAP); the 1 m centre-line head is scored as the layer centreline |
| Planner (later) after the system | later (no code yet) |  | a consumer of the output interface, not part of the system; the outputs are dense and differentiable, so a later joint training lets its loss flow back into the fusion |

## Arrows

| from | to | carries |
|---|---|---|
| BEV backbone + neck ResNet + FPN \| SECOND | Time mover velocity + ego splat | S |
| Radar BEV encoder raster \| delta slots \| pillars \| off | Temporal fusion per branch: camera, radar | radar |
| OUTPUT INTERFACE maps, object maps per dt, v_hat, features S, state | Planner (later) after the system | dense, differentiable |
| Ego steps, calibration (dt, v, omega), K, T | Visibility maps camera, radar fan, fusion area | calibration |
| Visibility maps camera, radar fan, fusion area | INPUT INTERFACE BEVInputs (typed dict + check): image features, lane image, K, T, ida, radar points, ego steps, gaps, flip | memory write masks |
| Visibility maps camera, radar fan, fusion area | Loss BEVLoss + Kendall | loss masks (D36; fusion_area_only, D31) |
| Visibility maps camera, radar fan, fusion area | Boxes, lines metrics, prototype | scoring masks (D37, D31) |
| Time mover velocity + ego splat | Map head (0.5 m) layers per dt | S at dt |
| Time mover velocity + ego splat | OUTPUT INTERFACE maps, object maps per dt, v_hat, features S, state | v_hat |
| Camera BEV encoder lift + collapse; student (option) | Motion block M velocity field, 3 levels | 2 camera BEVs |
| INPUT INTERFACE BEVInputs (typed dict + check): image features, lane image, K, T, ida, radar points, ego steps, gaps, flip | Motion block M velocity field, 3 levels | radar points |
| Motion block M velocity field, 3 levels | Time mover velocity + ego splat | velocity field |
| Time mover velocity + ego splat | 1 m head (option) centre-line | S at dt, pooled 2 x 2 |
| 1 m head (option) centre-line | OUTPUT INTERFACE maps, object maps per dt, v_hat, features S, state | sem_1m |
| Layer table name, kind, level | Loss BEVLoss + Kendall | targets, loss per kind |
| Split lists small \| large \| xl | Ground truth GtGenerator, cached | scene lists |

## Main path from input to output

main path (bold): 17 stages = 11 learned, 2 fixed-differentiable, 4 frozen, 0 rule steps.

| stage | class | code |
|---|---|---|
| Image features frozen encoder, cached | frozen | `net/prep/images.py`, `net/prep/image_cache.py`, `net/prep/build_feature_cache.py`, `net/prep/cameras.py`, `net/model/camera_encoder.py::FrontCameraEncoder`, `net/model/bevfusion_camera.py`, `net/prep/feature_cache.py::FeatureStore`, `net/prep/cache_bevfusion_features.py` |
| Radar points to now pose + Doppler, cached | frozen | `net/prep/nuscenes_dataset.py::FrontRadarDataset._radar`, `net/prep/feature_cache.py::build_radar_cache` |
| Ego steps, calibration (dt, v, omega), K, T | frozen | `net/prep/batches.py::gap_ego_motion`, `net/prep/batches.py::batch_ego`, `training/train.py::clip_step`, `net/prep/nuscenes_dataset.py::FrontRadarDataset` |
| Visibility maps camera, radar fan, fusion area | frozen | `net/prep/geometry.py`, `net/prep/precompute.py::Precompute`, `net/prep/precompute.py::radar_hits`, `net/prep/precompute.py::radar_static`, `net/prep/geometry.py::camera_visibility` |
| INPUT INTERFACE BEVInputs (typed dict + check): image features, lane image, K, T, ida, radar points, ego steps, gaps, flip | fixed-differentiable | `net/model/interface.py::check_inputs`, `net/prep/nuscenes_dataset.py::collate`, `net/prep/feature_cache.py::collate_cached` |
| Camera BEV encoder lift + collapse; student (option) | learned | `net/model/cached_model.py`, `net/model/lift_pd.py::LiftPD`, `net/model/model.py::BEVFusionModel.camera_bevs`, `net/model/model.py::BEVFusionModel.camera_visibility`, `net/model/blocks_new.py::ZSumReduce`, `net/model/student_encoder.py::StudentEncoder` |
| Radar BEV encoder raster \| delta slots \| pillars \| off | learned | `net/model/radar_bev.py::RadarBEV`, `net/model/radar_bev.py::rasterize`, `net/model/blocks_new.py::RadarPillars`, `net/model/radar_encoder.py::RadarEncoder` |
| Lane add-on (option) stride 4, after temporal | learned | `net/model/lane_branch.py::LaneBranch` |
| Temporal fusion per branch: camera, radar | learned | `net/model/temporal_fusion.py::TemporalFusion`, `net/model/temporal_fusion.py::recentre`, `net/model/splat.py::splat_window` |
| Motion block M velocity field, 3 levels | learned | `net/model/motion_block.py::MotionBlock`, `net/model/motion_block.py::block_inputs` |
| Fusion concat \| gate | learned | `net/model/fusion.py::ConcatFusion`, `net/model/fusion.py::GateFusion`, `net/model/fusion.py::drop_modality` |
| BEV backbone + neck ResNet + FPN \| SECOND | learned | `net/model/bev_trunk.py::BEVTrunk`, `net/model/blocks_new.py::SecondTrunk` |
| Time mover velocity + ego splat | learned | `net/model/timed_head.py::TimeMover`, `net/model/splat.py::splat_bev` |
| Map head (0.5 m) layers per dt | learned | `net/model/bev_trunk.py::SemanticHead`, `net/model/timed_head.py::TimedHead` |
| Object head CenterPoint maps per dt | learned | `net/model/object_head.py::ObjectHead`, `net/model/object_head.py::BoxRefiner` |
| 1 m head (option) centre-line | learned | `net/model/bev_trunk.py::SemanticHead`, `net/model/layers.py::layer_spec_1m` |
| Soft extraction peaks, box params, line map | fixed-differentiable | `net/model/extract.py::dense_boxes`, `net/model/extract.py::soft_line_map` |

## Where the path is not end to end, and what could stand in for it

| step | class | what it does and what it carries | building block that emulates it differentiably | catch |
|---|---|---|---|---|
| Image features frozen encoder, cached | frozen |  |  |  |
| Radar points to now pose + Doppler, cached | frozen |  |  |  |
| Ego steps, calibration (dt, v, omega), K, T | frozen |  |  |  |
| Visibility maps camera, radar fan, fusion area | frozen |  |  |  |

## Rule steps outside the main path

| step | what it does | catch |
|---|---|---|
| Line cells threshold |  |  |
| Select boxes local max, top-k, threshold |  |  |
<!-- archdraw:end -->
