# bev_fusion: camera + radar BEV system in five layers

![bev_fusion: camera + radar BEV system in five layers](architecture.svg)

Built 2026-11-01 in Mirko's layered design; the design record with its corrections is the git history of `training/architecture_target.*` (commits 91d4b92 to 08377e5), which this drawing replaces. `archdraw.py check training/architecture.json` confirms that the code every box names exists and that every `.py` of `training/` is in a box; it does not check the wiring between the boxes.

## Five layers
| layer | code | what it holds | its rule |
|---|---|---|---|
| 1 Precompute | `precompute.py`, the dataset and feature caches | calibration, ego steps and gaps, radar detections compensated to now, frozen image features, the **visibility maps** (camera now, radar now) | everything before the first learned weight: deterministic, no gradient; the camera map is cached per calibration (the same function runs live in a car) |
| 2 Input interface | `interface.py` (`INPUTS`, `CALL`, `check_inputs`) | the batch dict and the forward arguments, each named with shape, unit and frame | the only thing the network reads; the flip flag travels with it, the mirror is a BEV operation after the camera slot |
| 3 Network | `model.py` (builder tables, `check_config`) and the block files | slots: camera encoder, radar encoder, temporal fusion per branch, lane add-on (option), fusion, backbone + neck, heads (maps, objects, time mover), soft extraction | one kind per slot chosen in `ModelConfig`; no rule step between the interfaces; `tests/test_slots.py` builds every kind |
| 4 Output interface | `interface.py` (`OUTPUTS_PER_DT`, `check_outputs`), `extract.py` | per dt: maps, object maps, the dense box output (peak weight + box per cell), the soft line map; `v_hat`, features S, the temporal state | dense and differentiable: the loss, the post-processing and later the planner read only this |
| 5 Post-processing | `extract.select_boxes`, the line threshold (`evaluation/metrics.line_cells` stays the metric's definition) | the selection of boxes and line cells | for the metrics and the prototype only; with the hard gate it gives exactly the boxes of `object_head.decode` |

The **planner** is not part of the system: it comes after it and reads the output interface; because that interface is dense and differentiable, a later joint training lets its loss flow back into the fusion.

## The visibility maps: made once, used three times (D36)
| map | the network | the training | the evaluation |
|---|---|---|---|
| camera now (`vis_cam`) | write mask of the camera memory (mirrored with the camera BEV) | loss weight of the camera outputs (lines, areas): next step of D36 | scoring mask of the camera outputs: next step of D36 |
| radar fan (`vis_radar`, D36: "obs_radar stays the fan") | - | loss weight of the radar outputs (velocity, radar aux): next step of D36 | coverage classes |
| fusion area (`vis_fusion` = camera x radar, D37) | - | - | the area fusion results are reported on (camera-only baselines on the same area); the cells hidden behind a GT box are removed on top |
| radar hits (`radar_hits`, cells with detections now) | write mask of the radar memory | - | - |

`vis_radar` is a placeholder (every cell; the fans cover 99.6 % of this grid) until the GT Discussion chat's `autogt.coverage` exists (`camera_cells`, `radar_cells`, `hidden_by_boxes`, `coverage_for`: numpy, rig geometry only, so computed once per rig); it replaces the definitions inside `precompute.py`, nowhere else.

## Network slots
| slot | contract | kinds (builder table) |
|---|---|---|
| camera encoder | `BEVInputs` -> BEVs (B, F, C_cam, X, Y); visibility from layer 1 | `camera_features` dualbev / bevfusion / dual; `LIFTS` pd / p / s; `COLLAPSES` zconv / zsum |
| radar encoder | radar points -> (B, C_rad, X, Y); the raster is its first op, before any weight | `RADARS` raster / pillars / off |
| temporal fusion | (BEV now, past BEVs with their ego steps, state, step, visibility) -> (BEV, state); the memory re-centre is a whole-cell roll, the gradient passes | `TEMPORALS` window_memory / ego_gru / none, one block per branch (`cam_temporal`, `rad_temporal`) |
| lane add-on | (lane image, K, T, ida, the camera depth) -> a BEV added after the camera's temporal block | `lane_branch` res4 / rgb / off |
| fusion | (camera BEV, radar BEV) -> (B, C_fuse, X, Y), modality dropout | `FUSIONS` concat / gate (cmx: see backbone) |
| backbone + neck | fused BEV -> S | `BACKBONES` dualbev / second / cmx (cmx takes both branch BEVs and replaces fusion and backbone) |
| heads | S moved to each dt by the shared time mover -> maps, object maps | maps, objects (`object_direction`, `object_moving`, refiner option) |
| soft extraction | head maps -> dense boxes (peak weight, box, heading vector), soft line map (fixed math) | `dense_outputs` off (default: training, timing) / on (planner, prototype) |

## Checks of the build (2026-11-01)
- **Equivalence** (`tests/equivalence.py`, CPU, deterministic; the code before the build against the final tree): the `t10_dual_s0` checkpoint (step 3000) on 4 val frames, all 29 outputs identical (max abs difference 0.0); the hurdle model of the stopped long run (`t10_dual_long_s0`, step 5030, `object_moving`) on 3 frames, all 32 outputs identical; the state-dict keys identical in both (every checkpoint loads). On the GPU the same code differs from itself by up to 3e-3 between runs (scatter order), so the GPU cannot show bit equality. The dense outputs are off by default (`dense_outputs`: on for the planner and the prototype), so training, timing and op counts compute what they did before.
- **Slots:** `tests/test_slots.py` (CPU, 25 s): 28 kinds and switches build, pass both interfaces, give finite dense outputs, send a nonzero gradient through the trained outputs to every trainable parameter (zero-start layers given small values first) and carry state; 6 invalid combinations refused; the parts that cannot train are named with their reason (the radar memory's window part, the blocks of `camera_off` and `radar_kind` off).
- **Extraction:** `training/extract.py` self-test: the selection equals `object_head.decode` box by box (120 boxes; also on float32 maps), every head map gets a real gradient through the dense output (the direction class through the heading vector, also at logit 0), the soft line map >= 0.5 equals `line_cells` (exact ties and clamped negative distances included).
- **Fresh critic pass** (mode code) on the build: 10 findings, all fixed (the zero gradient of the soft direction sign, the unclamped line map, extra work on the training and timing paths, a weak gradient test, float32 selection, mixed frames of the maps for flipped samples, a radar-less batch, dead code).
- **Finding:** with no past radar frames the radar memory block's window part and velocity head get no gradient (named in `tests/test_slots.py`): the radar memory needs its own static gate (the measured Doppler) or past radar BEVs before it is a fair arm.

## What comes next inside this frame
1. D36: loss weights and scoring masks read `vis_cam` / `vis_radar` (with `autogt.coverage_for` once it exists).
2. The radar memory arm with a Doppler static gate.
3. A first training on the new frame as its real test, bugs fixed on the way.

<!-- archdraw:begin: generated from architecture.json by autoresearch/tools/archdraw.py; edit the spec, then `archdraw.py render` -->
## File map: every box and the code it stands for

| box | kind | code | note |
|---|---|---|---|
| Cameras now + past frames | sensor | `training/nuscenes_dataset.py::FrontRadarDataset` | today the front camera; six cameras are one more entry of the interface, not a new chain |
| Radars detections, 0.5 s | sensor | `training/nuscenes_dataset.py::FrontRadarDataset._radar` | all five radars |
| Ego motion poses / CAN | sensor | `training/train.py::gap_ego_motion` | from the poses in nuScenes; a deployed system supplies its own odometry |
| Image features frozen encoder, cached | precompute: deterministic, cached with an identity, no weights learned before it | `training/camera_encoder.py::FrontCameraEncoder`, `training/bevfusion_camera.py`, `training/feature_cache.py::FeatureStore`, `training/cache_bevfusion_features.py` | frozen DualBEV / BEVFusion image heads: depth probabilities + context per pixel; while frozen they are a cache (with a meta.json identity), once unfrozen the image encoder becomes the first network slot |
| Radar points to now pose + Doppler, cached | precompute: deterministic, cached with an identity, no weights learned before it | `training/nuscenes_dataset.py::FrontRadarDataset._radar`, `training/feature_cache.py::build_radar_cache` | the sweeps of 0.5 s compensated to the current time and cached as points (not as a grid): the radar slot keeps the choice of raster or point encoder |
| Ego steps, calibration (dt, v, omega), K, T | precompute: deterministic, cached with an identity, no weights learned before it | `training/train.py::gap_ego_motion`, `training/train.py::batch_ego`, `training/train.py::clip_step`, `training/nuscenes_dataset.py::FrontRadarDataset` | the ego step from every past frame to now, the true gap to the target time, the memory step between calls, camera and radar calibration, the image crop; today partly built in train.py from the GT meta (also at evaluation) |
| Visibility maps camera, radar fan, fusion area | precompute: deterministic, cached with an identity, no weights learned before it | `training/precompute.py::Precompute`, `training/precompute.py::radar_hits`, `training/lift_pd.py::camera_visibility` | the precompute step (precompute.py): vis_cam from calibration + grid (cached per calibration, the same function live), vis_radar = the radar fans (D36; placeholder until autogt.coverage.radar_cells), vis_fusion = camera x radar (D37: fusion results are scored there), radar_hits = cells with detections now (the radar memory's write mask); read by the network, the training (D36 loss masks) and the evaluation (scoring masks) |
| Ground truth autogt, cached | training only (data, GT, loss, trainer) | `../autogt/`, `training/gt_cache_build.py` |  |
| INPUT INTERFACE BEVInputs (typed dict + check): image features, lane image, K, T, ida, radar points, ego steps, gaps, flip | interface: one typed contract (names, shapes, units, frames) | `training/interface.py::check_inputs`, `training/nuscenes_dataset.py::collate`, `training/feature_cache.py::collate_cached` | BEVInputs = the batch dict with its contract in interface.INPUTS (names, shapes, units, frames) and check_inputs(); the forward arguments dts, state, step, ego in interface.CALL; the per-sample flip flag travels with it (the mirror is a BEV operation after the camera slot) |
| Datasets, caches clips, mirror, scratch | training only (data, GT, loss, trainer) | `training/feature_cache.py::CachedFrontRadarDataset`, `training/train.py::ClipDataset`, `training/augment.py::flip_batch`, `training/scratch.py` |  |
| Camera BEV encoder lift + collapse; mirror in BEV | network slot (learned, swapped by config) | `training/lift_pd.py::LiftPD`, `training/model.py::BEVFusionModel.camera_bevs`, `training/model.py::BEVFusionModel.camera_visibility`, `training/blocks_new.py::ZSumReduce` | contract: BEVInputs -> (BEVs (B, F, C_cam, X, Y), visibility (B, 1, X, Y)), mirrored in BEV for flipped samples (cached perspective features cannot be mirrored); kinds dualbev, bevfusion, dual |
| Radar BEV encoder raster + convs \| pillars \| off | network slot (learned, swapped by config) | `training/radar_bev.py::RadarBEV`, `training/radar_bev.py::rasterize`, `training/blocks_new.py::RadarPillars` | contract: radar points -> (B, C_rad, X, Y); the raster is the first op, before any weight, so it costs no gradient; radar history is the sweeps inside the points |
| Temporal fusion per branch: camera, radar | network slot (learned, swapped by config) | `training/temporal_fusion.py::TemporalFusion`, `training/temporal_fusion.py::recentre`, `training/splat.py::splat_window`, `training/fallbacks.py::EgoWarpGRU` | the same block for both branches (TemporalFusion takes channels and grid); camera: window + world memory; radar: none today, world memory as the first arm (its 0.5 s history is in the points); contract: (BEV now, past BEVs with their ego steps, state, step since the last call, coverage) -> (BEV, state); the re-centre is a whole-cell roll, the gradient passes; finding (tests/test_slots.py): with no past radar frames the window part and the velocity head of the radar block get no gradient, so the radar memory needs its own static gate (the measured Doppler) or past radar BEVs |
| Lane add-on (option) stride 4, after temporal | network slot (learned, swapped by config) | `training/lane_branch.py::LaneBranch` | reads the lane image, K, T, ida and the DualBEV depth of the current frame; its BEV is added to the camera BEV after the temporal block; mirrored on its own |
| Trainer + slot tables check_config | training only (data, GT, loss, trainer) | `training/train.py::train`, `training/model.py::check_config`, `training/feature_cache.py::CachedBEVFusionModel` | ModelConfig picks one kind per slot from the builder tables of model.py (LIFTS, COLLAPSES, TEMPORALS, RADARS, FUSIONS, BACKBONES); check_config refuses every unknown kind and unfitting combination in one place |
| Fusion concat \| gate | network slot (learned, swapped by config) | `training/fusion.py::ConcatFusion`, `training/fusion.py::GateFusion`, `training/fusion.py::drop_modality` | contract: (camera BEV, radar BEV) -> (B, C_fuse, X, Y); modality dropout lives here (off when a sensor is off) |
| BEV backbone + neck ResNet + FPN \| SECOND \| CMX | network slot (learned, swapped by config) | `training/bev_trunk.py::BEVTrunk`, `training/blocks_new.py::SecondTrunk`, `training/fallbacks.py::CMXTrunk` | contract: (B, C_fuse, X, Y) -> features S (B, C, X, Y), the optional 1x1 head reduction included; CMX replaces fusion and backbone together (a kind of this slot that takes both branch BEVs) |
| Map head layers per dt | network slot (learned, swapped by config) | `training/bev_trunk.py::SemanticHead`, `training/timed_head.py::TimedHead` | contract: (S moved to dt) -> map probabilities (B, L, X, Y); the aux heads read the branch BEVs, not S (training only) |
| Object head CenterPoint maps per dt | network slot (learned, swapped by config) | `training/object_head.py::ObjectHead`, `training/object_head.py::BoxRefiner` | contract: (S moved to dt) -> heatmap, offset, size, yaw, velocity, moving maps; the refiner (option, off in the base chain) decodes top-k and matches the GT inside the training path: a rule step when it is on |
| Time mover velocity + ego splat | network slot (learned, swapped by config) | `training/timed_head.py::TimeMover`, `training/splat.py::splat_bev` | moves S to a target time dt for every head (shared) and gives the velocity field v_hat at dt = 0 |
| Soft extraction peaks, box params, line map | fixed math, differentiable | `training/extract.py::dense_boxes`, `training/extract.py::soft_line_map` | built from today's decode and line extraction: per cell a peak weight (heatmap x exp(-beta (maxpool - heatmap))), the box parameters (centre with offset, size, yaw with a soft direction sign, velocity x sigmoid(moving)) and a soft line map (near-zero distance x soft local minimum); fixed math, differentiable |
| Loss BEVLoss + Kendall | training only (data, GT, loss, trainer) | `training/losses.py::BEVLoss`, `training/loss_extras.py` | one module: the Kendall weights and the cross-head terms (motion and time consistency, refine) stay together; heads declare their output keys |
| OUTPUT INTERFACE maps, object maps per dt, v_hat, features S, state | delivered output | `training/interface.py::check_outputs`, `training/model.py::BEVFusionModel.forward` | BEVOutputs = per dt the maps, the object maps, the dense boxes and the soft line map (interface.OUTPUTS_PER_DT / OPTIONAL_PER_DT, check_outputs), plus v_hat, the features S and the temporal state |
| Line cells threshold | rule step (no gradient) | `evaluation/metrics.py::line_cells` | thresholds the soft line map for the line metric; the metric's reference line_cells stays the definition and a test checks the two agree |
| Select boxes local max, top-k, threshold | rule step (no gradient) | `training/extract.py::select_boxes` | the only box step left outside the network: picks cells of the dense box output for the metrics and the prototype; with the hard local maximum it gives exactly today's boxes |
| Boxes, lines metrics, prototype | delivered output | `evaluation/bev_adapter.py::predictions`, `evaluation/board.py` |  |
| Planner (later) after the system | later (no code yet) |  | a consumer of the output interface, not part of the system; the outputs are dense and differentiable, so a later joint training lets its loss flow back into the fusion |

## Arrows

| from | to | carries |
|---|---|---|
| Camera BEV encoder lift + collapse; mirror in BEV | Temporal fusion per branch: camera, radar | visibility |
| Radar BEV encoder raster + convs \| pillars \| off | Temporal fusion per branch: camera, radar | radar |
| OUTPUT INTERFACE maps, object maps per dt, v_hat, features S, state | Planner (later) after the system | dense, differentiable |
| Ego steps, calibration (dt, v, omega), K, T | Visibility maps camera, radar fan, fusion area | calibration |
| Visibility maps camera, radar fan, fusion area | Loss BEVLoss + Kendall | loss masks |
| Visibility maps camera, radar fan, fusion area | Boxes, lines metrics, prototype | scoring masks |

## Main path from input to output

main path (bold): 15 stages = 9 learned, 2 fixed-differentiable, 4 frozen, 0 rule steps.

| stage | class | code |
|---|---|---|
| Image features frozen encoder, cached | frozen | `training/camera_encoder.py::FrontCameraEncoder`, `training/bevfusion_camera.py`, `training/feature_cache.py::FeatureStore`, `training/cache_bevfusion_features.py` |
| Radar points to now pose + Doppler, cached | frozen | `training/nuscenes_dataset.py::FrontRadarDataset._radar`, `training/feature_cache.py::build_radar_cache` |
| Ego steps, calibration (dt, v, omega), K, T | frozen | `training/train.py::gap_ego_motion`, `training/train.py::batch_ego`, `training/train.py::clip_step`, `training/nuscenes_dataset.py::FrontRadarDataset` |
| Visibility maps camera, radar fan, fusion area | frozen | `training/precompute.py::Precompute`, `training/precompute.py::radar_hits`, `training/lift_pd.py::camera_visibility` |
| INPUT INTERFACE BEVInputs (typed dict + check): image features, lane image, K, T, ida, radar points, ego steps, gaps, flip | fixed-differentiable | `training/interface.py::check_inputs`, `training/nuscenes_dataset.py::collate`, `training/feature_cache.py::collate_cached` |
| Camera BEV encoder lift + collapse; mirror in BEV | learned | `training/lift_pd.py::LiftPD`, `training/model.py::BEVFusionModel.camera_bevs`, `training/model.py::BEVFusionModel.camera_visibility`, `training/blocks_new.py::ZSumReduce` |
| Radar BEV encoder raster + convs \| pillars \| off | learned | `training/radar_bev.py::RadarBEV`, `training/radar_bev.py::rasterize`, `training/blocks_new.py::RadarPillars` |
| Lane add-on (option) stride 4, after temporal | learned | `training/lane_branch.py::LaneBranch` |
| Temporal fusion per branch: camera, radar | learned | `training/temporal_fusion.py::TemporalFusion`, `training/temporal_fusion.py::recentre`, `training/splat.py::splat_window`, `training/fallbacks.py::EgoWarpGRU` |
| Fusion concat \| gate | learned | `training/fusion.py::ConcatFusion`, `training/fusion.py::GateFusion`, `training/fusion.py::drop_modality` |
| BEV backbone + neck ResNet + FPN \| SECOND \| CMX | learned | `training/bev_trunk.py::BEVTrunk`, `training/blocks_new.py::SecondTrunk`, `training/fallbacks.py::CMXTrunk` |
| Map head layers per dt | learned | `training/bev_trunk.py::SemanticHead`, `training/timed_head.py::TimedHead` |
| Object head CenterPoint maps per dt | learned | `training/object_head.py::ObjectHead`, `training/object_head.py::BoxRefiner` |
| Time mover velocity + ego splat | learned | `training/timed_head.py::TimeMover`, `training/splat.py::splat_bev` |
| Soft extraction peaks, box params, line map | fixed-differentiable | `training/extract.py::dense_boxes`, `training/extract.py::soft_line_map` |

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
