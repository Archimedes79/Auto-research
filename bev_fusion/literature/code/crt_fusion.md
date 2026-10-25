# crt_fusion - code extraction (CRT-Fusion, NeurIPS 2024)

| field | value |
|---|---|
| repository | https://github.com/mjseong0414/CRT-Fusion (owner mjseong0414 = Minjae Seong, second author; the arXiv abs page has no code link, the HTML v3 of 2411.03013 does) |
| papers covered | `../papers/A30` is in the lane project: `C:\Auto-research\bev_fusion_lane_center\literature\papers\A30_crt-fusion-camera-radar-recurrent.md`; arXiv 2411.03013 |
| state read | commit `2fa611a5c8944c487e6fe78adbef63b3d0237fec`, 2024-12-05, message "Initial commit", branch `main` (31 commits listed on the page; the API returns this one as latest). Read through GitHub API / raw.githubusercontent.com pages, passed through the fetch tool's page reader. **No clone, no download.** Line numbers: counted by the page reader from the raw file top; the `fuse_history` and `reverse_bilinear` blocks were returned as code with numbers (checked, +-0 within the block, but the two passes disagreed by 4 lines on where `fuse_history` starts: 300 vs 304; trust 304); every other number is approximate (+-5) |
| licence | `LICENSE` = Apache License 2.0 (January 2004), header line "Copyright 2018-2019 Open-MMLab" (the stock mmdet3d file; GitHub API says Apache-2.0). Checkpoints: v0.0.0 release assets, no separate licence text seen; nuScenes data is CC BY-NC-SA, so weights trained on it are research-only in practice |
| stack | mmdet3d 0.x fork vendored in the repo (`mmdet3d/`), mmcv-full runner (`IterBasedRunner`, `WarmupFp16OptimizerHook`, `force_fp32`, `auto_fp16`), mmdet, a custom CUDA op `mmdet3d/ops/bev_pool` (BEVDet's), `cv2.fillPoly` for the GT rasterisation, Docker folder |
| size, checkpoints, data | 4.6 MB repo; code that matters about 1,500 lines (`crtfusion.py` 34 KB, `crtfusion_head.py` 32 KB, `view_transformer_crtfusion.py` 14 KB); checkpoints R50 (49.3 mAP / 57.9 NDS) and R50-light-cbgs (48.9 / 58.7) as release v0.0.0 (page does not prove the links resolve); data = nuScenes + `nuscenes_infos_crtfusion_{train,val}.pkl` made by `tools/create_data_crtfusion.py` |
| extracted by, date | literature scout (deep mode, code extraction), 2026-10-25 |

## Port card

| field | value |
|---|---|
| repo | https://github.com/mjseong0414/CRT-Fusion |
| commit | 2fa611a 2024-12-05 |
| licence | Apache-2.0 |
| portable | mixed (the temporal part is pure torch; around it mmcv / mmdet3d / cuda-op) |
| reuse | idea-only (the design and the GT recipe; the velocity shift as released is suspect, see 7) |
| effort | S (ego warp + pull in plain torch, hours); M with the velocity head and its GT (days) |
| port_targets | `bev_fusion/training/temporal_pull.py` (new), `bev_fusion/training/tests/test_temporal_pull.py` (new), later a BEV velocity / occupancy GT in the GT generator |
| status | read |
| papers | A30 (lane project), 2411.03013 |

## 1. Map of the code

| path | what it does | lines | portable to our env? |
|---|---|---|---|
| `mmdet3d/models/detectors/crtfusion.py` | class `CRTFusion(BEVDet)`: image encoder, radar pillars, view transformer call, motion + occupancy heads, **`fuse_history`**, **`reverse_bilinear`**, depth / PV-seg loss | about 620 | the temporal part (lines 59-216, 304-457) is pure torch but reads mmcv-style `img_metas`; the rest needs mmcv / mmdet3d |
| `mmdet3d/models/dense_heads/crtfusion_head.py` | CenterHead variant; **`get_targets_single`** builds the BEV velocity / occupancy GT, `loss` computes motion + occupancy losses | about 750 | GT part: numpy + cv2 + torch, easy; loss: pure torch |
| `mmdet3d/models/necks/view_transformer_crtfusion.py` | `ViewTransformerCRTFusion`: `RCA_Attention` (radar-camera azimuth attention before depth), depth net with D+2 channels (D depth bins + 2 PV-seg logits), `voxel_pooling_crtfusion` (BEVDet `bev_pool` + a BEV occupancy count channel), `RadarCameraGating` | about 380 | needs the `bev_pool` CUDA op (replaceable by a gather, see our plan 2.4) |
| `mmdet3d/models/utils/radar_camera_gating.py`, `point_gating.py` | sigmoid-gated sum of camera and radar BEV (two 3x3 conv-BN-ReLU stacks per modality, `img*w_c + pts*w_r`); point gating mixes radar with max-pooled image rows / columns | 52 + 30 | pure torch (only `@auto_fp16` from mmcv) |
| `mmdet3d/datasets/nuscenes_dataset.py` | `get_data_info`: `curr_to_prev_lidar_rt`, `sequence_group_idx`, `start_of_sequence`, `curr_lidar_to_ego_rt`; `_set_sequence_group_flag` | n/a | pure python |
| `tools/data_converter/nuscenes_converter_crtfusion.py` | info pkl: GT velocity, lidar sweeps, `make_multisweep_radar_data` (7 radar sweeps, time gap in s) | 30 KB | pure python / nuscenes-devkit |
| `configs/crt-fusion/crtfusion-r50-fp16_phase{1,2}.py` | the two training phases (see 3) | about 430 each | n/a |

Everything is BEVDet4D / BEVDepth lineage (the `fuse_history` skeleton, `history_sweep_time`, `sequence_group_idx` and the `+= 1` time bookkeeping are copied from BEVStereo / BEVDepth).

## 2. Key algorithms (where, how)

**2.1 Motion feature estimator** (`crtfusion.py`, `__init__` lines 157-169, used in `extract_img_feat` lines 498-500; checked).
- Input: the fused BEV feature after `pre_process_net` and `embed` (1x1 conv-BN-ReLU twice): `fused_bev_feat` [B, C=80, H, W] (config: grid 512 with out_size_factor 4 -> 128 x 128 cells of 0.8 m).
- `motion_net` = Conv3x3(C,C) - BN - ReLU - Conv1x1(C,2): output `curr_motion` [B,2,H,W], **raw (vx, vy) in the units of the GT (m/s), no activation**.
- `occupancy_net` = Conv3x3(C,C) - BN - ReLU - Conv1x1(C,1) - Sigmoid: `occ_pred` [B,1,H,W] in 0..1.
- Both heads run on the **current** frame's BEV (before temporal fusion) and their output is stored for the next step as the history's motion map (`his_motion`). Two heads, so no shared trunk beyond `embed`.

**2.2 Shifting by velocity x dt** (`fuse_history` lines 404-418 plus `reverse_bilinear` lines 174-216; checked, code read as code).
- It is a **forward shift (push / splat) of the past feature, not a pull**, and not `grid_sample`, not `roll`: every occupied history cell gets a target coordinate (grid index in the history frame + velocity shift, then the ego affine), and the feature is scattered to the 4 surrounding integer cells of the current grid.
- Weights are not standard bilinear: `w = 1 / (euclidean distance to the corner + 1e-6)`, normalised to sum 1 (lines 188-208); exact hit -> weight 1 on that corner.
- Accumulation `BEV_buckets[b,:,w0,h0] += w * feat` (lines 211-214) uses advanced indexing with repeated indices. In PyTorch `+=` on repeated indices does **not** accumulate: the last write wins and the order is non-deterministic on GPU. The paper's "average over source pixels" (eq. S(x,y)) is therefore not what runs. Holes (no source lands there) stay zero; no mask of coverage is returned.
- Velocity units: `his_occ_motion / voxel_size[0] / out_size_factor * history_cam_sweep_freq` (lines 411-412): m/s -> cells per second -> times dt. **dt is the constant `self.history_cam_sweep_freq = 0.5`** (line 102), i.e. the nominal keyframe interval, not the measured timestamp difference.
- Threshold: `moving_occ_mask = |v| > self.motion_th` (default 1.0, line 410), applied to the history frame's predicted speed, so slow cells are only ego-warped.

**2.3 Ego motion** (lines 382-418; checked). Separate from the velocity shift, and applied **after** it, as one 4x4 matrix on pixel coordinates (homogeneous 1):
```
rt_flow_p2c = inv(feat2bev) @ forward_augs @ inv(curr_to_prev_lidar_rt) @ inv(history_forward_augs) @ feat2bev
grid(prev pixel) -> grid += v*dt (moving cells) -> grid = rt_flow_p2c @ grid -> scatter to current grid
```
`feat2bev` = diag(dx, dy) with offset `bx - dx/2` (pixel index -> metres in the lidar frame); `curr_to_prev_lidar_rt` = `nuscenes_get_rt_matrix(info[index], info[index-1], "lidar", "lidar")` (BEVDepth helper, composes lidar -> ego -> global -> ego' -> lidar'; direction curr -> prev, hence the `inverse` to get prev -> curr; the helper's body is in `..utils` and was **not read**); `forward_augs` / `history_forward_augs` undo / redo the training-time BEV augmentation (rot, scale, flip) of the current and the previous sample. Order consequence: the velocity offset is in the **history frame's** axes and metres, then the rigid ego change is applied, which is physically right (target moves in the old ego frame, then the frame moves). Only keyframe-to-keyframe poses are used, so ego motion over 0.5 s, and no per-cell time.

**2.4 Memory bank / recurrence** (lines 330-457; checked).
- State on the module (not a tensor bank): `his_occ_feat` [N_occ_cells, history_cat_num * C] (sparse: only cells with `occ_pred > bev_seg_th`, default 0.05, i.e. object-like cells are kept, the background is dropped), `his_occ_mask` [B,H,W], `his_motion` [B,2,H,W], `history_sweep_time` [B, T], `history_forward_augs`, `history_seq_ids`.
- **One recurrent step per call**, not a loop over N frames: all `history_cat_num` stored frames (config phase 2: **6**, `history_cat_conv_out_channels=160`; class default 1) are stacked on the channel axis and **warped together by the single one-step transform** with the single previous motion map, then the stack is re-scattered each step (so a frame 6 steps old has been resampled 6 times: accumulated blur, and the shift of older frames uses the newest velocity map).
- Fusion (lines 426-447): `cat(curr_bev, prev2curr_bev)` [B,(1+T)*C,H,W] -> `motion_gating` (3x3 conv-BN-ReLU-Sigmoid, per-channel gate, multiplied in) -> reshape to [B,1+T,C,H,W], append a constant channel with `age_steps * 0.5` s -> `history_keyframe_time_conv` (1x1, C+1 -> C, applied per frame) -> `history_keyframe_cat_conv` (1x1, (1+T)C -> 160). So the paper's "multiply by occupancy" is a **sparse mask on what is stored** here, and a learned sigmoid gate on what is fused; no `x O_{t}` product.
- History update (lines 450-454): shift the channel groups down by one, put the current BEV on top, keep `detach()` (no gradient through time: training is truncated to one step).
- Sequence handling: `start_of_sequence` resets the history to a repeat of the current frame and zeroes `his_motion` (comment, line 380: "do not velocity-compensate in the first sequence frame"); `sequences_split_num=2` in training splits scenes.

**2.5 Losses and labels** (`crtfusion_head.py`; page-reader reading of `get_targets_single` ~L171-345 and `loss` ~L346-440, short fragments checked, rest digest).
- Velocity GT per BEV cell: for each GT box the **rotated footprint is rasterised** (`cv2.fillPoly` of four box corners in grid coordinates, `mask_corner = [True,False,False,True,True,False,False,True]` picks the footprint corners) into an occupancy map (1.0) and into two velocity maps filled with the box's `(vx, vy) = task_boxes[idx][k][7:]` (the nuScenes annotation velocity, columns 7-8 of the box tensor). Cells whose speed is below `motion_th` (1.0 m/s) are zeroed; where boxes overlap the faster one wins; the maps are merged over the 6 CenterHead tasks by the largest speed. Box centre outside the grid -> skipped. **The paper's IoU ratio r >= 0.5 per cell is not in the code**: the code marks every cell whose centre falls in the footprint.
- GT velocity origin (converter, ~L225-245, digest): `nusc.box_velocity(token)[:2]` (over-ground velocity from neighbouring annotations in the **global** frame), padded with z = 0, rotated by `inv(e2g_rot)` and `inv(l2e_rot)` only (no translation, **no ego-velocity subtraction**) into the lidar / ego axes of the sample. So the target is ground velocity expressed in ego axes: a parked car is 0, a car at 10 m/s ahead of a moving ego is +10, not relative speed.
- Motion loss: MSE on **foreground cells only** (`occ_maps != 0`), `avg_factor = number of foreground cells`, `loss_weight = 0.25`; plus an angle term: MSE on `atan2(vy, vx+1e-6)` of prediction and target on the same cells (no wrap handling). The paper's lambda_vel = 1 corresponds to the `loss_weight=0.25` MSE plus the angle term; the mapping is not exact.
- Occupancy loss: focal-weighted BCE, alpha 0.25, gamma 2, split into foreground and background sums, each divided by its count (floor 1) and multiplied by the hard-coded **30** (`loss_occ_fore`, `loss_occ_back`); the page reader notes `F.binary_cross_entropy_with_logits` applied to a sigmoid output (double sigmoid) and a non-detached focal weight - **unchecked, read the lines before trusting**.
- Depth: BCE on D bins, weight 3.0; PV segmentation: focal BCE, weight 25 (`get_depth_and_segment_loss`, lines ~250-301).

**2.6 Radar branch it inherits** (config + `view_transformer_crtfusion.py`; digest of the page reader).
- Not CRN's. The paper itself says it differs from CRN by enhancing camera features with radar **before** depth prediction. Chain: 6 radar sweeps (`N_sweeps=6`, `N_use=5`, drop_ratio 0.1; converter stores 7 sweeps with time gaps in s) -> `RADARPillarFeatureNet_MAX` (in_channels 18, feat 64, max 10 points per pillar, max 1500 voxels) -> `PointPillarsScatter` 512 x 512 -> SECOND (layers 3,5,5; channels 128,256,512) -> `SECONDFPN_v2` to 80 channels (BEV radar feature). Then `RCA_Attention` warps the radar BEV to each camera with `grid_sample`, picks for each image column the 128 radar cells closest in azimuth, mixes them with `PointGating` and adds to the image features with sigmoid gates -> depth net (SE module on camera parameters + ResNetForBEVDet, 1x1 conv to D+2) -> lift-splat with `bev_pool` (depth bins 2..58 m at 0.5 m = 112 bins) -> `RadarCameraGating` with the radar BEV -> occupancy-count channel concatenated and reduced by 1x1 conv.
- This is the RCM-Fusion lineage of the same group; radar-aware depth + gating, no Doppler channel handled explicitly in the files read.

## 3. Config versus paper

| item | paper | code / config |
|---|---|---|
| frames N | 6 past frames (R50), 3 (light) | `history_cat_num=6` in phase 2 config (class default 1); recurrent, one step per call |
| shift | per pixel, average of colliding sources | inverse-distance splat, repeated indices overwrite (non-deterministic) |
| shift active | yes | **line 413 `grid[moving_occ_mask][:, 0:2] = ...` writes into a copy made by boolean indexing and has no effect on `grid`** (my reading of PyTorch semantics, not run): the released temporal module most likely does ego warp only |
| occupancy | `concat(...) * O_{t-k+1}` | sparse mask at 0.05 on stored cells + sigmoid channel gate |
| ego compensation | not described in the paper text | explicit 4x4 (2.3) |
| velocity GT | cell labelled if IoU ratio >= 0.5 | footprint rasterisation, speed >= 1 m/s |
| dt | duration of one frame | 0.5 s constant (nominal 2 Hz keyframes) |
| loss weights | depth 3, seg 25, vel 1, occ 30 | depth 3.0, semantic 25, motion MSE 0.25 + angle MSE, occupancy x30 |
| numbers | R50 val 57.2 NDS / 50.0 mAP | README 57.9 NDS / 49.3 mAP "slightly different, further optimised" |
| training | 24 epochs, motion module off for 6 | phase 1 (`do_history=False`, `motion_est=True`, 10,548 iters = 6 epochs x 1,758) then phase 2 (`do_history=True`, resume from the EMA of phase 1, to 42,192 iters); AdamW lr 2e-4 wd 1e-7, grad clip 5, batch 4 x 4 GPUs, EMA 0.001, fp16 |

Paper ablation (checked on the HTML page): N = 6 chosen (57.2 / 50.0; 7-8 frames 57.3 NDS but lower mAP); tau_v 0 / 0.5 / 1.0 / 1.5 m/s -> NDS 57.0 / 56.6 / 57.2 / 56.8; no isolated ablation of the velocity shift alone or of the occupancy filter; MFE + MGTF together +1.1 NDS / +1.1 mAP (56.1 / 48.9 -> 57.2 / 50.0). Together with the line-413 finding this means **the evidence that the velocity shift itself helps is missing**; what is shown is that a recurrent 6-frame ego-warped gated BEV memory helps (night mAP 33.0 vs CRN 30.4). Latency: 67.1 ms (R50), 48.8 ms (light), 14.5 / 20.5 FPS.

## 4. Data and label pipeline
- nuScenes keyframes in scene order; `curr_to_prev_lidar_rt` only to the previous **keyframe** (index - 1), identity at a scene start; history valid inside a `sequence_group_idx`. BEV augmentation (rotation, scale, flip) is the reason for `forward_augs` in the warp: GT boxes and their velocities are augmented consistently with the BEV (standard mmdet3d behaviour, memory).
- Radar: 5 radars, up to 7 sweeps per keyframe via `prev` links, each with its transform into the lidar frame and its time gap in seconds; `invalid/dynprop/ambig` states are defined but not used for filtering in the converter. The radar BEV is therefore a multi-sweep accumulation at the keyframe time, with no velocity compensation of moving points (the per-point time gap enters the pillar features; the 18 input dims were not itemised).
- GT velocity: see 2.5. `nusc.box_velocity` returns NaN at the first / last annotation of an instance; mmdet3d zeroes NaN in the dataset (memory, not read in this repo).

## 5. Evaluation code
Standard nuScenes detection evaluation through mmdet3d (`--eval bbox`); nothing for velocity maps. No velocity-map metric exists in the repo: if we want one (velocity EPE on foreground cells, moving-mask IoU), write our own.

## 6. Port plan

Comparison with Mirko's design (three BEV maps with their own times, pull into camera time, static = ego only, moving = ego + flow, confidence gate):

| aspect | CRT-Fusion | Mirko's design |
|---|---|---|
| direction | forward push (scatter) of the past into the present | backward pull with `grid_sample` into camera time |
| ego vs velocity | separate: velocity shift in the old ego frame first, then rigid ego transform, one 4x4 | separate, same physics; see the corrected formula below |
| velocity source | previous frame's own predicted map, at the source cell | past state's map, sampled at the pulled position |
| dt | constant 0.5 s | per-source: radar cell mean age, past state t_cam - t_past |
| time sources | camera keyframes only; radar accumulated as sweeps with no flow | radar and past each with their own time |
| memory | 6 frames stacked in channels, rewarped each step, truncated gradient | one past state (recurrent) |
| gating | sparse occupancy mask + sigmoid channel gate | confidence gate |
| holes / collisions | present (non-deterministic) | none (pull gives a value for every target cell) |

**Answer to "shift or pull, ego separate?"**: CRT-Fusion shifts (pushes), does not pull; ego motion is a separate 4x4 applied after the velocity offset in pixel coordinates. Static content (|v| < 1 m/s) gets the ego transform only, so this part of Mirko's rule is the same as CRT's.

**Physics check for the pull formula** (derived here, not from the repo). Let `v_c` be the ground velocity of the content expressed in the axes of the camera-time ego frame (what the GT recipe of 2.5 produces), `T_{s<-c}` the rigid transform from camera-time ego coordinates to the ego coordinates at time `t_s`, `R` its rotation, `dt = t_c - t_s`. A cell at `x` at `t_c` was at `x_s = T_{s<-c}(x - v_c dt) = T_{s<-c}(x) - R v_c dt` at `t_s`. Mirko's `x_s = T x - v dt` is therefore exact only if `v` is expressed in the `t_s` axes (or `R = I`); over 0.5 s the heading change of the ego is a few degrees so the error is `|v| dt * yaw` (about 0.3 m at 15 m/s and 0.04 rad) - harmless in the model, but the spec test must use the exact form. Also `v` must be sampled where the content is at `t_c`; if the velocity field belongs to the past state, evaluate it at `x_0 = T x` and optionally iterate once (`x_s = T x - R v(x_0) dt`, then re-sample `v` at `x_s`): at 10-30 m/s and dt 0.5 s the displacement is 5-15 m, many cells, so one fixed-point iteration is not a luxury. Make the number of iterations a parameter.

| item | decision |
|---|---|
| licence consequence | Apache-2.0: adapt with attribution allowed, but there is little to adapt: the temporal block is rewritten as a pull anyway; we keep the Apache notice in the file header if any line is copied |
| what to lift | (1) GT recipe -> `bev_fusion` GT generator: rasterise rotated box footprints into the 0.5 m grid, fill with ground velocity rotated into ego axes by rotations only, zero below 1 m/s (make the threshold a parameter, paper ablation 0.5-1.5 flat within 0.4 NDS), faster box wins on overlap, NaN velocity -> 0 and a validity flag (CRT silently uses 0). (2) Velocity head: Conv3x3-BN-ReLU-Conv1x1(2) and the same with sigmoid(1), on the BEV feature; MSE on foreground only plus a mask-normalised focal BCE; copy the shapes, re-tune the weights (do not copy the x30 and 0.25 blindly). (3) The tuple (`curr_to_prev` 4x4 -> pixel-space matrix `inv(feat2bev) @ M @ feat2bev`) as a unit-test oracle for the ego warp. (4) Time channel: constant channel `age_s` per frame -> we have a real per-cell age map instead |
| what to re-write | the warp (pull with `F.grid_sample`, metre-based grid, no half-extent normalisation, see `fiery_ego_warp.md`), the memory (dense, no occupancy sparsification: lanes are static and are exactly what the occupancy mask would throw away), the gating (confidence from the per-cell weight / age), radar branch (our plan section 3, not RCA_Attention) |
| what to leave | `reverse_bilinear` (non-deterministic scatter), the sparse `his_occ_feat` bookkeeping, mmcv runner, `bev_pool` op, sequence-group machinery (replace by our own scene-ordered loader), the double use of detach (truncated BPTT) unless we choose it, the RCA / PointGating radar fusion |
| dependency gaps | none for the planned block: `torch.nn.functional.grid_sample` + `affine_grid`-free explicit grid; velocity GT needs `cv2.fillPoly` or `matplotlib.path` / own scanline (cv2 is fine in `gcar`) |
| effort, risk | S for `temporal_pull.py` (ego pull + velocity pull + spec tests); M for the velocity head + GT + loss. Risk: the paper gives no isolated gain for the velocity shift and the released shift is probably inactive, so the velocity pull in our design must earn its place with its own ablation (static-only pull vs ego + flow), and the lane/map target is mostly static, so the gain would be on vehicles / pedestrians as removed ghosts, not on lane quality |

Port interface (proposal, `bev_fusion/training/temporal_pull.py`):
```
pull_to_camera_time(feat_src [B,C,H,W], T_src_from_cam [B,4,4]  (ego(t_s) <- ego(t_cam)),
                    dt [B,1,H,W] or [B,1,1,1] (t_cam - t_s, seconds, per cell for radar),
                    vel_cam [B,2,H,W] (m/s, camera-time ego axes, sampled at target cells) or None,
                    grid_meta (x0, y0, dx, dy, H, W), iters=1) -> (warped [B,C,H,W], valid [B,1,H,W])
```
`valid` = 1 where the source point lies inside the source grid (so the gate can distinguish "empty" from "outside"). `align_corners=False`, explicit metre grid.

Minimal spec tests (toy, known answer; numpy reference or analytic):
1. Pure ego motion, static scene: a random feature field defined as a function of world coordinates (e.g. sum of sinusoids), sampled in ego(t_s) and ego(t_c) with a known `T` (translation 3.2 m forward, 0.7 m left, yaw 5 deg); `pull` of the source onto the camera grid must equal the analytic field at the target cell centres, away from the borders and where `valid=1`, max abs error < 1e-4 (bilinear error of a smooth field, so test with a linear field for exactness 1e-6). Also `T = I` and `v = 0` -> identity to 1e-7.
2. Moving target: a Gaussian blob moving at known `v_c = (8, -3)` m/s, `dt = 0.5 s`, zero ego motion: blob at `x_c + (-4, +1.5)` m in the source map (it was there earlier); with the true `v` pull gives a blob centred at the camera-time position (error < 0.05 cell); with `v = 0` the blob centroid is off by exactly `v dt` (the control that must lag). Then repeat with an ego rotation to check the `R v dt` term against the `T(x - v dt)` form.
3. Per-cell `dt` (radar): two halves of the map with ages 0.1 s and 0.4 s, same moving blob, check each lands at `x + v dt_i` independently.
4. Gradient: `feat_src.grad` and `vel_cam.grad` are non-zero and finite (gradient reaches the velocity map through the sampling grid).
5. Parity (optional, S): reproduce `rt_flow_p2c` of CRT (`inv(feat2bev) @ M @ feat2bev`, line 403) in numpy and check that our metre-grid pull and a push-forward of the same cells agree at the integer cell centres for pure ego motion.

## 7. Gotchas
- Line 413 of `crtfusion.py`: `grid[moving_occ_mask][:, 0:2] = ...` is an assignment into a temporary. The velocity shift appears to be dead code in the released version (to be confirmed by running a three-line check on `torch.zeros(5,4,1)`, which we may do locally without cloning the repo). Everything the paper says about the velocity-guided shift is therefore not independently supported by the released code.
- `reverse_bilinear`: repeated-index `+=` (non-deterministic last-writer-wins), argument names `bev_h, bev_w` are swapped relative to the grid columns (x, y) but the indexing `[b, :, w0, h0]` compensates (looks consistent; unchecked), non-standard 1/d weights.
- The history is object-sparse (occupancy > 0.05): background is dropped from memory. Good for objects, wrong for lanes.
- dt constant 0.5; real keyframe gaps vary slightly (memory: about 0.5 +- 0.05 s); radar sweeps carry their own time gap but the BEV memory does not use it.
- History features are `.detach()`ed each step: nothing is learned through time (a 1-step truncated recurrence); the history is also re-warped with only the newest motion map.
- GT velocity is ground velocity (not relative), NaN at track ends, box centre outside the grid skipped.
- Paper vs code differences (3) mean the repository, not the paper, is the reference for any number we reproduce.
- Config: `lr_config=None`, EMA hook, fp16 with loss scale 512 after warm-up; phase 2 must resume from the **EMA** swap of phase 1 (`tools/swap_ema_and_non_ema.py`).
- Leftover `breakpoint()` / `pdb` and undefined `QuickCumsum` in the non-`bev_pool` branch of the view transformer (digest): the code was not cleaned for release.

## 8. Snippets
Apache-2.0, at most 15 lines each, `mmdet3d/models/detectors/crtfusion.py` (line numbers within the file as read; the file carries no own header):

Lines 408-413 (velocity block, the suspect line is the last one):
```python
if curr_motion is not None:
    his_occ_motion = self.his_motion.permute(0,2,3,1).contiguous()[self.his_occ_mask]
    moving_occ_mask = torch.sqrt((his_occ_motion[:,0]**2)+(his_occ_motion[:,1]**2)) > self.motion_th
    his_occ_motion = his_occ_motion / self.voxel_size[0] / self.out_size_factor
    his_occ_motion = his_occ_motion * self.history_cam_sweep_freq
    grid[moving_occ_mask][:, 0:2] = grid[moving_occ_mask][:, 0:2] + his_occ_motion[moving_occ_mask].unsqueeze(-1)
```
Lines 403-404 (ego matrix in pixel space):
```python
rt_flow_p2c = (torch.inverse(feat2bev) @ forward_augs @ torch.inverse(curr_to_prev_lidar_rt)
           @ torch.inverse(self.history_forward_augs) @ feat2bev)
```
Source: https://github.com/mjseong0414/CRT-Fusion at 2fa611a (Apache-2.0, Copyright Open-MMLab / the CRT-Fusion authors).
