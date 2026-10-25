# CRN - code extraction, FUSION sections only (camera BEV + radar BEV + past frames)

Scope (Mirko's brief 2026-10-25): only where camera BEV, radar BEV and past BEVs are fused. Backbones, lift internals and heads are not covered except where they decide the fusion inputs. Extracted by the fusion scout (deep mode), 2026-10-25. Fact status: `checked` = read in the local clone files named; derived statements are marked.

| field | value |
|---|---|
| repository | https://github.com/youngskkim/CRN (local clone `\\wsl.localhost\Ubuntu-22.04\home\ml\work\CRN`, WSL path `/home/ml/work/CRN`) |
| papers covered | CRN, arXiv 2304.00670 (ICCV 2023); paper text not re-read here, only the code |
| state read | commit `5e9d2fa2f91c714b297e75ca666d27fc4ad0d13d`, branch `main`, shallow clone made 2026-10-12 (`.git/HEAD`, `.git/refs/heads/main`, `.git/logs/HEAD`; commit date not readable in the packed objects). Read as local files, no download |
| licence | MIT, "Copyright (c) 2023 Youngseok Kim" (`LICENSE`, checked). The deformable attention file says it is based on BEVFormer's `multi_scale_deformable_attn_function.py` (BEVFormer is Apache-2.0, from memory: keep both notices if copied). The bundled `mmdetection3d/` folder has its own LICENSE (Apache-2.0, not opened) |
| stack | mmcv-full runner/cnn/ops, mmdet, mmdet3d, mmseg 0.28, pytorch-lightning, custom CUDA ops `ops/average_voxel_pooling_v2`, `ops/voxel_pooling_v2` (build with `python setup.py develop`), MultiScaleDeformableAttn CUDA kernel |
| size, checkpoints | fusion files: `layers/fuser/multimodal_feature_aggregation.py` 235 lines, `layers/modules/multimodal_deformable_cross_attention.py` 200 lines, `layers/fuser/conv_fuser.py` 83 lines; checkpoints are on the GitHub release page (R18, R50, 4 key frames; README) |

## Port card
| field | value |
|---|---|
| repo | https://github.com/youngskkim/CRN |
| commit | 5e9d2fa2f91c714b297e75ca666d27fc4ad0d13d (local clone, 2026-10-12) |
| licence | MIT |
| portable | mixed (fuser = mmcv `build_*` helpers + a CUDA deformable-attention op, both replaceable by plain torch `F.grid_sample`; the BEV pooling is a CUDA op) |
| reuse | adapt (MFA block as a reference baseline; the multi-frame scheme is idea-only for us because it re-runs the backbone) |
| effort | M (MFA in plain torch, 1 to 2 days; the multi-frame variant is not worth porting) |
| port_targets | `training/fusion/mfa_deformable.py` (new, only if chosen as comparison baseline), spec test `tests/test_mfa_vs_numpy.py` |
| status | read |
| papers | CRN 2304.00670 |

## 1. Map of the code (fusion path only)
| path | what it does | lines | portable? |
|---|---|---|---|
| `models/camera_radar_net_det.py` | wiring: `backbone_pts` -> `backbone_img` (gets radar context + occupancy) -> `fuser` -> `head`; per-block timers | 141 | yes (glue) |
| `layers/backbones/pts_backbone.py` | radar pillars in the CAMERA PERSPECTIVE plane (width x depth), returns a context map and a 1-channel sigmoid occupancy per camera and sweep | 213 | needs mmcv Voxelization, mmdet3d PillarFeatureNet / SECOND |
| `layers/backbones/rvt_lss_fpn.py` (`RVTLSSFPN._forward_single_sweep` L237-350, `forward` L352-431) | radar-aware view transform; pools camera and radar features with the SAME frustum geometry into one BEV tensor with 80 + 80 channels; loops over the key frames | 432 | CUDA pooling op |
| `layers/fuser/multimodal_feature_aggregation.py` (`MFAFuser`) | the fusion: per key frame, 6 layers of deformable cross-attention between camera BEV and radar BEV, then a 3x3 conv over the concatenated key frames | 235 | after replacing the CUDA op |
| `layers/modules/multimodal_deformable_cross_attention.py` (`DeformableCrossAttention`) | one attention layer, queries from the fused BEV, values from both modality maps | 200 | after replacing the CUDA op |
| `layers/fuser/conv_fuser.py` (`ConvFuser`) | the simple alternative (3 conv layers on concat, same multi-frame reduce conv); not used by the CRN configs read | 83 | yes |
| `scripts/gen_radar_bev.py`, `scripts/gen_radar_pv.py` | offline: radar sweeps -> key-frame lidar frame (ego motion + velocity compensation), then -> camera perspective | 134 / n/a | numpy + nuscenes-devkit |
| `exps/det/CRN_r50_256x704_128x128_4key.py` | the main configuration | 362 | config |

## 2. Key algorithms

### 2.1 What goes into the fuser (inputs, shapes)
- Both modalities are pooled into BEV by the same `average_voxel_pooling`, with the same geometry (`geom_xyz` from `sensor2ego` of that frame; z collapsed to one slice): `rvt_lss_fpn.py` L333-341 concatenates image context (80 ch) and radar context (80 ch) per frustum point, then pools them together. Result per frame: `(B, 160, 128, 128)`.
- `RVTLSSFPN.forward` returns `(B, num_sweeps, 160, 128, 128)`; here "sweeps" = the 4 key frames. `MFAFuser.forward` slices channels 0:80 = camera BEV, 80:160 = radar BEV (`multimodal_feature_aggregation.py` L215-218).
- BEV grid of the main setting: x, y in [-51.2, 51.2] step 0.8 m -> 128 x 128 cells, z collapsed (`CRN_r50...py` L124-127); depth bins 2-58 m step 0.8 (70 bins, L127).

### 2.2 How radar occupancy is used (this is not in the fuser, it is in the lift)
- `PtsBackbone._forward_single_sweep` (`pts_backbone.py` L134-176): radar points are in the perspective plane `[w, d, h]`; pillars of size `[8, 0.4, 2]` over range `[0, 2, 0, 704, 58, 2]` -> a `(140, 88)` = (depth bins, width columns) scatter map; PointPillars scatter + SECOND + SECONDFPN (out 3 x 128). Two heads: `pred_context` (3x3 conv, BN, ReLU, 1x1 -> 80 ch) and `pred_occupancy` (same, -> 1 ch, `sigmoid`, bias initialised for prior 0.01, `occupancy_init`).
- No occupancy loss: `training_step` of the config adds only `loss_detection + loss_depth` (L314, depth weight 3.); the occupancy is learned through the detection loss.
- `rvt_lss_fpn.py` L299-319: camera depth softmax x image feature (the LSS frustum), height summed over the valid z range -> `img_feat_with_depth`; in parallel the image feature (height-collapsed to one row) x radar occupancy `(D, W)` -> `img_feat_with_radar`; both stacked on the channel axis (160 ch) and merged by `ViewAggregation` (3x3 conv stack on the (depth, width) plane, 160 -> 80). So radar occupancy replaces the camera depth distribution as a second depth hypothesis, then a conv merges the two; it is not a gate on the BEV.
- The radar context (80 ch, same (D, W) plane) is concatenated at the frustum points (L330-333) and pooled with the same geometry.

### 2.3 Multi-modal Feature Aggregation, `MFAFuser` (defaults in the class, values of the config in brackets)
- Constructor (`multimodal_feature_aggregation.py` L13-14): `num_sweeps=4, img_dims=80, pts_dims=128 [80], embed_dims=256 [128], num_layers=6, num_heads=4, bev_shape=(128,128)`. The config `fuser_conf` is `img_dims 80, pts_dims 80, embed_dims 128, num_layers 6, num_heads 4, bev_shape (128,128)` (`CRN_r50...py` L197-204).
- Per key frame (`_forward_single_sweep` L133-204): 
  1. LayerNorm over channels on each map (L145-146).
  2. Flatten each map to `(B, 16384, C)` (L150-155); `bev_queries = Linear(80 + 80 -> 128)(cat(img, pts))` (L162-163): the query is the channel-concat of the two BEVs.
  3. Learned 2D positional encoding (`LearnedPositionalEncoding`, 128 x 128, `num_feats = embed_dims/2`) is added inside the attention layer to the query (L165-168, `attn` L165-166).
  4. Reference points: the cell centre `(x+0.5)/W, (y+0.5)/H` for BOTH modalities (`ref_2d`, L120-130, repeated per modality L143). So there is no calibration or ego motion inside the fuser: the two maps are assumed to be roughly aligned and the learned offsets absorb the misalignment.
  5. Six layers, post-norm: `x = LN1(Attn(x, img, pts))`, `x = LN2(FFN(x))`, FFN width 256 (= 2 x embed), dropout 0.1 (L177-192).
- `DeformableCrossAttention` (`multimodal_deformable_cross_attention.py`):
  - values: `Linear(80 -> 128)` per modality, reshaped to `(B, N, heads=4, 32)`, the two modalities are concatenated along the key axis, i.e. treated as two "levels" of the same size 128 x 128 (L172-176; `spatial_shapes` = [(128,128),(128,128)]);
  - `sampling_offsets = Linear(128 -> modalities 2 x heads 4 x points 4 x 2)`, `attention_weights = Linear(128 -> 2 x 4 x 4)` with ONE softmax over modalities x points (8 weights per head) (L178-187). So per query and head the network decides jointly how much to read from camera and how much from radar, and where: this is the gate (a softmax over modalities, no separate confidence input);
  - sampling location = reference + offset / (W, H) (L189-193); initial offsets: head directions on a circle with radius `1..4` cells (L100-117), attention weights initialised to zero (uniform);
  - output = dropout(sampled sum) + identity (L199). The sampling is done by `MultiScaleDeformableAttnFunction_fp32` (CUDA); plain-torch equivalent is bilinear `F.grid_sample` per head, as in mmcv's `multi_scale_deformable_attn_pytorch` (not in this repo, from memory).
  - Cost: 1.2 M parameters for the whole fuser (README-style table in the config docstring), 6.8 ms of 43.97 ms per frame (fusion_layer 5.34 ms; config docstring, FP16, batch 1, GPU not stated in the repo, README says FP16 batch 1).

### 2.4 Multi-frame part (how past BEVs are used)
- Key frames: `key_idxes = [-2, -4, -6]` plus the current one (`CRN_r50...py` L275; the dataset prepends 0, `nusc_det_dataset.py` L193). `infos` has one entry per nuScenes sample (`scripts/gen_info.py` L16-23, one loop per sample), so the offsets -2/-4/-6 are 1, 2, 3 s in the past at 2 Hz (derived from the code; not checked against the paper). The camera sweeps (`sweep_idxes`) are empty in this setting.
- Ego alignment of the past is done in the GEOMETRY, not on BEV features: for every frame the dataset computes `sweepsensor2keyego = global2keyego @ sweepego2global @ sweepsensor2sweepego` (`nusc_det_dataset.py` L494-499) and `get_geometry_collapsed` puts that frame's frustum points into the key-frame ego coordinates (`rvt_lss_fpn.py` L187-214, L304-308). There is no `grid_sample` of a stored BEV and no cached state: all 4 frames are pushed through the whole camera + radar backbone and pooled into the key-frame grid. Past frames run under `torch.no_grad()` (L418, and in `PtsBackbone` L201); only the key frame gets gradients. Depth loss only on the key frame.
- Radar per frame: `scripts/gen_radar_bev.py` accumulates `N_SWEEPS = 8` radar sweeps per radar, each moved by `ref_from_car @ car_from_global @ global_from_car @ car_from_current` (L97), velocity rotated to the reference frame (L104-108), and moving points shifted by their compensated velocity times the time lag: `x += vx_comp * time_lag`, `y += vy_comp * time_lag` (L110-112); sweep index stored as a field (L115). Training keeps `N_use = 5` of the 6 first sweeps at random (`rda_aug_conf`, `base_exp.py` L59-63; `transform_radar_pv` L283-291), drops 10 % of the points, caps at 1536 points per camera (L199, L342). Per-point input to the pillar net is 5 numbers (`w, d, h`-swapped coordinates, normalised RCS, normalised speed, `transform_radar_pv` L334-335, L354).
- Merge of the frames (`MFAFuser.forward` L214-235): the key frame goes through MFA with gradient, each past frame through the same MFA under `no_grad`; outputs `(B, 128, 128, 128)` each are concatenated in the order [key, -2, -4, -6] along channels -> `(B, 512, 128, 128)` -> `reduce_conv`: one 3x3 conv 512 -> 128, BN, ReLU (L82-91). `ConvFuser` (the non-attention variant) ends in two 3x3 convs.
- Object motion: none in the fuser. Moving radar points are shifted by Doppler x time lag offline (above); for the camera frames nothing handles moving objects, the BEV concat conv has to cope.

## 3. Config versus paper
- README table (4 key frames, no CBGS): CRN R50 NDS 56.2 / mAP 47.3 / 22.7 FPS / 61.4 M; R18 54.2 / 44.9 / 29.4 FPS / 37.2 M (README, checked). The config docstring of the R50 file reports a slightly different NDS 0.5617, mAP 0.4725 and FPS 22.74 (same model, evaluation log).
- Resolution: README and config say 256 x 704 images, 128 x 128 BEV. The 100 m "far" setting uses the same grid at 0.8 m.
- The class default `pts_dims=128` differs from the config value 80: the config wins.
- Paper claims not re-checked here: 20 FPS real-time setting, 62.4 NDS test (README abstract text).

## 6. Port plan
| item | decision |
|---|---|
| licence consequence | MIT: adapt with notice. BEVFormer-derived attention: keep the BEVFormer header (Apache-2.0, from memory) |
| what to lift | `DeformableCrossAttention` + `MFAFuser._forward_single_sweep` as a plain-torch block `(cam_bev, radar_bev) -> fused` with `F.grid_sample` replacing the CUDA op; `ConvFuser` as the trivial control. NOT the multi-frame scheme (re-runs the backbone 4 times, 2 s history at 2 Hz, no cached state) |
| dependency gaps | mmcv `build_norm_layer`, `build_feedforward_network`, `LearnedPositionalEncoding` -> `nn.LayerNorm`, `nn.Sequential(Linear, ReLU, Dropout, Linear)`, `nn.Embedding` x/y; `MultiScaleDeformableAttnFunction_fp32` -> grid_sample loop over (modality, point) |
| spec test | (a) toy: if offsets are zero and weights are uniform the output equals the mean of the two value maps at the cell (+ residual); (b) numpy reference of one head: bilinear sample at `ref + offset`, weighted sum, compare with the torch block to 1e-5; (c) shuffle test: replace radar BEV by that of another frame, the gain over camera-only must vanish |
| effort, risk | M; risk = 16384 queries x 6 layers x 32 points is memory heavy with a pure-torch grid_sample at fp32, fine at 128 x 128, check at 120 x 104 / 0.5 m with 4x more cells (our grid is 12480 cells) |

## 7. Gotchas
- Fusion needs both maps in the SAME grid and with the SAME channels layout; the radar enters BEV through the camera frustum geometry (its range comes from radar occupancy along the camera ray), so the radar BEV is a camera-ray-shaped object, not a plan-view point map. Our radar density (plan section 3) is a different input.
- z is collapsed to one BEV slice (`geom_xyz[..., 2] = 0`, `rvt_lss_fpn.py` L337), valid-height filter only `z_bound` [-5, 3].
- Past frames use `no_grad` and share BatchNorm statistics with the key frame in train mode (running stats updated by the no-grad passes); the fuser contains `BatchNorm2d` in `reduce_conv`.
- `MFAFuser.times` is set in `forward`; calling `_forward_single_sweep` alone fails (uses `self.times`).
- The `float()` cast at the end of `forward` and `@auto_fp16` mean fp16 training; do not port the decorators.

## 8. Snippets (MIT; at most 15 lines each)
`layers/fuser/multimodal_feature_aggregation.py` L177-192 (the six-layer loop):
```python
for attn_layer, ffn_layer, norm_layer1, norm_layer2 in \
    zip(self.attn_layers, self.ffn_layers, self.norm_layers1, self.norm_layers2):
    # post norm
    bev_queries = attn_layer(
        bev_queries, feat_img, feat_pts, identity=None, query_pos=bev_pos,
        reference_points=ref_2d_stack, spatial_shapes=spatial_shapes,
        level_start_index=level_start_index)
    bev_queries = norm_layer1(bev_queries)
    bev_queries = ffn_layer(bev_queries, identity=None)
    bev_queries = norm_layer2(bev_queries)
```
`layers/modules/multimodal_deformable_cross_attention.py` L178-193 (offsets, one softmax over modalities and points):
```python
sampling_offsets = self.sampling_offsets(queries).view(
    bs, num_query, self.num_heads, self.num_modalities, self.num_points, 2)
attention_weights = self.attention_weights(queries).view(
    bs, num_query, self.num_heads, self.num_modalities * self.num_points)
attention_weights = attention_weights.softmax(-1)
attention_weights = attention_weights.view(bs, num_query, self.num_heads,
                                           self.num_modalities, self.num_points)
offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
sampling_locations = reference_points[:, :, :, :, None, :] \
    + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
```
`layers/fuser/multimodal_feature_aggregation.py` L227-235 (past key frames, no grad, channel concat + one conv):
```python
ret_feature_list = [key_frame_res]
for sweep_index in range(1, num_sweeps):
    with torch.no_grad():
        feature_map = self._forward_single_sweep(
            feats[:, sweep_index, :self.img_dims],
            feats[:, sweep_index, self.img_dims:self.img_dims+self.pts_dims])
        ret_feature_list.append(feature_map)
return self.reduce_conv(torch.cat(ret_feature_list, 1)).float(), self.times
```
