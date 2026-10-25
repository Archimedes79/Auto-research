# GaussianCaR - code extraction, FUSION sections only (camera BEV + radar BEV)

Scope (Mirko's brief 2026-10-25): the part where the camera BEV and the radar BEV meet. The Gaussian lift / rasteriser and the heads are described only as far as they define the inputs of the fuser. Extracted by the fusion scout (deep mode), 2026-10-25. Everything below was read in the local clone files named (`checked`) unless marked derived.

| field | value |
|---|---|
| repository | https://github.com/santimontiel/gaussiancar (local clone `\\wsl.localhost\Ubuntu-22.04\home\ml\work\gaussiancar`) |
| papers covered | GaussianCaR, ICRA 2026 (README: "accepted for publication at IEEE ICRA 2026", DOI in the BibTeX); no note in `literature/papers/` was opened |
| state read | commit `c0e748b9e95a6a6062579ddebdcd1c1e79500c2e`, branch `master`, shallow clone made 2026-10-13 (`.git/refs/heads/master`, `.git/logs/HEAD`); local files, no download |
| licence | project: Apache-2.0 (`LICENSE` header read, README states it). The CUDA rasteriser `gaussiancar/ops/diff-gaussian-rasterization/` carries the Inria / MPII "Gaussian-Splatting License" (`LICENSE.md`, first lines read: research-community use, non-commercial terms; rest not read) - do not copy it. `gaussiancar/ops/ptv3/model.py` is "Pointcept detached version" (Xiaoyang Wu; its licence is not stated in the file header - from memory MIT, to verify before use). `cmx.py` follows the CMX fusion modules (licence of the original not checked) |
| stack | Python, PyTorch, hydra / rootutils configs, lightning, timm (`efficientvit_l2.r384_in1k`), einops, custom CUDA `diff_gaussian_rasterization` (built from source), PointTransformerV3 (needs spconv / torch_scatter style deps, not checked) |
| size, checkpoints | fuser 110 lines (`modeling/components/fuser.py`), CMX modules 189 lines (`cmx.py`), model 256 lines; weights on Hugging Face `santimontieleu/gaussiancar` (README) |

## Port card
| field | value |
|---|---|
| repo | https://github.com/santimontiel/gaussiancar |
| commit | c0e748b9e95a6a6062579ddebdcd1c1e79500c2e (local clone, 2026-10-13) |
| licence | Apache-2.0 (project); NC research licence on the rasteriser op |
| portable | mixed (`cmx.py` + `fuser.py` are pure torch + timm init; the splat needs the Inria CUDA op, PTv3 needs sparse-conv packages) |
| reuse | adapt (fuser only); idea-only for the Gaussian splat (a plain-torch BEV splat of 2D Gaussians would replace the CUDA op) |
| effort | S for the fuser (a few hours); L for the splat with covariances if ever wanted |
| port_targets | `training/fusion/cmx_fuser.py` (new, only if chosen as a baseline), spec test `tests/test_cmx_fuser_shapes.py` |
| status | read |
| papers | GaussianCaR (ICRA 2026) |

## 1. Map of the code
| path | what it does | lines | portable? |
|---|---|---|---|
| `gaussiancar/modeling/model.py` (`GaussianCaR`) | forward: camera Gaussians + radar Gaussians -> two rendered BEV maps -> `fuser` -> `decoder` -> `head`, plus an `aux_head` on the finest fused map | 256 | glue |
| `gaussiancar/render.py` (`GaussianRenderer`) | orthographic BEV camera + Inria rasteriser: features are the "colours", covariance 3D given, opacity per Gaussian; one render per sample in a Python loop; output `(B, 128, 200, 200)` | 137 | needs the CUDA op |
| `gaussiancar/modeling/components/fuser.py` (`CMXFuser`) | the four-stage camera-radar fusion | 110 | yes (pure torch + timm `trunc_normal_`) |
| `gaussiancar/modeling/components/cmx.py` | `FeatureRectifyModule` (channel + spatial cross-gating) and `FeatureFusionModule` (cross-attention + conv embed) | 189 | yes |
| `gaussiancar/modeling/components/radar_encoder.py` (`PointsToGaussians`) | PointTransformerV3 on the radar points, MLP heads for mean offset, 6 covariance numbers, opacity, 128-d feature | 164 | needs PTv3 |
| `gaussiancar/modeling/components/dpt.py` (`DPTHead`) | decoder over the four fused scales | n/a | yes |
| `gaussiancar/data/transforms/nuscenes/loading.py` (`RadarDataLoader`) | 7 radar sweeps per radar via the devkit, points in the lidar then ego frame, 48+ features | n/a | numpy |

## 2. Key algorithms

### 2.1 Inputs of the fuser (shapes)
- Camera: 6 cameras, image 224 x 480 after a 46 px top crop (`configs/data/nuscenes.yaml` L18-21); EfficientViT-L2 (stages 1-3, channels 128 / 256 / 512) + a neck that upsamples to one resolution; per pixel: a 64-bin depth logit (depth 1 to 61 m, `depth_num: 64`, `depth_max: 61.0`; uniform bins), a 3D offset, an opacity, a 128-d feature. Mean of the Gaussian = softmax-weighted expected 3D point along the ray; covariance = depth-distribution covariance of the ray points times `error_tolerance^2 / 9` with `error_tolerance = 0.5` (`model.py` L180-188, `gaussiancar.yaml` L11). Then `means += offset` (L199-200).
- Radar: Gaussians per point (mean = point + learned offset, learned covariance by softplus diagonals and tanh-bounded correlations capped at 0.9 of the geometric mean, `radar_encoder.py` L142-161; sigmoid opacity, bias init 3.0). Up to 3500 points (`max_points`), padded.
- Both are rendered by `GaussianRenderer` into `(B, 128, 200, 200)` BEV maps: BEV 100 m x 100 m, 200 x 200 cells = 0.5 m (`nuscenes.yaml` L12-28). Gaussians with opacity <= 0.01 are dropped before rendering (`opacity_filter`, `render.py` L88). Fuser input: `camera_bev, radar_bev`, both `(B, 128, 200, 200)`.

### 2.2 The four-stage fusion `CMXFuser` (`fuser.py` L11-101; embed 128 in the config, 256 is only the class default)
For stage s = 1..4 with maps `c_s, r_s` (both `(B, 128, H_s, W_s)`, H_s = 200, 100, 50, 25):
1. `FeatureRectifyModule(c, r)` (`cmx.py` L49-77), cross-leak with learned gates:
   - channel weights `ChannelWeights` (L9-29): global average and max pool of `cat(c, r)` -> `(B, 4C)` -> MLP (Linear, ReLU, Linear, Sigmoid) -> `(2, B, C, 1, 1)`;
   - spatial weights `SpatialWeights` (L32-46): 1x1 conv on `cat(c, r)` (2C -> C -> 2 channels, Sigmoid) -> `(2, B, 1, H, W)`: a PER-CELL gate for each modality;
   - `c' = c + 0.5 * cw[1] * r + 0.5 * sw[1] * r` and `r' = r + 0.5 * cw[0] * c + 0.5 * sw[0] * c` (L75-76; lambda_c = lambda_s = 0.5). Each map is corrected by the other, scaled by a channel gate and a spatial gate.
2. `FeatureFusionModule(c', r')` (`cmx.py` L159-189): flatten to tokens, `CrossPath` (L111-132): each modality is projected by Linear to 2C and split into `y` and `u`; cross-attention in the "efficient / linear" form: context `ctx_i = softmax_over_channels(k_i^T v_i) * scale` of size `(heads, C/h, C/h)`, and `x1 = q1 @ ctx2`, `x2 = q2 @ ctx1` (L93-108), i.e. the query of one modality reads the key-value summary of the other (cost linear in the number of cells); `out = LN(x + Linear(cat(y, v)))`; 2 heads. Then `ChannelEmbed` (L136-156): concat the two modalities (2C), residual 1x1 conv plus 1x1 - depthwise 3x3 - ReLU - 1x1 conv, BatchNorm -> one fused map `(B, C, H, W)`.
3. Down-sampling between stages (separate for each modality, after the rectify module of the previous stage): stride-2 3x3 conv + 2 ResBlocks (stage 1 -> 2), + 4 ResBlocks (stages 2 -> 3 and 3 -> 4) (`fuser.py` L31-70, forward L79-101).
- Output: list `[cr1..cr4]` of the four fused scales, `(B, 128, 200/100/50/25, ...)`, consumed by `DPTHead` (`gaussiancar.yaml` L54-67). The aux head sees `fused_bev[0]` only (`model.py` L246).
- Gating summary: soft multiplicative gates (sigmoid) of the opposite modality, per channel and per cell; no confidence, uncertainty, range or occupancy input to the gates other than the features themselves; the Gaussian opacity and covariance act earlier, inside the splat.

### 2.3 Temporal fusion: confirmed absent
- Camera: a single frame (`batch["image"]`, `B, N, 3, H, W`, `model.py` L193). A grep over `gaussiancar/` for sweep / prev / history / temporal / ego_motion finds only the radar loaders. No recurrent state, no warp of a past BEV.
- Radar: `RadarDataLoader(num_sweeps=7)` (`nuscenes_dataset_generated.py` L48) aggregates 7 sweeps per radar into the key-frame lidar frame with the devkit `from_file_multisweep`, plus a time-delta feature column (`loading.py` L135-142, docstring L220-222). The velocity columns are rotated to the ego frame but the docstring says the multi-sweep compensation of the velocity is a TODO (L230-231). So temporal information is only radar point accumulation, ego-compensated, with the time lag as a feature.

## 3. Config versus paper
- README (checked): Table III on an RTX 4090: GaussianCaR vehicle 57.3 IoU / 75.6 ms (13.2 FPS), map 81.1 ms (12.3 FPS), against Simple-BEV 55.7 / 57.6 ms, BEVCar 58.4 / 245.6 ms. Map segmentation on nuScenes val: drivable 82.9, lane divider 50.1 IoU (Table II).
- The whole network is therefore not fast; the fuser share is not reported. Nothing about 20 or more FPS.
- `configs/module/gaussiancar.yaml` uses `embed_dims: 128`; `CMXFuser` default is 256 (only used in its `__main__` test, `fuser.py` L104-110).
- `radar.num_features: 51` and `PointTransformerV3(in_channels=51)` while the encoder feeds `radar_points[:, 3:]` as features; the column count of the output of `encode_radar_features` was not verified (possible off-by-three, to be checked before any run).

## 6. Port plan
| item | decision |
|---|---|
| licence consequence | Apache-2.0 for `fuser.py`, `cmx.py`, `model.py`: adapt with notice, but `cmx.py` is derived from the CMX paper's code (original licence not checked): verify before vendoring. The rasteriser is non-commercial research: read-only |
| what to lift | `FeatureRectifyModule` (about 30 lines) as the per-cell, per-channel cross-gate and `FeatureFusionModule` as a second fusion arm; the multi-scale decoder structure is optional. Idea to take: opacity + covariance as the per-source confidence of a splat (our plan section 2.4 / 3 already have footprints) |
| dependency gaps | `timm.models.layers.trunc_normal_` (replace by `torch.nn.init.trunc_normal_`), `rootutils` (drop), CUDA rasteriser (replace by a dense 2D Gaussian evaluation on our 120 x 104 grid, which our plan already needs) |
| spec test | (a) shapes: `(2,128,200,200)` pair -> `[ (2,128,200,200), (2,128,100,100), (2,128,50,50), (2,128,25,25) ]` (the file's own `__main__` test, with 256 channels); (b) gate test: with `r = 0` the rectified camera map must equal `c` (mod the zero-input gates), with `c = r` the fused output is symmetric under swapping the inputs up to the separate parameter sets; (c) shuffle test as in plan section 4 |
| effort, risk | S, risk low (pure convs / small linear attention) |

## 7. Gotchas
- Both BEVs are in the same ego frame and the same grid by construction (both rasterised by the same orthographic camera): there is no deformable alignment; misalignment is left to convs of the later stages.
- The renderer loops over the batch in Python and takes a masked subset of Gaussians per sample (`render.py` L92-103): the number of Gaussians is data dependent; wanted differentiability goes through the rasteriser backward only.
- `BEVCamera.world_view_transform` is a hand-written orthographic matrix, `tanfov` is set from the metre extent (`render.py` L112-113): the unit conventions of the Inria code are re-used in an unusual way; do not copy it into a plain-torch port without a numpy reference of the footprint.
- Padding of radar Gaussians with zero opacity hides the sigmoid bias: padded entries have opacity 0, valid ones start near 0.95 (bias 3.0).

## 8. Snippets (Apache-2.0; at most 15 lines each)
`gaussiancar/modeling/components/cmx.py` L72-77 (the cross-gating of the rectify module, per-channel and per-cell):
```python
def forward(self, x1, x2):
    channel_weights = self.channel_weights(x1, x2)
    spatial_weights = self.spatial_weights(x1, x2)
    out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
    out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
    return out_x1, out_x2
```
`gaussiancar/modeling/components/fuser.py` L79-87 (one stage; the others repeat it after stride-2 down-sampling):
```python
# Stage 0 of fuser.
c1, r1 = self.frm_1(camera_feats, radar_feats)
cr1 = self.ffm_1(c1, r1)
# Stage 1 of fuser.
c2 = self.camera_down_conv1(c1)
r2 = self.radar_down_conv1(r1)
c2, r2 = self.frm_2(c2, r2)
cr2 = self.ffm_2(c2, r2)
```
