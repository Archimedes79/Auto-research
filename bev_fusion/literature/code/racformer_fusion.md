# RaCFormer - code extraction, FUSION sections only (radar BEV, camera BEV, ConvGRU over radar BEV)

Scope (Mirko's brief 2026-10-25): where the radar BEV meets the camera features and the implicit dynamic catcher (ConvGRU over radar BEV across frames). Extracted by the fusion scout (deep mode), 2026-10-25, from GitHub raw files only, no clone.

Reliability note: the raw files were read through the page-fetch tool, which summarises with a small model and cannot give verbatim blocks longer than a sentence; line numbers below are the tool's counts and may be off by a few lines, shapes marked "derived" are inferred from constructor arguments. Treat this note as a map to verify against a checkout before any port; the unresolved points are listed in section 7.

| field | value |
|---|---|
| repository | https://github.com/cxmomo/RaCFormer |
| papers covered | RaCFormer, arXiv 2412.12725, CVPR 2025 (note `../papers/2412.12725_racformer-radar-depth-rcs-embedding-doppler-gru.md`) |
| state read | branch `main`, 9 commits (repo page); latest commit hash and date not visible on the page. Read as raw files `models/racformer.py`, `models/racformer_head.py`, `models/racformer_transformer.py`, `models/bev_self_attention.py`, `loaders/pipelines/loading.py`, `loaders/nuscenes_dataset.py`, `configs/racformer_r50_nuimg_704x256_f8.py`, `LICENSE` (2026-10-25); the file tree came from the GitHub API tree listing |
| licence | MIT, "Copyright (c) 2023 Multimedia Computing Group, Nanjing University" (`LICENSE`, fetched) |
| stack | per the code-sources table: torch 2.0.0 / CUDA 11.8, mmcv-full 1.6.0, mmdet 2.28.2, mmseg 0.30.0, mmdet3d 1.0.0rc6; custom CUDA ops in `models/csrc` (`bev_pool_v2`, `msmv_sampling`); built on SparseBEV (sampling code names `sparsebev_sampling.py`, from the file name: derived) and BEVFormer-style `BEVSelfAttention` |
| size | `racformer_transformer.py` about 715 lines (all fusion logic), `bev_self_attention.py` about 225 lines |

## Port card
| field | value |
|---|---|
| repo | https://github.com/cxmomo/RaCFormer |
| commit | web pages only (main, 9 commits; hash not visible) |
| licence | MIT |
| portable | mixed (mmcv / mmdet3d, two CUDA sampling ops; the ConvGRU itself is pure torch) |
| reuse | adapt (ConvGRU radar encoder, about 80 lines) / idea-only (query-based fusion is detection specific) |
| effort | S for the ConvGRU + temporal fusion conv; L for the sampling-based fusion (not wanted: it needs object queries) |
| port_targets | `training/temporal/convgru_radar.py` (control arm A of plan section 7, together with the ego warp of FIERY, which the other scout covers) |
| status | read |
| papers | 2412.12725 |

## 1. Map of the code (fusion only)
| path | what it does | lines | portable? |
|---|---|---|---|
| `models/racformer.py` | `extract_pts_feat` (L130-148): radar points -> pillars -> radar BEV `(B, 256, 128, 128)`; frame loop L309-342 stacks per-frame `bev_feats` `(B, T, C, X, Y)` and `radar_bev_feats` `(B, T, C_r, X, Y)`; the camera lift gets radar depth and RCS as extra maps (L320-334); `simple_test_online` keeps a 16-entry feature memory | about 450 | mmcv / mmdet3d |
| `models/racformer_head.py` | DETR-style head with 900 queries on polar rays; passes `mlvl_feats, lss_bev_feats, radar_bev_feats` to the transformer without touching them | about 520 | n/a |
| `models/racformer_transformer.py` | the fusion: `RaCFormerTransformerDecoderLayer` (L140-274), `BEVSampling` (L425-541), `RadarBEVTemporalEncoder` (L613-657), `ConvGRU` (L659-687), `ConvGRUCell` (L689-714) | about 715 | ConvGRU yes; sampling needs CUDA op |
| `models/bev_self_attention.py` | `BEVSelfAttention`: deformable sampling on a BEV with `num_bev_queue` frames (default 2, 8 here), no alignment inside | about 225 | CUDA op or torch fallback |
| `loaders/nuscenes_dataset.py` (`get_nu_radar`) | radar sweeps -> key-frame LIDAR_TOP frame via ego poses | n/a | numpy |

## 2. Key algorithms

### 2.1 Radar BEV (input of the catcher)
- Radar points per frame: `get_nu_radar` aggregates the radar sweeps of one sample into the key-frame lidar frame with `ref_from_car @ car_from_global @ global_from_car @ car_from_current` (the devkit scheme), removes points closer than 1 m, appends the time lag `ref_time - t_sweep` as a column. Velocity (`vx_comp, vy_comp`) is only rotated into the lidar frame; the file does not shift points by velocity x time lag (different from CRN) (fetch tool reading of `get_nu_radar`).
- `Loadnuradarpoints` / `LoadradarpointsFromMultiSweeps` give one radar point set per frame of the clip (`sweeps_num = 7`, `num_aggr_sweeps = 5`, frames 8 = `num_frames`; frame spacing "fixed interval 6 sweeps at test, random in [4, 8] at training" copied from the camera loader, i.e. about 0.5 s at 12 Hz; the config states no time intervals) (config + `loading.py`). Kept columns: rows `[0, 1, 2, 5, 8, 9, 18]` (x, y, z, rcs, vx_comp, vy_comp, time lag), z set to 0.
- Pillars: `PillarFeatureNet` 7 in -> 64 ch, pillar size 0.8 x 0.8 m, max 10 points per pillar, 30000/40000 pillars; `PointPillarsScatter` -> `(B, 64, 128, 128)` over x, y in [-51.2, 51.2]; `radar_bev_conv` = three 3x3 conv + BN blocks to `embed_dims = 256` (L80-98; config does not define it, the model builds it; derived: 128 x 128 x 256 per frame).
- Radar into the camera lift (not a BEV fusion): `RadarPointToMultiViewDepth` projects the radar points into each camera and writes depth and RCS into WHOLE image columns (vertical stretch), giving `radar_depth`, `radar_rcs` maps of shape `[B*NT, 1, H, W]` that enter the LSS view transformer next to the image features (`racformer.py` L191-195, L320-334).

### 2.2 Implicit dynamic catcher = `RadarBEVTemporalEncoder` + `ConvGRU` (`racformer_transformer.py` L613-714)
- Construction: only the radar sampling branch has it (`BEVSampling(..., temp_radar=True)`, L178 and L463-464): `RadarBEVTemporalEncoder(embed_dims=256, hidden_dims=64, num_frames=8)`; kernel 3, downsample ratio 2.
- Input `bev_feats` `(B, T, 256, 128, 128)`, T = 8 frames, one radar BEV per frame.
- Forward (fetch tool reading; line numbers from the tool):
  1. each frame: `Conv2d(256 -> 64, k=3, stride=2, pad=1)` -> `(B*T, 64, 64, 64)` (L630);
  2. `ConvGRU(input 64, hidden 64, k=3)` over the time axis: zero initial hidden state; cell: the previous hidden state passes a `matching_layer`, one conv outputs `3 x hidden` channels, update gate `z`, reset gate `r`, candidate `h~`; `h_next = (1 - z) * h_prev + z * h~` (L689-714);
  3. only the first 4 frames are processed recurrently (`num_t = 4 if T > 4 else T`, L674); frames with index 4 and above get a zero state `h0` (L676-678); steps with t > 1 run under `torch.no_grad()` (L680-682);
  4. upsample the GRU output to 128 x 128 (L632-634), `cat(bev_feats, h_feats)` -> 256 + 64 = 320 channels (L648), `Conv2d(320 -> 256, k=3)` (L627) -> the temporally enriched radar BEV, one per frame.
- Ego alignment of the hidden state: NONE found. The GRU receives the T radar BEVs as they come from the loader; the only alignment is that each frame's radar points were moved into the key-frame lidar frame by `get_nu_radar` (if `get_nu_radar` is called with the same key-frame reference for every frame of the clip - not verified, see section 7). The hidden state is not warped between steps, there is no `grid_sample` in the five files read.
- Object motion: not handled in the GRU. Elsewhere, `BEVSampling.inner_forward` shifts the sampling points of older frames by the query's predicted velocity x `time_diff` (L376-383 for the image sampling, L501-503 for the BEV sampling: `sampling_points = sampling_points[..., 0:2] - dist`), where `time_diff` comes from `img_timestamp` (L94-99). That is query-level motion compensation (a box moves, its sampling points follow), not a cell-level flow field.

### 2.3 Where radar BEV and camera meet (the fusion operator)
- Per decoder layer (6 layers, 900 queries, embed 256), three read-outs, then one concat:
  - `sampling_radar_bev = BEVSampling(num_levels=1, num_heads=4, num_points=4, depth_num 5, num_frames=8, temp_radar=True)` reads the temporally encoded radar BEV;
  - `sampling_lss_bev = BEVSampling(...)` without the temporal encoder reads the camera LSS BEV (`lss_bev_feats`, `numC_Trans = 256`, grid 0.8 m, 128 x 128);
  - `RaCFormerSampling` reads the multi-view image features (4 groups, 4 points, 3 depth bins, 8 frames, 4 FPN levels);
  - the three query features are normalised and fused: `query_feat = norm_fusion(Linear(3 x 256 -> 256)(cat(query_img, query_radar, query_lss)))` (L188, L252). This is a per-query concat + linear, not a cell-wise BEV fusion: radar BEV and camera BEV are never merged into one map.
- `BEVSampling` (L425-541): per query, `Linear(256 -> depth_num x heads x points x 2)` gives offsets in a polar-style frame (query centre -> polar), z-offset zero, depth axis `linspace(-d_region, d_region, depth_num)` plus a learned sigmoid-bounded `ray_points_offset`; weights softmax over depth x points; samples all `num_frames` BEVs through `BEVSelfAttention` (`num_bev_queue = num_frames`); the history is averaged over the queue (or a learned softmax with `queue_weight`, `bev_self_attention.py` L209-215, `output = torch.sum(output, dim=-1) / num_bev_queue`).
- So the temporal fusion of the camera is only the query-guided sampling of 8 frames of LSS BEV (with velocity shift); the radar additionally has the ConvGRU.

## 3. Config versus paper
- Config file name `racformer_r50_nuimg_704x256_f8` = R50, 704 x 256, 8 frames. Paper (our note): the real-time variant uses a 64 x 64 grid (1.6 m voxels) and 4 frames (51.0 mAP / 58.8 NDS, 12 FPS on an RTX 3090) - the released config is the 128 x 128, 0.8 m, 8 frame setting (config values fetched); no config for the real-time variant was seen in the tree (`configs/` holds one file).
- The paper note says the paper lists no separate Doppler channel for the catcher; the loader keeps `vx_comp, vy_comp` as point columns that go into the 7-channel pillar encoder, so Doppler enters the radar BEV through the pillar features, not through the GRU.
- Training: AdamW 4e-4, weight decay 0.01, 36 epochs, batch size 2 per GPU, fp16, cosine with 500 warm-up iterations, image backbone lr x 0.1 (config).

## 6. Port plan
| item | decision |
|---|---|
| licence consequence | MIT: adapt with notice (Nanjing University); the sampling code derives from SparseBEV / BEVFormer, check their headers (Apache-2.0 / MIT, from memory) before copying `sparsebev_sampling.py` or `bev_self_attention.py` |
| what to lift | `ConvGRU` / `ConvGRUCell` / `RadarBEVTemporalEncoder` as the "A: ego warp + ConvGRU" control of plan section 7 (put the FIERY ego warp in front of the GRU, which RaCFormer does not have); the idea of radar depth + RCS columns as a depth prior for the lift (already in our plan 2.4 / 3) |
| not lifted | query-based fusion (needs 900 object queries, polar assigner, denoising queries) |
| dependency gaps | mmcv `BaseModule`, `ConvModule` -> plain `nn.Module` / `nn.Sequential` |
| spec test | (a) GRU toy: constant input x, zero state: after one step `h = z * h~` with `z = sigmoid(...)`; compare with a numpy cell on random weights (1e-5); (b) memory test: a single radar blob in frame 0 and zeros afterwards must leave a decaying trace in `h_t` (and, with our ego warp in front, the trace must move with the ego motion); (c) shape: `(B, T, 256, 128, 128)` -> `(B, T, 256, 128, 128)` |
| effort, risk | S for the GRU; risk: the released code runs recurrence over only 4 of 8 frames and with `no_grad` for t > 1, so the gradient reaches the first two steps only |

## 7. Gotchas and open points
- Unresolved (not confirmed from the files read): whether the per-frame radar BEVs of the GRU are ALL expressed in one common frame. `get_nu_radar` is keyed to a reference sample; the per-sweep call passes `radar_sample_rec=sweep` (`loading.py`), so each frame's points may be in their own lidar frame, which would make the GRU unaligned. The camera matrices of history frames are reported as not re-expressed in the key-frame frame (`collect_sweeps` stores raw sweep dicts); the SparseBEV-style sampling then uses per-frame `lidar2img` from `sweeps`. Verify with a checkout before relying on "ego-aligned".
- `racformer.py`: in the `stop_prev_grad > 0` training branch `radar_bev_feats` seems never assigned (UnboundLocalError reported by the tool at L348) and the `collect_sweeps` previous-frame pairing looks off by one; `get_nu_radar` with `mutil_sweep=False` calls `RadarPointCloud.from_file()` without a path. Do not trust unverified.
- `models/csrc/msmv_sampling copy/` is a duplicate folder with a space in its name; the tree has a second `bev_pool_v2`.
- Frame spacing for the 8-frame clip is not stated in the config; the sampling uses the real timestamps (`time_diff`), so the spacing is data driven.

## 8. Snippets (MIT; only one-line quotes could be read verbatim, so no block is reproduced)
`models/racformer_transformer.py` L252 (the whole fusion operator):
```python
query_feat = self.norm_fusion(self.fusion(torch.cat((query_feat, query_radar_feat, query_lss_feat), dim=-1)))
```
`models/racformer_transformer.py` L648 (temporal fusion of the GRU output) and L674 (4 recurrent frames):
```python
bev_feats = torch.cat((bev_feats, bev_h_feats), dim=2)
num_t = 4 if T>4 else T
```
