# fastbev_plus_plus - code extraction (repository ymlab/advanced-fastbev)

| field | value |
|---|---|
| repository | https://github.com/ymlab/advanced-fastbev (owner ymlab), "FastBEV++" code on a BEVDet / mmdet3d fork; branch `fastbev1.0` |
| papers covered | 2512.08237 FastBEV++ - `../papers/2512.08237_fastbev-plus-plus-index-gather-depth.md` |
| state read | commit 04c2b3610f7c33ebc323723fe90998121e22cfac, committer date 2025-12-10, message "update FastBEV++ paper", from the GitHub API commits list (checked, through a page summariser); the repository page shows branch `fastbev1.0`, 1,119 commits. Read with L0 web reads only (tree pages, raw.githubusercontent.com, one Google Drive folder listing); no clone, no download. |
| licence | Apache-2.0 (the repository page shows "Apache-2.0 license"; the README states no licence for the weights and thanks BEVDet, BEVFusion, BEVDepth). Weights: no licence stated. |
| stack | Python, PyTorch, mmcv-full + mmdet + mmdet3d (BEVDet fork, "prepare the environment as in BEVDet"), **no custom CUDA op in the view transformer**; ONNX / TensorRT export for deployment (config `fastbev-r50-deploy.py`, exported ONNX on Google Drive) |
| size, checkpoints, data | the view transformer is one file of about 322 lines; weights on Google Drive (below); nuScenes (+ lidar for the depth variant) |
| extracted by, date | literature scout (deep mode, code extraction), session of 2026-10-25 |

Fact status: raw files read through a page summariser; **line numbers approximate (+-5)**; shapes reasoned from the code, nothing run. The summariser's answers to "is there interpolation / custom op / scatter" agree over three questions (no `grid_sample`, no `scatter_add`, no `index_add`, indexed assignment only).

## Port card

| field | value |
|---|---|
| repo | https://github.com/ymlab/advanced-fastbev |
| commit | 04c2b36 (2025-12-10) |
| licence | Apache-2.0 |
| portable | mixed |
| reuse | adapt |
| effort | S |
| port_targets | `bev_fusion/training/lift_pd.py` (gather core and table builder, pure torch) |
| status | read |
| papers | 2512.08237 |

## 1. Map of the code

| path | what it does | lines | portable to our env? |
|---|---|---|---|
| `mmdet3d/models/necks/fastray_transformer.py` | `FastrayTransformer`: the whole view transform: depth head (1 x 1 conv), index tables, gather of feature and depth probability, product, reshape to voxels, fusion over Z, depth GT + loss | about 322 | yes after removing `BaseModule` / `force_fp32` / registry (pure torch ops: `torch.inverse`, matmul, `where`, advanced indexing, `F.one_hot`) |
| `mmdet3d/models/detectors/fastbev.py` | `FastBEV(CenterPoint)`: prepare_inputs, image encoder, view transformer, BEV encoder, CenterHead, depth loss call | about 160 | glue |
| `configs/fastbev/paper/*.py` | 25+ configs: `fastbev-r50-cbgs` (+ `-depth`, `-wodepth`, `-softmax`, `-softmax-depth`, `-depth_l1/_l2`, `-zrange_l1..l3`, `-fuse_max`, `-fuse_s2c`, `-4d`, `-4d-longterm`, `-4d-afteraline`, R101 variants, `-deploy`) | each about 300 | config only |
| `mmdet3d/models/necks/fpn.py` (`CustomFPN`), `lss_fpn.py` (`FPN_LSS`, `CustomResNet`) | BEVDet neck and BEV encoder | | plain torch + mmcv ConvModule |
| `mmdet3d/datasets/pipelines/loading.py` | `PrepareImageInputs`, `PointToMultiViewDepth` (GT depth from lidar) | | needs lidar for depth GT |

## 2. Key algorithms (where, how)

Reference config `configs/fastbev/paper/fastbev-r50-cbgs-depth.py` (checked): detector `FastBEV`, camera only, `use_depth=True`; backbone ResNet-50 (`out_indices=(2,3)`, `with_cp`, ImageNet weights `resnet50-19c8e357.pth`); neck `CustomFPN` [1024, 2048] -> 256, `out_ids=[0]`; `FastrayTransformer(grid_config, in_channels=256, out_channels=64, image_size=[256, 704], feature_size=[16, 44], downsample=1, stride=16, fuse=dict(type='sum'))`; `grid_config`: x, y = [-51.2, 51.2, 0.8], z = [-2.5, 4.5, 1.0], depth = [1.0, 60.0, 1.0]; BEV encoder `CustomResNet` (64 -> 128/256/512) + `FPN_LSS` (640 -> 256); CenterHead with one 10-class task. AdamW 2e-4, 20 epochs, CBGS, EMA, 6 cameras, 256 x 704 from 900 x 1600 (same augmentation as DualBEV). The repo header of that config reports 0.3025 mAP / 0.3849 NDS for this variant (page header, not the README).

Sizes derived: feature 16 x 44 (stride 16); D = int((60-1)/1.0) = **59** bins (bin i = [1+i, 2+i) m, floor index); grid X = Y = 128, **Z = 7** slices of 1 m (z from -2.5 to 4.5, voxel centres at -2, -1, ..., 4 in the sensor-frame of the BEVDet lidar convention) -> **K = 128 x 128 x 7 = 114,688 voxels** per sample.

### 2.1 FastrayTransformer.__init__ (about L17-81)
- Signature (checked): `(grid_config, in_channels, out_channels, image_size, feature_size, downsample=1, stride=8, fuse=None, use_depth=True, loss_depth_weight=0.125, depth_act='sigmoid', sid=False, is_transpose=True, accelerate=False)`.
- `self.D = int((depth[1] - depth[0]) / depth[2])`; `self.depth_net = nn.Conv2d(in_channels, D + out_channels, kernel_size=1)`: **one 1 x 1 conv, no camera input, no SE gate**. First D output channels = depth logits, the rest = feature (64 ch).
- `self.voxel_coords = nn.Parameter(coords, requires_grad=False)` of shape (X*Y*Z, 3): voxel CENTRES in metres, `coords * interval + (lower_bound - interval/2)`. It is a Parameter, hence **part of the state dict** (key `img_view_transformer.voxel_coords`).
- `fuse`: `'s2c'` -> `Conv2d(out*Z, out, 1)`; `'sum'` / `'max'` -> no parameters. `downsample == 2` adds a 3-conv stack; else Identity.
- No index tables are registered: they are computed in `forward` (below).

### 2.2 Index-table construction - `get_fastray_input(input)` (about L110-219)
Per batch item (python loop over the batch), all in torch on the GPU:
1. `cur_coords = voxel_coords - bda_translation`, then `inverse(bda_rotation) @ coords` (undo the BEV augmentation).
2. For all N cameras at once: `camego2img = new_cam2img @ inverse(sensor2keyego)`; `cur_coords = camego2img[:, :3, :3] @ cur_coords + translation` -> camera coordinates (N, 3, K). The depth is the camera z; points with z <= 0 are set to `inf`.
3. Perspective divide `cur_coords[:, :2] /= cur_coords[:, 2:3]`, then `post_rots @ cur_coords` + `post_trans` (image augmentation), then `(row, col) = coords[..., [1, 0]] / stride` and **`.long()` (truncation = floor for positive values)**. No interpolation weights are made.
4. Validity `on_img`: row < image_h/stride, col < image_w/stride, both >= 0, depth in [d0, d1).
5. Depth index: `((dist - (d0 - step)) / step).long() - 1`, which equals `floor((dist - d0)/step)` (the `- (d0 - step)` / `- 1` pair keeps the value positive before truncation). **The depth bin is the one the voxel's own camera-axis depth falls in; the lookup is one integer bin, no interpolation.**
6. Duplicate-camera rule: loops over cameras; a voxel already claimed by an earlier camera is set invalid for later ones (`on_img[valid_i][on_img[valid_j]==True] = False`), so **each voxel is assigned to at most one camera** (the first valid), no averaging in overlaps. Voxels seen by no camera get a dummy index (0, 0, 0) and a zero feature.
7. Returns per batch item three index lists: voxel index (M,), image index (M, 3) = (cam, row, col), depth index (M, 4) = (cam, row, col, bin). With `accelerate=True`: asserts `batch_size == 1` and `use_depth`, sorts by voxel index (`.sort()`) so the gather comes out in voxel order and no scatter is needed.

### 2.3 forward (about L270-318)
| step | tensor |
|---|---|
| `x = input[0]` (neck features) | (B, N, C=256, 16, 44) |
| `depth_net` on `x.view(B*N, C, H, W)` | (B*N, D + 64, 16, 44) |
| `view(B, N, D+64, H, W).permute(0,1,3,4,2)` | (B, N, H, W, D+64) (channels last, makes the gather rows contiguous) |
| depth = `x[..., :D].sigmoid()` (default `depth_act='sigmoid'`) or `softmax(dim=-1)` (config `-softmax`) | (B, N, H, W, D) |
| feature = `x[..., D:D+64]` | (B, N, H, W, 64) |
| gather `feat[i][cam, row, col]` | (M, 64) |
| gather `depth[i][cam, row, col, bin]` then `unsqueeze(-1)` | (M, 1) |
| product (only `use_depth`) | (M, 64) |
| `voxel_feature = zeros(B, K, 64)`; `voxel_feature[i][voxel_idx] = product` (advanced-indexing assignment, not `scatter`, collision-free because each voxel appears once) | (B, K, 64) |
| `view(B, X, Y, Z, 64)` | (B, 128, 128, 7, 64) |
| fuse: `sum` / `max` over Z, or `s2c`: reshape (B, X, Y, Z*64) -> 1 x 1 conv | (B, 64, Y, X) |
Returns `(x, depth)`; the depth is returned also when `use_depth` is False (then the feature is copied unweighted: this is the "wodepth" arm).

**Equation (paper Eq. 3) in code:** `V[voxel] = Feat[cam, row, col] * Depth[cam, row, col, bin]` with the three indices from the table - the pure-gather PD, except that (i) all lookups are integer (no bilinear in u, v, no linear in depth), (ii) one camera per voxel, (iii) the "probability" is a sigmoid score by default (not a normalised distribution), (iv) heights are summed (`fuse='sum'`) in the reference configs, though `s2c` keeps them as channels (as our plan does).

### 2.4 Depth GT and loss (about L221-268)
- `get_downsampled_gt_depth`: input (B, N, H, W) lidar depth map; minimum non-zero depth per stride x stride block; bin index; `F.one_hot(., D+1)[:, 1:]` -> (B*N*h*w, D); no-label rows all zero.
- `get_depth_loss`: BCE (`reduction='none'`) between the sigmoid / softmax output and the one-hot, foreground rows only (`max > 0`), summed over rows and bins, divided by `max(1, n_fg)`, times `loss_depth_weight` (class default 0.125). The `depth` variant is trained with this; the paper text says that the depth is trained through the detection loss, the config `-depth` uses `PointToMultiViewDepth` and `use_depth=True` for supervision (config checked: the `wodepth` config keeps `PointToMultiViewDepth` in the pipeline but `use_depth=False`).
- `depth_from_lidar` is accepted by `forward` and never used.

## 3. Config versus paper
| item | paper note | code |
|---|---|---|
| Gather(D_prob, I_depth) | formula only | `depth[cam, row, col, bin]` with integer indices; the table is rebuilt in every forward (python loop over the batch) unless `accelerate=True` |
| depth supervision | "no explicit depth ground-truth loss" | a depth loss exists (`use_depth=True`, BCE, weight 0.125); the paper's baseline `-wodepth` has none. Which of the zoo checkpoints has it: only the R101 4D long-term depth model is named "depth" in the model zoo |
| activation | D_prob "distribution over bins" | sigmoid by default (independent per-bin scores), softmax is a separate config (`-softmax`) |
| D and bins | not given | D = 59 bins of 1 m over 1-60 m (`grid_config['depth'] = [1.0, 60.0, 1.0]`) |
| heights | reshaped to [Z, H, W, C] | Z = 7 slices of 1 m; configs `-zrange_l1..l3` vary the range; `fuse` sum / max / s2c |
| TensorRT | "no plugin" | the export config `fastbev-r50-deploy.py` and an ONNX file exist on Google Drive; not read |

## 4. Data and label pipeline
Same BEVDet pipeline as DualBEV: `PrepareImageInputs` (image RGB, mean `[123.675, 116.28, 103.53]`, std `[58.395, 57.12, 57.375]`, `to_rgb=True`, 256 x 704 crop, resize factor 704/1600 + U(-0.06, 0.11), rotation +-5.4 deg, flip), returning `(imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans)`; `FastBEV.prepare_inputs` asserts 7 inputs and returns `[imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda]` with `sensor2egos` as per-camera 4 x 4 matrices. Camera order in the configs: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT (`Ncams = 6`). `PointToMultiViewDepth(downsample=1)` makes `gt_depth` by projecting lidar into each camera.

## 5. Evaluation code
Standard nuScenes detection through mmdet3d; not relevant for the lift. FPS in the paper is for the whole network under TensorRT (see paper note).

## 2.5 Pre-trained checkpoints and the front camera (Mirko, 2026-10-25)

**Released checkpoints** (README "Model Zoo", Google Drive folders; nothing downloaded):
| zoo name | weights | size |
|---|---|---|
| `fastbev-r50-cbgs` (config `configs/fastbev/paper/fastbev-r50-cbgs.py`; whether its `use_depth` is True was not read) | https://drive.google.com/drive/folders/1AYtoX8XaNg8ZckFBgbo2TVmyXb26Dw2V (one file `epoch_20_ema.pth`) | 170.5 MB (Drive listing, checked) |
| `fastbev-r50-cbgs-4d` | https://drive.google.com/drive/folders/1XBcftoEn_2TkpG-qQAo66Oa8_hEnfoqi | not read |
| `fastbev-r101-cbgs-4d-longterm` | https://drive.google.com/drive/folders/1JLcU96Oimk7wSZLi7-FeImUGD_1fXWG1 | not read |
| `fastbev-r101-cbgs-4d-longterm-depth` | https://drive.google.com/drive/folders/1RG2hFFu0germP-8sIvMYpv-K-eVER6yB | not readable (login redirect) |
| exported ONNX | https://drive.google.com/file/d/1qefFlah6PkKPtz0Zgh9aqn0rqITjU2C8/view | not read |
- The README gives no mAP / NDS table in text (images only) and **no licence for the weights**; no GitHub releases exist. There is no checkpoint for `fastbev-r50-cbgs-depth` (the arm with a supervised depth head) in the zoo.
- Contents (derived from the code; file not opened): the full detector, EMA weights: image backbone, neck, view transformer (`depth_net` and `voxel_coords`, plus the optional `fuse` conv), BEV encoder, CenterHead.

**State-dict key prefixes** (derived from `FastBEV.__init__` and the `CenterPoint` base; verify with `list(ckpt['state_dict'])`):
| prefix | module | note |
|---|---|---|
| `img_backbone.` | ResNet-50 | transfer |
| `img_neck.` | `CustomFPN` (1024/2048 -> 256, stride 16, `lateral_convs`, `fpn_convs.0`) | transfer (256 ch) |
| `img_view_transformer.depth_net.weight`, `.bias` | the single 1 x 1 conv (256 -> D + 64 = 123 outputs; 59 depth logits + 64 features) | transfer only if D = 59 / 1 m bins and 64 ch are kept; trained without a camera input, so it is calibration-free |
| `img_view_transformer.voxel_coords` | (K, 3) buffer-like parameter | leave (geometry of that grid) |
| `img_view_transformer.fuse_conv.*`? (only for `s2c`) | | leave |
| `img_bev_encoder_backbone.`, `img_bev_encoder_neck.`, `pts_bbox_head.` | BEV encoder, CenterHead | leave |

**Camera-aware input:** none. `get_mlp_input` returns `None`; the calibration (intrinsics, extrinsics, image / BEV augmentation) enters only through the index tables, i.e. the network itself is calibration-free and works for one camera trivially: the depth head is a 1 x 1 conv per pixel, the backbone is per image, and the table builder loops over cameras (N=1 works; the duplicate-camera rule becomes void). Input format as in section 4: 256 x 704 RGB, mean / std above, `intrins` = original 3 x 3, `post_rots` / `post_trans` 3 x 3 / 3 for the resize-crop-flip-rotate, `bda` 3 x 3, `sensor2keyego` 4 x 4 into the ego frame of CAM_FRONT.
**Caveats:** (1) sigmoid scores, not a distribution: if we put a softmax PD on top, re-train the 1 x 1 head or at least the bias; (2) the only zoo model named "depth" is the R101 4D long-term one (folder not readable); the R50 checkpoint was trained either without a depth term or with the weak implicit one, so do not expect metric depth from it; DualBEV's checkpoint with a supervised DepthNet (BCE on lidar depth, weight 3.0) is the better donor for the depth head.

## 6. Port plan
| item | decision |
|---|---|
| licence consequence | Apache-2.0: adaptation allowed with attribution; we re-write anyway (the core is 25 lines) |
| what to lift | (1) the table layout: `(voxel_idx, cam, row, col, bin)` as five integer lists sorted by voxel, so the gather is in voxel order and needs no scatter -> optional integer fast path (`mode='nearest'`) of `lift_pd.py`; (2) not taken: its integer pixel convention (truncation); we use fractional coordinates with pixel centres at (i + 0.5) * stride and `grid_sample`; (3) `get_downsampled_gt_depth` + the foreground BCE as the depth-loss reference (same recipe as BEVDepth / DualBEV) |
| what to leave | the python batch loop, `torch.inverse` per sample (our calibration is one fixed camera: build the table once), the `voxel_coords` parameter, the first-camera rule (single camera), fusion by sum over Z (we keep Z in the channels) |
| dependency gaps | `BaseModule`, `force_fp32`, mmcv ConvModule; no CUDA op to replace |
| effort, risk | S for the gather core; the deployment recipe (ONNX export of Gather + Mul) is already there for later; risk: integer-only lookups give no gradient to learned elevations, so the default mode of our `lift_pd` is trilinear and the integer mode is the speed fall-back / comparison arm |

Spec test (shared with `dualbev.md` section 6.2): for the integer mode, the sorted five-list table must reproduce `fastray_transformer.get_fastray_input` on a random synthetic camera (numpy transcription of items 1-6 of section 2.2), and `lift_pd(mode='nearest')` with `depth = one_hot(bin)` and `feat = 1` must give 1 exactly at voxels whose floor bin is that bin and 0 elsewhere.

## 7. Gotchas
- The paper's accuracy note (0.359 mAP vs NDS) is inconsistent on the paper page; the config header reports 0.3025 / 0.3849 for R50 depth. Take numbers from our own runs.
- `.long()` truncates; negative coordinates truncate toward zero, so a voxel just outside the left / top edge (coordinate in (-1, 0)) maps to pixel 0 and passes a `>= 0` test unless the check is written on the float. In the reference the check `on_img` is on the already truncated integers (derived from the summary; check before copying).
- The perspective divide happens before `post_rots`; `post_rots` is applied to (u, v, depth): the third row of the 3 x 3 acts on depth, as in BEVDet. The (row, col) swap `[..., [1, 0]]` is easy to lose in a port.
- `accelerate=True` supports batch 1 only and only `use_depth=True`.
- `voxel_coords` is a Parameter: it is saved in the checkpoint and loading a checkpoint with another grid fails or silently mixes grids.
- The same name `FastBEV` is used by an unrelated earlier repository (Fast-BEV 2301.12511, licence not stated, plan says ideas only); this repository is the FastBEV++ (BEVDet-based) one.
- Time per call: the table is built with a python loop per sample and per camera in every forward; the 134 FPS of the paper uses the exported ONNX / TensorRT graph where the tables are constants.

## 8. Snippets
Apache-2.0, at most 15 lines, quoted through a page summariser (line numbers approximate, whitespace may differ).

`mmdet3d/models/necks/fastray_transformer.py`, `get_fastray_input`, about L170-200 (projection, truncation, depth bin):
```python
cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]
cur_coords[:, 2, :][cur_coords[:, 2, :] <= 0.0] = torch.inf
cur_coords = curr_post_rots.matmul(cur_coords)
cur_coords = cur_coords[..., [1, 0]] / self.stride
cur_coords = cur_coords.long()
...
depth_idx = ((dist[c, on_img[c]] - (self.grid_config['depth'][0] - self.grid_config['depth'][2]))
             / self.grid_config['depth'][2]).long() - 1
```

`.../fastray_transformer.py`, `forward`, about L290-310 (the gather-multiply-assign core, own words with the quoted lines):
```python
depth = x[..., :self.D].sigmoid()            # (B, N, H, W, D)   [softmax variant: .softmax(dim=-1)]
# feature   = x[i][cam, row, col]                      -> (M, C)
# depth_sel = depth[i][cam, row, col, bin].unsqueeze(-1) -> (M, 1)
voxel_feature[i][pre_voxel_coors_list[i]] = feature * depth_sel
```
