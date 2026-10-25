# dualbev - code extraction

| field | value |
|---|---|
| repository | https://github.com/PeidongLi/DualBEV (owner PeidongLi), ECCV 2024 implementation, built as an mmdetection3D / BEVDet fork |
| papers covered | 2403.05402 DualBEV - `../papers/2403.05402_dualbev-heighttrans-prob-sampling.md` (no deep note yet; this file is the code half of the deep pass) |
| state read | commit 301f6321a8e85286f9086f94894920e2d4627280, committer date 2024-07-05 (authored 2024-03-21), branch `main`, from the GitHub API commits list (checked, read through a page summariser). Read with L0 web reads only: GitHub tree pages and raw.githubusercontent.com files, no clone, no download. |
| licence | Apache-2.0 (LICENSE file at the top level, name seen on the repository page; the source header of `view_transformer_dual.py` names Zhijia Technology and Peidong Li). Weights / data licence: not looked at (weights not downloaded; nuScenes is non-commercial anyway). |
| stack | Python, PyTorch, mmcv-full + mmdet + mmdet3d (the repo IS mmdet3d, Docker recommended), custom CUDA op `mmdet3d/ops/bev_pool_v2` (BEVPoolV2), optional DCN in the depth net (switched off in the main config), nuScenes-lidarseg for the labels |
| size, checkpoints, data | the vision part that matters is about 680 lines (`view_transformer_dual.py`, 683 lines on GitHub) plus about 680 lines of the parent `view_transformer.py` (DepthNet, frustum, pooling). Needs nuScenes + nuScenes-lidarseg (`tools/generate_point_label.py` makes depth and semantic labels). Weights: not checked. |
| extracted by, date | literature scout (deep mode, code extraction), session of 2026-10-25 |

Fact status: everything below is `checked` in the sense that I read the raw file through a page summariser (WebFetch); the summariser cannot print long verbatim blocks, so **line numbers are approximate (+-3 lines; for the `forward` method +-25)** and every shape was reasoned by the summariser from the code, not run. The one conclusion that decides our port (nearest-bin lookup, no interpolation) was asked three times in different forms and the answers agree (including "the words `grid_sample`, `interp`, `floor`, `ceil` do not occur in the file").

## Port card

| field | value |
|---|---|
| repo | https://github.com/PeidongLi/DualBEV |
| commit | 301f632 (2024-07-05) |
| licence | Apache-2.0 |
| portable | mixed |
| reuse | adapt |
| effort | M |
| port_targets | `bev_fusion/training/lift_pd.py` (PD gather core + table builder, pure torch); `bev_fusion/training/depth_head.py` (DepthNet with camera gate, depth loss); depth labels need lidar, see section 6 |
| status | read |
| papers | 2403.05402 |
## 1. Map of the code

| path | what it does | lines | portable to our env? |
|---|---|---|---|
| `configs/dualbev/dualbev-r50-cbgs.py` | main nuScenes setting, 1 frame, R50, 256 x 704, 20 epochs CBGS (README: 36.8 mAP / 43.4 NDS, 1 frame; the 4D config `dualbev4d-r50-cbgs.py` is 38.6 / 50.5) | about 300 | config only (mmcv dict) |
| `mmdet3d/models/necks/view_transformer_dual.py` | `DualViewTransformerFull`: HeightTrans (the PD-like lookup), Prob-LSS, fusion, ProbNet (BEV probability), depth + semantic loss | 683 | no: mmdet3d registry, `force_fp32`, `bev_pool_v2` CUDA; the maths is plain |
| `mmdet3d/models/necks/view_transformer.py` | parent classes `LSSViewTransformer`, `LSSViewTransformerBEVDepth`, `DepthNet`, `ASPP`, `SELayer`, `Mlp`, frustum, `get_lidar_coor`, `voxel_pooling_v2` | about 680 | DepthNet is plain torch except the optional DCN; pooling needs the CUDA op |
| `mmdet3d/models/necks/fpn.py` | `CustomFPN` (image neck) | about 185 | plain torch + mmcv ConvModule |
| `mmdet3d/models/detectors/dualbev.py` | `DualBEV(BEVDet)` and `DualBEV4D`: encoder -> view transformer -> BEV encoder -> CenterHead; losses `loss_depth`, `loss_ce_semantic`, BEV mask loss | about 250 | glue |
| `mmdet3d/ops/bev_pool_v2/bev_pool.py`, `src/bev_pool_cuda.cu` | BEVPoolV2: weighted sum of (depth prob x feature) per BEV cell from sorted index tables | small | custom CUDA, replace by gather + `index_add_` or a dense gather |
| `tools/generate_point_label.py` | makes depth and semantic labels from nuScenes lidar + lidarseg | not read | needs lidar |

## 2. Key algorithms (where, how)

Main setting (`configs/dualbev/dualbev-r50-cbgs.py`, checked): detector `DualBEV`; backbone ResNet-50 (`out_indices=(2,3)`, strides 16 and 32, channels 1024 / 2048, `with_cp=True`); neck `CustomFPN` in [1024, 2048] -> 512, `num_outs=1`, `out_ids=[0]`; view transformer `DualViewTransformerFull`, 512 -> 80 channels (`numC_Trans`), `downsample=16`, `input_size=(256,704)`, `depthnet_cfg=dict(use_dcn=False)`, `pc_range=[-51.2,-51.2,-5.0,51.2,51.2,3.0]`; `grid_config` x, y = [-51.2, 51.2, 0.8], z = [-5, 3, 8] (one z cell of 8 m), depth = [1.0, 60.0, 0.5]; BEV encoder `CustomResNet` (80 -> 160/320/640) + `FPN_LSS` 720 -> 256; `CenterHead` (6 tasks). Six cameras, source 900 x 1600, resize (-0.06, 0.11), rot +-5.4 deg, flip; BDA rot +-22.5 deg, scale 0.95-1.05. AdamW lr 2e-4, wd 1e-2, clip 5, warmup 200 it, 20 epochs, EMA, 8 samples per GPU, 8 GPUs (README commands). The config does not set `loss_depth_weight`, `num_height`, `bev_h/bev_w`, `accelerate`: they take the defaults (`loss_depth_weight=3.0`, `num_height=13`, `bev_h=bev_w=128`, `accelerate=False`).

Derived sizes: feature map 16 x 44 (256/16, 704/16); depth bins D = len(arange(1, 60, 0.5)) = 118 (bin i has its LEFT edge at 1 + 0.5 i); BEV 128 x 128 at 0.8 m.

### 2.1 Image head and DepthNet
- `DepthNet` (`view_transformer.py`, class at about L458; `forward` about L503). Constructed as `DepthNet(in_channels=512, mid_channels=512, context_channels=80, depth_channels=D+2 =120, use_dcn=False, use_aspp=True)`. Layers: `reduce_conv` (3x3 conv-BN-ReLU 512 -> 512); `context_conv` (1x1, 512 -> 80); `bn` = BatchNorm1d(27) on the camera vector; `depth_mlp`, `context_mlp` = `Mlp(27, 512, 512)`; `depth_se`, `context_se` = `SELayer(512)` (sigmoid channel gate); `depth_conv` = 3 x BasicBlock(512) -> ASPP(512) (if `use_aspp`) -> [DCN if `use_dcn`] -> 1x1 conv 512 -> depth_channels.
- Camera-aware gate: `mlp_input` (B, N, 27) from `get_mlp_input` (`LSSViewTransformerBEVDepth`, about L589; exact order checked): `[fx, fy, cx, cy` (intrin[0,0], [1,1], [0,2], [1,2]) `, post_rot[0,0], post_rot[0,1], post_tran[0], post_rot[1,0], post_rot[1,1], post_tran[1]` (2 x 2 image-augmentation matrix and offset, row-wise) `, bda[0,0], bda[0,1], bda[1,0], bda[1,1], bda[2,2]` (5 BDA values) `, sensor2ego 3 x 4 flattened row-wise: r00 r01 r02 t0 r10 r11 r12 t1 r20 r21 r22 t2]`. No scaling is applied in `get_mlp_input`; raw values (pixels for intrinsics, metres for t) go into `BatchNorm1d(27)` inside DepthNet, whose running statistics are part of the checkpoint (so the checkpoint's gate expects the statistics of 6 nuScenes cameras). It is reshaped to (B*N, 27), batch-normed, passed through an MLP to (B*N, 512, 1, 1), and used as a sigmoid gate on the `reduce_conv` output, once for the depth branch and once for the context branch (BEVDepth's squeeze-excitation gate).
- Output: `torch.cat([depth, context], 1)` of shape (B*N, D + out, H, W) in the parent; in the dual class the depth branch has D+2 channels: first D channels depth logits, next 2 semantic logits (foreground / background of an image mask), then 80 context channels. Shape at the main setting: (B*6, 200, 16, 44).
- `DualViewTransformerFull.forward` (about L657-690, see the caveat on line numbers): `depth = softmax(depth_digit, dim=1)` over the D bins; `semantic = softmax(semantic_digit, dim=1)`; `filter_depth = where(depth < depth_threshold, 0, depth)` with `depth_threshold = 1 / D` (so bins below the uniform level are zeroed: a sharpening of the distribution, not renormalised); `img_mask = semantic[:, 1:2] >= 0.25` (`semantic_threshold`); `filter_feat = img_mask * tran_feat` (this is the paper's P_img). Then `view_transform(input, filter_depth, filter_feat)`. Returns the BEV feature, the unfiltered softmax depth (for the loss) and `(bev_mask_logit, semantic)`.

### 2.2 HeightTrans: the lookup (the part that decides our port)
All in `DualViewTransformerFull` (`view_transformer_dual.py`).

1. **Fixed heights** - `get_reference_points_3d(H, W, Z=8, num_points_in_pillar=13, bs=1)` (about L260-282). x, y are cell centres `linspace(0.5, W-0.5, W)/W` (and for H), normalised to [0, 1]. z: `zs_l = linspace(3, Z-1, 5)` and `zs_g = linspace(0.5, Z-0.5, 8)`, concatenated (13 values) and divided by Z=8. The callers never pass Z, so Z=8. Returned shape (bs, 13, H*W, 3), z-major. Mapped to metres by `pc_range` (z in [-5, 3]): the 13 heights are -2, -1, 0, 1, 2 and -4.5, -3.5, ..., 2.5, i.e. in the sensor-frame of the nuScenes lidar: 0.5 m steps from -2.5 to 2.5, 1 m below (derived from the formula, not run). They are fixed, not learned.
2. **Projection matrices** - `get_projection(rots, trans, intrins, post_rots, post_trans, bda)` (about L284-294): lidar2img (B, N, 3, 4) = `[K R^-1 BDA^-1 | -K R^-1 t]`; img_aug (B, N, 3, 4) = `[post_rots | post_trans]`.
3. **Points -> image** - `get_sampling_point(reference_points, pc_range, depth_range, lidar2img, img_aug, image_shapes)` (about L296-347): normalised points (B, 13, HW, 3) are scaled to metres, made homogeneous and repeated per camera (B, N, 13*HW, 4); `lidar2img.matmul` gives camera coordinates (u*z, v*z, z); the perspective divide uses `eps = 1e-5`; `img_aug` applies the image augmentation; (u, v) are divided by the image width / height (normalised to [0, 1]); the depth z is normalised `(z - 1) / (60 - 1)` with `depth_range = grid_config['depth']`. Output: coordinates (B, N, 13, HW, 3) = (u/Win, v/Hin, (z-1)/59) and mask (B, N, 13, HW, 1) that is true iff z > eps, 0 < u_n < 1, 0 < v_n < 1 and 0 < z_n < 1.
4. **The index table** - `fast_sample_prepare(coor, mask)` (about L389-454). **This is where the depth is looked up.** The normalised triple is scaled to the feature grid and the bin axis, then ROUNDED to the nearest integer (no floor, no bilinear weights):

   - `coor[..., 0] *= self.W; coor[..., 1] *= self.H; coor[..., 2] *= self.D` (about L407-409), then `coor = coor.round().long().view(num_points, 3)` (about L411), then `clamp_` of each axis to [0, W-1], [0, H-1], [0, D-1].
   - The depth index of a point is therefore the bin whose left-edge-based index is nearest to `(z - 1)/0.5` (since 118/59 = 2 bins per metre). The depth probability used for a point is the probability of ONE bin, at the point's OWN depth (z along the optical axis), at the pixel its (u, v) rounds to.
   - Flat indices: `ranks_depth = cam*(D*H*W) + d*(H*W) + v*W + u` into the depth tensor (B*N, D, H, W); `ranks_feat = cam*(H*W) + v*W + u` into the feature tensor (B*N, H, W, C); `ranks_bev` = the BEV cell (0..HW-1), repeated over cameras and heights. Masked points are dropped (`kept = mask.reshape(-1)`), the three lists are sorted by `ranks_bev` (`argsort`), `interval_starts` / `interval_lengths` give the run of points per occupied BEV cell.
5. **Pooling** - `get_ht_bev_feat` (about L493-523), `fast_sampling` (about L364-387): `bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape=(B, 1, 128, 128, C), interval_starts, interval_lengths)`. Kernel semantics (`src/bev_pool_cuda.cu`, checked): one thread per (interval, channel); `out[ranks_bev[start]*C + c] = sum_{i in interval} feat[ranks_feat[i]*C + c] * depth[ranks_depth[i]]`; a plain store, no atomics. Output permuted to (B, C, 1, Y, X), squeezed to (B, C, 128, 128) (the TensorRT wrapper returns (B, 128, 128, C)). **Z = 1: the 13 heights AND all cameras are summed into the same BEV cell** - the height is not kept as a channel, and the sum is neither normalised nor limited to one camera.
6. **Tables per forward or cached** - with `accelerate=False` (the config default) `get_ht_bev_feat` rebuilds the projection and `fast_sample_prepare` on every forward (tables computed in torch: matmul, round, argsort); `accelerate=True` computes them once in `pre_compute` (guard `initial_flag`) and reuses `ranks_*_ht` (valid only for a fixed calibration). Side effect: `fast_sample_prepare` modifies `coor` in place.

### 2.3 Prob-LSS branch and fusion
- `get_lss_bev_feat` (about L470-491): the classic LSS pooling of the frustum with `voxel_pooling_v2` / `bev_pool_v2`, (B,N,D,H,W) depth x (B,N,C,H,W) features into (B, C, 128, 128) (Z grid has one cell). The BEV probability P_bev (below) enters the branch through the optional `kept` argument; not traced further.
- `view_transform_core` (about L525-541): `lss_feat`, `ht_feat` -> `DualFeatFusion` (`MS_CAM` r=4 on the concatenation, 2C -> C weights w in (0,1), output `w*x1 + (1-w)*x2`) -> `LearnedPositionalEncoding` + `ProbNet` (3x3 conv C -> C/2, ResCBAMBlock, 1x1 -> 1: the BEV cell logit) -> `BEVGeomAttention` (channel mean and max, 7x7 conv, sigmoid(x + bev_logit)) -> multiply. Returns (B, 80, 128, 128), the depth, and the logits.

### 2.4 Losses
- Depth GT: `get_downsampled_gt_depth_and_semantic` (about L543-622). Per feature cell (16 x 16 pixels) the minimum non-zero lidar depth; bin index `long((d - (1.0 - 0.5)) / 0.5)`, 0 if out of range; `F.one_hot(., D+1)[:, 1:]` (out-of-range rows become all zero = ignored). For cells with semantic foreground the hard label is replaced by the normalised histogram of the depths in the block (soft label; division by a zero sum is a NaN risk, not seen running).
- `get_depth_and_semantic_loss` (about L624-655): `F.binary_cross_entropy(softmax depth, target, reduction='none')`, weighted per cell 0.9 for foreground and 0.1 for background, rows with no label dropped, divided by `max(0.1, sum of weights)`; semantic: focal loss (alpha 0.25, gamma 2) on the 2-class softmax / `max(1, N)`. Weights: `get_loss` returns `loss_depth_weight * depth_loss` (3.0, parent default) and `loss_semantic_weight (25) * semantic_loss`. `ProbNet.get_bev_mask_loss` with `loss_weight=6.0` for the BEV mask (needs `gt_bev_mask`).
- The detection loss is CenterHead's (Gaussian focal + 0.25 L1).

### 2.5 Pre-trained checkpoints and the front camera (Mirko, 2026-10-25: front camera only, pre-trained camera heads)

**Released checkpoints** (README "Model Zoo", checked; nothing downloaded):
| config | frames | mAP / NDS | weights | size |
|---|---|---|---|---|
| `configs/dualbev/dualbev-r50-cbgs.py` | 1 | 36.8 / 43.4 | Google Drive folder https://drive.google.com/drive/folders/15VIJDJyqnzTJJYGKgj4AQcWe9ZPslYkz (file `dualbev-r50-cbgs.pth`, plus the log); Baidu https://pan.baidu.com/s/1enTKsWlu_MRonSFthnjDaw?pwd=7umw | 316.7 MB (Drive folder listing, checked) |
| `configs/dualbev/dualbev4d-r50-cbgs.py` | 2 | 38.6 / 50.5 | same Drive folder (`dualbev4d-r50-cbgs.pth`); Baidu https://pan.baidu.com/s/14Z3dWwI3_d-_8VBUqQ7kVg?pwd=qbmn | 318.7 MB |
- Weight licence: **none stated** (the README has no licence statement; only the code carries the Apache-2.0 LICENSE file). The checkpoints are trained on nuScenes (non-commercial) and ImageNet-initialised ResNet-50 (`torchvision://resnet50`); treat as research-only until clarified. No GitHub releases exist.
- Contents (derived from the code, **the file itself was not opened**): one full detector state dict (the README test command loads `epoch_20_ema.pth`, i.e. the EMA weights, `MEGVIIEMAHook`; whether the file keeps a `state_dict` wrapper or `ema_` names is unverified). Modules: image backbone (ResNet-50), image neck, view transformer including the DepthNet and the Dual/Prob modules, BEV encoder (backbone + FPN), CenterHead. The 1-frame model is the right donor for a single frame; the 4D model adds a temporal BEV concatenation (BEV encoder input doubles), so its BEV encoder does not fit.

**State-dict key prefixes** (derived from the attribute names in `BEVDet.__init__` and the `CenterPoint` parent; not read from the file, check `list(ckpt['state_dict'])` after the first download):
| prefix | module | note for our port |
|---|---|---|
| `img_backbone.` | ResNet-50 (stem + layer1-4; only `out_indices=(2,3)` are used, so `layer4` is used; strides 16 / 32) | loadable into `torchvision.models.resnet50` after renaming (mmdet ResNet has `conv1`, `bn1`, `layer1..4` as torchvision; `with_cp` has no keys) |
| `img_neck.` | `CustomFPN`: `lateral_convs.{0,1}.conv/.bn?`, `fpn_convs.0` (3x3), output 512 ch at stride 16 | mmcv `ConvModule` keys end in `.conv.weight`, `.bn.*` if a norm is configured; only stride 16 is produced, so a stride-8 neck needs new weights |
| `img_view_transformer.depth_net.` | `DepthNet`: `reduce_conv.{0,1}` (conv, BN), `context_conv`, `bn` (BatchNorm1d 27), `depth_mlp.*`, `context_mlp.*` (Mlp: `fc1`, `fc2`), `depth_se.*`, `context_se.*` (SELayer: `conv_reduce`, `conv_expand`), `depth_conv.{0,1,2}` (BasicBlock x3), `depth_conv.3` (ASPP), `depth_conv.4` (final 1x1, D+2 outputs) | the one to transfer; sub-module names follow BEVDepth's DepthNet (checked in the code list above, exact `nn.Sequential` indices unverified) |
| `img_view_transformer.{fuser,prob,positional_encoding,geom_att}.` | DualFeatFusion, ProbNet, learned positional encoding, BEVGeomAttention | leave |
| `img_bev_encoder_backbone.`, `img_bev_encoder_neck.` | `CustomResNet` (80 -> 160/320/640), `FPN_LSS` (720 -> 256) | leave (input is the DualBEV BEV map, 80 ch) |
| `pts_bbox_head.` | CenterHead | leave |
Load only the first three rows with `strict=False` and filter keys by prefix; the final 1x1 conv of `depth_conv` (D+2 = 120 outputs: 118 depth, 2 semantic) must be kept at D = 118 / 0.5 m bins if the pre-trained depth is to stay meaningful, or re-initialised (new D, or no semantic channels).

**Input format** (`PrepareImageInputs`, `mmdet3d/datasets/pipelines/loading.py`, checked, line numbers about +-5):
- Image: RGB, normalised with mean `[123.675, 116.28, 103.53]` and std `[58.395, 57.12, 57.375]` (0-255 scale, `mmlabNormalize`; `to_rgb=True` is passed on a PIL (already RGB) image, so a channel swap may happen: check against one image when we load the checkpoint), CHW float. Size 256 x 704 (H x W) from 900 x 1600: train resize factor 704/1600 + U(-0.06, 0.11), crop width random, crop height from the bottom (`crop_h` range (0,0) puts the crop at the bottom of the resized image, so the top 396 - 256 = 140 rows are dropped; derived), rotation +-5.4 deg, random flip. Test: resize 704/1600 (+`resize_test=0`), centred width crop, no flip, no rotation.
- Augmentation matrices (`img_transform`): `post_rot` (3 x 3, identity with the 2 x 2 block set) and `post_tran` (3, first two entries): scale multiplies the matrix, the crop offset is subtracted from the translation, flip uses A = [[-1,0],[0,1]] with b = [crop width, 0], rotation uses [[cos, sin],[-sin, cos]] about the crop centre; (x, y) order = (column, row). Applied to the pixel coordinates of the ORIGINAL image, i.e. `u_aug = post_rot (u, v, 1) + post_tran`; in `get_sampling_point` the perspective-divided (u, v) are first computed with the raw intrinsics and then passed through `img_aug`.
- `intrins` = the original 3 x 3 `cam_intrinsic` of the nuScenes camera (not rescaled); `rots`, `trans` = `sensor2keyego[:3,:3]`, `[:3,3]`: camera -> the ego frame of the key frame's `ego_cam` (default CAM_FRONT) - so for one front camera the extrinsics are not identity but the front camera's mount in this ego frame. `bda` (3 x 3): BEV augmentation (rotation / scale / flip about the origin), identity at test.
- Depth GT file per camera: `samples_point_label/*.npy`, first 4 rows (x, y, depth, label); points scaled/cropped/flipped/rotated with the same matrices and written to a depth map at integer pixels (`gt_depth`, `gt_semantic = label >= 0`).

**Does DepthNet run per camera alone? Yes (checked):** the backbone and neck run on the images reshaped to (B*N, 3, H, W); `mlp_input` is reshaped to (B*N, 27); BatchNorm1d, MLP and SE gate act row by row; there is no attention or concatenation across cameras anywhere in `DepthNet` (N=1 works). Two caveats: (a) in training mode `BatchNorm1d(27)` and all BatchNorm2d take their statistics over all rows, so a batch of front-camera-only images has different statistics from 6-camera batches (use eval-mode BN or re-estimate the running statistics of `depth_net.bn`, or freeze it); (b) the 27-vector of one front camera is a narrow slice of the 6-camera distribution the gate was trained on, but within it. The HeightTrans / LSS branches also loop over N independently (camera index is only a flat-index term).

**Fit to the plan:** resolution (stride 16 on 256 x 704, neck output 512 ch, 80 context channels) differs from the plan's stride 8 of the half-resolution image; the backbone weights transfer at any resolution (fully convolutional), the DepthNet transfers if it is fed the 512-channel stride-16 map (so the neck + DepthNet pair is the unit to keep), the context channels (80) and D (118) are free choices only if `context_conv` and the last depth conv are re-initialised.

## 3. Config versus paper
| item | paper note (2403.05402) | code |
|---|---|---|
| sampling operator for P_proj | "trilinear grid sampler on the depth map" | **nearest neighbour on all three axes**: `round().long()` of (u, v, depth-bin); no `grid_sample`, no bilinear feature sampling, no interpolation between bins anywhere in the file |
| depth taken at the point's own depth | not stated explicitly in the text | **yes**, the normalised camera-axis depth z of the 3D point is the bin coordinate (checked) |
| bilinear feature sampling at (u, v) | "sampled bilinearly" | nearest pixel on the feature map (16 x 44) |
| heights | 13 points, -5..3 m, 0.5 m steps inside +-2 m, 1 m outside | 13 fixed heights from two `linspace` calls (union of 0.5 m steps in [-2.5, 2.5] and 1 m steps down to -4.5), fixed |
| pooling | sum per cell | sum over 13 heights x cameras into ONE BEV cell, no normalisation, no height channels |
| P_proj | the depth probability | `filter_depth`: softmax with entries below 1/D set to zero |

## 4. Data and label pipeline
- Train pipeline: `PrepareImageInputs(is_train=True, load_point_label=True, data_config)`; `Collect3D` keys include `gt_depth`, `gt_semantic`, `gt_bev_mask` (README: run `tools/generate_point_label.py` after downloading nuScenes-lidarseg). The depth label is the lidar point cloud projected into each camera (the BEVDepth recipe) - **needs lidar**, which our project has not downloaded (plan section 2.6).
- Augmentation: per-camera image resize / rotation / flip (enters the projection as `post_rots`, `post_trans` and the camera gate as 6 of the 27 inputs) and BEV augmentation (BDA, enters the projection as `bda`).

## 5. Evaluation code
Not relevant for the lift (standard nuScenes detection mAP / NDS through mmdet3d). Latency claim of the paper (0.32 ms for the operation) is for `bev_pool_v2` with the precomputed table, on a 3090.

## 6. Port plan
| item | decision |
|---|---|
| licence consequence | Apache-2.0: code may be adapted with attribution and the notice kept. In practice we re-write the lift (it is 60 lines of maths) and take only what is listed below as ideas; keep a header comment naming DualBEV and BEVDepth |
| what to lift | (1) the contract `V = c(u, v) * p(z_cam | u, v)` and the pre-processing `depth < 1/D -> 0` (optional ablation) -> `bev_fusion/training/lift_pd.py`. (2) The camera-aware gate of DepthNet: 27-vector -> BatchNorm1d -> Mlp -> sigmoid gate on the reduce_conv features for both branches (re-implement from BEVDepth, MIT, or from this file) -> `bev_fusion/training/depth_head.py`. (3) Depth GT downsampling (min non-zero depth per stride cell, left-edge bin index, one-hot D+1 and drop class 0) and the foreground-only BCE with weight 3.0 -> same file, only when lidar depth is available. (4) The 13 heights are a hint for the elevation set (0.5 m inside +-2.5 m, 1 m below), our 8 slices of 0.5 m from -1 to 3 m (car frame) replace them |
| what to leave | BEVPoolV2 CUDA (we have no pooling: PD is one gather per voxel); the sum over heights and cameras into one cell (we keep heights as channels); nearest-neighbour rounding (see below); Prob-LSS, ProbNet, MS_CAM fusion, BEVGeomAttention, the semantic mask head (P_img) - extra supervised heads; CenterHead, BEV encoder |
| dependency gaps | mmcv `ConvModule`, `force_fp32`, mmdet3d registry, DCN (switched off) -> plain `nn.Conv2d/BatchNorm2d`; `bev_pool_v2` -> not needed for PD (for an S baseline: `index_add_` over a flat voxel index with `depth * feat`) |
| what changes against the paper's practice | the reference (nearest rounding) gives NO gradient to the geometry and puts a hard half-cell error on the lookup (at most 0.5 feature cell = 4 px at stride 8 for our image, and half a depth bin). Our PD uses trilinear sampling (bilinear in u, v on the feature, linear in the depth bin on the depth probability), which is smooth in the elevations z_k and so supports Mirko's learned heights; cost is `grid_sample`, which TensorRT has. Ablation arm: nearest (DualBEV exact) vs trilinear |
| effort, risk | S for the lift + its spec test (hours); M for the depth head with supervision (lidar needed, else depth is trained only through the task loss as in FastBEV++, which gains little) |

### 6.1 Core of lift PD (target interface)
```
build_pd_table(calib, z_k, grid, depth_bins) -> (uv [N,K,2] float, zf [N,K] float, valid [N,K] bool)   # K = X*Y*Z
lift_pd(feat [B,N,C,H,W], depth_prob [B,N,D,H,W], table, mode='trilinear'|'nearest') -> V [B, C, X, Y, Z]
```
Shapes for our grid (plan 2.1): K = 120 x 104 x 8 = 99,840 voxels per camera; with C=64 the product is about 6.4 M multiply-adds. DualBEV reference sizes for comparison: 13 x 128 x 128 = 212,992 candidate points per camera, 80 channels, D = 118, feature 16 x 44.
- features: `F.grid_sample(feat[:, n], grid_uv, mode='bilinear', padding_mode='zeros', align_corners=False)` with `grid_uv` (B, K, 1, 2) normalised to [-1, 1] -> (B, C, K, 1).
- depth: the depth tensor as a 5-D volume (B, 1, D, H, W); `F.grid_sample(vol, grid_uvd, mode='bilinear', align_corners=False)` with `grid_uvd` (B, 1, 1, K, 3) ordered (u, v, d_bin) -> trilinear (B, 1, 1, 1, K). Convention to fix once: bin i is centred at `d0 + (i + 0.5) * s`, so the fractional coordinate is `(z - d0)/s - 0.5` and the normalised bin coordinate is `2 * (zf + 0.5)/D - 1`; DualBEV's own convention is bin i at the left edge (frustum `arange`) with rounding. Clamp `zf` into [0, D-1] and use `valid` for the out-of-range mask, otherwise `padding_mode='zeros'` tapers the weight in the first and last half-bin.
- output: `V = feat_s * depth_s * valid`, summed (or mean over visible) over cameras, then reshape to [B, X, Y, Z*C] and a 1 x 1 conv (plan 2.1).

### 6.2 Minimal spec tests (toy cases with known answers; they also validate the table)
1. Uniform depth: `depth_prob = 1/D` everywhere (constant) -> `lift_pd(...) == lift_p(...) / D` within 1e-6 for every voxel inside the depth range (constant is preserved by trilinear interpolation).
2. One-hot depth: feature map = 1 at one pixel (and a map of ones with the one-hot depth at bin k); only voxels whose table depth equals the bin centre of k (use voxel depths placed exactly at centres) are non-zero, value 1 x 1; voxels half a bin away receive 0.5 (linear interpolation) in trilinear mode and 0 or 1 in nearest mode.
3. Geometry: a synthetic point at a known (x, y, z) in the car frame, a pinhole camera with known K and pose: `uv` of its voxel equals the analytic projection, `zf` equals `(z_cam - d0)/s - 0.5`.
4. Numpy reference for the whole function on a 4 x 4 x 8 grid with random tensors (loop implementation of trilinear weights).
5. Gradient: `gradcheck` on the depth probability and the features in float64; non-zero finite gradient on the elevations z_k when the table is recomputed from them (this FAILS for the nearest mode, by construction).
6. DualBEV equivalence test (reference check of the nearest mode): for random calibration, the sorted `(ranks_depth, ranks_feat, ranks_bev)` of our nearest mode equals a numpy transcription of `fast_sample_prepare` lines 399-453, and `sum over heights, cameras` equals `bev_pool_v2` semantics (index_add reference).

## 7. Gotchas
- The paper says "trilinear"; the code rounds. Do not quote the paper's description as the evidence for the code's behaviour.
- Half-pixel bias in the reference: the frustum uses `linspace(0, Win-1, Wfeat)` for feature pixel positions (pixel i sits at input column about 16.35 i), but the HeightTrans lookup uses `round(u/Win * Wfeat)` = `round(u/16)`. The two conventions differ by up to a fraction of a cell; our table must use one convention for the whole system (align_corners=False, pixel centres at `(i + 0.5) * stride`).
- The depth axis: bin i has its left edge at `1 + 0.5 i`; the label formula `(d - (1 - 0.5))/0.5` puts the label in bin `floor((d - 1)/0.5) + 1` of the D+1 one-hot, after dropping class 0 the bin `floor((d - 1)/0.5)`; the HT lookup rounds the same coordinate: a point is therefore assigned to the nearest bin LEFT EDGE, half a bin shifted against the label convention (derived, not run).
- `depth < 1/D -> 0` after the softmax is applied to the depth passed to the BEV transform, while the loss uses the unfiltered softmax.
- `accelerate=False` by default: the tables are rebuilt each forward (argsort of about 1.3 M entries per sample); with `accelerate=True` they are built once and a changed calibration or BDA is silently ignored.
- `fast_sample_prepare` modifies `coor` in place; `get_ht_bev_feat` overwrites `self.H / self.W` with the actual feature size on each call.
- Normalisation: the sum over 13 heights x several cameras is not normalised; the scale of the BEV feature depends on how many points land in the cell (BN in the BEV encoder absorbs it).
- Z collapsed: DualBEV gives the BEV encoder a pure BEV map; there is no per-height information (the paper ties heights only to the depth weight).
- Docstring and shape comments in `bev_pool_v2` mix (B, Z, Y, X, C) with the unbind of dim 2; the accelerated path squeezes dim 2, which only works because Z = 1.
- The README mAP / NDS 36.8 / 43.4 (1 frame) differs from the paper's multi-frame numbers in the note; do not mix.

## 8. Snippets
Apache-2.0, each at most 15 lines; quoted through a page summariser, so whitespace may differ from the file. Line numbers approximate.

`mmdet3d/models/necks/view_transformer_dual.py`, about L406-414 (nearest-bin lookup of the depth probability at the point's own depth):
```python
# convert coordinate into the image feat space
coor[..., 0] *= self.W
coor[..., 1] *= self.H
coor[..., 2] *= self.D
# [B, N, Z, Nq, 3]
coor = coor.round().long().view(num_points, 3)
coor[..., 0].clamp_(min=0, max=self.W-1)
coor[..., 1].clamp_(min=0, max=self.H-1)
coor[..., 2].clamp_(min=0, max=self.D-1)
```

`.../view_transformer_dual.py`, about L426-428 (flat index into the depth tensor (B*N, D, H, W)):
```python
ranks_depth = coor[:, 3] * (self.D * self.W * self.H)
ranks_depth += coor[:, 2] * (self.W * self.H)
ranks_depth += coor[:, 1] * self.W + coor[:, 0]
```

`.../view_transformer_dual.py`, about L337-340 (depth normalisation and range mask):
```python
if depth_range is not None:
    referenece_depth = (referenece_depth-depth_range[0])/(depth_range[1]-depth_range[0])
    bev_mask = (bev_mask & (referenece_depth > 0.0)
                & (referenece_depth < 1.0))
```

`mmdet3d/ops/bev_pool_v2/src/bev_pool_cuda.cu` semantics (own words, not copied): `out[ranks_bev[s_j]*C + c] = sum_i feat[ranks_feat[s_j+i]*C + c] * depth[ranks_depth[s_j+i]]` over the interval j.
