# SN01 - GaussianCaR (our frozen camera-radar foundation)

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B.*

## Papers

GaussianCaR: Gaussian Splatting for Efficient Camera-Radar Fusion (Montiel-Marin, Antunes-Garcia, Sanchez-Garcia, Llamazares, Caesar, Bergasa; ICRA 2026) https://arxiv.org/abs/2602.08784 ; code https://github.com/santimontiel/gaussiancar ; project page https://www.santimontiel.eu/projects/gaussiancar ; weights https://huggingface.co/santimontieleu/gaussiancar

## Idea

Every camera feature-map cell (6 cameras, EfficientViT-L2 at 1/8 resolution) and every radar point (7 accumulated sweeps of 5 radars) becomes a 3D Gaussian carrying a 128-d feature. Both sets are splatted orthographically (alpha blending) onto a 200 x 200 BEV grid, fused by a 4-stage CMX-style module, decoded by a DPT-style conv decoder and read by segmentation heads. The released map checkpoint predicts seven binary map layers.

## Formulation

Camera: D = 64 uniform depth bins in [1, 61] m (config `depth_num` 64, `depth_max` 61.0); Gaussian mean = softmax-weighted bin depth back-projected with K and extrinsics + learned 3-D offset; covariance from the spread of the depth distribution scaled by 0.5 (6 numbers); opacity sigmoid, Gaussians with alpha < 0.01 dropped (the renderer drops <= 0.05). Radar: per point 3 coordinates + 51 feature columns (one-hots of dynprop / ambig_state / invalid_state, pdh0 ordinal, rcs, vx, vy, vx_comp, vy_comp, rms values, time lag, 2 flags); only |x|,|y| < 50 m and |z| < 10 m kept; past sweeps moved to the reference frame by ego poses (devkit `from_file_multisweep`, 7 sweeps, min distance 1 m); PTv3 U-Net (encoder depths 1 each, channels 64/64/128/128/256, patch 64, grid 2.0), four MLP heads (2 x 512, LN, GELU, dropout 0.1) give offset, 6-number covariance (softplus diagonal, bounded correlation), opacity, 128-d feature; one Gaussian per point, max 3500 points. Splat: F = sum_i f_i a_i prod_{j<i}(1 - a_j). Fuse: 4 stages at 200 / 100 / 50 / 25 cells, 128 channels (config `embed_dims` 128), rectify + fuse modules. Decoder: DPT-style (channels 2 x 128 = 256), main head `dim_last` 256, auxiliary head 128 on the stage-1 fused map. Map task: layers lane, road_segment, road_divider, lane_divider, ped_crossing, walkway, carpark_area, one BCE-type loss each (weight 1.0), `min_visibility` 0.

## Input

6 surround images (paper: 448 x 800; the repo config `scale_0_5` shows h 224, w 480, top_crop 46 - not reconciled) + radar point set (<= 3500 points x 54 columns) in the nuScenes LiDAR_TOP flat ego frame; no lidar at inference, no depth supervision stated.

## Output

7 x 200 x 200 map logits (0.5 m cells, +-50 m, heading-aligned ego frame). `forward` also returns `features_cam`, `features_radar` (per-modality BEV maps), `features_fused` (4-scale list), `aux_output`, Gaussian counts. Accumulated alpha / coverage is NOT returned (the renderer returns only the rendered feature channels). Meaning of the layers (code read): labels come from the nuScenes devkit `get_map_mask` on a 100 x 100 m patch into 200 x 200; in the devkit the "lane" mask uses only `lane` records - `lane_connector` is a lookup-only layer and not a mask layer, so junction connector areas have no own supervision; `road_divider` / `lane_divider` are polylines drawn 2 pixels thick (= 1 m at 0.5 m cells); the other layers are polygons.

## Training

regime matching_free (per-pixel segmentation). L = L_sem + auxiliary L_sem (BCE + Dice + centerness + offset in the paper; weights not stated); AdamW, peak LR 3e-4 (warm-up, linear decay), weight decay 1e-7, 40 epochs, effective batch 16, 4 x A100 80 GB, FP32 only (README: PTv3 part has no backward under mixed precision); BEV augmentation (config [30,20,0,20,0,0]) + image flip / zoom / rotation. nuScenes train split (paper: 700 / 150 / 150 scenes, results on validation). Image backbone `efficientvit_l2.r384_in1k`, `pretrained: true` (the paper does not state pretraining). Map layers are trained without visibility masking (`min_visibility` 0; the paper: visibility filtering does not apply to map evaluation).

## Differentiable?

yes (differentiable Gaussian rasteriser); hard operation: Gaussians below the opacity threshold are dropped. In our use the network is frozen.

## Sensors and data

6 cameras + 5 radars (7 sweeps, ~0.5 s at 13 Hz), nuScenes only (Boston, Singapore); BEV 100 x 100 m, 0.5 m, 200 x 200; camera Gaussians exist only for depth 1 - 61 m (+ offset); radar points cropped to +-50 m.

## Reported results

nuScenes val IoU (paper Tables I, II): vehicle 57.3 (visibility >= 40%), drivable area 82.9, lane divider 50.1 (BEVCar 58.4 / 83.3 / 45.3; CRN 58.8 vehicle; GaussianBeV 82.6 / 47.4). Radar sweeps 1 / 4 / 7: 54.9 / 56.2 / 57.3 vehicle IoU. PTv3 radar branch: 48.0 -> 55.0 vehicle IoU (Table III, XYZ scatter). Runtime 75.6 ms vehicle / 81.1 ms map (RTX 4090, FP32, batch 1). Only drivable area and lane divider are published for the map task; the other five layers (incl. the lane polygon layer) have no published numbers. No failure-case or limitation section; the qualitative figure shows day, rain and night frames without discussion.

## Compute

parameter count not stated; Gaussians converge to ~14k (vehicle) / ~24k (map) per frame.

## Code

https://github.com/santimontiel/gaussiancar - README says the project "is released under the Apache-2.0 License"; LICENSE file = Apache License 2.0 (no copyright holder filled in); HF page licence field Apache-2.0; project page states no licence; `pyproject.toml` has no licence field. CAUTION: the vendored CUDA rasteriser `gaussiancar/ops/diff-gaussian-rasterization/LICENSE.md` is the Gaussian-Splatting licence (Inria / MPII; non-commercial research and evaluation only); the vendored `ops/ptv3` folder shows no licence file (the original PTv3 repo shows MIT). Stack: Python >= 3.10 < 3.13, torch 2.8.0, Lightning 2.5, Hydra, spconv-cu126, torch-scatter, flash-attn, timm, peft, nuscenes-devkit; no mmcv / mmdet. Checkpoints `nuscenes_vehicle.ckpt`, `nuscenes_map.ckpt` on HF (model card empty). GitHub: 47 stars, 11 commits, no open issue (open view only).

## Fit to our project

adapt_code - effort S - it IS the frozen foundation; extra outputs need a small code change and no retraining: return per-modality BEV maps, the fused pyramid, and an alpha-coverage map (render the same Gaussians with all-ones features: F = sum a_i prod(1 - a_j) is the accumulated alpha; derived here from the paper's rendering formula, not tested).

## Needs

calibration + 6 images + radar sweeps at cache time; our cache = 32 PCA channels of the 256-ch decoder feature + 7 logits (+ a 32-ch fused feature).

## Conflicts

frozen (no gradient to sensors); features are cleaner on its own training scenes than on unseen scenes (project: F1@1 0.93 vs 0.82); logits in unobserved cells are learned priors (no visibility masking in training); single-frame camera, no temporal state; camera depth cap 61 m; FP32 only; non-commercial rasteriser licence; 1 m wide divider bands and the missing connector layer limit what the logits say about thin lines and junction interiors (lane_divider IoU 50.1 is the best published value in its table, still only half overlap).

## Tried in our experiments

E27 (detection-pretrained CRN vs GaussianCaR: F1@1 0.624 vs 0.867 -> kept); E17 / E18 decoders on the cache (U-Net kept); E39 / E41 temporal feature fusion on the cache (+0.001 / +0.008 F1); E40 / E48 map prior.
