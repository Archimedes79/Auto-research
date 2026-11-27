# VS08 - UniPAD: differentiable rendering as 3D pre-training (SDF-based opacity along rays)

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S3, B, T.*

## Papers

UniPAD (Yang et al., CVPR 2024) https://arxiv.org/abs/2310.08370

## Idea

Voxel features (lidar, camera or both) are rendered to RGB and depth along camera rays by a NeuS-style SDF volume renderer; the rendering loss pre-trains the backbone; block masking of the input is added.

## Formulation

sample points p_j = o + t_j d, features by trilinear interpolation of a 180 x 180 x 5 voxel grid (32 channels); s_j = MLP_SDF(p_j, f_j); opacity alpha_j = max((sigma_s(s_j) - sigma_s(s_{j+1})) / sigma_s(s_j), 0), sigma_s(x) = 1 / (1 + exp(-s x)) with learnable sharpness s; T_j = prod_{k<j}(1 - alpha_k); w_j = T_j alpha_j; RGB = sum w_j c_j; depth = sum w_j t_j; 96 samples per ray; L1 colour and L1 depth (depth only on rays with lidar), both weight 10. Ray sampling: random (512 rays per view in the implementation details), depth-aware (pixels with lidar depth) or dilation (every I-th pixel): NDS 32.5 / 32.9 / 31.9, memory 1x / 1x / 1.4x. Masking: images block 32, ratio 0.3 (NDS peaks at 0.3: 32.9), points block 8, ratio 0.8. AdamW, 12 epochs.

## Input

bev_raster-like voxel volume (lidar sparse conv or camera lift). output: pre-trained encoder.

## Training

self_supervised (render RGB + depth), then fine-tune.

## Differentiable?

yes.

## Sensors and data

nuScenes lidar + 6 cameras.

## Reported results

nuScenes val detection NDS / mAP: UVTR-L 67.7 / 60.9 -> 70.6 / 65.0; UVTR-C 45.0 / 37.2 -> 47.4 / 41.5; UVTR-M 70.2 / 65.4 -> 73.2 / 69.9; the ablation Table 6 gains +9.1 NDS (lidar), +7.7 (camera), +6.9 (fusion) (the page notes headline and Table 1 baselines differ); segmentation SpUNet 73.3 -> 79.4 mIoU.

## Compute

not stated beyond the memory ratios.

## Code

URL from memory, not opened: https://github.com/Nightmare-n/UniPAD

## Fit to our project

not_suitable as published (3D voxels, camera rays, pre-training of a trainable encoder; our foundation is frozen); reuse_idea - S - sample rays only where a label exists (depth-aware) = verify only rays with an available lane label.

## Needs

trainable BEV / voxel encoder; camera ray geometry.

## Conflicts

frozen foundation; no pre-training stage.

## Tried in our experiments

none.
