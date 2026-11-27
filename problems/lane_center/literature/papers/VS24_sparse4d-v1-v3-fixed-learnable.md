# VS24 - Sparse4D v1 / v3: fixed + learnable keypoints scaled by the instance box, bounded offsets, quality head

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2, S3, S8.*

## Papers

Sparse4D (Lin et al., arXiv 2022) https://arxiv.org/abs/2211.10581 ; Sparse4D v3 (Lin et al., arXiv 2023) https://arxiv.org/abs/2311.11722

## Idea

Each instance (anchor box + feature) samples a fixed keypoint set (box centre, face centres) plus learnable keypoints whose offsets are bounded and expressed in box units; v3 adds a quality head (centerness, yawness) whose product with the confidence ranks detections.

## Formulation

v1: 7 fixed + 6 learnable keypoints; D_m = R_yaw [sigmoid(Phi(F_m)) - 0.5], so every learnable offset lies in [-0.5, 0.5] box units; P_L = D_m * [w, h, l] + [x, y, z]; temporal keypoints shifted with constant velocity and ego motion; bilinear reads per view n, scale s, time t; weights W = Psi(F_m) (linear layer, groups G, value not stated); fusion order views / scales, then time, then keypoints. v3: centerness C = exp(-|pos_pred - pos_gt|_2), yawness = sin / cos product, loss lambda1 CE(Y) + lambda2 focal(C) (lambdas not stated), score = centerness x confidence; 5 denoising groups (3 temporal); decoupled attention (concatenate instead of add); 900 instances, 600 temporal, 6 layers.

## Input

image_feat pyramids + instance anchors. output: 3-D boxes, tracking ids.

## Training

bipartite matching + denoising (Hungarian family).

## Differentiable?

partial.

## Sensors and data

nuScenes cameras.

## Reported results

v3 ablation nuScenes val R50 (mAP / NDS): v2 0.439 / 0.539; + single-frame denoising 0.447 / 0.548; + decoupled attention 0.458 / 0.551; + temporal denoising 0.462 / 0.557; + centerness 0.463 / 0.554; + yawness 0.469 / 0.561; v1: learnable keypoints on vs off 0.537 vs 0.533 NDS (the text's percentage gains do not match this table).

## Compute

v3 R50 256 x 704 19.8 FPS.

## Code

URL from memory, not opened: https://github.com/linxuewu/Sparse4D

## Fit to our project

reuse_idea - S - cleanest published recipe for "teardrop = reference pattern scaled by (length, width) + bounded learned offsets"; the quality head is a calibrated nu for the beam.

## Needs

per-tip size. compat_conflicts: the perspective projection is not needed.

## Tried in our experiments

none.
