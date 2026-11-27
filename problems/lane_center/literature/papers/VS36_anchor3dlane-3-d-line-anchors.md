# VS36 - Anchor3DLane: 3-D line anchors projected to the image, iterative regression, per-point visibility

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S1, S2, S5.*

## Papers

Anchor3DLane (Huang et al., CVPR 2023) https://arxiv.org/abs/2301.02371

## Idea

3-D lane anchors (rays on the ground) are projected into the front-view feature map with the camera matrices, features are read at the projected points, class / x / z / visibility are regressed, and the regressed anchors are projected and read again.

## Formulation

anchor = ray from (x_s, 0, 0) with pitch theta and yaw phi, N points at fixed y (N = 10 for ApolloSim and ONCE-3DLanes, 20 for OpenLane); x_s spaced 1.3 m (range not stated); yaw set {0, +-1, +-3, +-5, +-7, +-10, +-15, +-20, +-30} degrees (17 values, DENSER NEAR ZERO), pitch set {0, +-1, +-2, +-5} degrees (7 values): 119 anchors per x_s; projection [u~, v~, d]^T = K T_{g->c} [x, y, z, 1]^T, u = (W_f / W) u~ / d, v = (H_f / H) v~ / d; front-view feature 45 x 60 x 64 from a 360 x 480 input; bilinear read, features of an anchor's points concatenated; iterative regression re-projects predictions as new anchors; positives = the 3 nearest anchors per GT lane (visibility-weighted distance; thresholds not stated); focal (alpha 0.5, gamma 2) + L1 on x / z + L1 on visibility, lambda_cls = lambda_reg = 1; NMS distance 2; equal-width constraint as post-processing (F1 54.8 -> 55.0).

## Input

image_feat + camera + point_set (anchors). output: 3-D polylines with visibility.

## Training

nearest-anchor assignment (matching_free).

## Differentiable?

yes except NMS.

## Sensors and data

monocular camera (OpenLane, ApolloSim, ONCE-3DLanes).

## Reported results

OpenLane val F1 53.1 (x error close / far 0.300 / 0.311 m), 53.7 with iterative regression, 54.3 with temporal; PersFormer 50.5; ablation OpenLane-300: iterations 1 / 2 / 3 F1 54.8 / 56.3 / 57.0 (the row mapping is ambiguous in the text); front-view sampling 53.1 vs BEV-feature sampling 47.6.

## Compute

not stated.

## Code

URL from memory, not opened: https://github.com/tusen-ai/Anchor3DLane

## Fit to our project

reuse_idea - S - (a) a NON-UNIFORM yaw fan, dense near forward, is a published precedent for a teardrop-like angular distribution; (b) per-point visibility trained with L1 (VS43); (c) two to three re-sampling iterations gain +1.5 / +2.2 F1.

## Needs

camera projection for the reads.

## Conflicts

perspective features; on a cached BEV raster the projection collapses to a direct grid_sample.

## Tried in our experiments

none.
