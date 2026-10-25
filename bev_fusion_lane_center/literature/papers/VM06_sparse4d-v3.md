# VM06 - Sparse4D v3 (temporal denoising, quality estimation, instance IDs)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: T, S8, S3.*

## Papers

Sparse4D v3: Advancing End-to-End 3D Detection and Tracking (Lin et al., arXiv 2311.11722) https://arxiv.org/abs/2311.11722

## Idea

(1) Temporal instance denoising: noisy anchors built from GT are propagated into the next frame together with the normal instances (ego + velocity compensation; their features initialise the next frame's), so the temporal path sees realistic stale instances. (2) Decoupled attention: concatenate (not add) anchor embedding and feature for q / k. (3) Quality estimation: centerness C = exp(-||xyz_pred - xyz_gt||) and yawness Y = dot of (sin, cos) of predicted and GT yaw, supervised by focal / CE; centerness x confidence ranks detections. (4) End-to-end tracking by propagating IDs of instances above a threshold.

## Formulation

noise per anchor component uniform: positive set in (-x, x), second set in (-2x, -x) U (x, 2x) (x NOT stated); M = 5 noise groups, M' = 3 random groups propagated; groups independent (attention mask); each group matched to GT by bipartite matching (at most one positive per GT per group; avoids DINO's fixed assignment). The N_t = 600 highest-confidence instances of the previous frame are carried (900 instances in the decoder); carried confidence c' = max(c', c * 0.6); ID given when confidence >= 0.25 and none yet; IDs persist; no tracking-specific training. 6 layers, dim 256, 7 fixed + 6 learnable keypoints; 100 epochs, AdamW.

## Input

multi-view image features + instance memory.

## Output

3D boxes + class + velocity + quality + IDs.

## Training

Hungarian per group + focal / CE quality losses; regime denoising + hungarian_set.

## Differentiable?

partial - top-k, matching.

## Sensors and data

camera; nuScenes.

## Reported results

nuScenes val R50 256 x 704: mAP 0.469, NDS 0.561, AMOTA 0.490, 19.8 FPS (v2: 0.439 / 0.539); R101 512 x 1408: 0.537 / 0.623 / AMOTA 0.567; test V2-99: 0.570 / 0.656.

## Compute

19.8 FPS (R50).

## Code

https://github.com/linxuewu/Sparse4D

## Fit to our project

reuse_idea - S - the confidence decay of carried instances (x 0.6) is a hand-made validity decay; a calibrated quality head for ranking and propagation of noised GT through the same operator as real instances are portable.

## Needs

instance geometry + features; bipartite matching for DN groups (replaceable by a nearest-GT rule). Cached-BEV: only the memory logic.

## Conflicts

3D boxes, not polylines.

## Tried in our experiments

none.
