# VM23 - MapTCL (bidirectional temporal consistency loss, training only)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M (HTML read)** (A abstract, M method, C code page). Verified: checked (HTML); detachment unknown.. Pipeline stage: L, S8, E.*

## Papers

MapTCL: Temporal Consistency Learning via Bidirectional Alignment for Vectorized HD Map Construction (arXiv 2608.05209, Aug 2026; authors not read) https://arxiv.org/abs/2608.05209

## Idea

An auxiliary training loss that penalises geometric and semantic differences between matched past and current instances (vector level, both directions after ego warp) and between warped past raster maps and the current one (raster level, mask focal). No inference cost.

## Formulation

BVCL: past instances with confidence > tau = 0.3 are warped into the current frame, current ones into each past frame by inverse motion; matched by MapTR's hierarchical scheme (instance Hungarian with L1 class + smooth-L1 position cost, then best point order, N = 100, N_p = 20), forward and backward; terms: smooth-L1 between matched points (no Chamfer) and KL divergence between class distributions; temporal weights w = [9, 7, 5, 3, 1] (position), u = [1.0, 0.7, 0.5, 0.3, 0.1] (semantic), N_t = 5 past frames. RCL: a BEV segmentation head (mask focal, beta = 1.0) and a weighted mask focal loss between warped past raster maps and the current one, weights [1.0, 0.7, 0.5, 0.3, 0.1]. Detachment of past predictions not stated. tau ablation (AV2 StreamMapNet): 0.1 -> 54.0 mAP / 23.8 C-mAP, 0.5 -> 56.0 / 36.5.

## Input

memory of N_t past predictions and rasters.

## Output

none at inference.

## Training

added to StreamMapNet (6 + 18 ep, loss only in the temporal stage), SQD-MapNet, MapTracker (6 / 2 / 16); nuScenes 24 ep, AV2 6 ep, 4 x A6000, batch 16.

## Differentiable?

partial (matching inside BVCL; RCL is fully differentiable).

## Sensors and data

camera; nuScenes, AV2, new and original splits, 60 x 30 m (and 100 x 50 m).

## Reported results

nuScenes new split mAP / C-mAP: StreamMapNet 35.2 / 25.7 -> 38.9 / 28.5; SQD-MapNet 36.6 / 25.4 -> 37.9 / 26.8; MapTracker 39.4 / 30.8 -> 41.0 / 32.4; AV2 new split StreamMapNet 53.6 / 35.4 -> 56.7 / 37.9; original nuScenes MapTracker 67.9 / 53.6 -> 70.2 / 59.8. Only C-mAP is reported as consistency metric (no mAS).

## Compute

no inference cost; FPS unchanged.

## Code

none on the page.

## Fit to our project

reuse_idea - S - consistency between the ego-warped previous prediction and the current one, applied both ways. We can drop the instance matching by using our point-to-polyline loss in both directions between the two predicted graphs (matching-free), and the raster part as a mask loss between warped rasterised graphs.

## Needs

memory of past predictions (detach policy open); ego warp. Cached-BEV: yes.

## Conflicts

the vector part as published needs Hungarian + permutation min.

## Tried in our experiments

none (temporal consistency is only a metric in our project).
