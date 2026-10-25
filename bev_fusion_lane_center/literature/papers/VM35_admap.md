# VM35 - ADMap (anti-jitter direction-difference loss)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: L, E.*

## Papers

ADMap: Anti-disturbance framework for reconstructing online vectorized HD map (Hu et al., arXiv 2401.13172; ECCV 2024 version retitled "Anti-Disturbance Framework for Vectorized HD Map Construction" per search) https://arxiv.org/abs/2401.13172

## Idea

Targets "point-order jitter" - jagged predicted point sequences from prediction bias - with three parts: a multi-scale perception neck (BEV features downsampled, upsampled and merged; per-level maps feed the decoder during training only), instance interactive attention (instance self-attention first, then point self-attention inside each instance) and a vector direction difference loss comparing the direction of consecutive predicted segments with the corresponding GT segments, weighting sharper GT turns more.

## Formulation

VDDL: cosine between predicted segment j and GT segment j (normalised dot product); weight W = exp((1 - cos_gt) / 2) where cos_gt compares each GT segment with its predecessor (endpoints weight 1.0); loss = sum (1 - cos_pred) * W over segments, range [0, 2]; training weight 1.0 (0.0 - 3.0 ablated, 1.0 best, 53.3 mAP camera-only); L1 weight 5.0; v2 adds dense and one-to-many losses (replication 3). Jitter measure (Table 8): among predicted instances whose internal chamfer-distance sum is below 1.5, the mean chamfer distance to GT: MapTR 0.597, ADMap 0.518, MapTRv2 0.560, ADMapv2 0.422. Point cloud range [-15, -30, 15, 30], voxel 0.15 m; R50 / SECOND; 8 GPUs, batch 8, lr 6e-4, AdamW cosine, 24 / 110 epochs.

## Input

MapTR-family BEV features (camera or lidar).

## Output

MapTR-family polylines.

## Training

MapTR-style matching (details not stated).

## Differentiable?

partial.

## Sensors and data

camera, lidar; nuScenes, AV2.

## Reported results

nuScenes camera mAP ADMap 54.5 vs MapTR 50.3; ADMapv2 62.9 vs MapTRv2 61.5; C&L ADMap 68.0 vs 62.5, ADMapv2 70.8 vs 69.0 (110 ep 72.7); AV2 camera 64.7 / 68.7.

## Compute

no inference change for the multi-scale supervision; FPS not stated.

## Code

https://github.com/hht1996ok/ADMap

## Fit to our project

reuse_idea - S - a turn-aware direction-consistency regulariser on consecutive vertex segments for E120-type position refinement; it needs GT segments aligned with predicted ones (for us: the projection of each predicted segment onto the GT polyline).

## Needs

index-aligned or projected GT segments.

## Conflicts

index alignment comes from matching in the published form.

## Tried in our experiments

E120 refinement (point-to-polyline both ways, +0.012 F1; vertex error 0.54 -> 0.43 m); a direction-difference regulariser: none.
