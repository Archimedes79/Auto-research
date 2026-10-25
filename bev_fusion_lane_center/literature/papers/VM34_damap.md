# VM34 - DAMap (distance-aware soft classification targets, task-modulated attention)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: L, S3.*

## Papers

DAMap: Distance-aware MapNet for High Quality HD Map Construction (Dong et al., ICCV 2025) https://arxiv.org/abs/2510.22675

## Idea

In one-to-many matching all candidates of a GT get label 1 although their localisation differs. DAMap (i) replaces the positive classification target by a localisation confidence P_dist = exp(-lambda * L1 distance between predicted and assigned GT points), lambda = 1, in a distance-aware focal loss DAFL(p, y) = -(y - p)^gamma (y log p + (1 - y) log(1 - p)); (ii) a hybrid scheme: ordinary focal loss in the first decoder layer(s), DAFL in the later ones (one FL layer best: 61.6 vs 60.9 all-DAFL); (iii) task-modulated deformable attention: query split into Q_cls / Q_loc with task-specific attention weights, shared sampling offsets.

## Formulation

see idea; query count, K, loss weights not stated; lr 3e-4 cosine, batch 16, 4 x 3090, 24 / 6 / 110 epochs, R50.

## Input

MapTRv2 / MapQR / Mask2Map hosts.

## Output

polylines + class (quality-aware scores).

## Training

one-to-many matching.

## Differentiable?

partial.

## Sensors and data

camera; nuScenes, AV2; range x +-15, y +-30 m.

## Reported results

nuScenes R50 24 ep, hard set {0.2, 0.5, 1.0} m / easy set {0.5, 1.0, 1.5} m: MapTRv2 36.6 / 60.4 -> 39.0 / 62.8; MapQR + DAMap 46.0 / 68.8; Mask2Map + DAMap 51.2 / 72.6; 110 ep MapTRv2 + DAMap 47.4 / 70.4.

## Compute

12.1 FPS, 52 M parameters (vs 40 M).

## Code

announced https://github.com/jpdong-xjtu/DAMap (not verified live)

## Fit to our project

reuse_idea - S - a calibrated VALIDITY target: train the keep probability toward the localisation quality (exp of the distance to the GT polyline) instead of a hard 0/1; with our point-to-polyline distance this needs no matching; same family as InteractionMap (VM17) and Sparse4D v3 (VM06).

## Needs

a distance of each output element to GT. Cached-BEV: the loss yes.

## Conflicts

the published form assumes an assigned GT.

## Tried in our experiments

keep gate of E120 (0.872) - soft versus hard targets not stated in PROJECT_CONTEXT.
