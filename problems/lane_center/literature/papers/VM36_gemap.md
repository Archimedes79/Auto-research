# VM36 - GeMap (Euclidean-invariant geometric clues, geometry-decoupled attention)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: L, S5.*

## Papers

GeMap: Online Vectorized HD Map Construction using Geometry (authors not on the page, ECCV 2024 from memory) https://arxiv.org/abs/2312.03341

## Idea

Describes each polyline by translation- and rotation-invariant quantities - edge lengths d_u, turning angles alpha_u, and inter-instance angles and point distances - and adds an Euclidean loss on them to the MapTR loss; geometry-decoupled attention applies a shape attention (same instance) then a relation attention (different instances).

## Formulation

L = 0.005 L_Euc + 2 L_cls + 5 L_pts + 0.005 L_dir + 1 L_seg + 3 L_dep (+ BEV 1, PV 2); angle term |cos a^ - cos a| + |sin a^ - sin a|; GDA beats two plain self-attention layers by +1.2 mAP (51.4 -> 52.6); N = 50 instances x 20 points (10 better for dividers), 6 blocks, GKT 1 layer, lr 6e-4 batch 4 (R50), 110 epochs nuScenes / 24 AV2.

## Input

MapTR-family.

## Output

polylines.

## Training

Hungarian (MapTR matching).

## Differentiable?

partial.

## Sensors and data

camera, lidar; nuScenes, AV2; no geographic split, no temporal part.

## Reported results

nuScenes camera R50 110 ep with dense losses 69.4 mAP (div 69.8, ped 67.1, bnd 71.4), 15.8 FPS; without dense 62.7; V2-99 76.0; Swin-T 72.0; AV2 71.8; weather split sunny / cloudy / rainy 73.1 / 71.0 / 59.3.

## Compute

15.8 FPS.

## Code

https://github.com/cnzzx/GeMap

## Fit to our project

reuse_idea - S - pose-invariant shape descriptors (lengths and turning angles) as a loss are invariant to start point and global shift (not to direction reversal of the angle sign); a vertex-shape regulariser for our graph.

## Needs

ordered polyline.

## Conflicts

ordered chains only.

## Tried in our experiments

none.
