# VM38 - Mask2Map (BEV masks -> instance queries -> key points; inter-network denoising)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S1, S2, T.*

## Papers

Mask2Map: Vectorized HD Map Construction Using Bird's Eye View Segmentation Masks (Choi et al., ECCV 2024) https://arxiv.org/abs/2407.13517

## Idea

Stage 1 (IMPNet) produces Mask2Former-style instance queries and BEV masks; stage 2 (MMPNet) refines point queries with (i) a positional query from the mask-weighted sinusoidal embedding (pixels > 0.3) and (ii) a geometric feature extractor: pixels > 0.8, strongest pixel per 4 x 4 window ("max survival"), farthest-point sampling of N_S = 20 key points, BEV features pooled at those points; then a deformable decoder. Inter-network denoising (INDT): noisy GT queries (class flip probability 0.2; GT masks built from noisy ordered points, magnitude not stated) are matched to their own GT without Hungarian, to fix inconsistent assignments between the two networks (query utilisation 24.7 % -> 74.7 %).

## Formulation

masks tau_M = 0.5 for masked attention; BEV encoder 6 layers, mask transformer 3 layers, decoder 6 layers; loss = L_seg + L_map (L1, focal, cosine direction) + L_aux (depth, 2D seg) + L_dn, apparently unweighted; range x +-15, y +-30 m; R50; AdamW 6e-4, 24 / 110 epochs, 4 x 3090.

## Input

BEV features + BEV masks.

## Output

polylines.

## Training

Hungarian for learnable queries, identity assignment for noisy ones.

## Differentiable?

partial.

## Sensors and data

camera, lidar fusion; nuScenes, AV2.

## Reported results

nuScenes camera R50 24 ep 71.6 mAP (MapTRv2 61.5), 10.1 FPS; 110 ep 74.6 (68.7); C + L 78.4; rasterised metric 54.7 vs 36.7; AV2 71.5 vs 67.4.

## Compute

10.1 FPS.

## Code

https://github.com/SehwanChoi0307/Mask2Map

## Fit to our project

reuse_idea - S - key-point selection from a thresholded mask (4 x 4 max survival + FPS) is a published analogue of our heat-map-peak seeds; INDT shows identity assignment for noised queries.

## Needs

a BEV mask (our dense head's heat map could play that role).

## Conflicts

stage-1 mask queries are trained jointly with the BEV encoder; the paper does not test external BEV masks.

## Tried in our experiments

seeds = heat-map peaks (192 peaks limited E94).
