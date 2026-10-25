# VM39 - MGMap (mask-guided instance queries and point-patch refinement)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S2, S5.*

## Papers

MGMap: Mask-Guided Learning for Online Vectorized HD Map Construction (Liu et al., CVPR 2024) https://arxiv.org/abs/2404.00876

## Idea

An enhanced multi-level neck (3 residual blocks with channel / spatial attention, upsample and fuse) feeds (i) a mask-activated instance decoder - 50 instance masks M_ins = sigmoid(Conv(F)), lane query = mask x features, then a deformable decoder with iterative point offsets - and (ii) position-guided mask patch refinement: for each point a patch of normalised size 0.1 around it is ROI-aligned to a 5 x 5 feature on a densified binary mask feature + position grid, refreshed by attention over 2 stages and regressed.

## Formulation

L_ins weight 2, L_b weight 15 (mask CE + Dice); other weights not stated; BEV 200 x 100 over 60 x 30 m (also 60 x 60, 30 x 90); 50 instances x 20 points; AdamW 6e-4, batch 6 per GPU, 8 V100, 30 / 110 epochs; no separate dense head (masks from the same feature, training auxiliary).

## Input

BEV multi-level features.

## Output

polylines.

## Training

bipartite matching (cost not stated).

## Differentiable?

partial.

## Sensors and data

camera, lidar; nuScenes, AV2 subset.

## Reported results

nuScenes camera R50 30 ep Chamfer avg 61.4 (ped 57.4, div 63.5, bou 63.3), raster avg 37.2, 11.6 FPS; 110 ep 66.5; lidar (SECOND) 67.9; fusion 71.7. No temporal part.

## Compute

11.6 FPS.

## Code

https://github.com/xiaolul2/MGMap

## Fit to our project

reuse_idea - S - local patch pooling around predicted points on a mask feature is the point-level refinement idea of our E120 (bilinear samples at predicted positions).

## Needs

a mask feature. Cached-BEV: yes.

## Conflicts

matching.

## Tried in our experiments

E120 refinement (+0.012 F1).
