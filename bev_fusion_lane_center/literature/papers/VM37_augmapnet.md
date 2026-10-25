# VM37 - AugMapNet (predicted raster fed back into the latent BEV grid)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: B, S1.*

## Papers

AugMapNet: Improving Spatial Latent Structure via BEV Grid Augmentation for Enhanced Vectorized Online HD Map Construction (Monninger et al., WACV 2026) https://arxiv.org/abs/2503.13430

## Idea

A parallel raster decoder predicts a 3-class segmentation (100 x 50 x 3) from the latent BEV grid; a 4-layer CNN encodes it ({32, 64, 128, 256} channels) and ADDS it back to the BEV grid before the vector decoder. The raster prediction is effectively detached ("immutable prior").

## Formulation

B_aug = B + e_raster(M^_raster); Dice loss for the raster branch (weight not stated); host StreamMapNet (also SQD-MapNet); lr 2.5e-4, batch 1, 8 V100, 24 epochs; 52.70 M parameters (+6.3 %). Argument: vector decoders supervise a sparse subset of BEV cells, raster decoders all cells.

## Input

BEV grid.

## Output

polylines (host) + raster.

## Training

hungarian_set (host) + Dice.

## Differentiable?

partial (raster path detached by design).

## Sensors and data

camera; nuScenes (location-disjoint split), AV2; 60 and 100 m ranges.

## Reported results

nuScenes location-disjoint mAP: StreamMapNet 33.8 -> 38.3 (+13.3 %) at 60 m, 23.2 -> 28.9 at 100 m; SQD-MapNet 36.9 -> AugSQD 40.5; oracle GT raster 91.4; AV2 60 m 57.1 -> 58.8.

## Compute

+6.3 % parameters.

## Code

https://github.com/tmonnin/augmapnet

## Fit to our project

reuse_idea - S - this is what our design already does (dense head heat map + direction channels concatenated to the 39 feature channels = 42-channel trace-network input); outside evidence that it helps vector decoding and that an oracle raster is a very high upper bound (91.4).

## Needs

a raster head on the BEV.

## Conflicts

none.

## Tried in our experiments

dense head + tracer (E31 / E32, grid kept).
