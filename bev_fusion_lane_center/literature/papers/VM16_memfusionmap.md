# VM16 - MemFusionMap (working memory + temporal overlap heatmap)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S8, B.*

## Papers

MemFusionMap: Working Memory Fusion for Online Vectorized HD Map Construction (Song et al., WACV 2025) https://arxiv.org/abs/2409.18737

## Idea

A fixed-lag working memory of the last 4 unified BEV features (warped by ego transform, concatenated, 3 conv layers) plus a "temporal overlap heatmap": a one-channel BEV map counting how long each cell has been inside the field of view, propagated recursively and fused as a 32-channel feature.

## Formulation

H_t0 = 1; H_t = Warp(H_{t-1}) + 1 with zero padding, so a cell that leaves view resets to 1 (a count of continuous visibility; unbounded, no normalisation; text and formula slightly disagree); Conv_H 1 -> 16 -> 16 -> 32 channels + sigmoid; fusion ConvMem: 3 x 3 conv x 3, dilation (2,2), 256 ch, LayerNorm; first BEV repeated 4 times at sequence start. Built on StreamMapNet: loss 5 focal + 50 line + 0.1 transformation; stage 1 single-frame (44 ep nuScenes), stage 2 temporal with detached memory. BEV 50 x 100 x 256, ranges 60 x 30 and 100 x 50 m, AdamW 5e-4, batch 32, 8 V100.

## Input

as StreamMapNet.

## Output

polylines.

## Training

as StreamMapNet; hungarian_set.

## Differentiable?

partial (Hungarian, detached memory); the heatmap recursion itself is differentiable.

## Sensors and data

camera; nuScenes and AV2 new splits.

## Reported results

new split mAP vs StreamMapNet: nuScenes 60 x 30 38.0 vs 34.1, 100 x 50 27.6 vs 22.2; AV2 60.6 vs 58.3, 54.3 vs 51.5; original split nuScenes 65.2 vs 63.4; 15.1 vs 15.3 FPS (A6000).

## Compute

FPS as above.

## Code

none; project page https://song-jingyu.github.io/MemFusionMap

## Fit to our project

reuse_idea - S - an explicit "how often observed" channel next to the fused evidence is the natural companion of a log-odds validity memory (tells the head how much to trust it).

## Needs

ego warp of a raster; conv fusion. Cached-BEV: yes.

## Conflicts

none technical; gain measured only on a StreamMapNet host.

## Tried in our experiments

E41 (feature fusion without a count channel).
