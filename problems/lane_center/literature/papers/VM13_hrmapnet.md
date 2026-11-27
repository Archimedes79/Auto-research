# VM13 - HRMapNet (global historical raster of past vector outputs)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S8, B.*

## Papers

Enhancing Vectorized Map Perception with Historical Rasterized Maps (Zhang et al., ECCV 2024) https://arxiv.org/abs/2409.00620

## Idea

A global BEV raster (3 class channels, 0.3 m per pixel, uint8) in world coordinates is updated additively from past vector outputs with the GLOBAL ego pose; the local crop is (i) concatenated into the BEV feature and used to place extra BEV queries (feature aggregation) and (ii) turned into prior embeddings that initialise the map queries.

## Formulation

pixel update +S+ = 30 if the local rasterised prediction is 1, -S- = 1 if it is 0 (occupancy-grid-style additive score, NOT log-odds); read-out 1 where score > S_th (value not stated). Local map downsampled to 0.6 m for query initialisation. F_BEV = Conv(Concat(F_img + F_M, M_local)). Training: the global map is reset to empty at the start of each epoch and filled from the model's own predictions; test maps start empty; no GT-derived maps in the main setting. Pose-noise test at inference: translation sigma 0 - 0.2 m, rotation 0 - 0.02 rad: 67.2 -> 63.8 mAP at worst.

## Input

BEV feature + retrieved local raster (GT pose by default).

## Output

polylines of the host (MapTRv2 or StreamMapNet).

## Training

host losses (hierarchical matching).

## Differentiable?

partial - rasterisation of vector outputs and the integer update are non-differentiable; the memory is an input only.

## Sensors and data

camera; nuScenes, AV2; 60 x 30 m, 0.3 m, R50.

## Reported results

nuScenes original split 24 ep: StreamMapNet 60.4 -> 66.3, MapTRv2 61.5 -> 67.2 (110 ep 68.7 -> 73.6); AV2 StreamMapNet 61.5 -> 64.3; NEW split: nuScenes StreamMapNet 33.9 -> 37.1, AV2 58.1 -> 61.5. Ablation (MapTRv2): feature aggregation +3.1, query initialisation +2.6. A pre-built training-area map gives 83.7 mAP but is leakage on the old split.

## Compute

map storage about 1 MB per km (120 MB Boston); training cost not stated.

## Code

https://github.com/HXMap/HRMapNet

## Fit to our project

reuse_idea - M - a raster memory of past graph OUTPUT in a rolling ego frame is a cheap second temporal channel; its additive score (uint8, so bounded; no log-odds) is a near relative of log-odds fusion (ours could use true log-odds with decay).

## Needs

rasterisation of our graph (0.5 m, optional direction channels), an ego-warp of the raster each frame, clamp / decay. Cached-BEV: the raster memory and query initialisation yes; the aggregator uses image projection (no).

## Conflicts

global-pose retrieval implies localisation in a world map (conflicts with the registration-free premise unless the memory is a rolling local window); rasterisation breaks gradients.

## Tried in our experiments

E39 / E41 (feature level), System B input memory (input level); raster memory of the OUTPUT graph: none.
