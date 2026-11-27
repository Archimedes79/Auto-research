# VM24 - DTCLMapper (dual temporal consistent learning: contrastive instances + occupancy consistency)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: L, S8.*

## Papers

DTCLMapper: Dual Temporal Consistent Learning for Vectorized HD Map Construction (arXiv 2405.05518; authors not read) https://arxiv.org/abs/2405.05518

## Idea

Two training-time consistency components on a MapTR-style host: instance consistency (a vector point pre-selection module selects reference points from an instance map; contrastive learning pulls the current instance embedding to historical embeddings of the same class that are close in position) and map consistency (rasterise the predicted vector instances to a grid map and compare ego-warped historical grid maps with the current one by an L1 "map occupancy" loss; self-supervised, no GT).

## Formulation

Loss_cst = log[1 + sum_c sum_{k+} sum_{k-} exp(v_c . k- - v_c . k+)]; Loss_ol = sum_{i=1..m} L1(V2GMap'_{t-i}, V2GMap_t) with ego rotation / translation warp (Eqs. 15 - 16); only instances with normal predicted class confidence are rasterised (threshold 0.4); total loss weights 2 / 5 / 0.005 (base) + 0.1 / 1 / 1 / 0.1 for the consistency terms (assignment to terms not read). Grid 0.15 m, map 400 x 200, range 60 x 30 m; batch 16, lr 1.5e-3, 24 ep nuScenes, 6 ep AV2, RTX A6000. Core temporal parts are training-time only; the Stream and Tracker variants still use memory at inference.

## Input

images + history grid maps (training).

## Output

as MapTR.

## Training

hungarian_set + consistency losses.

## Differentiable?

yes for the occupancy loss through the rasteriser (soft rasteriser details not read).

## Sensors and data

camera; nuScenes, AV2.

## Reported results

nuScenes mAP DTCLMapper-l (LSS) 61.9, -b 53.6 (MapTR 50.3, PivotNet 57.6, MapTRv2* 59.9); +StreamMapNet 60.3 -> 61.4, +MapTracker 65.4 -> 66.9; AV2 65.1. No temporal-consistency metric reported.

## Compute

not stated.

## Code

https://github.com/lynn-yu/DTCLMapper

## Fit to our project

reuse_idea - S - a GT-free, matching-free temporal consistency loss between rasterised outputs of consecutive frames after ego warp.

## Needs

a differentiable rasteriser of our graph (VM32); ego warp. Cached-BEV: yes.

## Conflicts

none for the raster loss; the contrastive part needs instance embeddings with positions.

## Tried in our experiments

none.
