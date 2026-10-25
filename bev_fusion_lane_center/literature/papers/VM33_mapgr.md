# VM33 - MapGR (global raster loss over all queries)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: L.*

## Papers

Learning Global Representation from Queries for Vectorized HD Map Construction (arXiv 2510.06969; authors not read) https://arxiv.org/abs/2510.06969

## Idea

Adds a whole-map objective: every query feature is mapped (MLP -> small 2D map), all are concatenated, passed through a light conv net and upsampled to a BEV raster supervised by the GT raster (BCE), so unmatched queries also get gradients; a guidance branch encodes the predicted global map and fuses it back into each query (first two of six decoder layers only, because the global constraint conflicts with final convergence).

## Formulation

GRL loss = BCE(pred raster, GT raster), weight ratio global : other = 1.0 : 0.1 in one place and 0.2 in a table (inconsistent); gradient weakening 0.8; global embedding dim 2048 best (512 -> 64.6, 2048 -> 65.0, 8192 -> 64.2 mAP2); 50 (MapTR) / 100 (MapQR) queries; range x [-15, 15], y [-30, 30] m; 24 / 110 epochs; lr 6e-4; R50.

## Input

queries of a MapTR-family decoder.

## Output

raster + polylines.

## Training

baseline matching kept for instance losses; the global loss needs no matching.

## Differentiable?

yes.

## Sensors and data

camera; nuScenes, AV2; standard splits (no new split).

## Reported results

nuScenes mAP1 / mAP2: MapTR 24 ep 27.3 / 50.3 -> 31.2 / 54.5; MapTRv2 24 ep 37.2 / 61.5 -> 41.1 / 65.0; MapQR (reproduced) 42.1 / 65.3 -> 45.3 / 68.1; ablation MapTRv2 mAP2: GRL only 63.7, GRL + GRG 65.0.

## Compute

+11 M parameters, FPS unchanged (MapTRv2 19.6 -> 19.4).

## Code

none given (supplement).

## Fit to our project

reuse_idea - S - a MATCHING-FREE loss on the aggregate of ALL output elements (rasterise the whole output graph with validity weights and compare with the GT raster) that supervises duplicates and false positives without any assignment.

## Needs

a rasteriser (VM32) or a learned small decoder as in the paper.

## Conflicts

none.

## Tried in our experiments

none (we supervise the dense head, not the raster of the output graph).
