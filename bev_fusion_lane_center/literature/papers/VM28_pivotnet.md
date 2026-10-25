# VM28 - PivotNet (dynamic number of pivot points, sequence matching by dynamic programming)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M+C (repository front page)** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: L, S1.*

## Papers

PivotNet: Vectorized Pivot Learning for End-to-end HD Map Construction (Ding et al., ICCV 2023) https://arxiv.org/abs/2308.16477

## Idea

Each element is represented by pivot points (the vertices of a Visvalingam-Whyatt simplification of the polyline; collinear points are removable) and a per-slot pivot classifier makes the vertex COUNT dynamic. Matching of predicted slots to GT pivots uses a dynamic-programming sequence alignment (O(N T)) with first and last point fixed, instead of Hungarian over permutations; a "point-to-line mask" module lets the point queries of one instance form an instance line feature that masks cross-attention over the BEV feature.

## Formulation

L_DVS = 5 L_pivot-point(L1) + 2 L_collinear-point(linearity) + 2 L_pivot-class(BCE); line-aware mask loss (BCE + Dice) weight 5; BEV segmentation (BCE + Dice) weight 3; max points per instance N = 10 (divider), 2 (crossing), 30 (boundary), max instances 20 / 25 / 15, 200 / 50 / 450 point queries; VW threshold not stated; BEV 64 x 32 features for 30 m front / back, 15 m left / right; backbones EfficientNet-B0, ResNet-50, Swin-T; AdamW 2e-4, wd 1e-4, 24 / 110 epochs, 8 x 2080Ti, batch 1 per GPU. Direction / start point: not discussed, fixed by the endpoints in matching.

## Input

surround images -> BEV features (4 enc / 4 dec).

## Output

variable-length pivot sequences per instance, set-based (no autoregression).

## Training

DP matching inside instances + instance assignment (not detailed).

## Differentiable?

partial (matching, pivot selection).

## Sensors and data

camera; nuScenes (AV2 45.0 at 10 epochs).

## Reported results

nuScenes mAP at thresholds {0.2, 0.5, 1.0}: SwinT 24 ep 40.8, 110 ep 49.2 (44.8 M parameters, 5.1 FPS) vs MapTR R50 27.3 (re-evaluated), VectorMapNet 21.3, HDMapNet 21.3; at {0.5, 1.0, 1.5}: 66.8 at 110 ep.

## Compute

44.8 M parameters, 5.1 FPS.

## Code

https://github.com/wenjie710/PivotNet (front page: official, MMCV 1.7.1 + Detectron2 + custom CUDA ops, MIT licence stated)

## Fit to our project

reuse_idea - S/M - (a) a learned dynamic vertex count per element via a "keep this slot" classifier is the nearest published relative of our validity gate, (b) a simplified GT (VW here, Douglas-Peucker in our oracle: 14.3 vertices per 100 m against 22 learned in E94, PROJECT_CONTEXT) defines the vertex budget the keep gate should learn, (c) a linearity loss for non-pivot points keeps dropped vertices on the chord.

## Needs

ordered slots per instance; GT simplification. Cached-BEV: the matching/loss part yes.

## Conflicts

the sequence DP assumes one ordered chain per instance (no branching).

## Tried in our experiments

none directly (vertex budget measured against a Douglas-Peucker oracle in loop 14+).
