# VS37 - CurveFormer: curve queries as ordered anchor points, deformable attention at projected points

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2, S5.*

## Papers

CurveFormer (Bai et al., arXiv 2022) https://arxiv.org/abs/2209.07989

## Idea

A query is a 3-D curve (confidence + two polynomials + start / end) whose positional part is an ordered set of anchor points at fixed Y positions; anchors are projected into multi-scale image features, bilinearly read and aggregated by deformable attention; anchors are refined layer by layer.

## Formulation

10 fixed Y positions between 5 and 100 m (inferred from the implementation statement); a binary indicator marks projected points that fall outside a feature level (Eq. 3) - an explicit "outside the map" flag; offsets from an MLP on the concatenated anchor feature and query; K and bounds not stated; per layer (dx, dz) offsets by a shared linear layer; 4 decoder layers best (F 94.1 vs 93.1 / 93.2 / 94.0 / 93.6 for 2 / 6 / 8 / 10 on the rarely-observed split); Hungarian, weights 2 / 5 / 2 / 2; EfficientNet, 360 x 480, range [-30, 30] x [3, 103] x [-10, 10] m; Adam 2e-4, 100 epochs.

## Input

image_feat + camera + queries. output: 3-D polylines.

## Training

hungarian_set.

## Differentiable?

partial.

## Sensors and data

monocular camera (OpenLane).

## Reported results

OpenLane F-score 50.5 (up-down 45.2, curve 56.6, night 49.1, merge and split 45.4); x error near / far 0.340 / 0.772 m, z 0.207 / 0.651 m; no ablation on the number of curve points, no FPS.

## Compute

not stated.

## Code

not checked.

## Fit to our project

reuse_idea - S - (a) curve-ordered anchor points as the query positional part; (b) the explicit per-level in-range indicator for projected points.

## Needs

camera projection, Hungarian.

## Conflicts

perspective features.

## Tried in our experiments

none.
