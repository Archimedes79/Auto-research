# VS30 - BezierLaneNet: one cubic Bezier per lane, sampled-point L1, column proposals

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2, S4, L.*

## Papers

Rethinking efficient lane detection via curve modeling (Feng et al., CVPR 2022) https://arxiv.org/abs/2203.02431

## Idea

A lane is a cubic Bezier curve (4 control points, 8 numbers); the loss compares 100 points sampled at uniform t on the predicted and GT curve, so the curve is never discretised inside the network.

## Formulation

Bernstein basis precomputed at fixed t, so sampling is one matrix multiplication; L1 over n = 100 samples; Hungarian matching with quality Q = score^(1-alpha) (1 - L1)^alpha, alpha = 0.8; weighted BCE (negative weight 0.4); total lambda1 = 1 (regression), lambda2 = 0.1 (classification), lambda3 = 0.75 (auxiliary segmentation); GT control points by least squares; feature flip fusion (feature map fused with its horizontal flip through a deformable conv); column proposals: average pooling along the vertical axis gives W/16 proposals (50 on CULane), two 1x3 1-D convs, then 1x1 classification and regression; layer-3 backbone features with dilated blocks (4, 8), 16x downsampling.

## Input

image_feat. output: bezier + existence per column proposal.

## Training

hungarian_set.

## Differentiable?

partial (matching).

## Sensors and data

images (TuSimple, CULane, LLAMAS).

## Reported results

LLAMAS test F1 95.17 (R34), 94.91 (R18); CULane F1 75.41 (R34 ablation row); 213 FPS (R18, 4.10 M parameters), 150 FPS (R34, 9.49 M).

## Compute

above.

## Code

URL from memory, not opened: https://github.com/voldemortX/pytorch-auto-drive

## Fit to our project

reuse_idea - S - Bernstein-matrix sampling is the cheapest differentiable curve resampler (adapter bezier <-> polyline); sampled-point L1 is a matching-free curve loss once correspondence is fixed by the parameter t.

## Needs

one curve per lane, Hungarian.

## Conflicts

a single cubic cannot cover a 24 m branch with a bend and a fork; column pooling presumes lanes run vertically.

## Tried in our experiments

E53 Bezier connectors.
