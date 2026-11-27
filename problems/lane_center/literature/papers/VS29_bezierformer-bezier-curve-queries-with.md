# VS29 - BezierFormer: Bezier curve queries with several reference points along the curve, matching-free assignment

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M (BezierFormer), A (TopoCurve)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2, S5, L.*

## Papers

BezierFormer: a unified architecture for 2D and 3D lane detection (arXiv 2404.16304) https://arxiv.org/abs/2404.16304 ; related TopoCurve (poster abstract only) https://neurips.cc/virtual/2026/poster/149504

## Idea

A query is a cubic Bezier curve; deformable attention is applied at N_ref = 5 reference points spread along the curve, so the receptive field covers the whole lane; assignment is SimOTA with top-k = 1 (NMS-free, no Hungarian); the regression loss is a Chamfer IoU on sampled curve points.

## Formulation

reference points at t = 0, 0.25, 0.5, 0.75, 1; N_sample = 5 sampling points per reference point per feature scale; two MLPs predict offsets and weights, offsets added to the reference points (bound not stated); attention = sum over reference points, scales and samples with weights normalised to 1; queries 10 (CULane), 16 (CurveLanes), 32 (OpenLane); loss: Chamfer IoU on 200 sampled points + length loss + endpoint loss, focal classification, lane width e = 10 (CULane, CurveLanes) or 0.9 (OpenLane); optional perspective projection of 3-D reference points onto the front view.

## Input

image_feat levels + queries (bezier). output: bezier lanes.

## Training

matching_free (SimOTA, top-k = 1).

## Differentiable?

partial (assignment).

## Sensors and data

images (OpenLane, CULane, CurveLanes).

## Reported results

OpenLane F1 / category acc: ResNet18 53.84 / 92.02, Swin-T 56.43 / 94.1; CULane F1 79.44 (R18), 80.63 (Swin-T); CurveLanes F1 89.06 (R18), 91.31 (Swin-T) (the text cites 90.72 / 91.06 elsewhere). Ablation CurveLanes / OpenLane F1: ordinary attention 82.76 / 46.53; deformable 85.81 / 49.08; Bezier attention one reference point (t = 0.5) 87.60 / 52.13; five reference points 89.06 / 53.84; representations: polynomial 80.90 / 50.30, row-based 83.10 / 51.27, Bezier 89.06 / 53.84; losses: sampling loss 78.78 / 47.11, sampling IoU 81.86 / 48.32, Chamfer IoU 89.06 / 53.84. TopoCurve poster: 50.6 OLS and DET_p 56.8 on OpenLane-V2 (abstract only).

## Compute

not stated.

## Code

not checked.

## Fit to our project

reuse_idea / reimplement - M - (a) the strongest evidence found for curve-aligned multi-point reads (CurveLanes / OpenLane F1: one Bezier reference point over plain deformable attention +1.8 / +3.1; five reference points over one +1.5 / +1.7; five over deformable +3.3 / +4.8); (b) Chamfer IoU is a matching-free curve loss close to our point-to-polyline loss; (c) SimOTA top-1 shows an NMS-free, Hungarian-free assignment for curve queries.

## Needs

curve queries with control points (bezier <-> polyline adapter).

## Conflicts

cubic Bezier cannot represent a branching ray.

## Tried in our experiments

E53 Bezier connectors (kept in system A); E120 point-to-polyline loss both ways is similar to a Chamfer loss.
