# VM32 - MapVR (differentiable rasterisation as supervision)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: L, E.*

## Papers

Online Map Vectorization for Autonomous Driving: A Rasterization Perspective (Zhang et al., NeurIPS 2023) https://arxiv.org/abs/2306.10502

## Idea

Vector outputs are rendered by a differentiable rasteriser and supervised with a Dice loss against the rasterised GT; the branch is removed at inference. Also proposes a rasterisation-based AP (AP_raster) that is more sensitive to small deviations.

## Formulation

polylines I = exp(-D / tau) with D = distance of a pixel to the nearest segment, tau = 2.0 best (0.5 - 6.0 tested); polygons sigmoid(C D / tau) with inside / outside sign; L = l1 L_render(Dice) + l2 L_cls(focal) + l3 L_dir + l4 L_reg(L1), weights not stated; Hungarian matching with cost l1 L_render + l2 L_cls + l3 L_reg, so supervision is per matched instance; training raster 256 x 128 (inferred default), metric raster 480 x 240 at 0.125 m, dilation 2 px, IoU thresholds 0.25:0.05:0.50 (lines), 0.50:0.05:0.75 (polygons). Direction / start point: not addressed; the distance-based rendering is order-independent (reader's inference).

## Input

MapTR host (R50, GKT).

## Output

polylines (raster branch training-only).

## Training

hungarian_set + raster loss.

## Differentiable?

yes through the rasteriser, Hungarian for pairing.

## Sensors and data

camera, camera + lidar; nuScenes, AV2; +-15 x +-30 m.

## Reported results

MapTR -> MapTR + MapVR on nuScenes camera 24 ep: Chamfer AP 50.3 -> 51.2, AP_raster 24.3 -> 31.2; 110 ep AP_raster 35.0 -> 38.5; no inference change (18.4 FPS).

## Compute

no inference cost.

## Code

https://github.com/ZhangGongjie/MapVR ; metric toolkit https://github.com/jiahaoLjh/MapVectorizationEvalToolkit

## Fit to our project

reuse_idea - S/M - the soft line rasteriser exp(-D / tau) is a differentiable bridge from our vertex output to the dense BEV (also needed by VM13 / VM24 / VM33-type losses and memories); related to but not the same as our point-to-polyline loss.

## Needs

rasteriser (a pure-PyTorch version is feasible at 200 x 200). Cached-BEV: yes.

## Conflicts

published use is per matched instance.

## Tried in our experiments

E120 point-to-polyline loss both ways (the geometric analogue).
