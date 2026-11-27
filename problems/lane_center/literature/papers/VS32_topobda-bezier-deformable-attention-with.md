# VS32 - TopoBDA: Bezier deformable attention (control points as reference points) with camera + radar

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M (about 90 % of the page read)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2, S5, B.*

## Papers

TopoBDA (Kalfaoglu et al., arXiv 2024; Neurocomputing 2026 per the earlier review) https://arxiv.org/abs/2412.18951

## Idea

Each Bezier control point is its own attention head and reference point, with K learned offsets per point; the curve's shape steers where the BEV is read, without resampling the curve.

## Formulation

BDA(q, V, C) = sum_{n=0}^{N} sum_{k=1}^{K} A_nk W_n V(c_n + dp_nk); 4 control points (the conclusion mentions 8; supplement text says 4); K = 32 attention offsets per control point; offset bounds not stated; control points of the first layer from an MLP + sigmoid, later layers add offsets in inverse-sigmoid space; decoder BDA then self-attention, 10 layers in the runtime study; auxiliary instance masks (training only); one-to-many auxiliary decoder sharing weights (repetitions R: 0 -> OLS_l 46.5, 4 -> 48.0, 5 -> 48.1); Hungarian with a mask + L1 mixed cost (Mask DINO style); weights lambda_reg 3 (matcher 5), mask BCE 5, Dice 5, class 2; BEV 200 x 104 cells of 0.5 m, 20 height bins of 1 m in [-10, 10], multi-height Lift-Splat; radar points (x, y, z + C_radar features) assigned to the nearest voxel, concatenated with camera features, flattened over height, 2-D conv; SD map concatenated in BEV; sweeps / radar representation not stated.

## Input

bev_raster + queries (bezier). output: bezier centrelines (+ topology MLP head).

## Training

hungarian_set + one-to-many.

## Differentiable?

partial.

## Sensors and data

OpenLane-V2 subsets A (Argoverse 2) and B (nuScenes): camera, + radar, + lidar, + SD map.

## Reported results

attention ablation SwinB OLS_l: SA 41.0, MA 42.6, SPDA 44.1, MPDA4 47.4, MPDA16 47.5, BDA 48.0 (BDA +3.9 over SPDA, +0.5 over MPDA16); subset B ResNet50 OLS_l: camera 49.9, + radar 54.8 (+4.9), + lidar 61.6; subset A: camera 44.1, + SD map 50.1; decoder runtime (10 layers, torch) BDA 18.47 ms, MA 24.76, SA 15.96; cost per BDA layer 10.0 GFLOPs, 1.48 GB, 14.78 M parameters vs SA 28.4 GFLOPs, 1.995 GB, 16.04 M and MA 41.6 GFLOPs, 5.07 GB, 16.24 M (Table S3); Table 5 columns could not be mapped on the extraction.

## Compute

above.

## Code

not checked.

## Fit to our project

reuse_idea - S - the only published camera + RADAR lane-topology result (radar +4.9 OLS_l on nuScenes subset B); BDA is the cheapest "read at hypothesis control points" pattern (K = 32 per control point, on a 0.5 m BEV).

## Needs

BEV raster + Bezier queries.

## Conflicts

Hungarian.

## Tried in our experiments

E53 Bezier connectors; E31 / E32.
