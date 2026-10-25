# SN19 - CenterLineDet: BEV heat maps from a separate stage, then a decision transformer

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (interface passages)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B, S1, S5.*

## Papers

CenterLineDet: CenterLine Graph Detection for Road Lanes with Vehicle-mounted Sensors by Transformer for HD Map Generation (Xu et al., ICRA 2023; from memory) https://arxiv.org/abs/2209.07734

## Idea

Stage 1 produces BEV centerline segmentation and an initial-vertex heat map (HDMapNet or its own FusionNet over 6 cameras + lidar sweeps), neighbouring-frame heat maps are warped and averaged; stage 2 crops an ROI around the current vertex and a transformer predicts the next vertices with validity.

## Formulation

FusionNet F_bev = max_i { IPM(F_i) + phi_i(F_i) }; ROI = fused heat map concatenated with the ego historical map, cropped around the vertex (size not stated); N_hat = 0 end, 1 continue, > 1 push candidates; behaviour cloning over a BFS of the GT graph with noise; "two stages and cannot be trained end to end".

## Input

BEV heat maps.

## Output

graph via traversal.

## Training

regime teacher_forcing (imitation).

## Differentiable?

no - separate stages, no gradient across.

## Sensors and data

camera + lidar sweeps; nuScenes (698 / 148 sequences).

## Reported results

Table I multi-frame P-F1 0.714, T-F1 0.518 (FusionNet), baselines 0.695 / 0.403 (FusionNet) and 0.685 / 0.400 (HDMapNet).

## Compute

0.41 s/frame.

## Code

not checked.

## Fit to our project

baseline - effort M - same interface (BEV raster -> graph by local steps) as our trace network E94 but with DETR queries; confirms that stage-wise training on a separate BEV stage is accepted practice.

## Needs

BEV heat map + ego history map.

## Conflicts

separate stage, no gradient across.

## Tried in our experiments

E94 - E99 (learned trace network, partial).
