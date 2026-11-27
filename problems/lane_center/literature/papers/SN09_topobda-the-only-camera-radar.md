# SN09 - TopoBDA: the only camera-radar lane-topology result found (radar rows)

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (radar and supplement passages; the first 200k of 223k characters)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B, S4.*

## Papers

TopoBDA: Towards Bezier Deformable Attention for Road Topology Understanding (Kalfaoglu et al., Neurocomputing 2026 per the earlier review) https://arxiv.org/abs/2412.18951

## Idea

A Bezier-control-point deformable-attention decoder for centerlines; the paper also fuses SD map, lidar and radar to show headroom. Radar (subset B = nuScenes) is "fused in voxel space"; the supplement says only that radar points are assigned to their nearest voxel and concatenated with the camera and lidar voxel features before height flattening and a 2D conv.

## Formulation

radar p = (x, y, z, F_radar), nearest-voxel assignment to H_bev x W_bev x Z x C_radar (200 x 104 x 20, 0.5 m, z in [-10, 10] at 1 m -> about 100 x 52 m); no radar encoder, sweep count, RCS / velocity use stated. Decoder: 4 Bezier control points, iterative refinement, one-to-many auxiliary matching R = 4 (training only).

## Input

6 images + radar (+ lidar, SD map in other rows).

## Output

Bezier centerlines + topology scores.

## Training

regime hungarian_set (one-to-one) + one-to-many auxiliary, end to end.

## Differentiable?

partial - Hungarian matching at training.

## Sensors and data

OpenLane-V2 subset B (nuScenes) has radar; subset A (Argoverse 2) has none; evaluation +-50 m forward, +-25 m lateral.

## Reported results

OpenLane-V2 subset B, ResNet-50 (Table 3): camera DET_l 45.1 / TOP_ll 35.6 / OLS 49.9; camera + radar 49.4 / 38.6 / 54.8; camera + lidar 57.5 / 46.0 / 61.6. Text: radar + lidar gives no gain over camera + lidar (row not seen).

## Compute

not stated.

## Code

not checked.

## Fit to our project

reuse_idea - effort S - evidence that radar adds about +4.3 DET_l / +3.0 TOP_ll to a camera BEV lane model without a dedicated radar encoder (a coarse point-to-voxel scatter).

## Needs

voxel-space fusion inside the encoder.

## Conflicts

decoder uses Hungarian; radar is fused inside the trained encoder.

## Tried in our experiments

E31 / E32 (matched sparse decoder dropped).
