# VM19 - Neural Map Prior (global neural map memory with GRU update)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M (read through ar5iv)** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S8, B.*

## Papers

Neural Map Prior for Autonomous Driving (Xiong et al., CVPR 2023) https://arxiv.org/abs/2304.08481

## Idea

A global feature grid (sparse tiles, 0.3 m, indexed by world coordinates) is read by pose, fused with the current BEV feature by cross-attention (patches 10 x 10 cells = 3 m, dim 256) and updated by a 2D-conv GRU; the updated local prior replaces the stored content.

## Formulation

z = sigma(Conv[p, o]), r = sigma(Conv[p, o]), p~ = tanh(Conv[r*p, o]), p_new = (1 - z) p + z p~; the paper's moving-average update is only an ablation baseline. Modules before the BEV features are frozen; test samples processed chronologically; inter-trip fusion +3.81 All mIoU vs intra-trip +2.30; 70 % less memory than dense tiles; resolution 0.3 / 0.6 / 1.2 m gives +3.86 / +3.25 / +1.78.

## Input

BEV feature, global pose.

## Output

refined BEV for a map segmentation / detection head (HDMapNet, LSS, BEVFormer, VectorMapNet hosts).

## Training

host losses; epoch counts not given.

## Differentiable?

yes end to end through the GRU; tiles written back by replacement.

## Sensors and data

camera (+ lidar variants); nuScenes, 60 x 30 m, also 100 x 100 and 160 x 100 m; assumes precise GPS / IMU pose.

## Reported results

HDMapNet 32.73 -> 37.05 All IoU; LSS 39.78 -> 44.80; BEVFormer mIoU 49.51 -> 55.01; VectorMapNet mAP 40.9 -> 44.8; range 160 x 100 m +4.65; night +7.73, rain +5.92 (All IoU).

## Compute

not stated.

## Code

none on the page.

## Fit to our project

reuse_idea - M - the gated conv-GRU as a learned fusion cell on a rolling local raster; not suitable as published (global pose, multi-traversal revisits of the same place).

## Needs

global pose; revisits of places. Cached-BEV: yes.

## Conflicts

conflicts with the registration-free premise; it is a place memory, so it is leakage-prone on random splits (VM46).

## Tried in our experiments

E40 / E48 map prior at the grid (+0.075 F1 exact pose, -0.057 unaligned) is the HD-map analogue.
