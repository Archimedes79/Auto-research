# SN17 - LaneSegNet: lane-segment queries with lane attention on a 200 x 100 BEV at 0.5 m

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B, S4, S6.*

## Papers

LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving (Li et al., ICLR 2024; from memory) https://arxiv.org/abs/2312.16108 ; code https://github.com/OpenDriveLab/LaneSegNet (URL from memory, not opened)

## Idea

Lane segments (centerline + left / right boundary + type) are predicted by DETR-style queries; "lane attention" lets each head sample a local region around reference points placed along the predicted segment; the BEV comes from three BEVFormer-style encoder layers.

## Formulation

BEV grid 200 x 100 cells for +-50 x +-25 m (0.5 m per cell, derived); 6 decoder layers, 200 queries, lane attention 8 heads x 32 sampling points per reference point (4 reference points per segment); instance-mask head by dot product with BEV features (mask supervision +2.0 APls); topology from predecessor / successor MLPs; Hungarian assignment; loss weights vector 0.025, seg 3.0, cls 1.5, line type 0.01, topology 5.0; AdamW 2e-4, 24 epochs, batch 8.

## Input

BEV features from ResNet-50 + FPN + 3 BEVFormer layers.

## Output

centerline + offsets (10 x 3 points), masks, adjacency.

## Training

regime hungarian_set, end to end.

## Differentiable?

partial - Hungarian.

## Sensors and data

camera, OpenLane-V2.

## Reported results

lane segment mAP 32.6 vs MapTRv2 28.5 (Table 1); lane attention ablation: 28.7 -> 32.6 mAP, TOP_lsls 6.9 -> 8.1.

## Compute

decoder 6.74 M parameters, 20.67 ms.

## Code

see papers line.

## Fit to our project

reuse_idea - effort M - its BEV grid is a crop of our raster (same 0.5 m cells: take 200 x 100 of 200 x 200); lane attention = reference points along the segment, close to our rays; Hungarian conflicts.

## Needs

BEV features of 256 channels in a 200 x 100 grid.

## Conflicts

Hungarian one-to-one assignment.

## Tried in our experiments

E31 / E32 (matched sparse decoder dropped); E42 / E45 (learned link heads within 0.01 of rules).
