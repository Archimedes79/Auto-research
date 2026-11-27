# SN37 - TopoNet: which parts read the BEV and which need the perspective image

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (interface passages)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B, S6.*

## Papers

TopoNet, graph-based topology reasoning for driving scenes (Li et al., arXiv 2023; title from memory) https://arxiv.org/abs/2304.05277 ; code https://github.com/OpenDriveLab/TopoNet (URL from memory, not opened)

## Idea

A BEVFormer-style view transformer (3 layers, no temporal input) produces a BEV feature on a 200 x 100 grid with four height levels (cell size not stated; 0.5 m for +-50 x +-25 m is my inference); a lane-centerline decoder reads the BEV feature, a traffic-element decoder reads the front-view image feature (the perspective image is required there), and a scene-graph network over the queries predicts lane-lane and lane-element topology.

## Formulation

two Deformable-DETR decoders (8 heads, 4 offset points, FFN 512), 200 lane queries, 100 element queries, dim 256; lane head 11 x 3 points; topology: two MLPs (128) per query type + sigmoid MLP on the pair; adjacency computed without gradients; one GNN layer is best; Hungarian matching with the loss as cost; loss weights lane cls 1.5, reg 0.025, topology 5.0; 24 epochs, batch 8, 8 x A100, AdamW 1e-4.

## Input

multi-view images.

## Output

11-point 3D centerlines, boxes, adjacency.

## Training

regime hungarian_set, end to end.

## Differentiable?

partial - Hungarian.

## Sensors and data

camera, OpenLane-V2.

## Reported results

subset A DET_l 28.5, TOP_ll 4.1, OLS 35.6; subset B 24.3, 2.5, 33.2 (older metric version; not comparable to later papers); 10.1 FPS (A100).

## Compute

10.1 FPS.

## Code

see papers line.

## Fit to our project

reuse_idea - effort M - the lane branch needs only the BEV raster (compatible with a 39 -> 256 adapter), the traffic-element branch needs images (not available); Hungarian conflicts.

## Needs

BEV features 256-ch, 200 x 100.

## Conflicts

Hungarian; the element branch needs the front image.

## Tried in our experiments

E42 / E45 / E46 (TopoLogic-style and learned scoring within 0.01 of the rules).
