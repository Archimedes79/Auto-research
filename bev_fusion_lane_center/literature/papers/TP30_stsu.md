# TP30 - STSU (first query-based centerline graph with a pairwise association classifier; nuScenes)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S6 L.*

## Papers

Structured Bird's-Eye-View Traffic Scene Understanding from Onboard Images (Can et al., ICCV 2021, arXiv 2110.01997) https://arxiv.org/abs/2110.01997

## Idea

A DETR-like transformer predicts centerlines as Bezier curves from ONE front camera and a pairwise MLP over per-curve association features gives the directed connectivity; this is the origin of the "adjacency between queries" formulation that all later piece-graph methods inherit (CGNet copies its BCE loss).

## Formulation

100 centerline queries (plus 100 object queries), each a quadratic Bezier curve (3 control points); each detected centerline gets a delta-dimensional association feature (delta < C); for every ordered pair (i, j) the two features are concatenated and passed through an MLP with sigmoid, giving P(i connects to j) (direction through the order of the pair). GT: x links to y when the start of y coincides with the end of x. The association step is trained only on Hungarian-matched detections; no class-imbalance measure, loss type not named. No cycle or acyclicity post-processing; junction points are averaged only for visualisation.

## Input

image 448 x 800, DeepLabv3+ backbone (Cityscapes pre-trained), BEV region x in [-25, 25] m, forward 1 - 50 m at 25 cm.

## Output

Bezier centerlines + directed association matrix (and BEV objects).

## Training

Hungarian matching with cross-entropy for detection and L1 for geometry; about 11 FPS for the chosen model.

## Differentiable?

partial - Hungarian matching.

## Sensors and data

camera (front only); nuScenes only (not evaluated on Argoverse in this paper).

## Reported results

nuScenes lane graph (M-Pre / M-Rec / Detect / C-Pre / C-Rec / C-IOU): STSU 60.7 / 54.7 / 60.6 / 60.5 / 52.2 / 38.9; polyline variant with estimated start points 54.7 / 51.2 / 40.5 / 58.4 / 16.3 / 14.6. Later reproductions on OpenLane-V2 give TOP_ll 2.9 (A, old metric as quoted by others) and on nuScenes 30 x 60 m TOPO F1 0.395 (LaneGAP table).

## Compute

about 11 FPS.

## Code

https://github.com/ybarancan/STSU

## Fit to our project

baseline - effort S - historical reference for the pair classifier; the connectivity IoU of about 39 shows the difficulty of learned association from single-frame features.

## Needs

Bezier curves with endpoints; Hungarian matching.

## Conflicts

front camera only; Hungarian.

## Tried in our experiments

E42 (learned link head).
