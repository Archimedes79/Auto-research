# GR26 - LaneGraphNet (adaptive resampling of the ground truth as a matching-free anchor loss)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: . Pipeline stage: L, S6.*

## Papers

Lane Graph Estimation for Scene Understanding in Urban Driving (Zuern, Vertens, Burgard, arXiv 2105.00195, 2021; venue RA-L 2021 from memory) https://arxiv.org/abs/2105.00195

## Idea

One-pass lane-graph estimation from lidar / camera BEV layers with a Graph R-CNN (nodes = lane anchor points, edges = connections, direction by a separate per-pixel direction net with 18 bins of 20 deg). The useful piece: *adaptive resampling* - each predicted anchor is projected orthogonally onto its nearest ground-truth lane segment, so position along the lane does not enter the loss.

## Differentiable?

yes (single pass). code: none found.

## Reported results

APLS 0.176, F1 0.574, IoU 0.420 on NuScenesGraph (Boston-Seaport, 256 x 256 crops, 0.2 m / px); nearest-neighbour connections baseline 0.088 / 0.448 / 0.298; direction accuracy 93.5 % for the direction net vs 50.8 % from the graph network. Failure: opposite lanes with the same direction, complex intersections ill-formed.

## Fit to our project

reuse_idea - effort S: a projection-based loss tolerant to the along-lane shift is the matching-free loss most growers lack (cf. point-to-polyline in E120).

## Tried in our experiments

E120 same principle. verified: checked.
