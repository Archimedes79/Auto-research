# TP24 - LaneGraphNet (anchor nodes + pairwise edges from nuScenes BEV)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S6 L.*

## Papers

Lane Graph Estimation for Scene Understanding in Urban Driving (Zuern, Vertens, Burgard, arXiv 2105.00195) https://arxiv.org/abs/2105.00195

## Idea

The closest setting to ours (nuScenes map expansion as ground truth, BEV raster input): nodes are lane anchor points (at most 2 m apart), edges are node pairs "directly reachable"; Graph R-CNN proposes nodes and classifies pairs as connected, a separate segmentation net gives direction.

## Formulation

BEV 256 x 256 px at 20 cm (51.2 m), 4 modalities (LiDAR intensity, RGB, 14 semantic classes, detected vehicle boxes); one feature extractor per modality, concatenated after conv 4; region proposal network (nodes), relationship proposal network (edges); node label single class "anchor point"; edge label binary has-connection (undirected). Adaptive resampling: each predicted anchor is projected orthogonally onto its nearest GT segment (t_proj = l1 + (l2 - l1)(t^ - l1)) before labelling, which removes the penalty for position along the lane (matching-free label). Direction: PSPNet, 18 bins of 20 deg + background, per-pixel CE; edge direction = the sign with larger dot product to the mean predicted angle. Post-processing: drop anchors < 0.5, edges < 0.2, edges longer than 1/4 of the image, isolated anchors. Class balance: most edges are 0; no balancing described.

## Input

BEV raster (Boston-Seaport only, random 256 x 256 crops; test region about 211 x 211 m).

## Output

undirected graph, then directed.

## Training

details (optimiser, epochs) not stated.

## Differentiable?

partial - proposals and NMS inside Graph R-CNN.

## Sensors and data

LiDAR + RGB + semantics from nuScenes; one city.

## Reported results

APLS / F1 / IoU: skeleton regression 0.033 / 0.202 / 0.114, nearest-neighbour anchors 0.088 / 0.448 / 0.298, LaneGraphNet 0.176 / 0.574 / 0.420; direction accuracy 93.5 % (LaneDirNet) vs 50.8 % (Graph R-CNN itself).

## Compute

not stated.

## Code

not checked

## Fit to our project

baseline - effort M - the weak numbers (APLS 0.18) show how hard pairwise edge scoring from raster features was before 2022; the projection-based label is a matching-free label recipe.

## Needs

BEV raster; anchor sampling.

## Conflicts

complex intersections give ill-formed graphs (authors); one-city evaluation.

## Tried in our experiments

E42 .. E56 (link heads on raster-derived nodes): no gain over rules.
