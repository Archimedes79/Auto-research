# TP26 - RoadPainter and TopoMask (points from masks; soft-argmax extraction)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2 S5 X.*

## Papers

RoadPainter: Points Are Ideal Navigators for Topology transformER (Ma et al., ECCV 2024) https://arxiv.org/abs/2407.15349 ; TopoMask: Instance-Mask-Based Formulation for the Road Topology Problem via Transformer-Based Architecture (arXiv 2306.05419, 2023 challenge entry) https://arxiv.org/abs/2306.05419

## Idea

A coarse polyline per query is turned into a BEV mask by a dot product with BEV features, and a second point set is read out of the mask with column-wise soft-argmax; both point sets are averaged. TopoMask drops Bezier / polyline regression altogether and extracts points from instance masks.

## Formulation

RoadPainter: 150 real + 150 virtual queries (virtual ones supervised only through masks, masked self-attention so real queries ignore them), 4 decoder layers, K = 11 points; mask M_i = B . Q'_i; column sampling C_(i,j) = [0..H-1]^T softmax(M_i(:, j)) (one expected row per column, differentiable), row-wise set for vertical lines; existence P = sigmoid(phi1(M)), direction D = sigmoid(phi2(Q')); keep the set with more valid points (> 0.5), remove outliers (nearest neighbour > 1.5 m), resample to K, average with coarse points. Topology = pairwise MLP over E_i = psi1(Q_i) + psi2(l_i). Loss weights top 5, cls 1.5, det 0.025, mask 1, mask-point 7. TopoMask: Mask2Former on BEV (200 x 104), direction label up / down / left / right, 11 points via expectation + second-order polynomial fit; relations with Sinkhorn normalisation.

## Input

BEV (RoadPainter grid not stated; TopoMask 200 x 104).

## Output

polylines + adjacency.

## Training

Hungarian (separate for real / virtual sets); 24 epochs.

## Differentiable?

yes for the soft-argmax readout; partial overall.

## Sensors and data

camera (+ optional SD map); OpenLane-V2.

## Reported results

RoadPainter subset A (DET_l / TOP_ll / OLS): baseline 26.9 / 7.7 / 37.2; +point-to-mask 28.1 / 7.9 / 37.6; +mask-to-point 30.7 / 7.9 / 38.9; +SD 36.9 / 12.7 / 42.6 (old metric); 6.5 FPS. TopoMask: masks vs Bezier OLS 36.0 vs 31.7 (F-score 45.2 vs 36.1); challenge 4th in OLS (39.2), 2nd in centerline F1 (45.9).

## Compute

RoadPainter 6.5 FPS (RTX 3090, FP32).

## Code

none found

## Fit to our project

reuse_idea - effort S - column / row soft-argmax is a differentiable way to read polylines from a dense map, relevant for a gradient path from vertex positions back to the heat map.

## Needs

per-instance masks.

## Conflicts

cannot read vertical lines with one pass; needs instance separation.

## Tried in our experiments

E120 / E121 (position refinement; planner loss through positions): small.
