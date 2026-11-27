# TP09 - TopoNet (scene-graph GNN with lane-lane and lane-traffic relations; TOP metric)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S6 E.*

## Papers

Graph-based Topology Reasoning for Driving Scenes (Li et al., arXiv 2304.05277) https://arxiv.org/abs/2304.05277

## Idea

Lane and traffic-element queries are nodes of a scene graph; a GNN with relation-type-specific weights passes messages along the previous layer's predicted adjacency (self loop, predecessor, successor channels), and an MLP head scores ordered pairs.

## Formulation

BEV 200 x 100 (BEVFormer-like, 3 layers, no temporal input); 200 centerline queries (11 x 3 points) and 100 traffic queries; initial adjacency T_ll^0 = I; T_ll^i = beta (A + A^T) + I, beta 0.6; GCN Q' = sigma(T Q W); knowledge-graph variant: K_ll = stack(A, A^T, I), per-class weights W_ll(c), messages weighted by traffic-element class scores; update Q_l~ = Q_l + downsample(ReLU(concat(Q_l', Q_l''))). Topology head: Q~a' = MLP_a(Q~a), Q~b' = MLP_b(Q~b) (128 ch), conf = sigmoid(MLP_top(concat)), separate MLP per relation (asymmetric for lane-lane). Loss L_top = 5 focal(lane-lane) + 5 focal(lane-traffic); focal as the only imbalance remedy (alpha, gamma not given); Hungarian matching assigns GT topology to matched query pairs; supervision every layer.

## Input

BEV features, query sets.

## Output

N x N and N x M adjacency probabilities.

## Training

24 epochs, AdamW 1e-4 cosine, batch 8, 8 A100, 2048 x 1550 x0.5; range x +-50, y +-25 m.

## Differentiable?

partial - Hungarian.

## Sensors and data

camera (front view for traffic); OpenLane-V2 A and B.

## Reported results

ablations (old metric, subset A; DET_l / TOP_ll / OLS): MLP baseline 25.7 / 4.0 / 34.6; +scene graph 27.7 / 3.7 / 35.0; +knowledge graph 28.5 / 4.1 / 35.6. GNN layers 1 / 2 / 3 / 4: OLS 35.6 / 35.3 / 28.3 / 22.6 (TOP_ll collapses to 0.5 and 0.0 at 3 and 4 layers). beta 0.5 / 0.6 / 0.7: 35.4 / 35.6 / 35.1. v2.1 values quoted elsewhere: TOP_ll 10.9, OLS 39.8 (A); TOP_ll 6.7 (B). Metric: TOP = vertex-level mAP - for each GT vertex the predicted neighbours are ranked by confidence and precision is accumulated over hits, normalised by the true neighbour count; edges with confidence > 0.5 count as positive.

## Compute

not stated (62.6 M parameters in Reusing-Attention's table).

## Code

https://github.com/OpenDriveLab/TopoNet (not opened)

## Fit to our project

baseline - effort S - GNN message passing on a link head added nothing measurable in its own ablation (TOP_ll 3.7 -> 4.1) and breaks at depth 3.

## Needs

per-lane query vectors; predicted adjacency from the previous layer.

## Conflicts

set-prediction pipeline; deeper GNN diverges.

## Tried in our experiments

E46 / E51 (node-conditioned head, path repair): within 0.01 of rules, dropped.
