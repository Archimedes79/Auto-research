# PL35 - LaneGCN (lane-graph convolution with four edge types and dilations)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M+C (README)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S6, S10.*

## Papers

Learning Lane Graph Representations for Motion Forecasting (Liang et al., ECCV 2020) https://arxiv.org/abs/2007.13732

## Idea

Nodes are the straight segments between consecutive centre-line points (location = midpoint); node feature x_i = MLP_shape(end - start) + MLP_loc(midpoint). Four adjacency matrices: predecessor, successor (shifted identities), left and right neighbour (closest node on the adjacent lane by l2 distance). LaneConv: Y = X W0 + sum over {pre, suc, left, right} A_i X W_i, with dilated pre / suc terms A^k for k in {1, 2, 4, 8, 16, 32} (left / right undilated).

## Formulation

MapNet = 4 LaneConv residual blocks, 128 channels; ActorNet 1-D CNN; FusionNet 4 modules A2L, L2L (another LaneGCN), L2A, A2A with distance thresholds 7 m, 6 m, 100 m; loss L_cls (max-margin to the best mode) + L_reg (smooth l1), alpha 1.0; actors and lanes within 100 m, frame centred on the agent with x along t-1 -> t0.

## Input

lane nodes + 4 adjacencies, actor histories.

## Output

K trajectories with confidences.

## Training

36 epochs, batch 128, Adam lr 1e-3 -> 1e-4; regime: winner-takes-all style max-margin.

## Differentiable?

yes.

## Sensors and data

Argoverse (GT map); 100 m radius.

## Reported results

Argoverse val (K = 6 minADE / minFDE): ActorNet only 0.91 / 1.66; + MapNet 0.79 / 1.29; full 0.71 / 1.08; K = 1 minFDE 2.97 (4.38 without map). LaneConv vs vanilla graph conv minFDE 1.10 vs 1.41.

## Compute

about 11.5 h on 4 TITAN-X (paper), about 8 h on 4 RTX 5000 (README).

## Code

https://github.com/uber-research/LaneGCN (archived January 2024)

## Fit to our project

reimplement - effort S - our edges map directly to pre / suc adjacency (edge nodes = LaneGCN nodes, the line graph of the vertex graph); left / right must be derived geometrically.

## Needs

segment nodes, a successor graph without gaps (dilation needs long chains), left / right neighbours.

## Conflicts

false forks and missing edges change the A^k powers; 6 m edges are coarser than typical LaneGCN segments (segment length not stated).

## Tried in our experiments

none (E42 / E45 / E46 learned link heads concern perception).
