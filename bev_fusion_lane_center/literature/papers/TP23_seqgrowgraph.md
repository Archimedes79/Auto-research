# TP23 - SeqGrowGraph (vertex graph as growing adjacency + geometry matrix)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S7 S6.*

## Papers

SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions (Xie et al., ICCV 2025) https://arxiv.org/abs/2507.04822

## Idea

The lane graph is stored as vertices (junction-like key points, incl. forks and merges) and directed edges (centerlines); an adjacency matrix grows from n x n to (n+1) x (n+1) per step in depth-first order together with a geometry matrix holding one quadratic-Bezier control point per edge.

## Formulation

A(i,j) = 1 iff a directed centerline runs from v_i to v_j; fork / merge = several non-zero entries in a row / column; loops and bidirectional lanes representable (A(i,j) and A(j,i) both 1). Step S_n = S_{n-1} + (v_n + F_n + T_n) (F incoming, T outgoing). Tokens: node coordinates 0-200 (0.5 m bins), node indices 200-350, control-point coordinates 350-570, <to> <SEP> <EOS> <BOS> <N/A>; point-position token loss weight 2. Model: LSS BEV (R50, pre-trained on centerline segmentation), 6-layer transformer decoder; negative log-likelihood given previous GT tokens. Ordering ablation (landmark F / reachability F): centre 52.4 / 63.7, coordinate 53.7 / 64.4, BFS 55.1 / 67.7, DFS 56.4 / 67.8.

## Input

LSS BEV; nuScenes x [-48, 48], y [-30, 30] m.

## Output

token sequence -> graph.

## Training

teacher forcing, 400 epochs nuScenes, 50 on Argoverse 2, 4 A100.

## Differentiable?

no (sequence decoding).

## Sensors and data

camera; nuScenes default and PON splits, Argoverse 2.

## Reported results

nuScenes default L-F / R-F 56.4 / 67.8 (LaneGAP / RNTR 53.2 / 47.5 best baselines); PON split 37.7 / 46.7; centerline metrics M-F 70.6, C-F 65.8; length-based re-segmentation changes the GT (not comparable).

## Compute

latency not reported.

## Code

none in the paper

## Fit to our project

reuse_idea - effort S (representation only) - it states precisely how the OpenLane-V2 piece graph relates to a vertex graph (key vertices + edge geometry), which is the conversion we need for evaluation.

## Needs

junction vertices; edge geometry.

## Conflicts

autoregressive decoding (not differentiable); sequence length scaling.

## Tried in our experiments

none
