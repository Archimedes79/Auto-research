# GR13 - SeqGrowGraph (lane graph as a chain of adjacency-matrix expansions)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: partly (training details absent in the paper text).. Pipeline stage: S4, S6 (token sequence).*

## Papers

SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions (Xie et al., ICCV 2025; arXiv 2507.04822) https://arxiv.org/abs/2507.04822

## Idea

The directed lane graph grows by one *junction vertex* per step; at step n the adjacency matrix expands from n x n to (n+1) x (n+1), so a step emits the new vertex position plus its incoming and outgoing links with a quadratic Bezier control point for each connecting centerline. Vertices are visited in DFS order; a 6-layer transformer decoder generates the token sequence from LSS BEV features. Handles loops and two-way single-lane roads because the graph is directed rather than a DAG.

## Formulation

S_n = S_{n-1} + (v_n + F_n + T_n) with F_n / T_n the incoming / outgoing centerlines to earlier vertices; tokens: coordinates 0 - 200 (0.5 m bins), node indices 200 - 350, Bezier control coordinates 350 - 570, special tokens to / SEP / EOS / BOS / N/A = 571 - 575. Loss = token NLL, weight 2 on point-coordinate tokens, 1 elsewhere (weights 1 / 2 / 3 / 4 give L-F 54.7 / 56.4 / 55.3 / 55.9). Ordering ablation (landmark-F / reachability-F): DFS 56.4 / 67.8, BFS 55.1 / 67.7, coordinate 53.7 / 64.4 (junctions sorted by x, y; key order not stated), centre-out 52.4 / 63.7 (junctions by increasing distance from the graph centre); random order not tested. Not stated (checked in three passes): DFS root, tie-breaking, how generation stops, decoding strategy, optimiser, learning rate, image size, BEV grid size, augmentation, post-processing, parameters, speed, limitations.

## Input

6 (nuScenes) or 7 (AV2) cameras; LSS BEV encoder with ResNet-50 pre-trained on centerline segmentation; BEV x in [-48, 48] m, y in [-30, 30] m (nuScenes). `image_feat` -> `token_sequence`.

## Output

`token_sequence` -> adjacency + Bezier centerlines.

## Training

`next_token` with GT prefix (teacher forcing implied, not named), 400 epochs nuScenes / 50 AV2 on 4 A100.

## Differentiable?

no - tokenisation (0.5 m bins), argmax / sampling at decoding.

## Sensors and data

cameras; nuScenes, Argoverse 2.

## Reported results

nuScenes default split landmark P / R / F 63.6 / 50.8 / 56.4, reachability 75.5 / 61.4 / 67.8 vs RNTR 57.1 / 42.7 / 48.9 and 63.7 / 45.2 / 52.8, LaneGraph2Seq 46.9 / 43.7 / 45.2, TopoNet 52.5 / 47.1 / 49.6 (reachability-R 10.8); Argoverse 2 landmark-F 56.4, reachability-F 55.7 (Tables 1 - 2). A re-segmentation study splits GT centerlines at junctions and optionally every 20 / 30 / 40 m (junction-level landmark-F 59.5 - 60.4, Table 4; different GT, not comparable with Table 1; which version the main table uses is not stated).

## Compute

not stated.

## Code

none found

## Fit to our project

not_suitable as a block (tokenised autoregression, junction-only vertices); reuse_idea for metrics (landmark / reachability) and the junction-level graph definition.

## Needs

a sequence order (DFS); coordinate quantisation.

## Conflicts

no gradient through tokens; vertices only at junctions (we need dense vertices).

## Tried in our experiments

none.
