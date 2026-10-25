# GR08 - LaneGNN (successor sub-graph per pose, rollout with Dijkstra pruning and aggregation)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: checked (full HTML text including supplement).. Pipeline stage: S4, S6, S7, S8.*

## Papers

Learning and Aggregating Lane Graphs for Urban Automated Driving (Buechner, Zuern, Todoran, Valada, Burgard, arXiv 2302.06175, Feb 2023; venue CVPR 2023 from memory, not on the arXiv page) https://arxiv.org/abs/2302.06175

## Idea

Instead of one vertex, a GNN predicts the whole *successor lane graph* reachable from a virtual agent pose (inside a 256 x 256 aerial crop, 15 cm / px, oriented by the pose). A rollout moves the pose forward along the best edge, transforms each prediction to world coordinates and merges it into a global graph; many start poses run in parallel and overlapping predictions are aggregated by weighted node updates.

## Formulation

Nodes = 400 Halton points filtered by an ego-lane mask; edges = pairs within [d_min, d_max]; GNN with causal message passing (predecessor and successor messages summed separately), L = 6 steps; edge features = geometry + a 4 x 32 x 32 oriented image patch (ResNet-18); outputs node, edge and terminal-node scores. Targets: node score (1 - d)^8 (d = distance to GT), binary edges from minimum-cost paths start -> endpoints; BCE sums; Adam 1e-3, 100 epochs, batch 2; pose noise sigma 5 px / 0.3 rad as the only exposure-bias remedy. Rollout per step: predict -> Dijkstra prune (paths to terminals, scores zeroed) -> world transform -> Laplacian smoothing -> merge (lateral distance a_thresh = 20 px; weights = merge counts) -> step forward one edge; depth of successor tree capped at 10 (loops); exploration depth first with confidence threshold; termination by steps (36), branches (4), branch age (12) or revisited pose; 178 start poses from predicted lane starts; unvalidated splits / merges and parallel branches shorter than 6 edges are removed.

## Input

aerial RGB crop + predicted lane / ego-lane masks (PSPNet-152 regressors).

## Output

successor sub-graph per step; global lane graph after aggregation.

## Training

`teacher_forcing` with pose noise (single-step supervision, rollout untrained); `matching_free` BCE on node / edge scores.

## Differentiable?

no for the rollout (Dijkstra, argmax pose choice, merge rules); inside one step yes (GNN).

## Sensors and data

aerial (Urban Lane Graph dataset derived from Argoverse 2 maps, six cities); no vehicle sensors.

## Reported results

Successor-LGP: TOPO P / R 0.600 / 0.699, SDA20 0.227, IoU 0.347 (skeleton baseline 0.597 / 0.613, SDA20 0.020); Full-LGP (Miami): TOPO P / R 0.481 / 0.670 and GEO P / R 0.649 / 0.689 vs naive aggregation 0.366 / 0.654 and 0.523 / 0.727 (Tables 2, 3). A lower a_thresh = 10 raises recall, lowers precision and Graph IoU by about 3 points.

## Compute

about 1 s per rollout step (RTX 3090), 7 min per 10^6 m^2 tile with parallel runs.

## Code

dataset and code announced at https://urbanlanegraph.cs.uni-freiburg.de (page not reachable during this session)

## Fit to our project

reuse_idea - effort M. A step that returns a *sub-graph with terminal scores* rather than one vertex is the same idea as our per-level children + rescoring; the aggregation of overlapping predictions with weights and lateral-only updates is a ready recipe for tree -> DAG and for time fusion.

## Needs

start poses; candidate node / edge proposals; a merge threshold.

## Conflicts

Dijkstra pruning and rule-based aggregation are discrete; single-step supervision.

## Tried in our experiments

E42 - E56 (learned link / path repair within 0.01 of rules, dropped) is the closest; temporal fusion E41 small.
