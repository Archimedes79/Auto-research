# PL39 - PGP (policy over lane-graph edges, traversals aggregated by attention; nuScenes map expansion)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M+C (README)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S6, S9, S10.*

## Papers

Multimodal Trajectory Prediction Conditioned on Lane-Graph Traversals (Deo, Wolff, Beijbom; CoRL 2021) https://arxiv.org/abs/2106.15004 ; code (re-implementation) https://github.com/nachiket92/PGP

## Idea

Lane graph from the nuScenes map: centre-lines inside [-50, 50] m laterally and [-20, 80] m longitudinally around the target, split into snippets of at most 20 m discretised at 1 m; node features per pose (x, y, yaw, 2-bit stop-line / crosswalk flag); edges: successor (incl. branches and merges) and proximal (neighbouring lanes within distance and yaw thresholds; values not given). A policy header scores every outgoing edge (MLP 2 x 32, softmax over outgoing edges plus an end state); traversals are sampled as roll-outs (200 trajectories clustered to K = 10).

## Formulation

GAT / GCN layers (dim 32), agent-node attention within a distance threshold; node embeddings along each path aggregated by 32-head attention into a 128-d context; 5-d latent z; decoder gives 24 values (x, y over 6 s at 2 Hz).

## Input

lane graph with successor + proximal edges, agent histories.

## Output

K = 10 trajectories.

## Training

L = L_BC (NLL of edges visited by the GT future; a node counts as visited if its yaw matches) + L_reg (winner-take-all); 100 epochs with GT traversals, then 100 epochs with policy-sampled paths.

## Differentiable?

partial - roll-out sampling of traversals.

## Sensors and data

nuScenes map expansion v1.3 (the same GT as our labels).

## Reported results

nuScenes minADE5 1.30, minADE10 1.00, miss rate (5, 2 m) 0.61, off-road rate 0.03.

## Compute

not stated.

## Code

https://github.com/nachiket92/PGP

## Fit to our project

baseline - effort M - the edge policy is a route probability on edges conditioned on the agent's motion; our route head could fill that slot and supply the traversal.

## Needs

successor + proximal edges.

## Conflicts

sampling of paths; false forks.

## Tried in our experiments

E102-E104 (route attribution variants) concern the perception side.
