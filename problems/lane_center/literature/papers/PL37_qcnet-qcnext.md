# PL37 - QCNet / QCNeXt (query-centric scene encoding with typed lane edges plus radius edges)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M (QCNeXt report) + C (dataset and map encoder)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S6, S10.*

## Papers

Query-Centric Trajectory Prediction (Zhou et al., CVPR 2023; paper PDF not readable here) ; QCNeXt: A Next-Generation Framework for Joint Multi-Agent Trajectory Prediction https://arxiv.org/abs/2306.10508 ; code https://github.com/ZikangZhou/QCNet

## Idea

Each scene element lives in its own local space-time frame; attention keys / values are concatenated with a relative pose embedding (distance, angle between the query orientation and the relative vector, relative orientation), so nothing depends on a global frame. Map: one polygon per lane segment (crosswalks two polygons); points on left boundary, right boundary and centre-line (raw map points, last point dropped; crosswalk centre-line resampled about every 2 m); polygon attributes position, orientation, type (4), is_intersection (3); point attributes orientation, magnitude, type (17), side (3).

## Formulation

polygon-to-polygon edges = radius graph (max 300 neighbours, type 0) merged (max) with map-graph edges of types predecessor, successor, left, right (embedding of 5 types); point-to-polygon edges by ownership; layers: point-to-polygon then polygon-to-polygon attention; missing neighbour ids are skipped. Decoder: two stages (anchor-free recurrent proposals, 3 recurrent steps of 2 s, then anchor-based refinement with detached proposals); mixture of Laplace with scene-level winner-take-all; D = 128, 8 heads, 2 blocks (QCNeXt); AdamW 50 epochs, batch 32, lr 5e-4.

## Input

polygons, points, typed edges, agent histories (50 steps).

## Output

K joint futures (60 steps) with scores.

## Training

mixture-of-Laplace NLL with winner-take-all; regime: wta.

## Differentiable?

yes (the recurrent stage-1 proposals are detached before refinement).

## Sensors and data

Argoverse 2 GT map.

## Reported results

Argoverse 2 validation (QCNet) minFDE6 1.25, minADE6 0.72, MR6 0.16, brier-minFDE6 1.87 (README); QCNeXt multi-agent avgMinFDE6 1.13 without ensemble, 1.02 with.

## Compute

training needs about 160 GB GPU memory in total (README: 8 x RTX 3090 example); parameters not stated.

## Code

https://github.com/ZikangZhou/QCNet

## Fit to our project

reuse_idea - effort M - relative geometry instead of a global frame, and radius edges next to topology edges so that a missing successor edge is not fatal; the av2 preprocessing is heavy.

## Needs

polygons with ordered points, typed edges.

## Conflicts

av2-style dataset class, PyG.

## Tried in our experiments

none.
