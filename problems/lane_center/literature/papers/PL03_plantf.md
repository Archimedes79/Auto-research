# PL03 - PlanTF (polygon tokens with on-route flag, state-dropout encoder)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M (partial: the HTML extraction lacked the map section) + C** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10, T.*

## Papers

Rethinking Imitation-based Planner for Autonomous Driving (Cheng et al., ICRA 2024) https://arxiv.org/abs/2309.10443 ; project page https://jchengai.github.io/planTF

## Idea

Transformer encoder (adapted from a masked-autoencoder forecasting model) over agent and map tokens; the ego plan is decoded from the ego token. Shows that kinematic ego state / history gives a shortcut: strong open-loop score, weak closed-loop score. Fixes: position-only ego input, state dropout encoder (SDE), perturbation augmentation with re-normalisation.

## Formulation

map input (from code): polygons = lanes + lane connectors + crosswalks within a +-100 m square (no cap in the builder; padded to the longest sample in the batch). Per polygon three polylines (centre, left, right) x 20 points (21 sampled, last dropped). The released encoder uses ONLY the centre polyline: per point [p - polygon_center (2, translation only), vector to next point (2), cos / sin heading (2)] = 6 channels -> point encoder (`PointsEncoder`, internals not read) -> 128-d; added embeddings: polygon type (3), on_route (2, roadblock-id membership), traffic-light status (4, UNKNOWN allowed), speed limit by MLP or learned "unknown" embedding. Positional embedding from (x, y, cos, sin) of the polygon centre. Ego frame at the current pose. Key padding: polygon masked if no valid point; point validity = inside the square. No adjacency; no map augmentation found. Agents: 2 s history, up to 64. Depth 4, dim 128, 8 heads, drop-path 0.2, 6 modes, 80 future steps (8 s), SDE dropout 0.75.

## Input

polygon feature dict (above) + agent histories.

## Output

6 trajectories with probabilities + predictions of other agents.

## Training

imitation (loss terms not found on the page); ego-state perturbation with probability 0.5; 25 epochs, batch 128, lr 1e-3 cosine, weight decay 1e-4, at most 1 M frames; regime: teacher forcing / imitation.

## Differentiable?

yes for the network; the map is GT, no gradient to a map head.

## Sensors and data

nuPlan, GT map + GT tracked objects; radius 100 m.

## Reported results

Test14-random OLS 87.07 / NR-CLS 86.48 / R-CLS 80.59; Test14-hard 83.32 / 72.68 / 61.70; Val14 (own Table VII) 89.18 / 84.83 / 76.78 (PLUTO's table lists PlanTF at 85.30 / 77.07 on Val14: differs between papers). SDE dropout 0.25 / 0.5 / 0.75 on Test14-random: NR-CLS 81.70 / 83.71 / 86.48 while OLS falls 89.11 / 89.12 / 87.07.

## Compute

cache of 1 M frames "20+ hours"; 4-6 GB GPU memory per GPU at batch 32 (README); parameters not stated.

## Code

https://github.com/jchengai/planTF

## Fit to our project

adapt_code - effort M - requires nuPlan devkit and Python 3.9, but the feature dict is a clean contract to emit from our graph.

## Needs

polygons with 20 centre points, polygon centre, on_route flag, per-point valid mask, ego-frame coordinates.

## Conflicts

no connectivity input (topology only implicit); range 100 m vs our 50 m; route flag is roadblock-level (coarser than our lane-level route); ego motion as input causes the copycat shortcut (E117 uses ego motion).

## Tried in our experiments

none for polygon consumers; E117 is the own consumer.
