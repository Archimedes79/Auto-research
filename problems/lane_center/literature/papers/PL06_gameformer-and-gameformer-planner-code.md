# PL06 - GameFormer (level-k interactive prediction + planning) and GameFormer-Planner code

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10.*

## Papers

GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning (Huang et al., ICCV 2023) https://arxiv.org/abs/2303.05760

## Idea

Level-0 decoder predicts every agent from scene context; level k re-conditions each agent's decoder on the level k-1 futures of all other agents. The nuPlan pipeline adds reference-path selection from the route and a differentiable Gauss-Newton refinement.

## Formulation

prediction model: agent history 11 steps (1 s), 22 agents; map up to 6 drivable lanes per target agent x 100 waypoints x 15 features (centre, left, right position and heading, speed limit, lane type, traffic light, stop-sign control), up to 4 crosswalks x 100 x 3, max-pooled to 60 lane + 20 crosswalk tokens; D = 256; K = 6 levels (K = 4 for open-loop planning). Route (paper): reference path from a cubic spline through traversed-lane waypoints, 100 m ahead, 1000 points at 0.1 m, 5 channels. Code (GameFormer-Planner): lanes (N, P, 3) = x, y, heading (+ 4 traffic-light one-hot = 7), route_lanes (N, P, 3) from route roadblock ids, crosswalks (N, P, 3); nearest N by minimum point distance; boundaries computed but not returned; `avails` (N, P) mask; ego frame at the present state; no other normalisation; N, P from config (not seen). Loss = L_IL (NLL of best mode per step + CE, every level) + 0.1 L_Inter (repulsive potential). Refinement: DIPP-style Gauss-Newton (max 30 iterations, step 0.3), cost weights learned through the optimiser.

## Input

ego / agent histories, lanes, route, crosswalks.

## Output

M joint trajectories (GMM) per level + refined ego plan.

## Training

planning 20 epochs, AdamW lr 1e-4, batch 32, one RTX 3080; regime: imitation + interaction loss.

## Differentiable?

yes (optimiser differentiable, cost weights learned).

## Sensors and data

nuPlan GT map; Waymo.

## Reported results

nuPlan test benchmark overall 0.8288 (OL 0.8400, CL-NR 0.8087, CL-R 0.8376; second behind 0.8745); Waymo interactive (J, M = 6) minADE 0.9161, minFDE 1.9373, MR 0.4531, mAP 0.1376; WOMD closed loop success 73.16 % -> 94.50 % with refinement.

## Compute

parameters not reported.

## Code

https://github.com/MCZhi/GameFormer-Planner (project https://mczhi.github.io/GameFormer/)

## Fit to our project

reuse_idea - effort M - reference path from the route plus differentiable refinement could post-process E117's path; the level-k part needs agents, which we do not predict.

## Needs

route lanes, lane centre-lines with heading, mask.

## Conflicts

no connectivity; needs agents for the game part; nuPlan devkit.

## Tried in our experiments

none.
