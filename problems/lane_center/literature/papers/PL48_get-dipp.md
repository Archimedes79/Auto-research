# PL48 - GET-DIPP (lane-graph encoder + differentiable integrated prediction and planning)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10.*

## Papers

GET-DIPP: Graph-Embedded Transformer for Differentiable Integrated Prediction and Planning (arXiv 2211.06031; authors not read) https://arxiv.org/abs/2211.06031

## Idea

Per agent 6 nearest lanes and 4 nearest crosswalks; lane waypoints (boundaries, centre, headings, speed limit, types, traffic light, stop signs) become DGCNN vertices connected to their k = 20 nearest neighbours in feature space (not map connectivity); proxy waypoints (averaged subsets) cut memory from O(L x L) to O((L/l)^2); the DIPP differentiable optimiser plans on the prediction.

## Formulation

loss = 0.5 L_predict + L_score + L_ADE + L_FDE.

## Input

lane and crosswalk waypoints, agent histories.

## Output

prediction and plan.

## Training

end-to-end; the gradient path into the encoder is not described explicitly.

## Differentiable?

yes.

## Sensors and data

Waymo (173 k / 25 k / 7 k frames).

## Reported results

integrated planning position error at 3 / 5 / 10 s 1.710 / 3.420 / 8.201 m vs DIPP 1.726 / 3.913 / 9.365; collision 3.76 % vs 5 %; prediction ADE / FDE 0.712 / 1.808 vs 0.740 / 1.814.

## Compute

not read.

## Code

none given.

## Fit to our project

not_suitable - effort L - Waymo GT map; relevant only as an example of a feature-space k-NN graph (no explicit lane topology) feeding a differentiable planner.

## Needs

waypoints.

## Conflicts

GT map.

## Tried in our experiments

none.
