# PL07 - DTPP (tree policy planning with learned cost)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, S10.*

## Papers

DTPP: Differentiable Joint Conditional Prediction and Cost Evaluation for Tree Policy Planning (Huang et al., ICRA 2024) https://arxiv.org/abs/2310.05885

## Idea

Ego plans a two-stage tree of candidate trajectories along reference paths; a query-centric transformer predicts all agents conditioned on every branch in one pass; a context-aware learned cost scores branches; softmax over costs + cross-entropy trains cost and prediction jointly.

## Formulation

3 reference paths (lane centre-lines from map and route, as in GameFormer); stage 1 (0-3 s) up to 30 target states over the paths, top 5 kept by learned cost; stage 2 (3-8 s) 6 targets per node, up to 30 branches; third-order polynomial speed profile; dynamic programming selects the first action. Encoder: 10 agents + ego, 20 past steps, 50 nearest map elements (lanes + crosswalks, MLP, dim 256, points per lane not stated), 3 self-attention layers; decoder with multi-axis (time / mode) causal attention. Cost = time-averaged weighted sum of ego features (jerk, acceleration, speed, lateral acceleration), a learned 5-d interaction feature and a Gaussian-RBF collision feature; weights from a 10-d MLP on the ego encoding. Loss = sum over stages beta^j (L_p smooth-L1 on the branch closest to the GT plan + 0.1 L_c cross-entropy), beta = 1.0, 0.2.

## Input

lane graph (50 elements), route reference paths, agent histories.

## Output

selected ego trajectory + predicted agent futures per branch.

## Training

100 k scenarios from the nuPlan validation split, 20 epochs, batch 16, AdamW lr 1e-4, one RTX 3080; regime: imitation / max-entropy IRL form.

## Differentiable?

yes (costs are differentiable functions of predicted states; softmax selection in training).

## Sensors and data

nuPlan GT; 200 test scenarios (20 per type).

## Reported results

nuPlan closed-loop non-reactive / reactive: DTPP 0.8964 / 0.8978; PDM 0.9061 / 0.9150; IDM 0.6396 / 0.6168; TPP 0.7388 / 0.7699; open-loop 0.7911.

## Compute

98.0 ms with pruning (199.6 ms without).

## Code

https://github.com/MCZhi/DTPP

## Fit to our project

reuse_idea - effort M - a tree of route-conditioned candidates scored by a learned cost parallels our hypothesis-tree thinking on the planner side.

## Needs

reference paths from lane graph + route; map elements as tokens.

## Conflicts

needs agent futures; nuPlan devkit.

## Tried in our experiments

none.
