# PL28 - GoalFlow (goal point from a vocabulary, drivable-area-scored)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10, S1.*

## Papers

GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving (Xing et al., CVPR 2025; first author from memory) https://arxiv.org/abs/2503.05689

## Idea

Goal points (x, y, theta) from a vocabulary of 4096 / 8192 trajectory endpoints are scored by a distance score (softmax over negative L2 to the GT endpoint) and a binary drivable-area score (shadow-vehicle box inside the drivable-area polygon); the best goal conditions a rectified-flow trajectory generator; a shadow trajectory generated without the goal detects unreliable goals.

## Formulation

scorer = transformer decoder (query F_v + F_ego, keys F_bev) + two MLPs; final score w1 log d_dis + w2 log d_dac; rectified flow x_t = (1 - t) x0 + t tau, target v = tau - x0; goal encoded sinusoidally and concatenated; random masking of conditions; 128-256 samples; selector score = -lambda1 norm(f_dis) + lambda2 norm(f_pg).

## Input

BEV features, ego features, goal.

## Output

trajectory (and goal).

## Training

flow matching + score losses; TransFuser-style scene encoder with HD-map cross-entropy, 3-D box classification and L1 as auxiliary losses (weights 10, 1, 10).

## Differentiable?

scoring / selection are hard.

## Sensors and data

NAVSIM cameras + lidar.

## Reported results

NAVSIM navtest PDMS 90.3 (NC 98.4, DAC 98.3, TTC 94.6, EP 85.0); with the GT endpoint as goal 92.1 (oracle; +1.8, mostly NC +1.4 and TTC +4.0, DAC -0.4); human 94.8; PARA-Drive 84.0 in the same table.

## Compute

not read.

## Code

https://github.com/YvanYin/GoalFlow

## Fit to our project

reuse_idea - effort M - a goal on the route graph: an endpoint vertex scored by distance and drivable-area compliance.

## Needs

drivable-area polygon.

## Conflicts

hard argmax of the goal.

## Tried in our experiments

none.
