# PL33 - SSR (16 navigation-guided scene tokens, no map supervision)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10, S9.*

## Papers

Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving (Li, Cui; ICLR 2025) https://arxiv.org/abs/2409.18341

## Idea

Dense BEV (100 x 100) is squeezed into 16 tokens (dim 256) by a TokenLearner whose BEV features are modulated by the command through a squeeze-and-excitation layer; future-BEV prediction as self-supervision; waypoint queries attend the tokens; the output set is chosen by the command. No detection or map supervision in the main model.

## Formulation

B^navi = SE(B, cmd); token = global-average-pool of B^navi times a learned spatial attention map; loss = L1 imitation + L2 future-BEV reconstruction (both weight 1.0).

## Input

BEV features + command.

## Output

trajectory.

## Training

imitation + self-supervised BEV prediction; 11 h on 8 x RTX 3090.

## Differentiable?

yes.

## Sensors and data

nuScenes, CARLA.

## Reported results

nuScenes (MAX metric) avg L2 0.75 m / collision 0.15 % vs UniAD 1.03 / 0.31 and VAD-Base 1.22 / 0.53; AVG metric 0.39 / 0.06 %; tokens 8 / 16 / 32 / 64: L2 0.69 / 0.75 / 0.77 / 0.84, collision 0.20 / 0.15 / 0.16 / 0.41; removing navigation guidance: straight 0.72 / 0.29 % -> 0.71 / 0.10 %, turns 1.09 / 0.78 -> 1.04 / 0.66; CARLA Town05 Long DS 78.9.

## Compute

19.6 FPS (RTX 3090); parameters not reported.

## Code

https://github.com/PeidongLi/SSR

## Fit to our project

reuse_idea - effort S - the command enters as a multiplicative channel gate on features; and a map-free planner reaching state-of-the-art nuScenes numbers shows the metric is no graph-quality test.

## Needs

BEV features.

## Conflicts

none.

## Tried in our experiments

none.
