# PL19 - Ego-status and open-loop critiques of nuScenes planning (BEV-Planner, AD-MLP) and the curb-collision metric

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: E.*

## Papers

Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving? (Li et al., CVPR 2024) https://arxiv.org/abs/2312.03031 ; Rethinking the Open-Loop Evaluation of End-to-End Autonomous Driving in nuScenes (Zhai et al., arXiv 2023) https://arxiv.org/abs/2305.10430

## Idea

On nuScenes a model using only past ego motion and a command matches perception-based planners on L2; ego status in the planner dominates. New metric: curb collision rate (CCR) = share of trajectories that intersect annotated road boundaries rasterised at 0.1 m. Recommendations: stricter, more diverse metrics; report where ego status enters.

## Formulation

AD-MLP input 21 values (4 past poses x, y, heading, velocity, acceleration, yaw rate) + one-hot command derived from whether the ego moves more than 2 m sideways in the next 3 s; Linear 21-512-512-18; 6 poses at 2 Hz; L1. BEV-Planner: R50, 704 x 256 images, BEV 128 x 128 (about 50 m), 4 past BEV frames concatenated without alignment, one learnable ego query, single L1 loss, no perception labels.

## Input

ego history + command (AD-MLP); camera BEV (BEV-Planner).

## Output

3 s trajectory.

## Training

L1 on trajectory points (AD-MLP down-weights samples whose predicted and GT points fall in the same 0.5 m occupancy cell).

## Differentiable?

yes.

## Sensors and data

nuScenes.

## Reported results

avg L2 m / collision % / CCR %: UniAD reproduced without ego status 1.03 / 0.77 / 1.93; official (ego status in BEV) 0.66 / 0.62 / 1.72; both 0.46 / 0.37 / 1.59; VAD-Base without 1.25 / 1.09 / 3.82, official 0.72 / 0.54 / 2.72, both 0.37 / 0.33 / 2.47; Ego-MLP 0.35 / 0.37 / 2.93; GoStraight 0.83 / 1.08 / 8.62; BEV-Planner 0.55 / 0.59 / 4.26, ++ 0.35 / 0.34 / 3.16; BEV-Planner + UniAD-style map task 0.96 / 0.89 / 2.60 (map helps turning, hurts straight driving: L2 and collision worse, CCR better). VAD with ego status: zeroing ego velocity gives L2 6.16 m; blank camera images leave planning almost unchanged (0.46 m / 0.54 %). AD-MLP: L2 0.29 m / collision 0.19 % vs UniAD 1.03 / 0.31 and VAD-Base 0.37 / 0.14 (quoted from the VAD table). 73.9 % straight driving in the main text, 87 % in Appendix E (not reconciled); about 70 % of headings within +-0.2 rad; the 0.5 m occupancy grid flags about 2 % of GT trajectories as colliding.

## Compute

not stated.

## Code

https://github.com/NVlabs/BEV-Planner (AD-MLP repository link not read)

## Fit to our project

reuse_idea - effort S - CCR-like boundary metric, ego-status ablation for E117, turning subset reported separately.

## Needs

GT road boundaries from the nuScenes map expansion.

## Conflicts

the command on nuScenes is derived from the future ego motion (leak); our route label (lane sequence driven in the next 6 s) is also future-derived.

## Tried in our experiments

none.
