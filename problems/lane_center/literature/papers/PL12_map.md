# PL12 - MAP (plan-enhancing online mapping + ego-status-guided planning)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10.*

## Papers

MAP: End-to-End Autonomous Driving with Map-Assisted Planning (Yin et al., ICCVW 2025) https://arxiv.org/abs/2509.13926

## Idea

Planning query Q_map = cross-attention of the mapping module's memory features with the ego-status embedding; a parallel branch Q_plan = cross-attention of BEV features with ego status; a weight adapter alpha in [0,1] (sigmoid of a 64-d vector from ego status + command) fuses Q_fused = alpha Q_plan + (1 - alpha) Q_map. The mapping module receives gradients from the mapping loss and from the trajectory loss through Q_map.

## Formulation

built on UniV2X (BEVFormer BEV, Panoptic SegFormer mapping), tracking / motion / occupancy removed; 5 s horizon (10 steps); loss = collision (BEV box overlap) + ADE + adaptive L2 / collision / off-road terms (weights not given); only the fused trajectory is supervised.

## Input

BEV features, mapping memory, ego status (CAN bus), command.

## Output

5 s trajectory.

## Training

initialised from UniV2X stage-1 checkpoint; regime: imitation.

## Differentiable?

yes (explicit statement of gradient into the mapping module).

## Sensors and data

DAIR-V2X-seq-SPD (cooperative driving), not nuScenes.

## Reported results

val L2 / collision / off-road: UniV2X (retrained) 3.46 / 0.34 / 0.74, MAP 2.48 / 0.94 / 0.89 (collision and off-road higher than the baseline on validation); test leaderboard MAP L2 2.56, collision 0.96, off-road 0.39 vs UniV2X 3.07 / 0.71 / 0.89. Ablation (data subsampled to 1/5): w/o plan-enhancing mapping 3.22 / 1.35 / 1.13; w/o ego-guided branch 3.52 / 0.45 / 1.35; full 2.88 / 2.48 / 0.45.

## Compute

2 x A800, about 50 h; transmission cost 1.69e5 BPS vs 8.09e5.

## Code

https://gitee.com/kymkym/map.git

## Fit to our project

reuse_idea - effort S - mapping memory as planner query and an ego-status-dependent fusion weight are simple, but collision got worse.

## Needs

dense mapping memory features.

## Conflicts

cooperative-driving pipeline; different dataset.

## Tried in our experiments

E119 / E121 (planner loss into route head / positions: no gain).
