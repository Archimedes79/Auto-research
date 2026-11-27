# PL08 - VAD (map vectors as planner input and planning constraints; gradient into map points)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10, L.*

## Papers

VAD: Vectorized Scene Representation for Efficient Autonomous Driving (Jiang et al., ICCV 2023) https://arxiv.org/abs/2303.12077

## Idea

Map queries predict 100 polylines x 20 points in 3 classes (lane divider, pedestrian crossing, road boundary). The ego query attends agent queries, then map queries (separate positional encoders); an MLP on [ego feature after agent attention, after map attention, ego status, command] outputs the plan. Training-only planning constraints use the predicted vectors.

## Formulation

ego-boundary hinge L_bd = max(0, delta_bd - d), d = distance of each waypoint to the nearest predicted boundary point, delta_bd = 1.0 m; ego-lane direction L_dir = angular difference between the ego step vector and the nearest lane-divider vector within 2.0 m; collision hinge with delta_X 1.5 m, delta_Y 3.0 m. Only predicted instances with class score >= 0.5 enter. Config weights: bound 1.0, direction 0.5, collision 1.0, regression 1.0; map cls focal 2.0, points L1 1.0, direction cosine 0.005; BEV 200 x 200 over 30 x 60 m; 4-frame queue; 3 discrete commands. Code: min over all map points of the Euclidean distance (gradient reaches the selected point); atan2 of gathered points for the yaw term (gradient reaches map points); scores only act as boolean masks (no gradient); `perception_detach` exists in the bound loss, default False, not set in the config.

## Input

multi-view BEV features, ego status (CAN bus), command.

## Output

6-step ego trajectory + map vectors + agent motion.

## Training

map and agent losses of its own heads (matching as in MapTR, from memory) + planning losses; fully end-to-end.

## Differentiable?

partial - planning gradients reach predicted map point coordinates; class scores get none; matching and thresholds are discrete.

## Sensors and data

nuScenes cameras, BEV 60 x 30 m.

## Reported results

nuScenes val avg L2 / collision: VAD-Base 0.72 m / 0.22 %, VAD-Tiny 0.78 / 0.38, UniAD 1.03 / 0.31. Ablation (Table 2): full 0.72 / 0.22; no map interaction 0.86 / 0.29; no agent interaction 0.82 / 0.26; no constraints 0.76 / 0.28; only overstep 0.80 / 0.24; only direction 0.75 / 0.25. Rasterised vs vectorised map 0.74 / 0.39 vs 0.72 / 0.22.

## Compute

VAD-Tiny 59.5 ms (16.8 FPS), VAD-Base 224.3 ms (4.5 FPS) on an RTX 3090.

## Code

https://github.com/hustvl/VAD

## Fit to our project

reuse_idea - effort S - hinge constraints on the nearest vertex / edge (1.0 m, 2.0 m, confidence 0.5) are easy; the boundary term needs road boundaries, which a centre-line graph lacks.

## Needs

ordered map points with a class score; an ego trajectory head.

## Conflicts

Hungarian matching in the original map head; mmcv / mmdet3d stack; ego-status leakage on nuScenes open loop.

## Tried in our experiments

E121 (planner loss through the positions: no gain) is the same idea; VAD's own gain from the constraints is small (L2 0.76 -> 0.72).
