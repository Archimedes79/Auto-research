# PL34 - ST-P3 (dense BEV cost-volume planner on a 200 x 200, 0.5 m grid)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10.*

## Papers

ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning (Hu et al., ECCV 2022) https://arxiv.org/abs/2207.07601

## Idea

Candidate trajectories sampled with a kinematic bicycle model are gated by the command and scored by f = f_o (hand-designed safety: overlap with occupied cells and road elements, lane-line distance) + f_v (learned cost volume from the prediction decoder, clipped) + f_r (comfort and progress), weights learned; a GRU refines the best candidate starting from the target point.

## Formulation

planning loss = max-margin over samples + L1 to the expert; BEV 200 x 200 cells at 0.5 m (100 m x 100 m) on nuScenes, identical to our grid; heads include drivable area and lanes.

## Input

BEV features, command, target point.

## Output

refined trajectory (3 s).

## Training

end-to-end; gradients appear to reach the shared network through the cost-volume head (inferred).

## Differentiable?

partial - candidate sampling and selection.

## Sensors and data

nuScenes cameras; CARLA.

## Reported results

nuScenes L2 1.33 / 2.11 / 2.90 m at 1 / 2 / 3 s (the table label "mm" is a unit error), collision 0.23 / 0.62 / 1.27 %; CARLA Town05 Short DS 55.14, Long 11.45.

## Compute

not read.

## Code

https://github.com/OpenPerceptionX/ST-P3

## Fit to our project

baseline - effort M - a dense-raster planner that can read the cached 39-channel features directly; a raster control for the graph consumer.

## Needs

BEV channels for drivable area and lane lines.

## Conflicts

sampling and argmin selection.

## Tried in our experiments

none (E117 is graph-based).
