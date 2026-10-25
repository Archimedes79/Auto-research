# PL47 - LFP (lane-level planning flags on double-edge lane queries; CARLA)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10.*

## Papers

LFP: Efficient and Accurate End-to-End Lane-Level Planning via Camera-LiDAR Fusion (arXiv 2409.14170; authors not read) https://arxiv.org/abs/2409.14170

## Idea

N_d double-edge lane structures (left / right edges of N_p/2 3-D points, intersection and direction flags); heads predict boundary points, occupancy and a planning flag per point; the planned path is the midpoint of left / right points with flag 1. No explicit topology between lanes; the route command handling is not described.

## Formulation

loss weights 3:2:1:3:4:5:1:0.1 over ROI, intersection, direction, occupancy, planning, edge, speed, signal terms; lane ROI selection by nearest pillars is a hard step.

## Input

camera + lidar lane queries.

## Output

lane flags, path, speed.

## Training

joint loss; whether planning gradients reach lane perception is not stated.

## Differentiable?

partial.

## Sensors and data

CARLA (about 126 K frames, 8 maps).

## Reported results

Town05 Long DS 91.1, RC 99.6, IS 0.92; Town05 Short DS 96.7; 19.27 FPS (51.90 ms).

## Compute

see results.

## Code

none found.

## Fit to our project

reuse_idea - effort S - a per-point "on the plan" flag on perceived lane points is a per-vertex route flag, pairing with our route probability.

## Needs

lane queries with points.

## Conflicts

CARLA only.

## Tried in our experiments

E102-E104.
