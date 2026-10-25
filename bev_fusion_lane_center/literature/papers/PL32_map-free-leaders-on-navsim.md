# PL32 - Map-free leaders on NAVSIM 2025-26 (RAP, DriveZero) and the privileged-teacher pattern

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10, T.*

## Papers

RAP: 3D Rasterization Augmented End-to-End Planning (arXiv 2510.04333; authors not read) https://arxiv.org/abs/2510.04333 ; DriveZero: End-to-End Driving Beyond Human Demonstrations (arXiv 2609.06055, Xiaomi) https://arxiv.org/abs/2609.06055

## Idea

RAP renders annotated lane polylines and cuboids as lightweight 3-D rasters to create recovery-oriented training views and aligns raster and real features (no map input to the planner; maps only generate training data). DriveZero trains a privileged teacher (5.7 M parameters) on a local vector map of up to 256 tokens (traffic-light states attached to the governed lane elements), up to 96 agents and two route-based goal points (near and far anchor along the route, look-ahead scaling with speed), then distils into a camera-only student whose only route signal is a command embedded in one ego token; the student has no map head.

## Formulation

student loss = winner-takes-all L1 over 64 proposals against the teacher rollout + BCE on six PDM-style sub-scores (weights not given); teacher: ego-to-agent cross-attention (2 layers), ego-to-map cross-attention (1 layer), 256-d tokens, Beta distributions over jerk and steering rate.

## Input

RAP cameras + ego history (route for WOD-E2E); DriveZero student 4 cameras, ego kinematics, command.

## Output

scored trajectories.

## Training

RAP synthetic-to-real alignment; DriveZero teacher by reinforcement learning (96 GPUs, about 21 h) then distillation; regime: rl + distillation.

## Differentiable?

n/a (distillation).

## Sensors and data

NAVSIM, Waymo, Bench2Drive (RAP); NAVSIM, HUGSIM (DriveZero).

## Reported results

NAVSIM v1 navtest PDMS: RAP-DINO 93.8 (NC 99.1, DAC 98.9, TTC 96.7, EP 90.3), DriveZero 94.8 (95.3 with extra simulated data), teacher with GT inputs 95.8, human 94.8; navhard EPDMS RAP-DINO 36.9 (the pseudo-simulation paper lists 39.6), DriveZero 51.5 / 57.1, PDM-Closed 56.6.

## Compute

RAP-DINO about 888 M parameters; DriveVFM pre-training 600 K + 200 K iterations at global batch 2,048.

## Code

none (project pages https://alan-lanfeng.github.io/RAP/ , https://xiaomiautol3.github.io/DriveZero)

## Fit to our project

reuse_idea - effort M - a consumer trained on GT graph + route as teacher, usable to distil or to score.

## Needs

GT graph tokens for a teacher.

## Conflicts

a student without a map head makes graph quality invisible.

## Tried in our experiments

none.
