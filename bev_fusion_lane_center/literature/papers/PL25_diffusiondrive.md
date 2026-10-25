# PL25 - DiffusionDrive (truncated diffusion from anchors; map only as auxiliary segmentation on NAVSIM)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10, S4.*

## Papers

DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving (Liao et al., CVPR 2025) https://arxiv.org/abs/2411.15139

## Idea

Noise is added to K-means trajectory anchors (20 on NAVSIM, 18 on nuScenes), truncated at 50 of 1000 steps; 2 DDIM steps at inference; a 2-layer cascade decoder (shared across steps) with deformable BEV / PV cross-attention, agent cross-attention, timestep modulation; outputs confidence + offset. NAVSIM has no map query (agent cross-attention only; BEV semantic segmentation only as an auxiliary task); on nuScenes (built on SparseDrive) object and map queries are used.

## Formulation

tau_k^i = sqrt(abar_i) a_k + sqrt(1 - abar_i) eps; loss = sum_k y_k L1(tau_hat_k, tau_gt) + lambda BCE(s_k, y_k), positive = anchor closest to the GT (winner-take-all-like nearest-anchor assignment); detection + BEV segmentation auxiliary losses.

## Input

BEV / PV features, agent (and map) queries.

## Output

top-1 scored trajectory.

## Training

loss above; ImageNet-initialised ResNet-34 on NAVSIM; nuScenes planner trained 10 epochs from stage-1 perception weights; gradient to perception not stated.

## Differentiable?

partial - nearest-anchor assignment and top-1 selection are hard.

## Sensors and data

NAVSIM camera + lidar; nuScenes cameras.

## Reported results

NAVSIM navtest PDMS 88.1 (NC 98.2, DAC 96.2, TTC 94.7, Comf 100, EP 82.2; TransFuser 84.0); nuScenes avg L2 0.57 m (1 s 0.27, 2 s 0.54, 3 s 0.90), collision 0.08 %.

## Compute

45 FPS on NAVSIM, 8.2 FPS on nuScenes (RTX 4090); about 60 M parameters (NAVSIM model).

## Code

https://github.com/hustvl/DiffusionDrive

## Fit to our project

reuse_idea - effort M - anchor-truncated denoising could refine E117's path, with anchors taken from route paths.

## Needs

BEV features, agent queries.

## Conflicts

nearest-anchor assignment is an argmin.

## Tried in our experiments

none.
