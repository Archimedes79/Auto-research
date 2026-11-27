# TP20 - FASTopoWM (fast-slow lane segment topology with latent world models)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S8 T.*

## Papers

FASTopoWM: Fast-Slow Lane Segment Topology Reasoning with Latent World Models (arXiv 2507.23325) https://arxiv.org/abs/2507.23325

## Idea

A slow branch propagates the previous frame's queries and BEV features through learned latent world models conditioned on the relative pose; a fast single-frame branch shares the same layers and heads, so the system falls back to single-frame output when pose is missing.

## Formulation

Q~^(T-1) = MLP([Q^(T-1), Psi]), F~^(T-1) = MLP([F^(T-1), Psi]); Q~^T = QueryWorldModel(Q~^(T-1)) (transformer blocks), F~^T = BEVWorldModel(F~^(T-1)) (temporal self-attention, average-pooled 200 x 100 -> 100 x 50 and upsampled), 2 layers each; stream BEV fused by GRU. Losses: L_bev = ||F~^T - F^T||_2 (target = current extracted feature, no labels), L_query = coordinate L1 + class focal / CE + mask CE / dice with GT warped from T-1; L_slow = L_ls + 0.3 L_latent, L_fast = L_ls, total = both; Hungarian on both branches. 30 % of 200 queries reserved for propagation.

## Input

BEV 200 x 100, 10 points per line.

## Output

lane segments + adjacency (head not described).

## Training

24 epochs, first 12 single-frame, AdamW 2e-4, V100.

## Differentiable?

partial - Hungarian.

## Sensors and data

camera; OpenLane-V2 A (lane segment) and B (centerline).

## Reported results

subset A (temporal mAP / TOP_lsls): baseline stream 34.0 / 28.1, full 37.4 / 29.6; single-frame branch 29.8 / 23.4 -> 34.1 / 26.6. Action condition (temporal): none 36.5 / 28.5, trajectory 31.2 / 24.5, relative pose 37.4 / 29.6. World-model architecture: linear 35.2, MLPs 36.2, transformer 37.4 mAP. Subset B centerline OLS: LaneSegNet 38.7, fast 41.8, slow 46.3 (TOP_ll 33.0). With GeoDist: TOP_lsls 31.6 (slow).

## Compute

slow 11.4 FPS, fast 14.0 FPS.

## Code

https://github.com/YimingYang23/FASTopoWM

## Fit to our project

reuse_idea - effort M - two lessons: pose conditioning must use the RELATIVE pose (trajectory conditioning hurts by 5 mAP), and a feature-prediction loss against the next frame's own features needs no labels.

## Needs

relative pose; a trainable BEV feature stream.

## Conflicts

pose-noise robustness untested (we have synthetic pose jumps); BEV world model needs pooling for memory.

## Tried in our experiments

E39 / E41 (temporal feature fusion +0.008 F1); pose jumps are a stated failure mode in our task.
