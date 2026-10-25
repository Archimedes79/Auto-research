# PL49 - BevAD study: what matters in end-to-end planners (BEV compression, path + speed outputs, commands as weak conditioning)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M (page cut at 100k characters)** (A abstract, M method, C code page). Verified: partly. Pipeline stage: S10, T.*

## Papers

What Matters for Scalable and Robust Learning in End-to-End Driving Planners? (arXiv 2603.15185; authors not read) project page https://dmholtz.github.io/bevad/ (card added last, hence the out-of-order number)

## Idea

Systematic study on Bench2Drive (CARLA, 220 short routes) and NAVSIM: (a) a high-resolution BEV (100 x 100) lets the planner latch onto spurious landmarks (causal confusion); masking 20 % of the left / right BEV cells and patchifying into scene tokens (patch size 4 best) removes this; (b) disentangled path + speed outputs beat entangled trajectories; (c) diffusion planners scale better; (d) navigation commands are weak conditioning (missed lane changes, route deviations), target points are suggested but not tested. Map-segmentation masks gave no significant difference from simple masking; online-map auxiliary supervision was pruned.

## Formulation

tokenizer ablation (Bench2Drive driving score): no mask, p = 1: 66.86 (success rate 36.36); mask p = 1: 72.40; p = 2: 74.98; p = 4: 82.62 (success rate 57.43); p = 5: 66.44. Planning representation success rate: point estimator trajectory 51.7, point estimator path + speed 57.4, diffusion trajectory 56.2, diffusion path + speed 59.4.

## Input

6 cameras -> BEV (100 x 100) -> masked, patchified tokens; command, ego state.

## Output

path + speed (disentangled) or trajectory.

## Training

imitation; point-estimator and diffusion variants.

## Differentiable?

yes.

## Sensors and data

Bench2Drive (CARLA), NAVSIM.

## Reported results

BevAD-S 80.63 DS / 55.30 SR, BevAD-M 88.11 / 72.73 on Bench2Drive; NAVSIM PDMS 86.6 vs ParaDrive 84.0 and UniAD 83.4 (85.9 without masking, 85.6 without patchifying). No route-format or map-quality ablation.

## Compute

not read.

## Code

project page only (no repository named)

## Fit to our project

reuse_idea - effort S - warns against feeding a dense raster to the consumer; our graph tokens are already a compressed form, and the route as a channel rather than only a command is the stated weak point.

## Needs

BEV tokenisation (masking + patchify) if a raster is used.

## Conflicts

CARLA-based evidence only.

## Tried in our experiments

none.
