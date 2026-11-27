# PL11 - UniAD (planner reads BEV + ego query; map queries reach it only through MotionFormer)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M (tail of the page not read)** (A abstract, M method, C code page). Verified: partly (appendix cut). Pipeline stage: S10.*

## Papers

Planning-oriented Autonomous Driving (Hu et al., CVPR 2023) https://arxiv.org/abs/2212.10156

## Idea

MapFormer produces map queries (300 thing queries for lanes / dividers / crossings with Hungarian matching + 1 stuff query for the drivable area) and a panoptic map; only the map queries go on to MotionFormer, where agent and map queries interact; the planner gets the ego query from MotionFormer, three learned command embeddings and the BEV feature, cross-attends the BEV in 3 decoder layers and regresses 6 steps (3 s). Inference-only occupancy optimisation picks the candidate minimising coordinate deviation + lambda_obs x Gaussian occupancy penalty (d = 5, sigma = 1, lambda_coord 1, lambda_obs 5).

## Formulation

BEV segmentation map not passed to the planner; max-pool over command-embedded modalities builds the plan query.

## Input

multi-view BEV features, command.

## Output

3 s plan; panoptic map; motion and occupancy.

## Training

stage 1 tracking + mapping about 6 epochs; stage 2 end-to-end 20 epochs; image backbone gradient stopped in stage 1, image backbone and BEV encoder frozen in stage 2 (as read on the page); no other detaching described.

## Differentiable?

partial - gradients reach map queries only through MotionFormer attention; matching discrete.

## Sensors and data

nuScenes cameras.

## Reported results

Table 2 (avg L2 / collision): no intermediate tasks (row 10) 1.131 m / 0.773 %; full UniAD (row 12) 1.004 / 0.430; the planning table does not isolate the map module; Table 10: occupancy optimisation lowers collision but raises L2 (0.44 -> 0.54 at 1 s); minADE 0.751 -> 0.736 when mapping is added to tracking + motion.

## Compute

125.0 M parameters, 1.8 FPS on an A100 (numbers from the SparseDrive comparison).

## Code

https://github.com/OpenDriveLab/UniAD (v2.0 branch: mmdet3d 1.x, torch 2.x per README)

## Fit to our project

baseline - effort L - the planner design (BEV cross-attention + command) is plain, the mmcv stack is the obstacle.

## Needs

map queries with positions; BEV feature.

## Conflicts

Hungarian matching; frozen BEV encoder in stage 2.

## Tried in our experiments

E31 / E32 (matched sparse decoder dropped).
