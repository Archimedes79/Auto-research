# PL45 - Neural Map Prior (global neural map memory updated by a GRU)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S8, S9.*

## Papers

Neural Map Prior for Autonomous Driving (Xiong et al., CVPR 2023) https://arxiv.org/abs/2304.08481

## Idea

A global grid of features in world coordinates (sparse tiles, 0.3 m default; 0.6 m and 1.2 m reduce the gain) is queried with the ego pose; current BEV features attend the retrieved prior (patches 10 x 10, about 3 m; dimension 256); a 2-D conv GRU fuses and writes the result back; tiles are downloaded and uploaded on demand (about 70 % less memory than dense tiles).

## Formulation

current-to-prior attention with separate position embeddings, so the model can lean on the prior at long range.

## Input

current BEV features, retrieved prior features, ego pose.

## Output

improved BEV features and updated prior tile.

## Training

not read in detail.

## Differentiable?

yes.

## Sensors and data

nuScenes.

## Reported results

nuScenes val mIoU (all classes), without -> with prior: HDMapNet 32.73 -> 37.05; LSS 39.78 -> 44.80; BEVFormer 43.01 -> 48.54; VectorMapNet mAP 40.9 -> 44.8; night +7.73, rain +5.92; single traversal +2.30 vs multiple traversals +3.81; Boston re-split 26.35 -> 33.04. Pose-noise robustness not evaluated.

## Compute

not read.

## Code

not found.

## Fit to our project

not_suitable - effort L - a memory prior needs repeated traversals and accurate pose; it does not address the route question.

## Needs

repeated traversals, accurate pose.

## Conflicts

pose assumed known.

## Tried in our experiments

E41 (temporal feature fusion, +0.008 F1) is a short-range cousin.
