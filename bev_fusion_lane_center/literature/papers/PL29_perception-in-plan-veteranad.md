# PL29 - Perception in Plan / VeteranAD (anchor-guided perception, dense BEV map)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10, S2.*

## Papers

Perception in Plan: Coupled Perception and Planning for End-to-End Autonomous Driving (arXiv 2508.11488, AAAI 2026; authors not read) https://arxiv.org/abs/2508.11488

## Idea

20 K-means planning anchors initialise trajectory queries; per future step the guiding points of the anchors are projected into image and BEV coordinates and used as reference points for position-guided cross-attention to image / BEV features (planning-oriented sampling of perception features); autoregressive near-to-far refinement predicts offsets; classification score only at the last step. Map = dense BEV segmentation (cross-entropy), no vector or lane queries.

## Formulation

P_ft = Delta P_ft + P_t per step; motion-aware layer norm moves queries across steps; agent cross-attention weighted by distance to the guiding points.

## Input

three front images, LSS BEV.

## Output

scored trajectories, BEV segmentation, boxes.

## Training

one total loss over BEV, agent, regression and classification terms; AdamW lr 2e-4, batch 32, 16 epochs, 8 x RTX 3090; gradients appear to reach perception (inferred, not stated).

## Differentiable?

yes.

## Sensors and data

NAVSIM, Bench2Drive; ResNet-34.

## Reported results

NAVSIM navtest PDMS 90.2 (NC 99.1, DAC 98.3, TTC 96.1, EP 83.1); Bench2Drive open-loop L2 0.60, driving score 64.22, success 33.85 % (DriveTransformer has higher success 35.01).

## Compute

see training.

## Code

https://github.com/fudan-zvg/VeteranAD

## Fit to our project

reuse_idea - effort M - sampling BEV features along the planned path is the same principle as our rays / fans, applied to planning.

## Needs

BEV feature pyramid (we have it).

## Conflicts

the image branch needs image-space projection.

## Tried in our experiments

none.
