# SN12 - Visible versus occluded BEV cells: how much of the BEV is unobserved and what happens there

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (visibility sections and Table 4)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: D, E, X.*

## Papers

Parametric Depth Based Feature Representation Learning for Object Detection and Segmentation in Bird's Eye View (arXiv 2307.04106; authors / venue not read) https://arxiv.org/abs/2307.04106

## Idea

Per-voxel visibility is derived from a predicted (or lidar-completed) parametric depth distribution along each camera ray: occlusion B(d) = F(d) - 0.5 exp(-mu / b) for a Laplacian depth model, visibility V = 1 - B, max over z for the BEV; thresholds tau_vis / tau_occ (values not given) split the BEV into visible and occluded regions for evaluation.

## Formulation

ground-truth visibility from lidar points projected to the images and a depth-completion network; BEV 200 x 200 at 0.5 m (intermediate volume 400 x 400 x 12).

## Input

depth distribution per ray.

## Output

soft visibility map in [0, 1].

## Training

used as feature weighting; the evaluation split is what matters here.

## Differentiable?

yes (soft visibility); the GT version is offline.

## Sensors and data

camera (+ lidar for GT), nuScenes.

## Reported results

nuScenes val (Table 4): visible cells 66.9 %, occluded 33.1 %. Lane IoU visible / occluded: LSS 23.1 / 6.5, M2BEV 39.8 / 12.8, proposed 41.9 / 12.4; road IoU visible / occluded: 79.4 / 47.1, 82.9 / 48.9, 84.8 / 48.9; overall lane 19.9 / 36.1 / 36.5.

## Compute

not stated.

## Code

not checked.

## Fit to our project

reuse_idea - effort S - quantifies how weak lane evidence is on unobserved cells (about a third of the BEV); a soft visibility V(x, y) in [0, 1] is the natural multiplier for survival terms.

## Needs

a depth distribution per camera ray or a lidar depth map (for GT).

## Conflicts

depth-based, not calibration-only.

## Tried in our experiments

none.
