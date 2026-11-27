# SN07 - RESAR-BEV: camera-radar map segmentation with progressive residual refinement

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: partly (reader flagged text-table mismatches; table values used). Pipeline stage: B.*

## Papers

RESAR-BEV: An Explainable Progressive Residual Autoregressive Approach for Camera-Radar Fusion in BEV Segmentation (arXiv 2025) https://arxiv.org/abs/2505.06515

## Idea

A coarse BEV map from a "Driver" transformer is refined by "Modifier" transformers that predict residuals at several semantic scales (road topology -> lane boundaries) against a ground truth decomposed offline into multi-scale token maps.

## Formulation

6 cameras 448 x 672, ResNet-101 (1/4, 1/8, 1/16); radar voxel features of 6 values (x, y, z, vx, vy, RCS), up to 10 points per voxel, 6 sweeps aligned by sensor -> global -> ego transforms; BEV +-50 m, 200 x 200, 0.5 m; vertical +-5 m in 50 bins -> 10; losses: residual token-map losses weighted 2 / 3 / 4 / 5 + Dice weighted 10.

## Input

6 images + radar sweeps.

## Output

7 classes trained (drivable, ped crossing, walkway, stop line, road divider, lane divider, vehicle); Table I reports three.

## Training

regime matching_free; nuScenes.

## Differentiable?

yes.

## Sensors and data

camera + radar, nuScenes, 100 x 100 m at 0.5 m.

## Reported results

Table I (their re-evaluation) drivable / vehicle / lane divider IoU: RESAR-Standard 83.53 / 56.87 / 44.43; BEVCar 79.40 / 54.17 / 43.07; CRN 81.34 / 54.80 / 39.69 (BEVCar's own paper has 83.3 / 58.4 / 45.3 - cross-paper tables are not protocol-matched). Lane divider sunny 45.9, rainy 44.1. Limitations named: multi-lane intersections, rain ground reflections, long range, sensor failure. 31.9 M parameters, 14.6 FPS (hardware not stated).

## Compute

31.9 M parameters.

## Code

not checked.

## Fit to our project

reuse_idea - effort M - multi-scale residual decomposition of the target is an idea for refining thin structures, but it needs the image path.

## Needs

images, radar voxels.

## Conflicts

end-to-end encoder, autoregressive refinement stages.

## Tried in our experiments

none.
