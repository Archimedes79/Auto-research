# SN41 - Horizon3D: range-binned radar-camera detection out to 150 m (TruckScenes)

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (first ~100k of ~105k characters)** (A abstract, M method, C code page). Verified: partly (reader summary; supplement failure cases cut off). Pipeline stage: B, E.*

## Papers

Horizon3D: Sparse Radar-Camera Fusion for Long-Range 3D Perception in Autonomous Driving (arXiv 2026) https://arxiv.org/abs/2606.31096

## Idea

Sparse radar BEV features (VoxelNeXt on 10 accumulated sweeps of six radars) and image-based Gaussian primitives are fused in a sparse BEV; radar keypoints (top 1024 BEV cells by a small MLP score) initialise the Gaussians; a dual-path temporal fusion is added. Evaluated on TruckScenes with range bins up to 150 m.

## Formulation

three modules: keypoint-guided Gaussian initialisation, object-centric sparse fusion, dual-path temporal fusion; per-point radar features: position, RCS, radial velocity; camera depth range not stated.

## Input

images + radar sweeps.

## Output

boxes.

## Training

not read in detail.

## Differentiable?

yes (top-K selection of keypoint cells is a hard operation).

## Sensors and data

camera + radar, TruckScenes (747 scenes, evaluation range 150 m), not nuScenes.

## Reported results

validation Table 9, mAP / NDS by range for BEVFusion (the compared camera-radar baseline): 0-25 m 28.8 / 35.7, 25-50 m 16.8 / 25.2, 50-100 m 11.7 / 20.1, 100-150 m 7.4 / 13.9; Horizon3D: 36.5 / 43.0, 23.3 / 37.8, 17.0 / 32.7, 8.9 / 18.4. Overall val (ResNet-50): Horizon3D 37.4 NDS / 23.6 mAP vs BEVFusion 30.4 / 18.2. Radar-only baseline 12.8 mAP / 27.1 NDS without range breakdown; no ablation of the radar contribution by range.

## Compute

8.5 FPS, 3.2 GB (BEVFusion 4.2 FPS, 6.52 GB).

## Code

not checked.

## Fit to our project

reuse_idea - effort S - a range-binned reference showing that even the best radar-camera detector loses about 53 % of its mAP between 0-25 m and 50-100 m (36.5 -> 17.0); the task is vehicles, not lanes, and the dataset is trucks; use only as a shape of the range curve.

## Needs

sparse BEV; images.

## Conflicts

different dataset and task.

## Tried in our experiments

none.
