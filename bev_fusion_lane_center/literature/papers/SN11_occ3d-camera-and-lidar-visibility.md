# SN11 - Occ3D: camera and lidar visibility masks by ray casting

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (Sec. 3.3.2 and appendix algorithm summary)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: D, E.*

## Papers

Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving (Tian et al., NeurIPS D&B 2023; authors / venue from memory) https://arxiv.org/abs/2304.14365 ; code https://github.com/Tsinghua-MARS-Lab/Occ3D (URL from memory, not opened)

## Idea

A voxel is "observed" for the lidar mask if it holds a return or a lidar ray passes through it; for the camera mask each occupied voxel is joined to the camera origin and the first occupied voxel along each pixel ray is observed, everything behind it and every voxel no pixel ray scans is unobserved. Evaluation counts only observed voxels.

## Formulation

camera mask: for each pixel a virtual point at a large depth, ray from the camera origin to it, traverse voxels, copy the lidar-derived state (occupied / free) else "not observed"; no explicit frustum culling (unscanned voxels stay unobserved); Waymo has a blind spot behind the car. Grid nuScenes [-40, 40] m, 0.4 m, 200 x 200 x 16.

## Input

aggregated lidar + camera poses.

## Output

binary mask per camera view.

## Training

none (label tool); the paper gives no training procedure for the masks (a search summary of another paper says training with visible masks can make a model ignore regions beyond the visible area).

## Differentiable?

no - label tool.

## Sensors and data

lidar + camera calibration, nuScenes / Waymo.

## Reported results

none used.

## Compute

not stated.

## Code

see papers line.

## Fit to our project

reuse_idea - effort M - same logic for lane rays, but it needs a 3D occupancy (lidar) to know occluders; the camera-frustum part can be taken alone.

## Needs

3D occupancy or boxes for occluders.

## Conflicts

lidar-derived (project: no lidar input).

## Tried in our experiments

none.
