# SN20 - BEVFusion (MIT): unified BEV with map-segmentation head

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B.*

## Papers

BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation (Liu et al., ICRA 2023; from memory) https://arxiv.org/abs/2205.13542 ; code https://github.com/mit-han-lab/bevfusion (URL from memory, not opened)

## Idea

Camera features are lifted LSS-style (depth distribution scattered along rays), pooled to BEV with a precomputed, interval-reduction BEV pooling kernel (about 40 x faster), fused with VoxelNet lidar BEV by an elementwise operator + conv BEV encoder; one head per task.

## Formulation

Swin-T + FPN at 1/8 of 256 x 704; depth [1, 60] m, 0.5 m step, D = 118; BEV pooling cell example 0.4 m; lidar voxel 0.075 m (detection) / 0.1 m (segmentation); map segmentation = one binary map per class (6: drivable, ped crossing, walkway, stop line, carpark, divider) with focal loss, evaluated on [-50, 50] m^2; grid-sample converts the fused BEV to the range / resolution of each head.

## Input

6 images + lidar.

## Output

3D boxes; 6 BEV maps.

## Training

not stated in the page read.

## Differentiable?

yes.

## Sensors and data

camera + lidar, nuScenes, Waymo.

## Reported results

map segmentation val mean IoU: camera-only 56.6, lidar-only CenterPoint 48.6, fusion 62.7; camera-only beats lidar-only on maps while lidar beats camera by ~20 mAP on detection.

## Compute

119.2 ms latency (detection config).

## Code

see papers line.

## Fit to our project

reuse_idea - effort S - map segmentation depends on the camera path; "grid-sample the fused BEV to each head's range / resolution" is exactly our bilinear `grid_sample` contract.

## Needs

lidar for the published fusion.

## Conflicts

lidar.

## Tried in our experiments

none.
