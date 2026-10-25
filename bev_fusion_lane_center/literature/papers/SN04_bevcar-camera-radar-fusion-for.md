# SN04 - BEVCar: camera-radar fusion for BEV map and object segmentation

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B.*

## Papers

BEVCar: Camera-Radar Fusion for BEV Map and Object Segmentation (Schramm et al., IROS 2024) https://arxiv.org/abs/2403.11761

## Idea

A frozen DINOv2 ViT-B/14 with a learnable adapter gives multi-scale image features; a voxel volume centred on the front camera is filled by projecting image features along camera rays; sparse radar points initialise deformable-attention queries so that radar guides the image lifting ("radar-aided lifting"); six transformer blocks and a ResNet-18 BEV encoder produce the BEV map.

## Formulation

images 448 x 896; radar: 5 sweeps (following the Simple-BEV code), features x, y, z, uncompensated vx, vy, RCS; points grouped in voxels matching the BEV grid, FCN + max pooling per voxel, height compression -> f_rad; fusion Q^F = f_rad + position + BEV query, deformable attention onto the lifted image features; grid 100 x 100 m, 200 x 200 cells, 8 height bins up to 10 m, reference frame = front camera; head = two conv layers + 1x1 conv with (M + 1) channels. Map loss: alpha-balanced focal (alpha 0.25, gamma 3); vehicle: BCE.

## Input

6 images (448 x 896) + radar sweeps.

## Output

(1 + 7) x 200 x 200 maps (vehicle + drivable area, carpark, ped crossing, walkway, stop line, road divider, lane divider).

## Training

regime matching_free; nuScenes official split (28,130 train / 6,019 val samples); epochs, optimiser, LR, batch not stated; backbone frozen (DINOv2) + trained adapter.

## Differentiable?

yes.

## Sensors and data

camera + radar, nuScenes, 100 x 100 m at 0.5 m.

## Reported results

val IoU (Table I), camera-only ViT-B/14 -> camera+radar ViT-B/14: vehicle 48.8 -> 58.4, drivable area 81.1 -> 83.3, lane divider 40.6 -> 45.3, map mean 50.5 -> 54.3; Simple-BEV++ (ViT) 54.5 / 81.2 / 40.4. Range (Table III, vehicle IoU): 0-20 m 74.0 vs 68.7 camera-only, 20-35 m 56.5 vs 45.8, 35-50 m 41.0 vs 29.1. Map IoU day / rain / night (Table IV): 57.3 / 48.8 / 42.4 (camera-only 53.3 / 45.1 / 39.2). Ablation (Table II): radar-driven lifting and fusion +4.2 map mIoU over parameter-free lifting. Runtime A100 forward: 382 ms (ViT-B/14), 137 ms (ResNet-101). Authors: radar-only not evaluated, sensor failure not evaluated.

## Compute

parameters not stated; 2.6 FPS (ViT variant).

## Code

https://github.com/robot-learning-freiburg/BEVCar (URL from memory, not opened).

## Fit to our project

baseline - effort L - cannot run on a cache (needs images, deformable attention, DINOv2); evidence that radar adds about +4.7 lane-divider IoU and that map IoU drops in rain / night even with radar.

## Needs

image features (frozen DINOv2 + adapter), radar voxel encoder.

## Conflicts

queries and lifting tied to the image path; not usable on a cached BEV raster.

## Tried in our experiments

none (foundation choice E27 compared CRN and GaussianCaR only).
