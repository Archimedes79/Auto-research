# SN06 - Simple-BEV: parameter-free bilinear lifting + rasterised radar

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B.*

## Papers

Simple-BEV: What Really Matters for Multi-Sensor BEV Perception? (Harley et al., ICRA 2023; authors / venue from memory) https://arxiv.org/abs/2206.07959

## Idea

A fixed 3D voxel grid is projected into every camera and bilinearly samples ResNet-101 features (no learned depth), the volume is folded into BEV channels, radar returns are rasterised into the same grid, and a small ResNet-18 style decoder predicts vehicle segmentation. Message: batch size, resolution and radar matter more than the lifting method.

## Formulation

grid 100 x 100 m x 10 m, 200 x 8 x 200 voxels (0.5 x 1.25 x 0.5 m), reference = front camera; radar: each point to the nearest X-Z cell, 15 non-position fields as channels (R = 15), concatenated with RGB BEV features and reduced by a 3x3 conv; sweeps aligned to the current frame (3 sweeps 55.7 IoU, 1 sweep 53.1; nuScenes outlier filter off 55.7 vs 53.7; binary occupancy only 55.0); decoder 3 ResNet-18 blocks, segmentation CE + centerness + offset L1 with uncertainty weights; 25,000 iterations AdamW 5e-4 one-cycle.

## Input

6 images (448 x 800) + radar sweeps.

## Output

BEV vehicle map (200 x 200).

## Training

regime matching_free; larger batch gives about +14 IoU (2 -> 40); crop / resize augmentation 45.8 -> 47.4.

## Differentiable?

yes.

## Sensors and data

camera, radar, optionally lidar occupancy; nuScenes vehicle superclass.

## Reported results

val vehicle IoU: C 47.4, C+R 55.7, C+L 60.8; lifting: bilinear 47.4, deformable attention 46.5, depth splatting 44.4. Radar called "sparse and noisy", single sweep too sparse.

## Compute

~42 M parameters (bilinear), 83 ms at 448 x 800.

## Code

https://github.com/aharley/simple_bev (URL from memory, not opened).

## Fit to our project

reuse_idea - effort S - confirms that the radar value comes from several aligned sweeps and from keeping the metadata fields (GaussianCaR does both); batch-size and resolution findings explain why heads on frozen features cannot repair a weak encoder.

## Needs

radar fields incl. metadata; several sweeps aligned to the current frame.

## Conflicts

vehicle only, no map layers evaluated.

## Tried in our experiments

none.
