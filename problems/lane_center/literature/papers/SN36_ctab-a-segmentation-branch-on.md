# SN36 - CTAB: a segmentation branch on RCBEVDet's coarse 0.8 m BEV upsampled to 0.5 m

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B.*

## Papers

Radar-Camera BEV Multi-Task Learning with Cross-Task Attention Bridge for Joint 3D Detection and Segmentation (arXiv 2026) https://arxiv.org/abs/2604.12918

## Idea

Adds a BEV segmentation branch (instance-norm decoder, learnable upsampler) and a bidirectional detection-segmentation attention bridge (0.58 M parameters) to RCBEVDet; the shared BEV has 128 x 128 cells at 0.8 m ([-51.2, 51.2] m) and segmentation is upsampled to 200 x 200 at 0.5 m.

## Formulation

seven classes (drivable, carpark, ped crossing, walkway, stop line, divider = road + lane dividers merged, vehicle); homoscedastic uncertainty loss weights; radar via RadarBEVNet (sweeps and feature dimensions not given).

## Input

6 images + radar.

## Output

boxes + 7 BEV maps.

## Training

regime matching_free; joint, nuScenes 28,130 / 6,019.

## Differentiable?

yes.

## Sensors and data

camera + radar, nuScenes.

## Reported results

val IoU baseline -> with CTAB: drivable 72.7 -> 73.3, divider 33.7 -> 34.4, vehicle 49.8 -> 49.8, mIoU-7 42.9 -> 43.5 (Table III); detection NDS 55.7 -> 55.6. Authors: features at 0.8 m cannot recover detail lost in the BEV projection; class sets differ across methods (cross-paper comparison is unsafe).

## Compute

+0.58 M parameters.

## Code

not checked.

## Fit to our project

reuse_idea - effort S - a coarse BEV upsampled to a finer grid caps thin structures (divider 34 here vs 45 - 50 for native 0.5 m camera-radar models in other papers; not protocol-matched); supports keeping the native 0.5 m cache rather than re-sampling.

## Needs

images.

## Conflicts

detection-oriented coarse BEV.

## Tried in our experiments

none.
