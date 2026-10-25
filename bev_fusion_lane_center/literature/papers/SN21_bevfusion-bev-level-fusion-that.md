# SN21 - BEVFusion (PKU): BEV-level fusion that survives lidar failure

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B, T.*

## Papers

BEVFusion: A Simple and Robust LiDAR-Camera Fusion Framework (Liang et al., NeurIPS 2022) https://arxiv.org/abs/2205.13790 ; code: ADLab-AutoDrive organisation named on the abstract page, https://github.com/ADLab-AutoDrive/BEVFusion (URL not opened)

## Idea

An independent camera stream (Dual-Swin-T, LSS-style projector, spatial-to-channel BEV encoder) and any lidar-to-BEV network are concatenated and fused by a conv plus squeeze-and-excitation gate (dynamic fusion), so the camera stream still works if lidar fails.

## Formulation

images 448 x 800; fusion: concat, 3x3 conv, SE-style sigmoid gate; two-stage training (streams separately, then fusion; lidar stream frozen for CenterPoint / TransFusion-L); robustness fine-tuning with limited-FOV and object-failure lidar augmentations (12 epochs, LR 1e-4).

## Input

images + lidar.

## Output

3D boxes.

## Training

staged; frozen lidar stream during fusion.

## Differentiable?

yes.

## Sensors and data

camera + lidar, nuScenes.

## Reported results

val 67.9 mAP / 71.0 NDS (TransFusion-L stream); test 69.2 / 71.8; lidar FOV failure (-pi/2, pi/2): TransFusion-L 27.8 -> 46.4 mAP.

## Compute

camera projector > 957 ms, ~1.5 s per sample on a V100 (their implementation).

## Code

see papers line.

## Fit to our project

reuse_idea - effort S - staged training with a frozen first stream is the same regime as ours; their robustness recipe (sensor-failure augmentation) matches our synthetic failures.

## Needs

lidar.

## Conflicts

lidar.

## Tried in our experiments

none.
