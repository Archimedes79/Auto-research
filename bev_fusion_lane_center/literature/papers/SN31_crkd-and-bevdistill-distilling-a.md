# SN31 - CRKD and BEVDistill: distilling a lidar BEV into a camera-radar or camera BEV

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (CRKD), M (BEVDistill)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: T.*

## Papers

CRKD: Enhanced Camera-Radar Object Detection with Cross-modality Knowledge Distillation (Zhao, Song, Skinner; CVPR 2024) https://arxiv.org/abs/2403.19104 , project page https://song-jingyu.github.io/CRKD ; BEVDistill: Cross-Modal BEV Distillation for Multi-View 3D Object Detection (Chen et al., ICLR 2023) https://arxiv.org/abs/2211.09386 , code https://github.com/zehuichen123/BEVDistill (named in a search result)

## Idea

CRKD distils a frozen BEVFusion lidar-camera teacher into a camera-radar student with four losses: cross-stage radar distillation (radar features vs the teacher objectness heat map, L1), mask-scaling feature distillation (L2 on foreground-mask-weighted fused features, masks enlarged for far / fast objects), relation distillation (L1 between cosine affinity matrices), response distillation (quality focal + SmoothL1). BEVDistill distils a frozen lidar detector into BEVFormer-type camera students with a Gaussian-heat-map-weighted L2 on BEV features (dense) plus an InfoNCE-style contrastive loss on matched query features (sparse), because logit matching fails across modalities.

## Formulation

CRKD weights 100 / 10 / 0.25 / 1 / 1 (detection); BEV 180 x 180 over [-54, 54] m; teacher keyframe + 9 sweeps, student keyframe + 6 radar sweeps; 20 epochs, LR 1e-4, 4 x A100. BEVDistill: Gaussian mask sigma 2 around GT centres, loss weights 1.0 / 0.25, BEV 128 x 128, 2 x 24 epochs, 8 x A100.

## Input

teacher and student inputs.

## Output

boxes.

## Training

teacher frozen; regime none / self-distillation; labels only for the detection loss.

## Differentiable?

yes.

## Sensors and data

CRKD: camera + radar student, camera + lidar teacher; BEVDistill: camera student, lidar teacher; nuScenes (BEVDistill also Waymo).

## Reported results

CRKD val mAP / NDS: baseline CR 43.2 / 54.1, student 44.9 / 55.9, CRKD (SwinT) 46.7 / 57.3; test 48.7 / 58.7; gains by range 0-20 m +3.05 mAP, 30-50 m +1.03 mAP; sunny / rainy / night mAP +1.22 / +2.32 / +0.20. BEVDistill val BEVFormer-R50 NDS / mAP 42.3 / 35.2 -> 45.7 / 38.6; BEVFormer-T 48.8 / 38.3 -> 51.5 / 40.7; test BEVDepth 58.9 / 49.1 -> 59.4 / 49.8; ablation dense only 44.7 / 38.0, instance only 43.9 / 37.1.

## Compute

not stated.

## Code

see papers line.

## Fit to our project

not_suitable - effort L - needs a lidar teacher; reuse_idea: radar distillation does not rescue far range (CRKD 30-50 m +1.03 mAP only).

## Needs

a teacher BEV with the same grid.

## Conflicts

lidar teacher.

## Tried in our experiments

none.
