# SN30 - OccFeat: pretraining a BEV network and evaluating map segmentation at 1 / 10 / 100 % labels

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: T.*

## Papers

OccFeat: Self-supervised Occupancy Feature Prediction for Pretraining BEV Segmentation Networks (Sirko-Galouchenko et al., CVPRW 2024; from memory) https://arxiv.org/abs/2404.14027

## Idea

Pretrain a camera BEV network to predict lidar-derived occupancy and DINOv2 features at occupied voxels, then fine-tune on vehicle and map segmentation; the gain is large with few labels and small with all labels.

## Formulation

L = L_occ + 0.01 L_feat (negative cosine to DINOv2 ViT-S/14 features); BEV 200 x 200 over 50 m; Simple-BEV and BEVFormer; 50 epochs.

## Input

images (+ lidar for the pretext).

## Output

BEV maps.

## Training

regime self_supervised pretraining, then full fine-tuning (no frozen / probe experiment).

## Differentiable?

yes.

## Sensors and data

camera, nuScenes.

## Reported results

aggregated map IoU (Simple-BEV EN-B0) none / OccFeat at 1 %: 20.1 / 25.4; 10 %: 28.1 / 33.7; 100 %: 45.3 / 47.3 (no per-layer numbers).

## Compute

not stated.

## Code

not checked.

## Fit to our project

reuse_idea - effort S - label-efficiency evidence for pretrained BEV features; ours is pretrained for the same layers, so the low-data regime of our 212 training scenes should favour frozen features.

## Needs

nothing for us.

## Conflicts

no frozen-encoder result.

## Tried in our experiments

none.
