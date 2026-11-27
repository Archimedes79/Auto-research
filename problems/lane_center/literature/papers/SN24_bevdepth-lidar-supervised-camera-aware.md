# SN24 - BEVDepth (and the BEVDet family): lidar-supervised, camera-aware depth for LSS

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (BEVDepth), M (BEVDet)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B.*

## Papers

BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection (Li et al., AAAI 2023; from memory) https://arxiv.org/abs/2206.10092 ; BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View (Huang et al., 2021; title from memory) https://arxiv.org/abs/2112.11790 ; code https://github.com/Megvii-BaseDetection/BEVDepth , https://github.com/HuangJunJie2017/BEVDet (URLs from memory, not opened)

## Idea

Explicit depth supervision from projected lidar points, a camera-parameter-aware depth module (intrinsics / extrinsics through an MLP into an SE gate), a depth refinement conv, a CUDA voxel pooling 80 x faster than the cumsum trick, multi-frame BEV concatenation. The camera-radar detectors here (CRN, RCBEVDet, HyDRa) start from this stream.

## Formulation

BEVDepth: BCE depth loss (L1 changes little); ResNet-50 256 x 704, AdamW 2e-4, batch 64; depth bin range and BEV grid not stated in the page read. BEVDet: four modules (backbone + neck, LSS view transformer, ResNet-style BEV encoder with FPN-LSS, first-stage CenterPoint head); ROI 51.2 m at 0.8 m, depth [1, 60] m; image-view augmentation (IDA) and BEV-space augmentation (BDA: flip, rotation within +-22.5 deg, scale 0.95 - 1.05) are decoupled by the view transformer and IDA helps only together with BDA; Scale-NMS (class-specific box enlargement before NMS); AdamW 2e-4, batch 64, 20 epochs, CBGS, MMDetection3D.

## Input

images (+ lidar for training).

## Output

boxes.

## Training

regime matching_free (heatmap detection head); depth supervised by lidar.

## Differentiable?

yes.

## Sensors and data

camera (lidar at training), nuScenes.

## Reported results

ablation val (mAP / NDS): baseline 0.282 / 0.327, + depth loss 0.304 / 0.344, + camera awareness 0.314 / 0.357, + refinement 0.322 / 0.367, + multi-frame 0.330 / 0.442; val R50 0.351 / 0.475, R101 0.412 / 0.535. Replacing learned depth by GT depth gives nearly +20 % mAP / NDS. BEVDet val (Table 2): Tiny 31.2 mAP / 39.2 NDS at 704 x 256, 15.6 FPS; Base 39.3 / 47.2 at 1600 x 640, 1.9 FPS; test 42.2 / 48.2; without augmentation Tiny peaks at 23.0 mAP and ends at 17.4 (overfits), with IDA + BDA 31.2; attribute error is worse than image-view methods (AAE 0.247 vs 0.170).

## Compute

pooling 80 x faster; training 5 -> 1.5 days.

## Code

see papers line.

## Fit to our project

reuse_idea - effort S - not usable as a block (image stream, lidar-supervised depth), but two lessons: depth quality limits camera BEV (the far-range loss), and BEV-space augmentation (BDA) is what regularises the BEV encoder and head (BEVDet without augmentation overfits within a few epochs); a head-side analogue on the cached raster (flip / small rotation / scale of raster and labels together) is cheap to test, but the sensor layout (blind spots, camera-specific artefacts) makes rotation only approximately valid (not tested).

## Needs

lidar depth at training.

## Conflicts

lidar.

## Tried in our experiments

none.
