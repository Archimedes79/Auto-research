# SN15 - HDMapNet: dense BEV decoder (semantic, instance embedding, direction) + vectorisation

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B, S6.*

## Papers

HDMapNet: An Online HD Map Construction and Evaluation Framework (Li et al., ICRA 2022; from memory) https://arxiv.org/abs/2107.06307 ; code https://github.com/Tsinghua-MARS-Lab/HDMapNet (URL from memory, not opened)

## Idea

A fully convolutional BEV decoder with three heads (semantic segmentation, instance embedding, per-pixel direction) is applied to BEV features from cameras (MLP view transform), lidar (PointPillars) or both; vectorisation clusters embeddings, applies NMS and links pixels by following the predicted directions.

## Formulation

decoder = ResNet-style, 3 blocks; direction head has N_d classes on the unit circle with two positive labels per lane pixel (forward and backward unknown), non-lane pixels excluded from the loss; instance loss L = L_var + L_dist (alpha = beta = 1, delta_v 0.5, delta_d 3.0); Adam 1e-3; post-processing: DBSCAN on embeddings, NMS, greedy step C_next = C_now + step * D. BEV extent: not in the ar5iv text; the 60 x 30 m at 0.15 m (400 x 200) convention is documented by MGMap and MV-Map (search summaries).

## Input

BEV features from any encoder.

## Output

seg + embedding + direction rasters -> polylines.

## Training

regime matching_free (CE, discriminative loss, direction CE).

## Differentiable?

partial - the network is differentiable; DBSCAN / NMS / greedy linking are not.

## Sensors and data

camera, lidar, fusion; nuScenes (lane boundary, divider, ped crossing); radar only mentioned as a possible extension.

## Reported results

nuScenes "All classes": Surr IoU 32.9, mAP 22.7; LiDAR 29.5 / 11.6; Fusion 44.5 / 30.6; temporal fusion (Surr) IoU 32.9 / 35.8 / 36.4 for 1 / 2 / 4 frames; fusion normal / night / rainy IoU 44.9 / 39.3 / 38.7.

## Compute

not stated in the text read.

## Code

see papers line.

## Fit to our project

reuse_idea - effort S - the direction head is what our dense head (heat map + 2-channel direction) already does; the multi-class direction loss with two positive labels is the one detail we do not use.

## Needs

a BEV feature raster only.

## Conflicts

DBSCAN / greedy linking are not differentiable.

## Tried in our experiments

E17 / E18 (U-Net dense head kept; dense head + tracer beats sparse).
