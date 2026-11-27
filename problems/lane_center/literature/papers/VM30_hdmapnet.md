# VM30 - HDMapNet (dense heads + clustering + direction-guided tracing)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked (paper text + code).. Pipeline stage: S1, S6, S7.*

## Papers

HDMapNet: An Online HD Map Construction and Evaluation Framework (Li et al., ICRA 2022) https://arxiv.org/abs/2107.06307

## Idea

Per-pixel BEV prediction of semantic class, instance embedding and lane DIRECTION (classification over angle bins, forward and backward both positive), followed by classic post-processing: DBSCAN clustering of the embedding, max-pool NMS, and greedy tracing of vertices along the predicted direction with a fixed step. The vectors are never trained; splits and merges are not evaluated.

## Formulation

losses: CE (semantic), discriminative loss (alpha = beta = 1, delta_v = 0.5, delta_d = 3.0), softmax CE on lane pixels (direction, weight not stated). Code: angle_class = 36 (10 degree bins, 37 channels with background), label line thickness 5 px, canvas 200 x 400 at 0.15 m (60 x 30 m), DBSCAN eps = 1.5 and min_samples = 50, NMS by 1 x 5 / 5 x 1 max-pooling, tracing step 7 pixels (1.05 m), top-2 direction bins with a 5-bin suppression window, instance confidence = mean probability over the mask, no resampling interval.

## Input

images (EfficientNet-B0) + MLP/geometric view transform, optional lidar pillars; BEV decoder ResNet.

## Output

raster heads, then polylines by rules.

## Training

dense losses, no matching; regime none (supervised dense).

## Differentiable?

no after the heads (clustering, NMS, greedy tracing are rules).

## Sensors and data

camera + lidar; nuScenes; 60 x 30 m.

## Reported results

nuScenes semantic IoU: fusion 44.5, surround camera 32.9; instance AP at Chamfer {0.2, 0.5, 1.0} m: fusion 30.6, camera 22.7; bidirectional Chamfer distance of fusion 0.639 m. In MapTR's comparison (thresholds 0.5 - 1.5 m): camera 23.0, fusion 31.0 mAP.

## Compute

0.8 FPS camera (MapTR's table).

## Code

https://github.com/Tsinghua-MARS-Lab/HDMapNet (PyTorch + torch-scatter, GPL-3.0 stated, no mmcv)

## Fit to our project

baseline - S - the dense-heads-plus-rule-tracing design is the ancestor of our dense head + rule system A (heat map, direction channels, greedy trace); direction as a CLASSIFICATION over bins with both directions positive is the published way to avoid a direction ambiguity without matching.

## Needs

any BEV raster. Cached-BEV: yes (heads and post-processing are independent of the encoder).

## Conflicts

the rule post-processing is not differentiable; no fork / merge handling.

## Tried in our experiments

dense head + tracer (E31 / E32 grid kept), rule system A.
