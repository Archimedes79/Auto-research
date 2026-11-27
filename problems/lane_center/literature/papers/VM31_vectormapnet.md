# VM31 - VectorMapNet (two-stage detector + autoregressive polyline generator)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: L, S5, X.*

## Papers

VectorMapNet: End-to-end Vectorized HD Map Learning (Liu et al., ICML 2023; authors from memory) https://arxiv.org/abs/2206.08920

## Idea

A DETR-style detector predicts per element a class and keypoints (bounding-box corners, k = 2, best), then an autoregressive transformer decodes the vertices of each polyline as discrete tokens (coordinates quantised on a 200 x 100 grid of 0.3 m cells, EOS token ends the polyline). Order encodes direction.

## Formulation

detector Hungarian loss (NLL class + smooth-L1 + IoU on keypoints; coefficients garbled on the page); generator NLL over quantised tokens, teacher forcing on GT keypoints by default, a second stage fine-tunes on predicted keypoints (exposure bias); augmentation: Gaussian noise added to input vertices with probability 0.3; vertices sampled by curvature (RDP), fixed 1 m spacing worse; no canonical-direction rule or start-point rule described; decoding rule (greedy / sampling) not stated. IPM camera BEV at heights -1, 0, 1, 2 m, 200 x 100 x 128 features, ResNet-50, N_max = 100 elements, 6 + 6 layers, dropout 0.2, 110 epochs, batch 32, 8 x 3090.

## Input

BEV (camera IPM / lidar PointPillars / fusion).

## Output

variable-length vertex sequences.

## Training

Hungarian + NLL, teacher forcing then fine-tuning.

## Differentiable?

no end to end (tokenisation, argmax decoding).

## Sensors and data

camera, lidar; nuScenes, 60 x 30 m; thresholds {0.5, 1.0, 1.5} m.

## Reported results

nuScenes mAP: camera 40.9 (ped 36.1, div 47.3, bound 39.3); camera + fine-tune 46.0; lidar 34.0; fusion 45.2, fusion + fine-tune 53.7.

## Compute

not stated.

## Code

https://github.com/Mrmoore98/VectorMapNet_code (linked as "Code" on the project page https://tsinghua-mars-lab.github.io/vectormapnet/; repository not opened)

## Fit to our project

not_suitable - L - autoregressive tokens are not differentiable through the sequence; reuse only the exposure-bias remedy (second-stage fine-tuning on own predictions) and the vertex-noise augmentation.

## Needs

a sequence order of the vertices.

## Conflicts

argmax / tokenisation break gradients; no branching.

## Tried in our experiments

E94 teacher-forced tracer (partial: F1 0.851 vs rules 0.867).
