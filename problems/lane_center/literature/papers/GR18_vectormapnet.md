# GR18 - VectorMapNet (detector plus autoregressive polyline generator; exposure-bias fine-tuning)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: checked (paper HTML).. Pipeline stage: S4, T.*

## Papers

VectorMapNet: End-to-end Vectorized HD Map Learning (Liu et al., arXiv 2206.08920; ICML 2023 from memory) https://arxiv.org/abs/2206.08920

## Idea

A DETR-style detector places map elements as keypoint sets (bbox corners by default); a second stage generates each polyline's vertices autoregressively as discrete coordinate tokens conditioned on keypoints, class and BEV features. Teacher forcing uses GT keypoints; an optional two-stage fine-tuning feeds predicted keypoints to remove the train-test mismatch.

## Formulation

100 element queries (6 layers, hidden 256, deformable cross-attention); generator: next-token NLL on a 200 x 100 grid of 0.3 m cells, EOS token; detector loss with Hungarian matching; Gaussian noise on generator input vertices with probability 0.3; 110 epochs, batch 32, AdamW. Discrete vs continuous (Gaussian mixture) vertex modelling: Chamfer mAP 32.5 vs 26.5, Frechet mAP 50.4 vs 41.9 (Table 6; the appendix quotes other numbers).

## Input

BEV features (camera, lidar or fusion); output: polylines.

## Training

`hungarian_set` detector + `next_token` generator; `teacher_forcing` with a fine-tune on predicted inputs.

## Differentiable?

no - tokenisation, argmax.

## Sensors and data

nuScenes, Argoverse 2; BEV about 60 x 30 m at 0.3 m.

## Reported results

nuScenes mAP camera 40.9 -> 46.0 with the predicted-keypoint fine-tune; fusion 45.2 -> 53.7; HDMapNet fusion 31.0 (best baseline); no temporal consistency, hallucination in occluded areas (stated).

## Compute

latency not stated.

## Code

project page https://tsinghua-mars-lab.github.io/vectormapnet (repo not checked)

## Fit to our project

reuse_idea - effort S: size of the exposure-bias gain by retraining the second stage on the first stage's own output (+5.1 / +8.5 mAP) is a data point for our cascade.

## Needs

detector keypoints; quantisation.

## Conflicts

tokens; Hungarian.

## Tried in our experiments

none.
