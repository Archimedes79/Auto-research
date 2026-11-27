# SN38 - MapTRv2: auxiliary dense losses (depth, BEV segmentation, PV segmentation) and one-to-many matching

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (first ~100k of ~104k characters)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: T, L.*

## Papers

MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction (Liao et al., arXiv 2023; title from memory) https://arxiv.org/abs/2308.05736

## Idea

Keeps the MapTR decoder and adds three auxiliary dense losses on the encoder side (depth from lidar-rendered depth maps on perspective features, BEV segmentation on the BEV feature, perspective-view segmentation on image features), a one-to-many auxiliary query group, and decoupled attention; the auxiliary parts exist to train the encoder.

## Formulation

one-to-many: T = 300 extra instance queries, each GT repeated K = 6, beta 1.0; decoupled self-attention O(N^2 + Nv^2) instead of O((N Nv)^2); default BEV encoder LSS with BEVPoolv2, cell 0.3 m; ranges x in [-15, 15] m, y in [-30, 30] m (a 100 x 50 m setting is not confirmed on the page read).

## Input

multi-view images.

## Output

classified point sets.

## Training

regime hungarian_set + one-to-many; the page does not say whether auxiliary heads run at inference.

## Differentiable?

partial - Hungarian.

## Sensors and data

camera (lidar for the depth target), nuScenes, Argoverse 2.

## Reported results

nuScenes val mAP, 24 epochs R50: 61.5; ablation chain: MapTR 50.3, + LSS 50.1, + depth 55.2, + BEV seg 56.5, + PV seg 57.1, + decoupled attention 57.6, + one-to-many 61.5; 110 epochs R50 68.7.

## Compute

14.1 FPS (R50).

## Code

not checked (same repository family as MapTR).

## Fit to our project

reuse_idea - effort S - tells which supervision belongs to the encoder: the BEV-segmentation auxiliary is already what GaussianCaR is trained on; perspective-view and depth auxiliaries cannot be recovered from a cached BEV raster.

## Needs

image features and lidar depth for the auxiliary losses.

## Conflicts

Hungarian one-to-one and one-to-many assignment.

## Tried in our experiments

E31 / E32 (matched sparse decoder dropped).
