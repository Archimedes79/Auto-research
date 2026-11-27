# SN18 - StreamMapNet: BEV-level GRU fusion and a geographically new split

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B, S8.*

## Papers

StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction (Yuan et al., WACV 2024; from memory) https://arxiv.org/abs/2308.12570 ; code https://github.com/yuantianyuan01/StreamMapNet (URL from memory, not opened)

## Idea

The previous BEV feature is warped by the ego-pose transform and fused with the current one by a GRU; top-k queries (k = 33) are propagated with their reference points transformed; multi-point attention keeps cost O(N_p). The paper also introduces new disjoint splits.

## Formulation

ResNet50 + FPN + one BEVFormer layer; N_q = 100, N_p = 20, N_off = 1; loss = 50 L_line + 5 L_focal + L_trans (weights as printed); first 4 epochs single frame; "100 x 50 m" = 50 m front / back and 25 m lateral, "60 x 30 m" = 30 / 15.

## Input

BEV features of consecutive frames.

## Output

polylines.

## Training

regime hungarian_set, streaming (no gradient into previous frames).

## Differentiable?

partial - Hungarian.

## Sensors and data

camera; nuScenes, Argoverse 2.

## Reported results

nuScenes new split mAP 33.9 (60 x 30 m), 23.0 (100 x 50 m); original split 62.9 at 30 m; overlap of the old split 85 % (nuScenes) / 54 % (AV2), new split 11 % / 0 %; AV2 ablation (50 m): direct prediction 41.7 -> + query propagation 42.8 (+1.1) -> + transformation loss 43.7 (+0.9) -> + BEV fusion 46.1 (+2.4).

## Compute

13.2 FPS (nuScenes), 8 x GTX3090 training.

## Code

see papers line.

## Fit to our project

reuse_idea - effort S - BEV-level temporal fusion gave +2.4 mAP, our cache analogue E41 gave +0.008 F1; the 30 m -> 50 m drop (33.9 -> 23.0) shows range sensitivity.

## Needs

ego poses between frames.

## Conflicts

needs a trainable BEV encoder to learn the fusion (we have a frozen cache; fusion must be head-side).

## Tried in our experiments

E39 / E41 (temporal feature fusion on the cache, +0.001 / +0.008 F1).
