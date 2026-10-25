# SN16 - MapTR: DETR-style map decoder reading a BEV feature map

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B, S4.*

## Papers

MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction (Liao et al., ICLR 2023; from memory) https://arxiv.org/abs/2208.14437 ; code https://github.com/hustvl/MapTR (URL from memory, not opened)

## Idea

Hierarchical queries (instance query + point query) with deformable attention over the BEV feature map predict each map element as an ordered point set; a permutation-equivalent Hungarian matching handles the start-point ambiguity. The BEV encoder (GKT by default) is a pluggable block.

## Formulation

q_ij = q_i(ins) + q_j(pt); MapTR-tiny 50 instance x 20 point queries, 6 decoder layers; loss = 2 L_cls + 5 L_p2p + 5e-3 L_dir; point-to-point cost 50.3 mAP vs Chamfer cost 47.5; range x in [-15, 15] m, y in [-30, 30] m (30 x 60 m), BEV cell 0.3 m (tiny) / 0.75 m (nano), about 100 x 200 cells (derived).

## Input

BEV features (any encoder: IPM, LSS, deformable attention, GKT).

## Output

classified point sets.

## Training

regime hungarian_set (instance matching + permutation-minimising point matching), end to end with the encoder.

## Differentiable?

partial - Hungarian matching at training.

## Sensors and data

camera, lidar, both; nuScenes.

## Reported results

val mAP: tiny C R50 24 ep 50.3, 110 ep 58.7; tiny L 55.6; tiny C&L 62.5; lifters: IPM 46.2, LSS 49.5, deformable 49.7, GKT 50.3. The paper says it is "compatible with LiDAR and RADAR" (extension "straightforward"), but it does not claim the decoder works on any BEV features.

## Compute

tiny 35.9 M parameters, 11.2 FPS.

## Code

see papers line.

## Fit to our project

baseline - effort M - decoder is separable from the encoder (needs a 39 -> 256 channel adapter and normalised-coordinate reference points), but the Hungarian loss conflicts with our matching-free losses (E31 / E32 dropped this family).

## Needs

BEV feature map B in R^(H x W x C), C = 256 in practice (from memory).

## Conflicts

Hungarian matching; ordered-point-set labels.

## Tried in our experiments

E31 / E32 (matched sparse decoder dropped).
