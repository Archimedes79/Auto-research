# TP18 - SCORE (stack: SD map + one-to-many + denoising + GeoDist + temporal)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S6 S8 S9 T.*

## Papers

Coherent Online Road Topology Estimation and Reasoning with Standard-Definition Maps (Bonn, arXiv 2507.01397; project page https://www.ipb.uni-bonn.de/html/projects/score) https://arxiv.org/abs/2507.01397

## Idea

A challenge-style stack on LaneSegNet whose incremental table shows that SD-map fusion, hybrid queries, one-to-many supervision, denoising, a geometric link post-process, resampling and ConvGRU temporal fusion add up.

## Formulation

SD queries: n_sd = 50 reference points from SD edge midpoints (OSM maps), initial p = MLP(q_pos) + p_sd, attention mask so SD queries do not leak into others, SMERF-style cross-attention. Hybrid queries = SD-enhanced + base + denoising (60 DN queries, noise U(-1,1) scaled by lambda_dn, groups = floor(n_dn / N_GT), masked attention). Topology head: A^S = sigmoid(MLP(pair features)); distance score A^D = 2 / (1 + exp(d / alpha)), alpha 2.5; A = min(A^S + beta A^D, 1), beta 0.8. Temporal: ConvGRU on BEV with ego-motion compensation after 30 single-frame epochs + 30 temporal epochs. Hungarian (LaneSegNet costs); o2m weight 1.0.

## Input

BEVFormer BEV (size not stated), 200 base queries.

## Output

lane segments + topology.

## Training

AdamW cosine, 8 V100, batch 8, images at 50 %.

## Differentiable?

partial - Hungarian; distance score is a differentiable closed form.

## Sensors and data

camera + OSM SD map; OpenLane-V2 subset A (subset B lacks lane segments).

## Reported results

incremental (DET_ls / DET_ped / TOP_ll): LaneSegNet 32.3 / 32.9 / 25.4; +SMERF 35.8 / 41.7 / 31.3; +SD queries 37.3 / 41.2 / 31.8; +OSM 38.2 / 42.0 / 32.4; +one-to-many 39.8 / 43.5 / 34.8; +denoising 39.8 / 44.1 / 34.6; +topology post-processing 39.8 / 44.1 / 36.2; +resampling 40.0 / 45.1 / 36.2; +temporal 44.0 / 47.3 / 40.0. Final val: OLUS 54.9 (DET_ls 44.0, TOP_ll 40.0, TOP_lt 39.1); test OLUS 55.8. Geo-disjoint split drops DET_ls to 23.8 (LaneSegNet 19.3). Traffic elements: YOLOv9 61.4 > DN-DETR + YOLO 57.0 > DN-DETR 53.0.

## Compute

8 V100 training; no inference numbers.

## Code

none (project page only)

## Fit to our project

baseline - effort S - gives a stacking order and magnitudes (temporal +4.0 DET_ls and +3.8 TOP_ll; GeoDist-style post-process +1.6 TOP_ll).

## Needs

LaneSegNet-type host.

## Conflicts

one test split with geographic overlap inflates numbers.

## Tried in our experiments

E39 / E41 (temporal fusion small gain), E40 (map prior).
