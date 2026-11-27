# VM27 - MapTRv2 (decoupled attention, one-to-many, dense auxiliaries, centerlines)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly (last 4 k characters of the page not read).. Pipeline stage: L, S4, B.*

## Papers

MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction (Liao et al., IJCV 2024; authors / venue from memory) https://arxiv.org/abs/2308.05736

## Idea

Keeps MapTR's hierarchical query + permutation-equivalent matching and adds (1) decoupled self-attention (inter-instance and intra-instance separately, O(N^2 + N_v^2)), (2) an auxiliary one-to-many branch (T = 300 extra queries, each GT repeated K = 6 times) with the same hierarchical matching, (3) dense supervision (depth from lidar projected to perspective views, BEV segmentation, perspective-view segmentation), (4) a fourth class "centerline": direction is given, so the permutation group has one element; centerlines are treated as paths via a graph traversal (following LaneGAP).

## Formulation

L = 1 L_one2one + 1 L_one2many + 1 L_dense, one2one weights 2 / 5 / 0.005 (class / p2p / direction); dense = 3 depth + 1 BEV seg + 2 PV seg; default BEVPoolv2 (LSS), range x [-15, 15], y [-30, 30] m, 0.3 m grid, 50 instances x 20 points, 6 layers, AdamW wd 0.01, lr 6e-4, batch 32, 24 / 110 epochs.

## Input

surround images -> BEV (LSS / GKT / BEVFormer variants).

## Output

as MapTR (+ centerline class).

## Training

hungarian_set (+ one-to-many).

## Differentiable?

partial.

## Sensors and data

camera, lidar fusion; nuScenes, AV2.

## Reported results

nuScenes val mAP: R18 110 ep 52.3 (33.7 FPS), R50 24 ep 61.5, R50 110 ep 68.7 (14.1 FPS), V2-99 110 ep 73.4; camera + lidar R50 24 ep 69.0; with centerline class, 24 ep: 54.0 average mAP (as read); AV2 R50 6 ep 67.4 (2D), 64.7 (3D), centerline 62.6.

## Compute

14.1 FPS (R50).

## Code

https://github.com/hustvl/MapTR

## Fit to our project

reuse_idea - S - (a) a centerline class with fixed direction and path-wise graph traversal is the published way to carry lane graphs in this family, (b) the one-to-many auxiliary is the published counter to sparse positive supervision (our dense heat map already supervises all cells).

## Needs

Hungarian; perspective-view features for the depth / PV auxiliary losses (not available).

## Conflicts

PV-specific heads unusable on cached BEV.

## Tried in our experiments

E31 / E32 (matched sparse decoder dropped).
