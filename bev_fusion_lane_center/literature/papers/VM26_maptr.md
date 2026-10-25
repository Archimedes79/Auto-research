# VM26 - MapTR (permutation-equivalent point sets, hierarchical matching)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked.. Pipeline stage: L, S4, S5.*

## Papers

MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction (Liao et al., ICLR 2023; authors / venue from memory) https://arxiv.org/abs/2208.14437 ; code read: decoder.py and maptr_assigner.py on GitHub

## Idea

Every map element is an ordered set of N_v = 20 points with an EQUIVALENT-PERMUTATION group: polylines have 2 permutations (forward, reversed), polygons 2 N_v (cyclic shifts x 2 orientations). The loss uses, per matched pair, the permutation with the lowest point-to-point cost, so start point and direction are not forced. Hierarchical queries q_ij = q_i^ins + q_j^pt (N x N_v) go through a deformable decoder.

## Formulation

instance matching cost = focal(class) + point2point Manhattan cost (Chamfer cost 47.5 vs 50.3 mAP), Hungarian; point level: gamma* = argmin over Gamma of sum_j Manhattan(v^_j, v_gamma(j)). Loss L = 2 L_cls(focal) + 5 L_p2p + 0.005 L_dir with L_dir = - sum cos(e^_j, e_gamma*(j)) (edge vectors, cyclic). Fixed order 44.4 vs permutation-equivalent 50.3 mAP; N_v 10 / 20 / 40 -> 48.0 / 50.3 / 50.0; queries N 25 / 50 / 75 -> 41.6 / 50.3 / 52.2; layers 1 / 2 / 3 / 6 / 8 -> 29.1 / 42.4 / 44.0 / 50.3 / 47.6. Code: reference points are updated per layer as sigmoid(offset + inverse_sigmoid(ref)) and DETACHED before the next layer; the assigner builds the cost per (query, GT, order) and takes the minimum over orders, then scipy `linear_sum_assignment` on CPU (the assigner's default point cost is a Chamfer class, the paper states Manhattan; the config was not seen).

## Input

surround images (1600 x 900 scaled 0.5), ResNet-50, GKT BEV (alternatives IPM / LSS / deformable attention), 6 decoder layers.

## Output

N = 50 instances x 20 points (normalised BEV coordinates) + class (3 classes).

## Training

Hungarian + focal + L1 + direction; regime hungarian_set.

## Differentiable?

partial (Hungarian; the minimum over permutations passes gradients to the chosen order).

## Sensors and data

camera (+ lidar variants); nuScenes; range x [-15, 15] m, y [-30, 30] m (30 x 60 m), BEV grid 0.3 m (tiny) / 0.75 m (nano); AP at Chamfer thresholds {0.5, 1.0, 1.5} m.

## Reported results

nuScenes val: MapTR-tiny R50 110 ep 58.7 mAP (ped 56.2, div 59.8, bound 60.1), 24 ep 50.3, 11.2 FPS; nano R18 45.9 at 25.1 FPS; VectorMapNet 40.9, HDMapNet 23.0 (camera); extrinsic noise sigma_t 0.5 m -> 34.0 mAP (50.3 clean), sigma_rot 0.02 rad -> 42.0.

## Compute

35.9 M parameters (tiny), 89.3 ms per frame (backbone 62 %).

## Code

https://github.com/hustvl/MapTR

## Fit to our project

reuse_idea - S - only the point-level "minimum over the permutation group" and the edge-direction cosine loss; the full model is the matched sparse decoder that E31 / E32 dropped.

## Needs

polyline element with a fixed point count; Hungarian for the instance level. Cached-BEV: the decoder yes, the GKT lifting no.

## Conflicts

needs Hungarian; our vertex sets have variable lengths and branching, which a fixed 20-point set cannot hold.

## Tried in our experiments

E31 / E32 (matched sparse DETR-style decoder dropped: grid +0.058 F1, +0.18 route F).
