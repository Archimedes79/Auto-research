# VS38 - LATR: lane-aware queries (lane x point) with a dynamic ground-plane positional embedding

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2, S5.*

## Papers

LATR (Luo et al., ICCV 2023) https://arxiv.org/abs/2308.04583

## Idea

Queries are the broadcast sum of a lane-level embedding (pooled from image features by instance activation maps) and a point-level embedding per predefined height; a ground-plane grid refined layer by layer gives a 3-D positional embedding for the image features, so no BEV transform exists.

## Formulation

Q = Q_lane (+) Q_point (N x M queries, M not stated); plane initialised horizontally at z = -1.5, projected into the image, scattered to a canvas, MLP -> positional embedding; each decoder layer predicts (d theta_x, d h) of the plane; 6 layers (LATR-Lite 2), deformable attention 4 heads x 8 points, 256-d; losses w_s = 5 (segmentation), w_p = w_l = 1; lane loss: x 2.0 L1, z 10.0 L1, class 10.0 focal (gamma 2, alpha 0.25), visibility BCE 1.0; dice-based bipartite matching reused for the predictions.

## Input

image_feat + camera. output: 3-D polylines with visibility.

## Training

hungarian_set.

## Differentiable?

partial.

## Sensors and data

monocular camera (OpenLane).

## Reported results

OpenLane val F1 61.9 vs PersFormer 50.5; x error near / far 0.219 / 0.259 m; Lite F1 61.5; ablation OpenLane-300: lane-only queries 66.5 vs lane + point queries 70.4 F1; ground embedding none 67.9, fixed frustum 69.1, fixed plane 69.2, dynamic plane 70.4.

## Compute

FPS not stated.

## Code

URL from memory, not opened: https://github.com/JMoonr/LATR

## Fit to our project

reuse_idea - S - lane x point factorised queries (each sample point has its own embedding) and a per-point visibility BCE.

## Needs

camera model, Hungarian.

## Conflicts

perspective features.

## Tried in our experiments

none.
