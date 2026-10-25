# VS28 - Polar R-CNN: local polar poles propose anchors, global head, NMS-free O2O branch

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M (all pages read, including the appendix)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S1, S2, S4.*

## Papers

Polar R-CNN: end-to-end lane detection with fewer anchors (arXiv 2024) https://arxiv.org/abs/2411.01499

## Idea

Anchors are lines in polar form (radius r, angle theta) relative to a pole; every lattice point of a coarse feature map ("local pole") predicts the (r, theta) of the nearest lane and a confidence, so only the top-K (about 20) anchors reach the second stage instead of about 192; the second stage samples features along each anchor; an O2O branch trained with stopped gradients removes duplicates without NMS.

## Formulation

local polar module: highest FPN level downsampled, 1x1 conv regression (theta, r), classification of two 1x1 convs; pole positive if its radius < lambda_l (value not stated); labels r^ = distance pole -> lane, theta^ = direction to the nearest lane point; local map H x W = 4 x 10 (6 x 13 for CurveLanes); K = 20 (CULane, TuSimple, LLAMAS), 12 (DL-Rail), 50 (CurveLanes). Global polar module: one global pole near the static vanishing point; 36 sampled points per anchor (72 regression points), features of three FPN levels combined by a learned softmax-weighted sum, then a linear projection; GNN-style relation block (not attention). Triplet head: O2M cls + reg (SimOTA, dynamic k = 4), O2O cls (Hungarian) with adjacency A = A^C (confidence) x A^G (|dtheta| < tau_theta, |dr| < lambda_g), max-pool over neighbours + 3-layer MLP; dual selection s~ > tau_o2o and s > tau_o2m (CULane 0.46 / 0.48). Losses BCE + smooth-L1 (local), focal + rank + GIoU + smooth-L1 endpoints + auxiliary segment loss (global); the appendix redefines the lane IoU (GLaneIoU: per-point semi-width from the local direction, overlap / gap / union), g = 0 for assignment costs and g = 1 for the GIoU loss. 800 x 320, AdamW lr 0.006, cosine, one A100.

## Input

image_feat levels. output: polyline lanes, no NMS.

## Training

SimOTA (O2M) + hungarian_set (O2O); gradient stopped on the O2O branch.

## Differentiable?

partial.

## Sensors and data

images (TuSimple, CULane, LLAMAS, CurveLanes, DL-Rail).

## Reported results

CULane F1@50 / F1@75 R18 80.81 / 63.96 (NMS-free), DLA34 81.49 / 64.97; CurveLanes DLA34 F1@50 87.29; 20 proposed anchors vs 192 (CLRNet); ablation CULane F1@50: fixed anchors 79.90, fixed + aux loss 80.38, proposals only 75.85, full 80.81; stop-gradient on the O2O branch 71.68 -> 80.81; CurveLanes NMS threshold 50 px 85.38, best 15 px 86.81, NMS-free 87.29. FPS and parameters not stated numerically; no ablation of sampled-point number or K in the text.

## Compute

one A100-40G for training.

## Code

not found on the pages read (URL from memory, not opened: https://github.com/ShqWW/PolarRCNN)

## Fit to our project

reuse_idea - S / M - every grid point proposing its nearest lane in polar form (r, theta) is a possible tip / seed generator; the stop-gradient between the ranking branch and the main branch is a case where cutting gradients is essential (71.68 -> 80.81).

## Needs

Hungarian for O2O.

## Conflicts

matching-free rule for the O2O head; anchors are image-plane lines.

## Tried in our experiments

none directly (E94 seeds = heat-map peaks, 192).
