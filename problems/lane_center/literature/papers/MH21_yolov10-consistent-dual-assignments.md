# MH21 - YOLOv10 consistent dual assignments (one-to-many head for training, top-1 one-to-one head for NMS-free inference)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L, T.*

## Papers

YOLOv10: Real-Time End-to-End Object Detection (Wang et al., 2024) https://arxiv.org/abs/2405.14458 (arXiv HTML)

## Idea

Two heads on one backbone: a one-to-many head (rich supervision) and a one-to-one head with the same structure and objectives; both use one matching metric with the same exponents, so the best one-to-many positive is the one-to-one top-1. Only the one-to-one head is used at inference.

## Formulation

m(alpha, beta) = s * p^alpha * IoU(b_hat, b)^beta (s spatial prior, p class score); consistent setting alpha_o2o = r * alpha_o2m, beta_o2o = r * beta_o2m with r = 1; supervision gap measured by the 1-Wasserstein distance between classification targets; one-to-one assignment is top-1 selection (not Hungarian, "same performance as Hungarian" with less training time). - input / output: detector heads; boxes.

## Training

joint training; regime `matching_free`.

## Differentiable?

yes.

## Sensors and data

COCO (images).

## Reported results

YOLOv10-S COCO val: one-to-many only (with NMS) 44.9 AP, 7.07 ms; one-to-one only 43.4 AP, 2.44 ms; dual 44.3 AP, 2.44 ms; mismatched metrics 42.7 (beta = 2.0) or 44.2 (beta = 4.0, alpha = 0.5); a gap to NMS-based training remains for small models.

## Compute

no inference overhead.

## Code

not checked

## Fit to our project

reuse_idea - effort S - "dense head trains the features, sparse head picks one per object with the same metric" is the NMS-free pattern for a ray fan.

## Needs

two heads sharing features and one scalar matching metric.

## Conflicts

designed for box detection.

## Tried in our experiments

none
