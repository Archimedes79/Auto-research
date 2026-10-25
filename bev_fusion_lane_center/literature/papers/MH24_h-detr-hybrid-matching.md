# MH24 - H-DETR hybrid matching (one-to-one branch + one-to-many branch with repeated GT)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L, T.*

## Papers

DETRs with Hybrid Matching (Jia et al., CVPR 2023) https://arxiv.org/abs/2207.13080 (read via ar5iv)

## Idea

Two query groups share the decoder: N = 300 queries one-to-one, T = 1500 queries one-to-many by repeating every GT K = 6 times and applying Hungarian matching to the repeated targets; masked self-attention separates the groups; only the one-to-one branch is used at inference.

## Formulation

total = lambda * L_one2many + L_one2one, lambda = 1 (0.1: 47.8, 1: 48.7, 5: 48.3 AP); K: 0 -> 47.0, 1 -> 46.4, 3 -> 48.1, 6 -> 48.6, 8 -> 48.6 (gain only for K > 3); T: 300 -> 47.8 ... 1500 -> 48.7. Variants hybrid epoch (one-to-many in the first 2/3 of epochs, K~ = 10) and hybrid layer (first 4 decoder layers one-to-many). - input / output: Deformable-DETR-like set predictors; also pose (PETR), 3D detection (PETRv2), tracking.

## Training

COCO 12 / 24 / 36 epochs; Hungarian per branch; regime `hungarian_set`.

## Differentiable?

yes.

## Sensors and data

COCO, LVIS, nuScenes (PETRv2 camera 3D, NDS 50.68 -> 52.38).

## Reported results

Deformable-DETR R50 43.7 / 46.4 / 46.8 (12 / 24 / 36 ep) -> hybrid branch 45.9 / 47.6 / 48.0; COCO test Swin-L 59.4 AP; one-to-many alone with NMS 49.4 / 50.2 / 48.8 (5.3 FPS) vs hybrid one-to-one 48.7 / 49.9 / 50.0 (6.7 FPS).

## Compute

training time +23.1 % (R50) / +6.4 % (Swin-L), memory 5.5 -> 7.7 GB (R50); no inference cost.

## Code

not checked

## Fit to our project

reuse_idea - effort M - training-only redundancy: extra children per tip in training (K repeats) that are dropped at test time.

## Needs

Hungarian on repeated GT (replace by nearest assignment).

## Conflicts

the no-Hungarian rule; extra children multiply tree size.

## Tried in our experiments

none
