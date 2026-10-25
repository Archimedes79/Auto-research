# VS18 - Cascade R-CNN: stages with increasing IoU thresholds

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S5, L.*

## Papers

Cascade R-CNN (Cai, Vasconcelos, CVPR 2018) https://arxiv.org/abs/1712.00726

## Idea

A sequence of detectors; stage t is trained on the boxes output by stage t-1 with a higher IoU threshold for positives. Regression raises the IoU of the sample set, so the number of positives stays about constant while the threshold rises; the same cascade runs at test time (no train / test mismatch).

## Formulation

thresholds U = {0.5, 0.6, 0.7} (3 stages after the RPN); b^t = f_{t-1}(x^{t-1}, b^{t-1}); features re-pooled at the new boxes; per-stage loss CE + lambda [y >= 1] L_loc, lambda = 1; regression targets normalised per stage (mean / variance values not stated); class-agnostic regression; final score = ensemble of the three stage classifiers; detach between stages not stated. Iterative bbox (one regressor repeated) degrades boxes above IoU 0.85 and gives no gain beyond two applications; integral loss (several classifiers on one distribution) leaves high-IoU classifiers short of positives.

## Input

point_set of proposals + feature map. output: refined boxes + scores.

## Training

IoU-threshold assignment per stage; matching_free.

## Differentiable?

partial (sample sets are re-selected by IoU between stages).

## Sensors and data

images (COCO).

## Reported results

COCO val ablation baseline: FPN+ 34.9 AP -> iterative bbox 35.4 -> integral loss 35.4 -> cascade 38.9; AP90 7.1 -> 8.1 / 6.9 / 15.8; test-dev ResNet-101 FPN+: 38.8 -> 42.8 AP, AP75 41.9 -> 46.3. (Text says ResNet-50 for the ablations but Table 6 gives 36.5 for FPN+ R50, not explained.)

## Compute

FPN+ R50 parameters 165 M -> 272 M; test 0.095 -> 0.115 s per image (Titan Xp).

## Code

URL from memory, not opened: https://github.com/zhaoweicai/cascade-rcnn

## Fit to our project

reuse_idea - S - tightening the positive radius per stage (spec: 1.0 / 0.5 / 0.3 m) is the direct analogue; key evidence is "same distribution at train and test".

## Needs

a hypothesis set per stage with positives defined by a geometric score (IoU -> point-to-polyline distance).

## Conflicts

IoU resampling between stages is discrete; in a fixed-shape tree it becomes re-labelling with a tighter radius.

## Tried in our experiments

E39 cascade refiner (+0.001 F1 / +0.014 route F, small, kept); E120 position refinement (+0.012 F1@1).
