# MH20 - OneNet (classification cost makes one-to-one assignment NMS-free; min-cost without bipartite matching)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L.*

## Papers

What Makes for End-to-End Object Detection? (Sun et al., ICML 2021) https://arxiv.org/abs/2012.05780 (read via ar5iv)

## Idea

Location cost alone yields redundant predictions; adding a classification term to the matching cost gives one prediction per object so NMS is unnecessary. The assignment can be "MinCost": each GT takes its minimum-cost sample, no bipartite matching.

## Formulation

C = lambda_cls * C_cls + C_loc, C_loc = lambda_iou C_iou + lambda_L1 C_L1 (lambda_iou = 0 for point candidates); one-to-one: each GT gets the single sample with minimum cost (usually bipartite matching to avoid conflicts). Appendix D: MinCost matched bipartite matching on COCO for RetinaNet and FCOS but failed on CrowdHuman. Coefficient values not given. - input / output: dense-detector heads (RetinaNet, FCOS, CenterNet) and set predictors.

## Training

regime `matching_free` (MinCost) or `hungarian_set`.

## Differentiable?

yes (labels constant).

## Sensors and data

COCO, CrowdHuman.

## Reported results

Table 2 AP (with NMS in brackets): RetinaNet location-only 33.6 (36.8), + classification 36.0 (36.2), + predicted location 37.5 (37.5); FCOS 34.9 (37.7) -> 38.9 (38.9); CenterNet 23.4 (32.0) -> 34.9 (34.8); DETR 12.6 -> 40.0, Deformable DETR 12.0 -> 44.0, Sparse R-CNN 20.1 -> 45.0 when the classification cost is added.

## Compute

not stated.

## Code

not checked

## Fit to our project

reuse_idea - effort S - direct evidence that argmin-per-GT labelling replaces Hungarian in sparse scenes; the CrowdHuman failure is the warning for parallel lanes and junction mouths.

## Needs

a cost combining score and geometry.

## Conflicts

crowded scenes need conflict resolution (bipartite).

## Tried in our experiments

none
