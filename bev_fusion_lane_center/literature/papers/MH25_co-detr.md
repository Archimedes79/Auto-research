# MH25 - Co-DETR (auxiliary dense one-to-many heads + positive queries without matching)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L, T.*

## Papers

DETRs with Collaborative Hybrid Assignments Training (Zong, Song, Liu, ICCV 2023) https://arxiv.org/abs/2211.12860 (read via ar5iv)

## Idea

Attach conventional one-to-many heads (ATSS, Faster R-CNN RPN / RoI, ...) to the DETR encoder features during training, and feed their positive anchors / proposals as extra decoder query groups where each query is supervised directly by the GT assigned to it (no Hungarian). Everything auxiliary is dropped at inference.

## Formulation

loss = sum_layers L_one2one + lambda1 * L_aux-decoder-queries + lambda2 * sum_heads L_encoder-head; lambda1 = 1.0, lambda2 = 2.0, K = 2 heads (ATSS + Faster R-CNN) or 1; positive query = linear(PE(box)) + linear(encoder feature at the location); about 8.8 (ATSS) and 18.7 (Faster R-CNN) positives per GT. Argument: one-to-one matching supervises the encoder sparsely and Hungarian matching is unstable. - input / output: multi-scale encoder features -> boxes.

## Training

COCO 12 / 36 epochs; regime `hungarian_set` (main) + `matching_free` (auxiliary queries).

## Differentiable?

yes.

## Sensors and data

COCO.

## Reported results

Deformable-DETR 12 ep R50 37.1 -> 42.9 AP (K = 2); 36 ep 43.3 -> 46.8 (K = 1); DINO-Deformable 12 ep 49.4 -> 51.2; Swin-L DINO 58.5 -> 59.5. K = 3: 49.5, K = 6: 48.9 (conflicts between diverse heads).

## Compute

GPU hours 70 -> 86 (K = 1) -> 120 (K = 2); memory 12.8 -> 13.9 -> 14.4 GB; no inference overhead.

## Code

not checked

## Fit to our project

reuse_idea - effort M - dense auxiliary supervision on the shared features and positive queries with their own assigned GT are matching-free; our heat map + tracer already follows the dense-head pattern (E31 / E32).

## Needs

dense auxiliary heads on the same features; a positive rule.

## Conflicts

published code is mmdet-style (mmdet / mmcv not in our environment).

## Tried in our experiments

E31 / E32 (matched sparse decoder dropped, grid kept)
