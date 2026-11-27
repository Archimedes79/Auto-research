# VM05 - SparseMeXt (physical-prior denoising of map polylines, query memory bank)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: T, S8, B.*

## Papers

SparseMeXt (arXiv 2505.08808; full title and authors not on the pages read, title from memory: "Unlocking the Potential of Sparse Representations for HD Map Construction") https://arxiv.org/abs/2505.08808

## Idea

A sparse map decoder (no dense BEV at inference) with a query memory bank and a "physical-prior denoising" (PPDN) adapted to polylines instead of boxes: four noise types on GT polylines - rotation about the mean point, one common x/y shift of all points, scale of the x/y coordinates, and a per-point curvature-based displacement (curvature from second-order tangent differences). No categorical noise (rationale: static geometry, strong dependence among road elements).

## Formulation

noise magnitudes, number of groups and queries, assignment rule, matching and loss weights: not stated. One non-temporal decoder stage + four temporal stages (ablation). 100 epochs, lr 0.006 half-cosine, AdamW wd 0.01, grad clip 35, 4 x A100, input 704 x 256, nuImages-pretrained R50 init for Base / Tiny.

## Input

images (SiMo neck on ResNet C5); a BEV segmentation head exists only as a training auxiliary (+0.9 mAP).

## Output

sparse queries -> polylines + class.

## Training

DN-DETR-style reconstruction of the source GT (assumed, not confirmed); matching not stated.

## Differentiable?

unknown (matching not stated).

## Sensors and data

camera; nuScenes only (official 700 / 150 split); 60 x 30 m base range, 90 x 60 m long range.

## Reported results

nuScenes val mAP: Tiny R18 55.5 (32.9 FPS), Base R50 65.2 (25.4 FPS), Large R101 68.9 (20.2 FPS); MapTRv2-Base 68.7; centerline average mAP 58.8 vs MapTRv2 54.0; long range 47.6.

## Compute

FPS above (RTX 3090).

## Code

none given.

## Fit to our project

reuse_idea - S - the noise TAXONOMY (rotate, shift, scale, curvature jitter on a polyline) is the closest published model for noising a previous lane set; magnitudes must be chosen by us.

## Needs

GT polylines; a recursion block that accepts noised priors. Cached-BEV: yes for the noise part, no for the image-sampling decoder.

## Conflicts

matching behind the denoising targets is unknown.

## Tried in our experiments

none.
