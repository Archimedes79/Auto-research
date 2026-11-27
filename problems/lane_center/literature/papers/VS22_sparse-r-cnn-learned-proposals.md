# VS22 - Sparse R-CNN: learned proposals, instance-conditioned dynamic head, 6 stages with blocked box gradient

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S5, S2.*

## Papers

Sparse R-CNN (Sun et al., CVPR 2021) https://arxiv.org/abs/2011.12450

## Idea

N learnable proposal boxes plus learnable proposal features replace the RPN; each box pools RoI features and a dynamic head whose two 1x1 convolutions are generated from the proposal feature transforms them; boxes and features are refined over 6 stages.

## Formulation

N = 100 (300 / 500 in variants), d = 256; RoIAlign S x S bins (7 x 7 not stated); self-attention among object features, then dynamic conv (C -> C/4 -> C, ReLU, norm) with parameters from the proposal feature; 3-layer MLP box, linear class; stage t+1 takes refined boxes and features of stage t; gradient blocked at the proposal boxes of every stage except the initial ones; loss 2 focal + 5 L1 + 2 GIoU on matched pairs (optimal bipartite matching).

## Input

pyramid + queries. output: boxes + classes.

## Training

hungarian_set.

## Differentiable?

partial (matching; box gradient blocked).

## Sensors and data

images (COCO).

## Reported results

COCO val ResNet-50 FPN 36 epochs: 42.8 AP at 23 FPS (45.0 with 300 boxes); stages AP / FPS: 1: 21.7 / 35, 2: 36.2 / 33, 3: 39.9 / 29, 6: 42.3 / 23, 12: 41.6 / 17; box initialisation centre 41.5, image 42.3, grid 41.0, random 42.1 ("relatively robust").

## Compute

V100 FPS above.

## Code

URL from memory, not opened: https://github.com/PeizeSun/SparseR-CNN

## Fit to our project

reuse_idea - S - evidence that a few stages saturate and the starting hypotheses hardly matter.

## Needs

Hungarian. compat_conflicts: matching-free rule.

## Tried in our experiments

E31 / E32 matched sparse decoder dropped.
