# MH23 - Group DETR (K groups of queries, each matched one-to-one; one group at inference)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L, T.*

## Papers

Group DETR: Fast DETR Training with Group-Wise One-to-Many Assignment (Chen et al., ICCV 2023) https://arxiv.org/abs/2207.13085 (read via ar5iv)

## Idea

Keep Hungarian one-to-one matching, but run K independent groups of N queries through the same decoder (self-attention restricted to the group); each GT gets K positives, one per group. Only group 1 is used at inference. Naive one-to-many assignment inside one group makes duplicates.

## Formulation

L = (1/K) * sum_k sum_n l(y_sigma_k(n), ybar_kn), sigma_k = Hungarian matching of group k; K = 11 by default (gain saturates); weights shared; separate self-attention per group is essential (group-wise one-to-many alone 34.8 mAP; with separate self-attention 37.6; naive one-to-many 8.4 vs 32.6 baseline, Conditional DETR 12 epochs). - input / output: DETR-family decoders; sets of boxes + classes.

## Training

12 / 36 / 50 epochs on COCO; regime `hungarian_set` per group.

## Differentiable?

yes.

## Sensors and data

COCO val2017.

## Reported results

12-epoch mAP baseline -> Group DETR: Conditional DETR 32.6 -> 37.6; DAB-DETR 35.2 -> 39.1; DN-DETR 38.6 -> 40.6; DAB-Deformable 44.2 -> 45.7; DINO 4-scale 49.4 -> 50.1; at 50 epochs DINO +0.4.

## Compute

training only: about 1.2 GB (Conditional) / 1.7 GB (DAB) more memory, about 5 min more per epoch.

## Code

https://github.com/Atten4Vis/GroupDETR (named in a search result)

## Fit to our project

not_suitable as published (Hungarian in every group) - effort - ; the transferable idea is K positives per target in training, one query set at inference.

## Needs

per-group Hungarian; independent self-attention per group.

## Conflicts

Hungarian is excluded by project rule; a tree has no query set.

## Tried in our experiments

E31 / E32 (matched sparse decoder) dropped
