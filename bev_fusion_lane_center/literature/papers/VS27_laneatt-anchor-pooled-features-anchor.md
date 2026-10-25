# VS27 - LaneATT: anchor-pooled features + anchor-based attention

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2, S4.*

## Papers

Keep your eyes on the lane (Tabelini et al., CVPR 2021) https://arxiv.org/abs/2010.12035

## Idea

Fixed line anchors (origin on the image border + angle) pool the feature column the anchor passes through at each row; an attention layer lets each anchor aggregate the features of the other anchors.

## Formulation

N_pts = 72 points per lane at fixed y; 2,782 anchors in total, top N_anc = 1000 kept by a training-based filter (angles and origins not stated); backbone feature reduced by a 1x1 conv to C_F channels; a_i^loc = feature at the anchor's x for each feature row (zero padding outside; C_F * H_F values); attention weights w_ij = softmax over j != i of an FC layer on a_i^loc; a_i^glob = sum_j w_ij a_j^loc; concatenate local and global; heads: class (K + 1 = 2) and (length + N_pts offsets); positive if mean x-distance < tau_p = 15, negative if > tau_n = 20; focal + smooth-L1; NMS with the same distance (thresholds not stated).

## Input

image_feat + anchors. output: polyline with start and length.

## Training

threshold assignment (matching_free).

## Differentiable?

yes except NMS.

## Sensors and data

images (TuSimple, CULane, LLAMAS).

## Reported results

CULane F1 R18 / R34 / R122 75.09 / 76.68 / 77.02 (250 / 171 / 26 FPS); ablation R34: without anchor-based pooling 64.89 (vs 76.68), without attention 75.78, without focal 75.54; anchors 250 / 500 / 750 / 1000 / 1250: F1 68.68 / 75.45 / 75.80 / 76.66 / 75.91.

## Compute

22.13 M parameters (R34), 18.0 GMACs.

## Code

URL from memory, not opened: https://github.com/lucastabelini/LaneATT

## Fit to our project

reuse_idea - S - the largest ablation effect is line-aligned pooling (+11.8 F1); anchor count saturates near 1000 and degrades beyond.

## Needs

anchors as lines with fixed row sampling. compat_conflicts: one x per row; anchors from a border.

## Tried in our experiments

none.
