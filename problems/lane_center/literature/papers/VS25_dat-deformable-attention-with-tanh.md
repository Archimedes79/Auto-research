# VS25 - DAT: deformable attention with tanh-bounded offsets

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2.*

## Papers

Vision transformer with deformable attention (Xia et al., CVPR 2022) https://arxiv.org/abs/2201.00520

## Idea

Reference points form a uniform grid; a small network predicts shifts; the shifts are bounded by a scaled tanh to keep sampling local.

## Formulation

offset network = 5x5 depthwise conv, GELU, 1x1 conv without bias -> 2 channels per group; Delta p <- s * tanh(Delta p), s = 2 ("without additional tuning"); G offset groups (3 / 6 for DAT-T and DAT-S stages 3 / 4); bilinear sampling; relative position bias interpolated at continuous displacements. No ablation with versus without the bound; a sweep of s from 0 to 16 is stable (values only in a figure). Learning-rate scaling of the offset net: not stated.

## Input

feature map. output: sampled keys / values.

## Training

image classification; none matching.

## Differentiable?

yes.

## Sensors and data

ImageNet-1K.

## Reported results

DAT-T top-1: no offsets, no position embedding 81.4; offsets only 81.8; relative position bias only 81.7; both 82.0.

## Compute

not stated.

## Code

URL from memory, not opened: https://github.com/LeapLabTHU/DAT

## Fit to our project

reuse_idea - S - offset = r_max * tanh(.) with r_max a fraction of the ring gap.

## Needs

none. compat_conflicts: none.

## Tried in our experiments

none.
