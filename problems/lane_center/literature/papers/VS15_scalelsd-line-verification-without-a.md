# VS15 - ScaleLSD: line verification without a neural verifier (support degree)

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S3.*

## Papers

ScaleLSD (Ke, Tan, Zheng, Shen, Wu, Xue, arXiv 2025) https://arxiv.org/abs/2506.09369

## Idea

Keeps the HAWP attraction-field parametrisation but drops LoI pooling: a candidate line is scored by its support degree (number of pixels whose decoded junction pair matches it); training is self-supervised at scale.

## Formulation

proposals with fewer than tau_l = 10 supporting pixels are discarded in training and pseudo-labelling, tau_l = 5 at inference; the paper states there are no neural modules for verification; BCE only for endpoint regression; pseudo labels from a synthetic bootstrap model (8 primitives) and an "LSD-Rectifier" that replaces the direction field by the classical level-line orientation; data about 16k synthetic, 20k Wireframe, over 10M images; 128 samples per line only for a 3-D reconstruction evaluation.

## Input

image. output: scored segments.

## Training

self_supervised.

## Differentiable?

partial.

## Sensors and data

images.

## Reported results

zero-shot repeatability HPatches Rep-5 (S) 0.367 vs 0.322 (HAWPv3), RDNIM 0.337 vs 0.278; vanishing-point error 1.55 vs 1.63 (DeepLSD); sAP and inference speed not stated.

## Compute

pseudo-label generation 0.607 s per batch with the rectifier vs 3.653 s with 10-iteration homography adaptation.

## Code

URL from memory, not opened: https://github.com/ant-research/scalelsd

## Fit to our project

reuse_idea - S - evidence that agreement with dense fields can replace a learned verifier.

## Needs

dense fields. compat_conflicts: none.

## Tried in our experiments

none.
