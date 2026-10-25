# VS14 - DeepLSD: distance and angle fields with per-sample inlier verification

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S3, S5.*

## Papers

DeepLSD (Pautrat et al., CVPR 2023) https://arxiv.org/abs/2212.07766

## Idea

The network predicts a line distance field (DF) and angle field (AF); a classical detector runs on a surrogate gradient built from them; each candidate line is verified by the fraction of 50 samples that agree with DF and AF; accepted lines are refined by least squares on 10 samples.

## Formulation

DF = r e^{-D_n}, r = 5 px; surrogate gradient magnitude r - D, angle AF - pi/2; filter: n_f = 50 points per line, keep the line if more than 50 % of samples have DF < eta_DF = 1.5 and angle within eta_theta = pi/9; refinement: n_opt = 10 samples, cost = angle + mean DF + vanishing-point term, weights 1 / 1 / 0.2, only orientation and midpoint offset change; loss L1 on normalised DF + circular L2 on AF within r of a line; pseudo ground truth from homography adaptation (N not stated); Adam 1e-3, about 12 h on one RTX 2080.

## Input

image. output: DF, AF, segments.

## Training

self_supervised pseudo labels.

## Differentiable?

partial (verification and refinement are classical, after the network).

## Sensors and data

images (Wireframe, MegaDepth).

## Reported results

HPatches structural repeatability 0.367, orthogonal localisation error 0.818; York Urban vanishing-point error 1.63, AUC 85.6; refinement lowers HAWP / TP-LSD orthogonal error by up to 32 % (paper).

## Compute

about 12 h training.

## Code

URL from memory, not opened: https://github.com/cvg/DeepLSD

## Fit to our project

reuse_idea - S - free, non-learned verifier for a ray: fraction of samples where heat map and direction channel agree with the ray; a training-free baseline and sanity check for nu.

## Needs

dense distance / angle fields (our dense head: heat map + 2 direction channels).

## Conflicts

threshold on the inlier fraction is a hard decision.

## Tried in our experiments

none.
