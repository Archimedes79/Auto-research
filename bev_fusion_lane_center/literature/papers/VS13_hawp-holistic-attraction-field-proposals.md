# VS13 - HAWP: holistic attraction field proposals + the same LoI verification

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S3, S4.*

## Papers

Holistically-attracted wireframe parsing (Xue et al., CVPR 2020) https://arxiv.org/abs/2003.01663

## Idea

Instead of enumerating all junction pairs, a 4-D attraction field proposes lines directly (about 4k per image against 22k for L-CNN); the same LoI head verifies them; a line survives only if both endpoints match junction proposals.

## Formulation

AFM per pixel (d, theta, theta1, theta2), d normalised by d_max = 5; up to 3 candidates per point d' = d + kappa * delta_d, kappa in {-1, 0, 1}; junction head as in L-CNN (lambda_msk 8, lambda_off 0.25). Verification: s = 32 uniform points, bilinear (RoIAlign-like), 1-D max-pool, small FC head, BCE; positive if the endpoint-matched maximum distance to a GT line < eta = 1.5 (GT lines are added as positives), 300 positives + 300 negatives per image; refinement: both endpoints within tau = 10 of junction proposals. L = L_LS (channel-wise L1) + L_Junc + L_Ver. Stacked hourglass (2 stacks), 512 x 512, Adam 4e-4, 30 epochs, batch 6, one V100.

## Input

feature map. output: scored segments.

## Training

matching_free (thresholds).

## Differentiable?

partial (junction NMS and matching).

## Sensors and data

images.

## Reported results

Wireframe sAP10 / sAP15 / msAP / APH / FH: HAWP 66.5 / 68.2 / 65.7 / 84.5 / 80.3 vs re-trained L-CNN 63.6 / 65.3 / 62.9 / 81.6 / 77.9; YorkUrban msAP 28.1 vs 26.8; 29.5 FPS vs 15.6 FPS. Ablation: HAWP is less sensitive to negative samplers because the proposals are informative; no ablation of proposal count or K.

## Compute

29.5 FPS on one GPU.

## Code

URL from memory, not opened: https://github.com/cherubicXN/hawp

## Fit to our project

reuse_idea - S - informative proposals ease the verifier = the teardrop's direction prior before verification; head unchanged from VS12.

## Needs

dense field head for proposals. compat_conflicts: as VS12.

## Tried in our experiments

none (see VS12 for E116).
