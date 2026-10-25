# VS17 - ELSD: midpoint heat map + angle / length, scoring without any sampling along the line

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S3, S2.*

## Papers

ELSD (Zhang, Luo, Qin, He, Liu, arXiv 2021) https://arxiv.org/abs/2104.14205

## Idea

A segment is (midpoint, angle, length); a Gaussian-peak midpoint heat map multiplied by a "line-centerness" gives the only confidence; there is no verification or classifier; a separate descriptor branch pools features along the segment with a rotated RoI.

## Formulation

endpoints = centre +- (rho / 2)(cos theta, sin theta); line-centerness = sqrt(min(d1, d2) / max(d1, d2)); final centre confidence = P_mid * P_centerness^0.5 (non-centerness suppression); fine sub-pixel offset; shared features 128 x 128, heads at 256 x 256. Losses: lambda_mid 25 (CornerNet-style focal, alpha 2, beta 4), lambda_cen 10 (weighted BCE), lambda_geo 1 (angle L1 x 300, length smooth-L1 x 10), lambda_off 3; descriptor: rotated RoI bilinear sample (width and number of sample points not stated), 1 x W' depth-wise conv, stride-S max-pool, FC, L2 norm, descriptor length 256 / 64 / 36, hard-negative triplet loss; Adam, 170 epochs, lr 1e-3, batch 16.

## Input

image feature maps. output: segments + descriptors.

## Training

dense heat-map targets, homography self-supervision for descriptors; matching_free.

## Differentiable?

yes except peak picking.

## Sensors and data

images (Wireframe, YorkUrban).

## Reported results

Wireframe sAP5 / 10 / 15 64.3 / 68.9 / 70.9, APH 87.2, FH 82.3, 42.6 FPS (Res34, 512); Lite (256) 57.4 / 63.1 / 65.5 at 107.5 FPS; YorkUrban sAP10 30.2; descriptor line matching F-score 74.7 vs 66.4 for LBD (Table 3).

## Compute

38 FPS with the descriptor branch.

## Code

not checked.

## Fit to our project

reuse_idea - S - a symmetric "centerness" ratio (min / max of the two end distances) as a cheap centred-ness prior; confirms that later line detectors (LETR, ELSD, ScaleLSD) dropped sampled verification once proposals are informative.

## Needs

none. compat_conflicts: none.

## Tried in our experiments

none.
