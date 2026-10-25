# VS41 - PolarBEVDet: polar BEV grid with equal cell budget, polar-aware temporal alignment

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M (PolarBEVDet), A (adaptive BEV)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B, S8, S2.*

## Papers

PolarBEVDet (Yang et al., arXiv 2024) https://arxiv.org/abs/2408.16200 ; related polar adaptive BEV (Yang et al., ICRA 2023) https://arxiv.org/abs/2304.00525 (search summary only)

## Idea

Lift-splat into a polar grid (azimuth x radius) with the same cell count as a Cartesian one, so cells are dense near the ego and sparse far away; temporal fusion converts polar cells to Cartesian, applies ego motion, converts back and samples bilinearly.

## Formulation

azimuth [-pi, pi] in N_theta bins, radius [r_min, r_max] in N_r bins (values not stated); default 256 x 64 = 16,384 cells (large backbones 384 x 96), same as a 128 x 128 Cartesian grid; centre-based head predicting (theta, r, z, w, l, h, alpha_p = alpha_c - theta, v_theta, v_r); alignment: theta_t = -pi + i delta_theta, r_t = r_min + j delta_r, x = r cos theta, y = r sin theta, ego-motion transform, back to polar indices, bilinear sampling, 1x1 conv fusion.

## Input

image_feat from six cameras. output: 3-D boxes.

## Training

heat-map targets (no Hungarian).

## Differentiable?

yes (bilinear sampling).

## Sensors and data

nuScenes cameras.

## Reported results

val R50 256 x 704: baseline 0.496 NDS / 0.397 mAP; + polar 0.509 / 0.414; + 2-D auxiliary supervision 0.521 / 0.429; + SAE 0.527 / 0.434; under an azimuth revolve test Cartesian drops 2.4 NDS, polar 0.4; porting to BEVDet 0.360 -> 0.382 NDS, BEVDet4D 0.426 -> 0.450, BEVDepth 0.457 -> 0.474. Text and Table I disagree on the gain over the best baseline (1.9 % mAP / 2.1 % NDS in the text vs +2.3 mAP / +1.9 NDS in the table); no measured speed or memory comparison.

## Compute

none stated.

## Code

URL from memory, not opened: https://github.com/Yzichen/PolarBEVDet

## Fit to our project

reuse_idea - S - the cell-budget argument (dense near, sparse far) and the polar -> Cartesian -> ego motion -> bilinear recipe, the same as our ego-warping by grid_sample.

## Needs

raster.

## Conflicts

none.

## Tried in our experiments

E41 temporal feature fusion (+0.008 F1).
