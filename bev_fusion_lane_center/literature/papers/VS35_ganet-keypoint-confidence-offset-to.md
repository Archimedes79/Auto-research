# VS35 - GANet: keypoint confidence + offset to lane start, lane-aware aggregation along neighbours

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2, S6.*

## Papers

A keypoint-based global association network for lane detection (Wang et al., CVPR 2022) https://arxiv.org/abs/2204.07335

## Idea

Lane detection as keypoint estimation; each keypoint regresses an offset to its lane's start point (association in parallel); a lane-aware aggregator reads features at M = 9 neighbouring keypoints on the same lane with predicted offsets.

## Formulation

confidence map (Gaussian splat, element-wise max on overlaps), penalty-reduced focal loss (alpha 2, beta 4, weight 1.0), quantisation L1 (1.0), start-point offset L1 (0.5), LFA auxiliary SmoothL1 with Hungarian matching of predicted to GT neighbour offsets (1.0); association: a keypoint joins a lane if its estimated start lies within theta_dis = 4 of that lane's start; LFA: F^(p_i) = sum_m w_m F(p_i + dp_i^m), M = 9; score threshold 0.4; 800 x 320; Adam 1e-3 poly, 300 epochs (TuSimple), 40 (CULane).

## Input

image_feat / raster. output: keypoints + start offsets.

## Training

dense + auxiliary hungarian_set (LFA only).

## Differentiable?

yes except grouping.

## Sensors and data

images (TuSimple, CULane).

## Reported results

CULane F1: S 78.79 (153 FPS), M 79.39, L 79.63 (63 FPS), curve F1 75.92 / 76.32 / 77.37; TuSimple F1 97.71 / 97.68 / 97.45; ablation (S): no LFA 77.84, LFA 78.30, LFA + aux 78.79.

## Compute

FPS above.

## Code

URL from memory, not opened: https://github.com/Wolfwjs/GANet

## Fit to our project

reuse_idea - S - "offset to a lane-level anchor" as a way to group sampled points into instances without Hungarian.

## Needs

start points per lane.

## Conflicts

lane graphs have several starts per path.

## Tried in our experiments

E42 / E45 learned link heads within 0.01 of the rules.
