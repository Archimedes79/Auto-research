# VS39 - PolarMask: rays from a centre with regressed lengths and the Polar IoU loss

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2, L.*

## Papers

PolarMask (Xie et al., CVPR 2020) https://arxiv.org/abs/1909.13226

## Idea

An instance mask is a centre plus n rays at fixed angles; only the ray lengths are regressed; a loss built on the ratio of the sum of minima to the sum of maxima of predicted and target lengths approximates mask IoU.

## Formulation

n = 36 rays at 10 degree spacing; contour point x_i = cos(theta_i) d_i + x_c, y_i = sin(theta_i) d_i + y_c; Polar IoU = sum d_min / sum d_max, loss = log(sum d_max / sum d_min) (the squared form changes results by +-0.1 mAP); polar centerness = sqrt(min d / max d), multiplied with the class score at inference; centre samples within 1.5 x stride of the mass centre (about 9 - 16 pixels per instance); a ray that hits the contour several times takes the maximum length; a ray from an outside centre that never meets the contour gets eps = 1e-6; FCOS-style head, box branch not needed.

## Input

image_feat levels. output: point_set (n ray lengths per location).

## Training

dense centre sampling, no Hungarian.

## Differentiable?

yes.

## Sensors and data

images (COCO).

## Reported results

COCO minival ResNet-50: rays 18 / 24 / 36 / 72 -> AP 26.2 / 27.3 / 27.7 / 27.6; Polar IoU loss 27.7 vs best smooth-L1 25.1 (Table 1b); adding polar centerness raises 27.7 to 29.1 (Table 1c); test-dev 32.9 AP (ResNeXt-101). Limits named: concave and donut-shaped instances (neither centre lies inside), multiple intersections.

## Compute

not stated.

## Code

URL from memory, not opened: https://github.com/xieenze/PolarMask

## Fit to our project

reuse_idea - S - evidence for the number of rays (18 -> 36 gains 1.5 AP, 72 adds nothing) and for a scale-free ratio loss on lengths; our teardrop uses 12 rays, below the smallest tested count, but serves forks, not mask area.

## Needs

a centre.

## Conflicts

closed star-shaped targets; a lane graph is open and branching.

## Tried in our experiments

none.
