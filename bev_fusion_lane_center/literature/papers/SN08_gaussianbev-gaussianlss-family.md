# SN08 - GaussianBeV / GaussianLSS family (camera-only Gaussian view transform)

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (GaussianBeV), not found (GaussianLSS)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B.*

## Papers

GaussianBeV: 3D Gaussian Representation meets Perception Models for BeV Segmentation (Chabot et al., WACV 2025) https://arxiv.org/abs/2407.14108 ; GaussianLSS (only seen as a table row and as the rasteriser source in GaussianCaR; not read)

## Idea

One Gaussian per image pixel per camera; a depth head and a 3D offset head place it, an allocentric quaternion gives its rotation; a feature-rendering rasteriser with orthographic projection splats them to a 200 x 200 BEV that goes into an LSS or Simple-BEV backbone and segmentation heads.

## Formulation

EfficientNet-b4 + Simple-BEV neck, 128 channels; z = (f_n / f)(1/d - 1) with d a disparity in [0, 1]; scale abs, opacity sigmoid; loss = segmentation (BCE, centerness, offset) + early loss on the rasteriser output + log-depth loss; AdamW 3e-4, wd 1e-7, up to 100 epochs; BEV 100 x 100 m at 0.5 m.

## Input

6 images (224 x 448 or 448 x 800).

## Output

vehicle, pedestrian, drivable area, lane boundary maps.

## Training

regime matching_free; nuScenes (28,130 / 6,019 images stated; the text says 750 sequences).

## Differentiable?

yes.

## Sensors and data

camera only, nuScenes, 200 x 200 at 0.5 m.

## Reported results

val IoU 448 x 800: drivable 82.6, lane boundaries 47.4 (MatrixVT drivable 83.5, lane 44.8); vehicles (visibility > 40 %) 47.5 at 224 x 448, 50.3 at 448 x 800; without a BEV backbone 45.9.

## Compute

24 fps and 13 fps (A100) for the two resolutions.

## Code

not checked.

## Fit to our project

reuse_idea - effort S - GaussianCaR's camera branch is this design with a coarse-to-fine depth; GaussianCaR's map protocol follows this paper and BEVCar (evaluation at 448 x 800).

## Needs

images.

## Conflicts

none new.

## Tried in our experiments

none.
