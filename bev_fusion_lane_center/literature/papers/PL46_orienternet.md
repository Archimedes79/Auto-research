# PL46 - OrienterNet (localisation in OSM by neural matching; the "no pose given" step)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S9.*

## Papers

OrienterNet: Visual Localization in 2D Public Maps with Neural Matching (Sarlin et al., CVPR 2023) https://arxiv.org/abs/2304.02009

## Idea

OSM rasterised (3 channels at 0.5 m, 128 x 128 m training maps; 50 classes: 7 area, 10 line, 33 point types, learned embeddings) and a neural BEV (32 x 32 m, 0.5 m, 8 channels + confidence) inferred from a single gravity-aligned image are correlated exhaustively over (x, y, theta) in the Fourier domain.

## Formulation

P = softmax(M + prior) over the pose grid, K = 64 rotations in training, 512 at test; the pose is the argmax; sequences are fused by multiplying warped probability volumes.

## Input

OSM raster, one image with known gravity.

## Output

pose with a probability volume.

## Training

loss = -log P at the GT pose (trilinear interpolation), supervised.

## Differentiable?

yes in training; argmax at inference.

## Sensors and data

single camera + OSM (MGL dataset).

## Reported results

MGL validation: position recall 15.78 % at 1 m, 47.75 % at 3 m, 58.98 % at 5 m; orientation 22.14 / 52.56 / 66.32 % at 1 / 3 / 5 deg.

## Compute

94 ms per localisation on an RTX 2080 (37 ms BEV, 51 ms matching); parameters not reported.

## Code

announced, no URL on the page.

## Fit to our project

reuse_idea - effort M - a probability volume over pose for the registration front end gives explicit uncertainty instead of regression.

## Needs

SD raster and BEV features.

## Conflicts

the exhaustive pose grid costs memory.

## Tried in our experiments

E48 (learned registration, kept) and E74 (registration front end).
