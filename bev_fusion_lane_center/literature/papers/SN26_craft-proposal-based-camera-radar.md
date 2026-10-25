# SN26 - CRAFT: proposal-based camera-radar fusion in polar coordinates

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B.*

## Papers

CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer (Kim et al., AAAI 2023; from memory) https://arxiv.org/abs/2209.06535

## Idea

Image-based 3D proposals (CenterNet-style, DLA-34, 448 x 800) gather radar points through soft polar association (azimuth between the corners, adaptive radial margin from depth variance, up to 128 points); a spatio-contextual fusion transformer updates the proposals; no BEV raster exists in the method.

## Formulation

gamma 5, delta 10 for the margin; radar PointNet++ on six sweeps (~0.5 s), 2048 points, RCS + ego-compensated radial velocity; 4 layers per encoder, 8 heads; camera detector 140 epochs then frozen, fusion 24 epochs.

## Input

image proposals + radar points.

## Output

boxes.

## Training

staged; frozen camera detector; regime matching_free.

## Differentiable?

partial - proposal threshold 0.05 and hard association.

## Sensors and data

camera + radar, nuScenes.

## Reported results

test NDS 52.3 / mAP 41.1, 4.1 FPS; polar beats Cartesian association (AP 41.3 vs 27.7).

## Compute

4.1 FPS (RTX 3090).

## Code

not checked.

## Fit to our project

not_suitable - effort L - needs per-object proposals; the idea of associating radar to a hypothesis in polar coordinates is the same geometry as our ray / tip.

## Needs

image proposals.

## Conflicts

object-centric, not for lanes.

## Tried in our experiments

none.
