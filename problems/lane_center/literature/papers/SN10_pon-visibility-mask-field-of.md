# SN10 - PON: visibility mask = field of view AND lidar ray passes; entropy loss on invisible cells

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (visibility, loss, extent, split)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: D, L, E.*

## Papers

Predicting Semantic Map Representations from Images using Pyramid Occupancy Networks (Roddick, Cipolla; CVPR 2020; authors / venue from memory) https://arxiv.org/abs/2003.13402 ; code https://github.com/tom-roddick/mono-semantic-maps (URL from memory, not opened)

## Idea

BEV ground truth is rasterised from map and box annotations; a cell counts as visible only if it lies in the camera field of view and at least one lidar ray passes through it. Invisible cells are ignored by the cross-entropy and by the evaluation, and get an entropy-maximising loss so the network is not pushed to a confident answer there.

## Formulation

balanced BCE (weights = sqrt of inverse class frequency); entropy-maximising uncertainty loss with weight 0.001 applied only to non-visible cells; IoU at 0.5 probability with non-visible cells ignored; BEV 50 m ahead, 25 m to each side, 25 cm per pixel.

## Input

monocular image per camera.

## Output

BEV occupancy per class + high entropy where unobserved.

## Training

regime matching_free; train / val sequences redistributed because the default splits share road segments.

## Differentiable?

yes (loss terms; the mask is a fixed label-side tensor).

## Sensors and data

camera (lidar only to build the mask), nuScenes, Argoverse.

## Reported results

none needed here (mask design).

## Compute

not stated.

## Code

see papers line.

## Fit to our project

reuse_idea - effort S - template for censoring: no loss on invisible samples plus an optional entropy term; the lidar-ray condition is not available from calibration alone (see SN13).

## Needs

a per-cell visibility mask at training time.

## Conflicts

lidar rays for the second condition.

## Tried in our experiments

none.
