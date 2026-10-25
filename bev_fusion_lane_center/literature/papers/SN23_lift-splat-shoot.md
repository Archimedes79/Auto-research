# SN23 - Lift-Splat-Shoot (and the frozen-encoder baseline inside it)

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B.*

## Papers

Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D (Philion, Fidler; ECCV 2020; from memory) https://arxiv.org/abs/2008.05711 ; code https://github.com/nv-tlabs/lift-splat-shoot (URL from memory, not opened)

## Idea

Per-pixel categorical depth distribution times a context vector lifts features into a frustum, pillar pooling (cumulative-sum trick) splats into a BEV grid, a small BEV CNN decodes; planning is a cost-volume classification over 1000 template trajectories.

## Formulation

images 128 x 352, depths 4 - 45 m at 1 m; BEV +-50 m, 0.5 m cells, 200 x 200; EfficientNet-B0 + ResNet-18 style BEV net, 14.3 M parameters; Adam 1e-3, 300k steps; BCE weights 1 vehicle, 5 lane, 1 road.

## Input

6 images.

## Output

BEV vehicle / drivable / lane-boundary maps, cost map.

## Training

regime matching_free.

## Differentiable?

yes.

## Sensors and data

camera, nuScenes, Lyft.

## Reported results

map IoU (Table 2): drivable 72.94, lane boundary 19.96; baselines CNN 68.96 / 16.51, OFT 71.69 / 18.07, Frozen Encoder 61.62 / 16.95; vehicles 32.07; transfer nuScenes -> Lyft vehicle: frozen encoder 15.82 vs LSS 22.59; performance falls roughly linearly with distance; night performance drops; single timestep only.

## Compute

35 Hz on a Titan V.

## Code

see papers line.

## Fit to our project

reuse_idea - effort S - the "frozen encoder" row (frozen ImageNet encoder, trained lift and decoder) loses 11 IoU on drivable area against end-to-end; but it freezes a generic encoder, not a map-pretrained one like ours.

## Needs

images.

## Conflicts

none new.

## Tried in our experiments

none.
