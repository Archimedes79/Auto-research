# PL40 - HiVT (lane segments as independent vectors; common consumer of online maps on nuScenes)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **C (dataset code + README); paper not read** (A abstract, M method, C code page). Verified: partly. Pipeline stage: S10.*

## Papers

Hierarchical Vector Transformer for Multi-Agent Motion Prediction (Zhou et al., CVPR 2022; paper not read) ; code https://github.com/ZikangZhou/HiVT

## Idea

In the dataset code each lane centre-line becomes N - 1 segments (start point and difference vector, no resampling); lane attributes (is_intersection, turn direction, traffic control) repeat for all segments of the lane; every lane segment connects to every agent whose position at the last observed step lies within local_radius = 50; one scene-level rotation (AV position and heading at the last step); no lane-to-lane edges, no successor information.

## Formulation

not read beyond the dataset code.

## Input

lane segment vectors + attributes, agent histories.

## Output

multimodal agent trajectories.

## Training

not read.

## Differentiable?

yes.

## Sensors and data

Argoverse 1.1 GT map; Gu et al. (PL15) run it on nuScenes with online maps as a point set.

## Reported results

Argoverse 1.1 val minADE / minFDE / MR: HiVT-64 0.69 / 1.03 / 0.10, HiVT-128 0.66 / 0.97 / 0.09 (README).

## Compute

README: 35-40 min per epoch on an RTX 2080 Ti; preprocessing about 3.5 h; parameters not stated.

## Code

https://github.com/ZikangZhou/HiVT

## Fit to our project

baseline - effort M - external consumer through the PL15 pipeline; PyTorch 1.8 / PyG 1.7 stack.

## Needs

segments with attributes.

## Conflicts

old framework pins.

## Tried in our experiments

none.
