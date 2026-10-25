# SN03 - Localization is all you evaluate: location leakage in nuScenes / Argoverse 2 splits

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: D, E.*

## Papers

Localization Is All You Evaluate: Data Leakage in Online Mapping Datasets and How to Fix It (Lilja, Fu, Stenborg, Hammarstrand; CVPR 2024) https://arxiv.org/abs/2312.06420 ; splits https://github.com/LiljaAdam/geographical-splits (named in a search result, not opened)

## Idea

On the official splits most validation / test samples lie within 5 m of a training sample, so online-mapping models can score well by recalling a memorised implicit map. Geographically disjoint splits (near and far extrapolation) lower every method sharply and change which design choices look useful.

## Formulation

leakage = share of val / test samples with a training sample within 5 m. Near-extrapolation split: regions defined manually from map attributes, 70 / 15 / 15 train / val / test, original test set kept; far-extrapolation: city folds for cross-validation.

## Input

sample locations; models re-run with their original configs.

## Output

new splits + re-evaluation.

## Training

unchanged regimes (HDMapNet, VectorMapNet, MapTR, MapTRv2, StreamMapNet).

## Differentiable?

no - evaluation protocol.

## Sensors and data

nuScenes, Argoverse 2; camera (one MapTRv2 camera + lidar row).

## Reported results

within 5 m of a training sample, original splits: nuScenes val 79.4 % / test 85.5 %, AV2 45.0 % / 41.9 %; near-extrapolation 0.9 % / 1.1 % and 0 % (Table 2). nuScenes mAP val / test original -> near split: MapTR R50 48.8 / 59.9 -> 19.0 / 19.7; MapTRv2 61.5 / 72.7 -> 26.7 / 26.2; StreamMapNet 62.6 / 74.4 -> 26.1 / 27.9; MapTRv2 camera + lidar 54.0 / 65.8 -> 20.0 / 20.1; HDMapNet IoU 31.3 / 38.8 -> 17.1 / 21.2. Average drop > 35 mAP on nuScenes; far-extrapolation CV mean for MapTRv2 21.4. Lifter ranking flips (GKT 19.0, BEVFormer 20.6, LSS 21.0); the gain from depth supervision shrinks. The authors show a model drawing dividers in occluded areas that no image shows.

## Compute

not stated.

## Code

splits repository named above (not opened).

## Fit to our project

reuse_idea - effort S - our TRAIN / VAL / TEST are scene-disjoint, not necessarily location-disjoint, and the foundation saw its own train scenes: report a location-distance-binned or location-disjoint TEST result next to the scene-disjoint one.

## Needs

sample ego positions in the global frame (nuScenes ego poses).

## Conflicts

none; it is an evaluation protocol.

## Tried in our experiments

none (our splits are scene-disjoint only).
