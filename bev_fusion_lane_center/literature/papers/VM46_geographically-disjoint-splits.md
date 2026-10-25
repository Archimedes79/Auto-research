# VM46 - Geographically disjoint splits (StreamMapNet split, Near / Far Extrapolation)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly (HTML read; sample counts of the Far folds not read). ---. Pipeline stage: D, E.*

## Papers

Localization Is All You Evaluate: Data Leakage in Online Mapping Datasets and How to Fix It (Lilja et al., CVPR 2024) https://arxiv.org/abs/2312.06420 ; the new splits of StreamMapNet (VM01); the comparison in MapLightning (VM41)

## Idea

Random scene splits of nuScenes and AV2 put most validation / test samples next to training samples, so models can recall the training-area map; geographically disjoint splits remove this.

## Formulation

leakage measure = share of val / test samples within 5 m of a training sample: nuScenes 79.4 % (val) / 85.5 % (test), AV2 45.0 % / 41.9 % (Lilja Tab. 2). Lilja Near Extrapolation: regions defined by hand from map attributes, target about 70 / 15 / 15, zone classes (residential / commercial / industrial) and for nuScenes weather and time of day preserved, Boston and Singapore NOT separated, val carved from the original training data, test = original test; nuScenes 28 k / 6 k / 6 k samples, no buffer (filtering val / test within 60 m of training changed results negligibly). Far Extrapolation: whole cities as folds for cross-validation (Boston 55 % of nuScenes samples). StreamMapNet split (area, 30 m radius): nuScenes overlap 85 % -> 11 % (train 1.64 km2, val 0.54 km2), AV2 54 % -> 0 %, 700 / 150 scenes. MapLightning: the StreamMapNet split has 2.1 % and Near-Extrapolation 0.9 % overlap (5 m criterion) on nuScenes.

## Input

frame positions of all splits.

## Output

overlap shares, splits.

## Training

none.

## Differentiable?

no.

## Sensors and data

nuScenes, AV2.

## Reported results

test mAP original -> Near (Lilja): VectorMapNet 54.8 -> 18.2, MapTR (R50 GKT) 59.9 -> 19.7, MapTRv2 72.7 -> 26.2, StreamMapNet 74.4 -> 27.9 (largest drop 46.5 points); AV2 MapTRv2 65.3 -> 55.2; HDMapNet IoU 38.8 -> 21.2; rankings largely preserved; lifting-method and depth-supervision conclusions weaken. Best published on the StreamMapNet split (60 x 30 m): MapLightning 45.8, Skeptic 45.4, PredMapNet 42.1, HisTrackMap 41.5, MapTracker 40.3, MambaMap 40.1 (3(a)-C).

## Compute

not applicable.

## Code

https://github.com/LiljaAdam/geographical-splits

## Fit to our project

reuse_idea - S - our split is scene-disjoint (212 / 22 / 21 scenes from the first three blobs) but NOT location-disjoint; with only two cities (a handful of map areas), VAL / TEST frames can lie within a few metres of training frames. Compute the 5 m overlap from nuScenes ego poses and report an overlap-filtered subset or a region split as a robustness check; the frozen foundation GaussianCaR may also have seen these scenes (not checkable here).

## Needs

ego poses (global) of all frames.

## Conflicts

with 255 scenes a region split is expensive.

## Tried in our experiments

none (split is scene-disjoint by design).
