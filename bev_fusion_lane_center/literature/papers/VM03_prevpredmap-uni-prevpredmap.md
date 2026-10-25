# VM03 - PrevPredMap / Uni-PrevPredMap (previous vector predictions as priors)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly (fetched text; transform and loss details missing).. Pipeline stage: S8, T.*

## Papers

PrevPredMap: Exploring Temporal Modeling with Previous Predictions for Online Vectorized HD Map Construction (arXiv 2407.17378; authors not read) https://arxiv.org/abs/2407.17378 ; Uni-PrevPredMap: Extending PrevPredMap to a Unified Framework of Prior-Informed Modeling (Peng et al., IEEE RA-L vol. 11 no. 10, Oct 2026, arXiv 2504.06647) https://arxiv.org/abs/2504.06647

## Idea

The previous frame's top-k VECTOR predictions (class, confidence, location), ego-transformed, become content + position queries of the current frame. Training uses no GT-derived prior: each sample's prior is a model prediction stored in a dictionary from earlier iterations, so it has the real error distribution. Dual-mode training: p = 0.5 of steps run single-frame; inference is single-frame on the first frame, temporal afterwards. Uni- version: predictions (and an optional corrupted HD-map prior) are stored in a tile-indexed global vector map retrieved by UTM position, rasterised into heatmaps and concatenated with the BEV feature (as in HRMapNet); tri-mode training none / temporal / temporal + map (ratio 0.5 : 0.3 : 0.2).

## Formulation

k = 10 (nuScenes), 8 (AV2) of 100 queries, saturating near 12. Category + confidence -> linear -> content query; location -> initial location prediction -> sinusoidal + linear -> position query; position queries re-encoded per decoder layer; group-wise one-to-many branch with 6 groups. 100 queries x 20 points, 6 layers, R50 + LSS BEVPoolv2, AdamW 6e-4, wd 0.01, batch 16, 4 A100, 24 / 110 epochs. Uni-: the only training perturbation of the map prior is instance-level displacement with magnitude uniform in [-6, 6] m on subsets (about half of the vectors, mean about 3 m); additions, deletions, rotation, scaling appear only as inference robustness tests. Transform formula and loss weights are not on the pages.

## Input

previous predicted polylines (top-k) + pose (Uni-: global pose); current BEV.

## Output

polylines + class (fixed set).

## Training

MapTRv2-style hierarchical matching + one-to-many group branch; regime hungarian_set with stored predictions as prior.

## Differentiable?

partial - top-k, stored predictions carry no gradient, Hungarian.

## Sensors and data

camera; nuScenes (old and non-overlapping), AV2 2 Hz / 10 Hz; 60 x 30 m (and 100 x 50 m for Uni-), 0.3 m grid.

## Reported results

nuScenes 24 ep: PrevPredMap 66.3 mAP vs SQD-MapNet 65.0, StreamMapNet 63.4 (30 ep); 110 ep: 71.3 vs MapTRv2* 69.2; AV2 2D 2 Hz: 68.2 vs 63.3 vs 61.5. Uni-PrevPredMap, nuScenes standard split, no prior: 74.0 (24 ep) / 77.0 (72 ep); with map prior 80.9 / 81.8; non-overlapping split 39.7 -> 52.5 with map prior.

## Compute

not stated.

## Code

https://github.com/pnnnnnnn/PrevPredMap ; https://github.com/pnnnnnnn/Uni-PrevPredMap (not opened)

## Fit to our project

reuse_idea - S - vector-level prior independent of the BEV encoder; the stored-predictions training and the single-frame / temporal mode switch are what a recursion trained on its own outputs needs, with a single-frame fallback for frame 0 and for failures.

## Needs

a per-frame store of model predictions refreshed across epochs; ego transform; a prior-input path in the recursion block. Cached-BEV: yes for the prior path.

## Conflicts

stored predictions are stale by up to one epoch and need a warm-up epoch; the decoder is query / Hungarian based; Uni- needs a global pose.

## Tried in our experiments

none.
