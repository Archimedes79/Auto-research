# VM15 - HisTrackMap (per-track history maps, global-map metric)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S8, E.*

## Papers

HisTrackMap: Global Vectorized High-Definition Map Construction via History Map Tracking (arXiv 2503.07168; authors not read) https://arxiv.org/abs/2503.07168

## Idea

Each track query owns a rasterised history map (H x W = BEV size). New instances initialise it from their rasterised vector scaled by score; tracked instances decay it by lambda = 0.95 and add the new rasterisation; it is warped to the new ego position each step. "Map-Trajectory Prior Fusion" samples PV and BEV features where the history map exceeds tau_map = 0.5 and cross-attends the track queries to them. Also defines a global-map metric (G-mAP, VM45).

## Formulation

tau_det = 0.4, tau_track = 0.5, lambda = 0.95, Nq = 100, Np = 20, C = 512; the Motion-MLP transformation loss was removed, other losses as MapTracker (weights not stated); 72 epochs nuScenes (18 / 6 / 48), 35 AV2, 8 A100; BEV range, lr not stated.

## Input

as MapTracker.

## Output

polylines with IDs.

## Training

as MapTracker; regime hungarian_set with identity.

## Differentiable?

partial.

## Sensors and data

camera; nuScenes, AV2; old and non-overlapping splits.

## Reported results

nuScenes 72 ep: mAP 76.6, C-mAP 68.7, G-mAP 50.2, 10.3 FPS vs MapTracker 76.1 / 69.1 / 49.4, 10.9 FPS; AV2 35 ep: 77.7 / 69.5 / 48.6 vs 76.8 / 68.3 / 48.0; non-overlapping splits: nuScenes 41.5 / 32.8 / 28.5, AV2 71.4 / 62.7 / 49.2 (mAP / C-mAP / G-mAP).

## Compute

10.3 FPS.

## Code

none; project page https://yj772881654.github.io/HisTrackMap/

## Fit to our project

reuse_idea - S - decayed per-instance raster memory (same family as VM14); G-mAP defines how to evaluate a MERGED multi-frame map.

## Needs

as VM14.

## Conflicts

as VM14; the prior fusion samples perspective-view features (not available on cached BEV).

## Tried in our experiments

none.
