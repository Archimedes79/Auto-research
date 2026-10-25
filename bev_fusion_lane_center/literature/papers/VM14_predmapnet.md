# VM14 - PredMapNet (decayed instance-mask history + short-term future polylines)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S8, E.*

## Papers

PredMapNet: Future and Historical Reasoning for Consistent Online HD Vectorized Map Construction (arXiv 2602.16669, Feb 2026) https://arxiv.org/abs/2602.16669

## Idea

Built on MapTracker. (i) A History Rasterized Map Memory keeps per-instance BEV masks, decayed (beta = 0.9) with confidence weighting and warped to the next frame by a grid-based ego warp; a mask-guided module (threshold 0.5) lets each track query cross-attend to BEV features under its own warped mask. (ii) A short-term future module predicts per-point offsets from the stacked past polylines (P_{t+1} = P_t + dP) and mean-pools the predicted polyline into the track query; supervised by Chamfer to the GT future polyline. (iii) A Mask2Former-style semantic-aware query generator.

## Formulation

tracks pruned below tau_t = 0.5, births above tau_d = 0.4; tau_L = 0.5; Nq = 100, Np = 20, C = 512; loss = BEV + track (MapTracker) + seg (2 Dice + 1 BCE) + future Chamfer + depth, unweighted sum; 72 epochs nuScenes (18 + 6 + 48), 35 AV2; 4 x A100; BEV range, lr, batch not stated.

## Input

as MapTracker; memory = instance masks + last n polylines per track.

## Output

polylines with class and track IDs.

## Training

as MapTracker plus future Chamfer; whether history is GT or predicted during training is NOT described.

## Differentiable?

partial (thresholds, Hungarian for births).

## Sensors and data

camera; nuScenes, AV2; old and non-overlapping splits.

## Reported results

nuScenes old split val, 24 ep: 73.2 mAP / 64.3 C-mAP (Mask2Map 71.6 / 55.8, MapTRv2 61.5 / 41.7, HRMapNet 67.3 / 49.2, StreamMapNet 63.4 / 38.4); 72 ep: 76.9 / 69.7 at 10.1 FPS vs MapTracker 76.1 / 69.1 at 10.9; AV2 77.3 / 69.1; non-overlapping nuScenes 42.1 / 33.7 vs MapTracker 40.3 / 32.5.

## Compute

10.1 FPS.

## Code

project page https://astronirav.github.io/predmapnet (no repository link on the page)

## Fit to our project

reuse_idea - S/M - the decayed, confidence-weighted, ego-warped per-instance mask memory is the closest relative of a recursively fused validity; the "future" offsets are a motion prior (static lanes need none).

## Needs

per-instance masks (our pieces / edges can be rasterised); ego warp; track identity. Cached-BEV: memory yes.

## Conflicts

relies on MapTracker identities and Hungarian births; the decay constant is hand-set, not a log-odds model.

## Tried in our experiments

none.
