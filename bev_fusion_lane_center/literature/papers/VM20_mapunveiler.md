# VM20 - MapUnveiler (clip tokens and memory tokens)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S8.*

## Papers

Unveiling the Hidden: Online Vectorized HD Map Construction with Clip-Level Token Interaction and Propagation (Kim et al., NeurIPS 2024) https://arxiv.org/abs/2411.11002

## Idea

Processes clips (T = 3 frames, stride 2, overlap 1 frame): 50 clip tokens generated from the frame-level map queries interact with the clips' BEV features (BEV updater: BEV as queries, clip tokens as keys / values) and a map generator produces map tokens; 96 memory tokens, read and written by token summarisers, carry information between clips. No ego-pose alignment is mentioned.

## Formulation

Nc = 50 (25 -> 69.2, 100 -> 68.5 mAP), M = 96; MapTRv2-style generator (one-to-one loss 2 / 5 / 0.005, PV seg 2; frame-level network also depth 3, BEV seg 1); frame-level MapNet pre-trained, then clip training; ResNet-50 + LSS, BEV 100 x 200; 24 ep nuScenes, 6 AV2; 8 A100, batch 16, AdamW 6e-4. First clip: memory written from learnable embeddings.

## Input

frame BEV + queries for T frames.

## Output

map tokens -> polylines.

## Training

Hungarian (MapTR one-to-one).

## Differentiable?

partial.

## Sensors and data

camera; nuScenes, AV2; 60 x 30 and 100 x 50 m; a geo-disjoint nuScenes split is reported.

## Reported results

nuScenes mAP 60 x 30: 68.0 (SQD 63.9, StreamMapNet 62.9, MapTRv2 61.5); 100 x 50: 68.7; occlusion subset 63.8 vs MapTRv2 47.1; geo-disjoint split 39.4 vs MapTRv2 36.6, StreamMapNet 33.9; 12.7 FPS (A100).

## Compute

12.7 FPS.

## Code

none; project page https://mapunveiler.github.io

## Fit to our project

reuse_idea - M - compact learned memory tokens instead of full BEV memories; low priority because gains over tracker-style memory are unclear and pose is ignored.

## Needs

clip dataloader; token summariser. Cached-BEV: yes (works on per-frame BEV + queries).

## Conflicts

map tokens are DETR queries.

## Tried in our experiments

none.
