# VM40 - MapQR (scatter-and-gather query)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S4, S5.*

## Papers

Leveraging Enhanced Queries of Point Sets for Vectorized Map Construction (MapQR; authors not on the page, ECCV 2024 from memory) https://arxiv.org/abs/2402.17430

## Idea

Self-attention runs among the N = 100 INSTANCE queries only (cheaper than MapTR's N x N_v point queries); each instance query is then scattered into n = 20 copies with a shared content vector and individual positional embeddings from the reference points, cross-attended to the BEV, and gathered back by concatenation + MLP. Pluggable into MapTR, MapTRv2 and StreamMapNet.

## Formulation

layer = gather(CA(scatter(SA(q)) + {P_ij}, F_bev)); P_ij = LP(PE(A_ij)); matching and loss as MapTRv2 (one-to-one + one-to-many + dense); GKT-h BEV 200 x 100 over +-15 x +-30 m; lr 6e-4; 24 / 110 epochs.

## Input

BEV features.

## Output

polylines.

## Training

hungarian_set + one-to-many.

## Differentiable?

partial.

## Sensors and data

camera; nuScenes, AV2; no temporal part (host-only).

## Reported results

nuScenes mAP1 (0.2 / 0.5 / 1.0 m) / mAP2 (0.5 / 1.0 / 1.5 m): R50 24 ep 43.3 / 66.4 at 17.9 FPS; 110 ep 50.5 / 72.6; SwinB 24 ep 45.6 / 70.1; plug-in gains: MapTR 27.3 / 50.3 -> 34.4 / 58.0, MapTRv2 37.2 / 61.5 -> 40.1 / 63.4, StreamMapNet 36.7 / 61.1 -> 38.4 / 62.1; AV2 2D mAP2 68.2 vs MapTRv2 67.4.

## Compute

17.9 FPS.

## Code

https://github.com/HXMap/MapQR

## Fit to our project

reuse_idea - S/M - an instance token that scatters into points and gathers back is our "step token -> vertex positions" pattern.

## Needs

Hungarian (as published). Cached-BEV: the decoder yes.

## Conflicts

Hungarian.

## Tried in our experiments

none.
