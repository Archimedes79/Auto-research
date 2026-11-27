# VM41 - MapLightning (1D map tokens instead of a BEV grid; near-extrapolation split)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M (first 100 k characters)** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: B, S8.*

## Papers

MapLightning: Online Vectorized HD Map Construction with 1D Map Tokens (arXiv 2610.01905, Oct 2026; authors not read) https://arxiv.org/abs/2610.01905

## Idea

K learnable map tokens (K = 50 x cameras x timesteps = 1200 on nuScenes) are concatenated with flattened image tokens of the current and 3 past frames and processed by a 4-layer full self-attention Transformer; only the map tokens are kept and decoded by a MapTRv2-style 6-layer decoder with full cross-attention. No intrinsics, extrinsics, depth or ego-pose alignment. Past-frame features are cached.

## Formulation

learned row / column, camera and time embeddings; matching, loss weights, epochs, lr not in the visible text; 16.7 x fewer intermediate tokens than a dense BEV (20 k vs 1.2 k).

## Input

image tokens of T = 4 frames.

## Output

polylines.

## Training

not stated (MapTRv2-style decoder).

## Differentiable?

partial.

## Sensors and data

camera; nuScenes, AV2; default range 60 x 30 m on the Near-Extrapolation split, also StreamMapNet split and original split.

## Reported results

(mAP / FPS) Near-Extrapolation split, ResNet-50: MapTR 19.0, StreamMapNet 26.1, MapTracker 27.5, MapTRv2 26.7, AugMapNet 27.5, ours (self-attention) 35.3 at 18.9 FPS (nuScenes); Argo2 63.3. StreamMapNet split: 45.8 (nuScenes 60 x 30), 42.1 (100 x 50); original split 77.7 (paper notes memorisation). Real-time variants 30.3 - 30.6 mAP at 40 - 43 FPS.

## Compute

18.9 FPS (R50).

## Code

none ("will be released").

## Fit to our project

not_suitable - S - removes the BEV grid altogether; only the split protocol and the fact that the best published numbers on location-disjoint splits are about 45 mAP are useful.

## Needs

raw image tokens.

## Conflicts

no use of cached BEV.

## Tried in our experiments

none.
