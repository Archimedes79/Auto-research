# VM43 - MapDiffusion (diffusion over polyline vertices, sample-based uncertainty)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S5, X.*

## Papers

MapDiffusion: Generative Diffusion for Vectorized Online HD Map Construction and Uncertainty Estimation in Autonomous Driving (arXiv 2507.21423; authors not read) https://arxiv.org/abs/2507.21423

## Idea

Polyline vertex coordinates are diffused (forward noise on GT, reverse by a DETR-style decoder with deformable cross-attention to a BEV grid and a time-step embedding); class scores are not diffused. Several samples give a per-location variance map as uncertainty.

## Formulation

T = 1000, cosine schedule; inference DDIM eta = 0.5, 5 steps; outputs with score below 0.5 are re-noised; n = 10 samples rasterised and Gaussian-smoothed (3 m kernel, sigma 0.6 m), U = sum over classes of the sample variance (unnormalised); fixed l queries, padding GT with Gaussian noise (mean 0.5, sd 0.25) best; permutation handling and matching not described; BEV 100 x 50 over 60 x 30 m, StreamMapNet configuration (24 ep, lr 2e-4).

## Input

BEV grid (trained encoder).

## Output

sampled polylines.

## Training

denoising.

## Differentiable?

partial.

## Sensors and data

camera; nuScenes (StreamMapNet split); temporal only via StreamMapNet BEV aggregation.

## Reported results

nuScenes (location-disjoint) mAP 35.6 vs StreamMapNet 33.8, SQD-MapNet 33.1, MapTRv2 29.9, MapVR 22.8, MapTR 22.1, VectorMapNet 18.0; 8.0 FPS with 5 steps (12.8 with 1); uncertainty is 31 % higher in invisible than visible areas (Occ3D visibility); a frozen pretrained encoder reached only 31.7 mAP.

## Compute

8.0 FPS.

## Code

none found.

## Fit to our project

not_suitable - M - iterative generative sampling is costly; the sample-variance uncertainty map and the -3.9 mAP for a FROZEN encoder are the only transferable facts.

## Needs

BEV grid.

## Conflicts

stochastic sampling vs a deterministic recursion.

## Tried in our experiments

none.
