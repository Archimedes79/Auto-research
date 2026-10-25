# GR25 - MapDreamer and SLEDGE (latent diffusion over lane tokens with a connectivity decoder)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: . Pipeline stage: S4, X.*

## Papers

MapDreamer: Aerial Imagery Conditioned Latent Diffusion for Lane-Level Map Generation (arXiv 2607.01370, 2026) https://arxiv.org/abs/2607.01370 ; SLEDGE: Synthesizing Driving Environments with Generative Models and Rule-Based Traffic (arXiv 2403.17933, 2024; authors and ECCV 2024 venue from memory) https://arxiv.org/abs/2403.17933

## Idea

MapDreamer denoises one latent per lane (VAE of 20-point centerlines + four adjacency matrices: successor, predecessor, left, right), conditioned by cross-attention on frozen aerial features; a cardinality head predicts the lane count, "ghost" latents pad a variable count to fixed shape; global maps by raster-order tiling with boundary tokens. SLEDGE generates a rasterised latent map with a DiT, vectorised by a transformer decoder with N <= 30 polylines of 20 points and an existence score; adjacency recovered by linking endpoints within 1.5 m whose orientations differ by < 60 deg.

## Formulation

MapDreamer: DDPM 100 steps or DDIM 20 steps; losses eps-prediction + focal existence + cardinality CE; local tiles 76.8 m at 512 px; results local GEO / TOPO F1 at 1.2 m: 0.640 / 0.562 vs BGFormer 0.458 / 0.345; fixed 64 slots without cardinality: precision collapses (0.498 / 0.484); 326 ms (DDPM) / 73 ms (DDIM) per tile on a V100 (240 M parameters; small 31 M).

## Differentiable?

yes in training; stochastic sampling.

## Sensors and data

aerial + Argoverse 2 lane graphs (MapDreamer), nuPlan (SLEDGE).

## Code

https://github.com/autonomousvision/sledge ; MapDreamer none found

## Fit to our project

reuse_idea (ghost / padding for variable counts; endpoint-linking adjacency rule 1.5 m / 60 deg) - effort S; generative approach itself not suitable.

## Tried in our experiments

none. verified: checked (MapDreamer text truncated at the appendix; some percentage claims in the text do not match its tables).
