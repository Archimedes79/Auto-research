# GR24 - LaneDiffusion (diffusion of a lane prior at the BEV-feature level)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: checked.. Pipeline stage: B, S5.*

## Papers

LaneDiffusion: Improving Centerline Graph Learning via Prior Injected BEV Feature Generation (Wang et al., ICCV 2025; arXiv 2511.06272) https://arxiv.org/abs/2511.06272

## Idea

A diffusion model (ResShift-style, starts near the original BEV feature) generates "prior-injected" BEV features from the plain ones; the prior-injected target is made by encoding GT centerlines (20 points each, 6-layer transformer) and cross-attending them in the BEV constructor; at inference no GT is needed, the generated features are refined by concat + encoder-decoder and a CGNet decoder produces centerlines and topology. Three training stages (prior module, diffusion, decoder).

## Formulation

T = 15 steps (5 / 15 / 30: TOPO-F1 44.8 / 46.8 / 46.0), kappa = 2.0, geometric schedule p = 0.3; Swin-UNet denoiser predicting x0; loss w_t ||f - x0||^2; sampling run 3 x and averaged; BEV feature 256 x 200 x 100, range x +-15 m, y +-30 m.

## Differentiable?

yes in the network; sampling of the diffusion is stochastic.

## Sensors and data

cameras; nuScenes, Argoverse 2.

## Reported results

nuScenes point-level GEO / TOPO / JTOPO F1, APLS, SDA: 58.9 / 46.8 / 38.8 / 37.1 / 10.6 vs CGNet 54.7 / 42.2 / 34.1 / 30.7 / 8.8; segment-level TOPll 3.4 vs 1.3 (Tables 1 - 2).

## Code

none found

## Fit to our project

reuse_idea (not growth) - effort L: a learned BEV-feature "restoration" stage before the heads; a different cluster should judge.

## Tried in our experiments

E39 / E41 refinement and temporal fusion small gains.
