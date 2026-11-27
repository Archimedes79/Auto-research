# TP21 - LaneDiffusion (diffusion in BEV feature space with a lane prior)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S5 B T.*

## Papers

LaneDiffusion: Improving Centerline Graph Learning via Prior Injected BEV Feature Generation (Wang et al., ICCV 2025, arXiv 2511.06272) https://arxiv.org/abs/2511.06272

## Idea

A diffusion model generates a "lane-prior-injected" BEV feature from the ordinary BEV feature, conditioned on it; the GT lane prior is used only to build the training target, and the downstream CGNet decoder works on the generated feature. Residual shifting (ResShift) is used instead of noise-to-data diffusion.

## Formulation

prior = GT centerlines (M x 20 points) -> sinusoidal embedding + 6-layer transformer -> cross-attended into the BEV constructor (LPIM). Diffusion: x0 = prior-injected feature, xc = original feature, x_res = xc - x0; q(x_t | x0, xc) = N(x0 + eta_t x_res, kappa^2 eta_t I), kappa 2.0, T = 15 steps, schedule exponent p = 0.3; denoiser = Swin-UNet predicting x0; loss w_t ||f(x_t, xc, t) - x0||^2. Sampling: start from xc + noise, 15 steps, run 3 times and average. Refinement LPR fuses generated and original feature (concat + encoder-decoder). Three training stages: (I) LPIM + CGNet losses 24 ep, (II) frozen LPIM, train diffusion 24 ep (lr 6e-5), (III) frozen diffusion, train decoder 24 ep.

## Input

BEV 256 x 200 x 100 (x +-15, y +-30 m); host CGNet.

## Output

refined BEV feature for the lane decoder.

## Training

three stages, 8 V100, batch 2 per GPU.

## Differentiable?

partial - stage-wise; the sampler is a rule-free iterative network, no gradient to the BEV encoder in stage III.

## Sensors and data

camera; nuScenes and Argoverse 2.

## Reported results

nuScenes 24 ep (GEO F1 / TOPO F1 / JTOPO F1 / APLS / SDA): CGNet 54.7 / 42.2 / 34.1 / 30.7 / 8.8 -> 58.9 / 46.8 / 38.8 / 37.1 / 10.6; segment level DET_l 22.0 -> 28.8, mAPcf 35.2 -> 41.6. Steps 5 / 15 / 30: TOPO 44.8 / 46.8 / 46.0 (sampling paradigm). Refinement: none 43.9, concat + FC 45.6, concat + encoder-decoder 46.8.

## Compute

about 45 denoiser evaluations per sample; no latency reported.

## Code

not found on the page

## Fit to our project

reuse_idea - effort L - the idea "train a generator that maps ordinary BEV features to GT-lane-conditioned features" can operate on cached 200 x 200 features (the foundation stays frozen), but 45 Swin-UNet passes on a 3060 is heavy; closest in spirit to our cascade refiner (E39).

## Needs

GT lane prior at training; a feature-space refiner.

## Conflicts

compute; only one host (CGNet) tested.

## Tried in our experiments

E39 (cascade refiner +0.001 F1): small.
