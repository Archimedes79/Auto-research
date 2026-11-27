# VM12 - L2G-Map (hand-specified log-odds fusion of warped local maps, nuScenes)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly (extracted from the HTML; closed-loop numbers not used).. Pipeline stage: S8, E.*

## Papers

L2G-Map: Local-to-Global Mapping via Hierarchical Diffusion Refinement and Elliptical Bayesian Fusion (arXiv 2503.02578; authors not read) https://arxiv.org/abs/2503.02578

## Idea

Builds a GLOBAL BEV semantic map (divider, crossing, boundary) from per-frame local maps by a closed-form recursive Bayesian log-odds update, weighted by the distance of each cell from the ego (observations near the perception edge count less); each local map is first refined by a conditional diffusion U-Net conditioned on the previous-frame map and BEV features warped into the current frame and on an OSM-centerline prior.

## Formulation

L_t = L_{t-1} + dL_t; dL_t = logit(p_obs) - logit(p_prior); class priors 0.1 (divider), 0.02 (crossing), 0.09 (boundary); weighted increment dL_w = dL x c x f(P) with c = 1 / (1 + d_n^2), d_n = sqrt((l_x/r_x)^2 + (l_y/r_y)^2) the elliptical distance in units of the perception radii (c = 1 at the vehicle, 0.5 at the boundary) and f(P) bilinear weights that spread a sub-cell-shifted observation over 4 neighbours; final map sigma(L_0 + sum dL_w). The page describes no decay, forgetting or clamping. Diffusion U-Net: DDIM, two-stage training (20 epochs with 50 % masking, 10 fine-tune), focal loss, Adam + StepLR, RTX A6000. The fusion itself is NOT learned (fixed formulas and empirical priors).

## Input

local BEV semantic maps from HDMapNet / P-MapNet / GenMapping + vehicle pose.

## Output

global semantic raster (and a closed loop with StreamMapNet).

## Training

diffusion network only; regime denoising; fusion none.

## Differentiable?

the log-odds sum is a closed form (differentiable), but the page does not train through it.

## Sensors and data

camera; nuScenes, Argoverse 2; local extents 60 x 30, 120 x 60, 240 x 60 (units as printed); global resolution not stated.

## Reported results

nuScenes 60 x 30 semantic mIoU 39.03 vs 37.13 (P-MapNet); Argoverse 2 48.93 vs 48.23 (GenMapping); nuScenes 240 x 60 48.37 vs 45.50; global fusion ablation (Table IV, Global IoU): Bayesian only 35.06, + bilinear 35.13, + confidence weight 35.57; the paper states 5.51 % Global IoU and 7.11 % mIoU gain over a probabilistic-mean fusion.

## Compute

about 0.13 s per frame for the panoramic input (as printed; ambiguous).

## Code

announced https://github.com/lynn-yu/L2G-Map (not released when the paper was written)

## Fit to our project

reuse_idea - S - the only log-odds recursion with a reliability weight on nuScenes map layers that I found. For vertex validity: p_obs = the per-frame keep gate, prior = its base rate, weight = distance-to-ego (the BEV border is less reliable) times bilinear spread; add a decay (absent here) to bound memory.

## Needs

pose alignment; cells instead of moving vertices (association is implicit on a grid). Cached-BEV: yes (works on any raster).

## Conflicts

for a graph the validity lives on vertices that move, so a soft association (nearest vertex or splat to a grid) is needed; the published version assumes a global pose.

## Tried in our experiments

none.

## Correction from the level-2 extraction (2026-10-24)

- The repository named above returns 404 (page, API, raw README): no code and no licence. The extraction also reports that version 1 of arXiv 2503.02578 is a different paper (TS-CGNet, repository with a README only); catalogue entry: no repository.
- "Tried in our experiments: none" is out of date: a log-odds fusion of the vertex validity (K24) ran in loop 16 inside the time recursion (E173 and E176, `p16_builder.fuse_prior`); see the Experiment Table.
- The fusion rule is hand-specified: L += logit(p_obs) - logit(p_class_prior), weighted by 1 / (1 + d_n^2) and a bilinear footprint; no clamp, decay or pose-error test was found.

Details: [../deep/VM12_l2g-map.md](../deep/VM12_l2g-map.md), [../code/l2g-map.md](../code/l2g-map.md).
