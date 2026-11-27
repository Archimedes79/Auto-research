# GR31 - LSR-DM (conditioned diffusion refinement of a lane mask before graph extraction)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: checked (paper HTML).. Pipeline stage: S5.*

## Papers

Lane Graph Extraction from Aerial Imagery via Lane Segmentation Refinement with Diffusion Models (Ruiz et al., arXiv 2405.00620; Remote Sensing 2025) https://arxiv.org/abs/2405.00620

## Idea

A D-LinkNet gives a lane mask; a DDPM conditioned on the aerial patch and on the initial mask refines it (DDIM, 25 steps, started from the initial mask with partial noise), then thresholding, thinning and graph extraction (no training) yield the lane graph. Recall rises, precision stays.

## Formulation

denoising network sees image (3-layer CNN) + time embedding; target = GT mask; T = 1000 training steps, sigmoid schedule; sampling starts after 50 % of the forward steps (best: GEO F1 0.839, TOPO F1 0.775; 90 % collapses, 100 % = pure noise gives TOPO 0.562); steps 10 - 1000 give almost the same (GEO F1 0.838 - 0.841); diffusion run at 256 x 256.

## Input

aerial RGB + mask; `bev_raster`. output: refined mask -> graph by skeleton rules. training: `denoising`, matching-free.

## Differentiable?

yes inside the net; graph extraction rule-based.

## Sensors and data

aerial, 12.5 cm / px, 35 tiles (24 train / 11 test), undirected, no intersections.

## Reported results

TOPO F1 0.774 vs 0.713 (reproduced baseline), TOPO recall 0.789 vs 0.673, GEO F1 0.841 vs 0.813 (Table 1); mask-conditioned DDIM only 0.784 / 0.703.

## Code

none found

## Fit to our project

reuse_idea - effort M: a cheap, few-step refinement of the dense head output before seeding; relevant to the seed-limited E94 (192 peaks). Not growth.

## Needs

initial mask as condition. compat_conflicts: stochastic sampling; 11 test tiles only.

## Tried in our experiments

E39 cascade refiner (+0.001 F1) found little.
