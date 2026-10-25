# VM04 - MapTracker (mapping as tracking, strided memory, consistent GT, C-mAP)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M+C** (A abstract, M method, C code page). Verified: partly (fetched text + front page).. Pipeline stage: S8, E, L.*

## Papers

MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping (Chen et al., ECCV 2024 oral) https://arxiv.org/abs/2403.15951

## Idea

Every map element is a track: its 512-d vector latent is carried frame to frame; a raster latent (BEV, 50 x 100 x 256) accumulates BEV evidence. Buffers keep the last 20 frames but fusion uses only 4 memories chosen by travelled distance (nearest to 1, 5, 10, 15 m ago); BEV memories are warped by the ego affine transform (bilinear), vector latents by a pose MLP (quaternion + translation) and fused by cross-attention with relative-frame encoding. Assignments of tracked elements are inherited from t-1 (identity persists); only new elements pass through Hungarian matching. Proposes a "consistent GT" (elements merged over time) and the consistency-aware mAP (VM45).

## Formulation

BEV loss focal + Dice (10, 1); vector loss 5 x focal + 50 x line; transformation loss from StreamMapNet 0.1; Gaussian noise added to the inter-frame transform P^t_{t-1} that feeds the vector PropMLP during training (sigmas not on the page; whether the BEV warp is perturbed is not stated; no noisy-pose test); consistent GT = per-frame cleaning, then bipartite matching of IoU between motion-aligned rasterised masks of adjacent frames, chained into tracks (look-back 1); thresholds 0.4 (first frame), 0.5 (propagated) / 0.6 (new); 100 new-candidate embeddings, 20-point polylines, 6 decoder layers (instance self-attention -> BEV multi-point cross-attention -> memory cross-attention). Stages: BEV pretraining 18 ep, vector warm-up 6, joint 48 (nuScenes; AV2 12 / 3 / 20); clips: 4 of the previous 10 frames, length 5; AdamW 5e-4; 8 RTX A5000, about 3 days.

## Input

images -> BEV (StreamMapNet code base); 20-frame BEV and vector buffers; pose per frame.

## Output

segmentation raster + 20-point polylines with class and persistent IDs.

## Training

hierarchical Hungarian for new elements, inherited assignment for tracked ones; focal + line + BEV segmentation.

## Differentiable?

partial - Hungarian for new elements, thresholded birth / death, distance-rule memory selection.

## Sensors and data

camera; nuScenes, AV2; 60 x 30 m (BEV 50 x 100, derived 0.6 m cells), also 100 x 50 m; original and non-overlapping splits.

## Reported results

nuScenes consistent GT mAP / C-mAP 76.1 / 69.1 vs StreamMapNet 70.4 / 56.4; original GT 74.7 vs 69.1; non-overlapping split (60 x 30 m) nuScenes 40.3 / 32.5 vs 33.5 / 22.2, AV2 71.3 / 63.2 vs 64.4 / 54.4. Ablation: single memory 70.8 / 62.4; fusion without stride 74.9 / 68.1; with stride 76.1 / 69.1; stride sets {1,5,10,15} 76.1, {1,2,3,4} 75.8, {5,10,15,20} 73.8, {0,0,0,0} 75.0. 11.5 FPS (RTX 6000, batch 1) vs StreamMapNet 14.2.

## Compute

FPS; parameters not stated.

## Code

https://github.com/woodfrog/maptracker (README: official, built on StreamMapNet, GPL-3.0 stated, framework not named)

## Fit to our project

reuse_idea - M - the distance-strided memory selection, the consistent-GT idea and C-mAP are portable; the tracker as a whole is not (query identity).

## Needs

ego pose per frame; element IDs or tokens; frame buffers; consistent GT (our piece graph has ID continuity). Cached-BEV: the memory modules yes; stage 1 trains the BEV encoder.

## Conflicts

identity inheritance has no meaning at forks / merges of a graph unless defined; the three-stage 72-epoch training is far beyond our budget.

## Tried in our experiments

E41 (single recent memory, +0.008 F1); distance-strided multi-memory: untested.
