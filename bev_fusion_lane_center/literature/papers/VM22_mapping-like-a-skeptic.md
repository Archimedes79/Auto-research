# VM22 - Mapping like a Skeptic (probabilistic BEV projection, confidence-weighted temporal merge)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: B, S8.*

## Papers

Mapping like a Skeptic: Probabilistic BEV Projection for Online HD Mapping (arXiv 2508.21689; authors not read) https://arxiv.org/abs/2508.21689

## Idea

Two small CNNs on the camera features predict per-pixel offsets of a static geometric mapping and a covariance + confidence map; each BEV cell samples K image locations from the Gaussian and weights features by confidence x likelihood; the cell confidence is the weight sum. Historical BEV features and their confidences are warped to the current time and merged by confidence-weighted averaging (memory: last 20 BEV maps, T = 4 chosen by vehicle position), then a learned temporal module fuses them. The decoder follows MapTracker.

## Formulation

K, losses, epochs and BEV grid are deferred to the supplement; 63.8 M parameters.

## Input

camera features + historical BEV + confidences.

## Output

polylines (MapTracker head).

## Training

as MapTracker.

## Differentiable?

yes in the projection and merge; partial in the head.

## Sensors and data

camera; nuScenes, AV2; 60 x 30 and 100 x 50 m, StreamMapNet-style new splits.

## Reported results

mAP on the new splits: nuScenes 60 x 30 45.4 vs MapTracker 40.3 and StreamMapNet 33.5; 100 x 50 42.8 vs 36.2 / 23.0; AV2 60 x 30 72.4 vs 71.3 / 64.4; ablation nuScenes 41.0 -> 43.5 (probabilistic projection) -> 45.4 (full); 15.9 FPS (A100).

## Compute

63.8 M parameters.

## Code

https://github.com/Fatih-Erdogan/mapping-like-skeptic

## Fit to our project

reuse_idea - S - confidence-weighted merging of warped past BEV evidence (weights = confidence) is a non-learned relative of log-odds fusion at feature level.

## Needs

per-cell confidence maps.

## Conflicts

the projection needs camera lifting (not available on cached BEV); the merge itself is portable.

## Tried in our experiments

E41 (unweighted GRU).
