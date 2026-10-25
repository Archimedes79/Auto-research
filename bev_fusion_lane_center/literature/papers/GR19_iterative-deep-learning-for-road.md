# GR19 - Iterative deep learning for road topology (best-first tracing with a local heat map)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: checked (paper HTML).. Pipeline stage: S4, S1.*

## Papers

Iterative Deep Learning for Road Topology Extraction (Ventura, Pont-Tuset, Caelles, Maninis, Van Gool; arXiv 1808.09814, Aug 2018; no venue on the page) https://arxiv.org/abs/1808.09814

## Idea

A global segmentation network gives the start and a cost map; a local hourglass network sees a 64 x 64 patch centred on the query point and outputs a heat map of *border locations connected to the centre*, which makes several branches per patch natural. Candidates above a confidence threshold go into a bag; the most confident one is visited next (best-first), joined to its predecessor by Dijkstra over the segmentation probability map.

## Formulation

training on 130 random 64 x 64 patches per image centred on road pixels; targets Gaussian peaks at skeleton intersections with the patch border square; visited set blocks re-expansion; near-visited points are discarded or connected but not expanded; when the bag is empty a new start is chosen if far from explored areas and confident.

## Input

64 x 64 patch; output: heat map of connected border points; `bev_raster` / `scores`.

## Training

`matching_free` heat-map regression, `teacher_forcing` (patches from GT).

## Differentiable?

no (bag / Dijkstra), but each step is a dense heat map (no set matching).

## Sensors and data

Massachusetts roads (1500 x 1500 tiles), DRIVE retina.

## Reported results

Massachusetts F^C 74.4 / F^R 81.6 vs best VGG threshold 49.3 / 72.4; DRIVE F^C 85.5 vs 79.8; patch-level F 84.5 on 700 validation patches.

## Compute

not stated. code: none found

## Fit to our project

reuse_idea - effort S: best-first order by confidence (instead of FIFO / DFS) and heat-map-valued branch targets (no assignment).

## Needs

confidence queue.

## Conflicts

sequential.

## Tried in our experiments

heat-map seeds in E94.
