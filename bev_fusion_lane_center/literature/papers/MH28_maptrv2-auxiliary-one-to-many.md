# MH28 - MapTRv2 auxiliary one-to-many branch (map elements on nuScenes)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L, T.*

## Papers

MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction (Liao et al., 2023) https://arxiv.org/abs/2308.05736 (arXiv HTML)

## Idea

Add a training-only group of T = 300 extra instance queries sharing the decoder; GT elements are repeated K = 6 times (padded to size T) and the same hierarchical bipartite matching is applied, so one GT can match 6 predictions.

## Formulation

loss weight 1; base matching: instance-level Hungarian (focal class cost + Manhattan point cost) then point-level permutation with the lowest Manhattan sum (2 permutations for undirected polylines, 1 for directed centrelines). - input / output: BEV features -> vectorised map elements.

## Training

nuScenes 24 epochs; regime `hungarian_set`.

## Differentiable?

yes.

## Sensors and data

nuScenes (camera, optionally lidar).

## Reported results

nuScenes val R50 24 ep: mAP 57.6 (no branch) -> 61.5 (K = 6); loss weight 0.1 -> 58.5, 1.0 -> 61.5, 2.0 -> 59.9; the three dense auxiliary losses (depth, perspective-view and BEV segmentation) together 56.6 -> 61.5 (+4.9) as read from the page.

## Compute

not stated.

## Code

https://github.com/hustvl/MapTR (named; licence not stated on the page)

## Fit to our project

baseline - effort M - proves the "dense positives in training only" trick in the nuScenes map domain.

## Needs

Hungarian.

## Conflicts

Hungarian.

## Tried in our experiments

E31 / E32 (matched sparse decoder dropped)
