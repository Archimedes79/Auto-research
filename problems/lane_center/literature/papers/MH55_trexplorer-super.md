# MH55 - Trexplorer Super (recurrent DETR tracking of tree centrelines; super-trajectory training; duplicate branches)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M (Trexplorer Super), A (RefTr)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, S6, T.*

## Papers

Trexplorer Super (Naeem et al., MICCAI 2025, arXiv 2507.10881; full title not on the page read) https://arxiv.org/abs/2507.10881 (arXiv HTML read); the original Trexplorer: Recurrent DETR for Topologically Correct Tree Centerline Tracking (MICCAI 2024; PDF not decodable); RefTr: Recurrent Refinement of Confluent Trajectories for 3D Vascular Tree Centerlines (Naeem et al., arXiv 2511.20823) https://arxiv.org/abs/2511.20823 (abstract only)

## Idea

A DETR decoder tracks each branch breadth-first, predicting the child nodes of the next graph level (end / intermediate / bifurcation); patches are placed at the endpoints of the graph built so far and the query outputs of the previous patch are carried over. Known problems of the first version: duplicate branches and premature termination.

## Formulation

Hungarian loss between predicted and GT nodes with self-attention among queries to suppress duplicates (individual loss terms not specified); history = past-trajectory token of up to 10 previous node positions; Super Trajectory Training: a sample is a super trajectory of 54 nodes split into six sub-trajectories of 10 nodes with shared link nodes, query outputs passed between them (so the model trains on its own carried state, not only GT); target augmentation samples Laplace offsets at bifurcations (scale proportional to radius) to create more bifurcation targets; Focal Cross Attention (high-resolution features, attention restricted to a focal region around the tracked branch); inference: 26 queries assigned at a bifurcation, some discarded, at most 196 tokens tracked concurrently, no NMS, no post-processing. The page does not describe teacher forcing or multi-hypothesis methods. - input / output: 3-D image patches -> tree nodes.

## Training

regime `hungarian_set` per step; STT gives a form of free rollout inside super trajectories.

## Differentiable?

partial - Hungarian assignment; carried queries are differentiable.

## Sensors and data

synthetic (368 / 32 / 100), ATM'22 (220 / 16 / 60), Parse 2022 (72 / 8 / 20) CT volumes.

## Reported results

point F1 (Vesselformer / Trexplorer / Trexplorer Super): synthetic 48.18 / 39.40 / 77.83; ATM'22 26.77 / 3.34 / 60.45; Parse 2022 16.43 / 10.01 / 39.46; branch F1 15.95 / 26.26 / 77.12 (synthetic), 1.95 / 3.34 / 41.15 (ATM'22); Betti errors zero for both Trexplorer variants.

## Compute

not reported.

## Code

https://github.com/RomStriker/Trexplorer-Super (paper CC BY-NC-ND 4.0; code licence not stated on the page)

## Fit to our project

baseline - effort L - closest published tree tracker with several children per step; shows duplicates and early stops as the failure modes of the matched version, and training on carried (own) state as the fix.

## Needs

Hungarian; 3-D patches.

## Conflicts

Hungarian; patch-based inputs.

## Tried in our experiments

E94-E99 (learned trace network, teacher forced)
