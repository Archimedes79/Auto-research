# GR15 - LaneGraph2Seq (vertex-centric DFS plus edge-partition sequence, classifier-free guidance)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked.. Pipeline stage: S4, S6.*

## Papers

LaneGraph2Seq: Lane Topology Extraction with Language Model via Vertex-Edge Encoding and Connectivity Enhancement (Peng et al., AAAI 2024; arXiv 2401.17609) https://arxiv.org/abs/2401.17609

## Idea

A lane graph is written as a vertex sequence (junction points in DFS order) followed by an edge sequence in which each edge is [child index, Bezier midpoint x, y] and a Split token closes the children of each parent. A transformer decoder generates it from BEV features; at inference classifier-free guidance (alpha_c = 4) contrasts a vertex-conditioned and an unconditioned stream to improve connectivity.

## Formulation

sequence padded to 200 vertex + 400 edge tokens; class-weighted log-likelihood with lowered weights of EOV / Split / EOE (against early termination; values not stated); <N/A> excluded from the loss; AdamW lr 2e-4, 300 epochs, 8 V100; 6 decoder layers (8 layers slightly better in Table 3: M-F 74.0 vs 73.3). Ordering ablation M-F / Detect / C-F (VoVNetV2, nuScenes): DFS 73.3 / 68.7 / 67.6, BFS 72.1 / 67.2 / 67.2, coordinate 69.8 / 61.7 / 61.2, random 70.2 / 61.3 / 62.1. Logits: uncond + alpha_c (cond - uncond). Nucleus sampling mentioned, parameters not stated.

## Input

LSS BEV (ResNet-50 or VoVNetV2, pre-trained on centerline segmentation), x in [-48, 48] m, y in [-32, 32] m; `image_feat` -> `token_sequence`.

## Output

`token_sequence` -> DAG.

## Training

`next_token`, teacher forcing implied; BEV augmentation.

## Differentiable?

no - tokenisation, argmax / sampling.

## Sensors and data

6 / 7 cameras; nuScenes, Argoverse 2.

## Reported results

nuScenes M-P / M-R / M-F 64.6 / 63.7 / 64.1, connectivity C-F 63.2 (R-50); VoVNetV2 73.3 / 67.6; AV2 61.7 / 60.1 (Table 1). Failure: premature termination before all edges are emitted.

## Compute

not stated.

## Code

https://github.com/fudan-zvg/RoadNet

## Fit to our project

not_suitable as a block; reuse_idea: ordering result (DFS best), reduced weights of terminating tokens, the observation that early termination is the dominant AR failure.

## Needs

sequence order; quantised coordinates.

## Conflicts

tokens; sampling.

## Tried in our experiments

none.
