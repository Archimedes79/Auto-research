# GR17 - Topo2Seq (training-only autoregressive topology decoder as regulariser)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: checked (paper HTML).. Pipeline stage: L, T.*

## Papers

Topo2Seq: Enhanced Topology Reasoning via Topology Sequence Learning (Yang et al., AAAI 2025; arXiv 2502.08974) https://arxiv.org/abs/2502.08974

## Idea

A DETR lane-segment decoder (LaneSegNet style, 200 queries) and a Pix2Seq-style autoregressive *topology sequence decoder* share the BEV features; only the first is used at inference. The sequence decoder receives the predicted key points as unordered prompts and must reconstruct the graph, which forces the BEV features to carry topology. Randomising the prompt order matters.

## Formulation

sequences as in RoadNet (six integers per edge: x, y, category, connection index, 2 Bezier controls; DFS from the front-right point; noise edges marked <ncls>); prompts 201 + edges 601 = 802 tokens; L = 0.025 L1 + 1.5 cls + 3.0 seg + 0.1 lt + 5.0 top + 1.0 seq. Ablation TOPls-ls: ordered GT prompts 25.1, randomised GT prompts 27.8, predicted key points early in training 26.5, full setup 29.9. BEVFormer encoder, BEV x in [-50, 50] m, y in [-25, 25] m on a 200 x 100 grid; 14.7 FPS inference.

## Input

BEV features; `queries`.

## Output

lane segments + adjacency (inference); sequence only in training.

## Training

`hungarian_set` (main decoder) + `next_token` auxiliary.

## Differentiable?

yes for the auxiliary path (it only adds a loss); the sequence is never sampled at inference.

## Sensors and data

7 cameras, OpenLane-V2 subset A (subset B not stated).

## Reported results

lane segment 48 epochs mAP 37.7 / TOPlsls 29.9 vs LaneSegNet 36.4 / 27.3; centerline TOPll 30.0 vs 27.3 (Table, subset A).

## Compute

no inference cost.

## Code

none found

## Fit to our project

reuse_idea - effort M: an auxiliary order-agnostic path-reconstruction loss from the cached pyramid costs nothing at inference; our tree could receive a training-only DFS-sequence head on its node tokens.

## Needs

key-point prompts from the main decoder.

## Conflicts

Hungarian main decoder; token loss needs quantisation.

## Tried in our experiments

none.
