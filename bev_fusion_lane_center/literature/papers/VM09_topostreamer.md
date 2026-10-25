# VM09 - TopoStreamer (streaming lane-segment topology, OpenLane-V2)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S8, S6.*

## Papers

TopoStreamer (arXiv 2507.00709; title "Temporal Lane Segment Topology Reasoning in Autonomous Driving" and authors from memory / repository owner) https://arxiv.org/abs/2507.00709

## Idea

Extends a LaneSegNet-style lane-segment decoder with (1) streaming attribute constraints (stream queries must predict coordinates, class, boundary type and BEV mask of the NEXT frame, supervised by previous-frame GT mapped by the pose), (2) lossless streaming supervision (follow the IDs of stream queries that were positively matched instead of re-matching pose-transformed GT; dropping it costs 36.6 -> 34.4 mAP), (3) a dynamic lane-boundary positional encoding (8 boundary reference points per query, re-encoded every layer), plus lane-segment denoising incl. topology denoising.

## Formulation

carried queries top-66 of 200 (30 %: 10 % 34.8, 30 % 36.6, 50 % 34.3, 75 % 32.9 mAP); query transform = MLP over [query, pose] + residual; BEV warp + GRU as StreamMapNet. DN: box-shift factor 0.2, class-label flip probability 50 %; 240 DN queries (120 -> 35.0, 240 -> 36.6, 360 -> 35.1); topology denoised by two MLP projections (predecessor / successor) -> pairwise sigmoid logits, topology itself not noised. Lane segment = centerline + left / right boundary, 10 ordered points each, adjacency thresholded at 0.5. Weights: vector L1 0.025, seg 3.0, class 1.5, boundary type 0.01, topology focal 5.0. 6 layers, 200 queries, 24 epochs (first 12 single-frame), AdamW 2e-4, batch 8, V100.

## Input

7 cameras -> R50 + FPN + BEVFormer-style, BEV 200 x 100 over +-50 x +-25 m.

## Output

lane segments (3 polylines) + adjacency matrix + class + boundary type.

## Training

Hungarian + DN + streaming supervision; regime hungarian_set + denoising.

## Differentiable?

partial (top-k, Hungarian).

## Sensors and data

camera; OpenLane-V2 subsets A (Argoverse 2) and B (nuScenes).

## Reported results

subset A lane segments mAP 36.6 (AP_ls 35.0, AP_ped 38.1), TOP_lsls 28.5, 13.6 FPS vs LaneSegNet 32.6 / 25.4, SQD-MapNet 26.0 / 16.6; centerline subset A OLS 44.4, DET_l 35.2, TOP_ll 28.8; subset B (pseudo lane-segment labels from a 4 m width) 42.6 / 30.9 / 29.4 vs LaneSegNet 38.7 / 27.5 / 24.9. No temporal-consistency metric.

## Compute

13.6 FPS.

## Code

https://github.com/YimingYang23/TopoStreamer

## Fit to our project

reuse_idea - M - the only streaming method that propagates and denoises TOPOLOGY (adjacency from predecessor / successor projections) and uses ID-following supervision; centerline direction is fixed by data (no permutation set).

## Needs

queries with identity; adjacency head; segment IDs in GT. Cached-BEV: the temporal parts yes.

## Conflicts

Hungarian; segment-level not vertex-level; OpenLane-V2 labels differ from nuScenes map-expansion pieces.

## Tried in our experiments

topology heads E42 / E45 / E46 / E51 / E56 (within 0.01 of the rules, dropped); streamed topology: none.

## Correction from the level-2 extraction (2026-10-24)

Topology is not carried between frames: the adjacency matrix is recomputed every frame by a pairwise head over the decoder queries; it is never stored, warped or used as a prior, and the streaming loss has no topology term. What is carried: the 66 most confident of 200 queries, updated by a pose MLP with residual, with identity-followed targets. The repository holds a README and four GIFs only (no code, no LICENSE file). Details: [../deep/VM09_topostreamer.md](../deep/VM09_topostreamer.md), [../code/topostreamer.md](../code/topostreamer.md).
