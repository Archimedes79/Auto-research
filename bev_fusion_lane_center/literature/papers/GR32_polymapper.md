# GR32 - PolyMapper (ConvLSTM vertex-sequence decoder with beam search; edge check by heat-map integral)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: checked (paper HTML). ---. Pipeline stage: S4, S3, S7.*

## Papers

Topological Map Extraction from Overhead Images (PolyMapper; Li et al., arXiv 1812.01497, 2018; authors and ICCV 2019 venue from memory, not on the fetched page) https://arxiv.org/abs/1812.01497

## Idea

(The vertex-by-vertex ConvLSTM with beam search follows the Polygon-RNN family used for object outlines, from memory.) A VGG backbone gives a boundary / centerline heat map B and a candidate-keypoint mask V at 1/8 resolution; a multi-layer ConvLSTM (4 layers for roads) walks over vertices on a 28 x 28 grid, each step choosing the next vertex (or an explicit end symbol) from the previous two vertices, so the traversal direction is unique. Roads are linearised into polygon sequences by a maze "wall follower" rule (turn right at intersections, turn around at dead ends; every segment is visited twice). Decoding uses beam search (width 5); after decoding every predicted segment is scored by the integral of B along it and dropped below 0.7.

## Formulation

start candidates for roads lie on the image border, the best one starts the outer polygon, inner polygons are seeded from segments passed only once; per-step classification trained with cross-entropy (softmax over the grid is the natural reading, not stated explicitly); max sequence length 30; Adam, batch 4, lr 1e-4; 300 x 300 patches, city scale by 50 % overlapping patches, splicing, small-loop removal and duplicate merging.

## Input

aerial patch features + B + V + the first one or two vertices; `image_feat`, `bev_raster`. output: ordered vertex sequence -> graph.

## Training

`teacher_forcing` implied (not named), no RL; `next_token`-like CE.

## Differentiable?

no - beam / argmax over grid, wall-follower traversal.

## Sensors and data

aerial; crowdAI buildings, RoadTracer road set (25 / 15 cities), own multi-city set.

## Reported results

RoadTracer set: SP +-10 % 61.1 vs RoadTracer 61.8, AP85 65.5 vs 64.9, AR85 84.2 vs 85.3 (accuracy on par) with far fewer vertices: 31,749 vertices / 35,998 edges vs 271,244 / 281,518 for RoadTracer; 0.29 s per road patch.

## Compute

single GPU, 12 h training for roads.

## Code

none found

## Fit to our project

reuse_idea - effort S: (a) the only beam-search decoder over a graph-growth sequence found in this cluster (aerial, sequence of vertices, not lane hypotheses), (b) edge validation by the *integral of a heat map along the candidate segment* is a direct, sample-along-the-line precedent for ray verification, (c) explicit vertex choice yields sparse graphs (compare our 22 vs 14.3 vertices per 100 m).

## Needs

heat map B along segments; keypoint mask V.

## Conflicts

sequential traversal order (wall follower) and softmax over a coarse grid.

## Tried in our experiments

none (E94 compactness issue noted in the project context).
