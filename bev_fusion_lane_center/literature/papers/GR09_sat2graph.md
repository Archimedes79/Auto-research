# GR09 - Sat2Graph (graph-tensor encoding: a graph as a dense multi-channel raster)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked (paper HTML, README).. Pipeline stage: S6, S7, L (dense alternative to growth).*

## Papers

Sat2Graph: Road Graph Extraction through Graph-Tensor Encoding (He et al., ECCV 2020; arXiv 2007.09547) https://arxiv.org/abs/2007.09547

## Idea

One forward pass of a plain segmentation network predicts a tensor that encodes the graph; no agent, no recurrence. Each cell stores a vertex score and, for each of D_max = 6 angular sectors, an edge score and the offset to the neighbour vertex in that sector. Decoding = local maxima + sector offsets + nearest-vertex matching. The sector slots give a fixed number of outgoing-edge slots per vertex without any matching.

## Formulation

tensor (W/lambda) x (H/lambda) x (1 + 3 D_max), lambda = 1 m -> 19 channels. Encode: segments are interpolated to < d = 20 m (15 - 25 m best); stacked roads shifted so overlapping vertices are > 5 m apart; each edge encoded once from one endpoint, at most one edge per 60 degree sector. Loss: sum over cells of CE(p_v) + [vertex present] * sum_i (CE(p_ei) + L2(offsets)); edge terms only where GT has a vertex (long segments are not label-consistent). Decode: vertices = maxima above p_thr; for vertex v and sector i choose the nearest candidate u by d = ||(v + offset_i) - u|| + w cos_dist(offset_i, u - v), w = 100 (F1 75.87 - 76.72 for w from 1 to 150), reject beyond 15 m. Backbone DLA, receptive field 352 px, 300 k iterations, lr 1e-3 halved every 50 k, one V100.

## Input

aerial RGB, 1 m / px; `image_feat`.

## Output

`graph(nodes+edges)` from a dense tensor; undirected, direction as attribute.

## Training

`matching_free` dense supervision; no rollout, no exposure bias.

## Differentiable?

partial - the network is differentiable; decoding (maxima, nearest matching) is not.

## Sensors and data

20 US cities (OSM, 720 km^2, 144 / 27 / 9 tiles of 2 km), SpaceNet 2,549 tiles.

## Reported results

City-scale TOPO-F1 / APLS 76.26 / 63.14 (DLA) vs RoadTracer 66.16 / 57.29; SpaceNet 80.97 / 64.43; the encoding lifts a U-Net from 70.36 to 76.40 TOPO-F1 (Table 1 and text). Stacked roads: precision 83.11 %, recall 49.81 %. Failure modes: stacked roads in complex scenes, missing small roads under wide highways; the RNGDet text adds (second-hand) that it cannot separate segments with a small intersection angle and is weaker on curves - relevant to fork resolution.

## Compute

not stated; training on one V100.

## Code

https://github.com/songtaohe/Sat2Graph

## Fit to our project

reuse_idea - effort S / M: (a) angular-sector slots as a *matching-free* way to give each tip b children (our fan has 9 bearings; 2 children per tip = 2 of the sectors), (b) the "edge terms only at vertex cells" mask, (c) baseline for a "spray + snap + link" dense route.

## Needs

dense head with extra channels (vertex + 6 x 3); decoding rules.

## Conflicts

non-iterative, so it does not give hypotheses with verification; undirected.

## Tried in our experiments

E31 / E32 (dense grid + tracer won over a matched sparse decoder); dense heat + direction head already exists (2.7 M U-Net).
