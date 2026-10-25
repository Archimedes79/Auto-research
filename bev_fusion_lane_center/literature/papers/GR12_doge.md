# GR12 - DOGE (differentiable Bezier graph optimisation with topology adaptation)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: checked (paper HTML).. Pipeline stage: S5, S7, L.*

## Papers

DOGE: Differentiable Bezier Graph Optimization for Road Network Extraction (arXiv 2511.19850, 2025) https://arxiv.org/abs/2511.19850

## Idea

No network is trained for the graph. A fine-tuned SAM2 gives a road mask; a graph of nodes and cubic-Bezier edges is rendered differentiably (DiffVG) and its positions, widths and curve parameters are optimised by Adam to cover the mask, with interleaved discrete operators that add, merge, split and prune nodes and edges. Matching-free, label-free at the graph level; test-time optimisation.

## Formulation

loss = 1.0 cover (L2 between union of rendered edges and mask) + 0.3 overlap (sum of renders above 1) + 0.012 G1 continuity at degree-2 nodes + 6e-3 offset + 6e-3 spacing (control points near 1/3, 2/3); 300 outer iterations, rendering 512 x 512. TopoAdapt: add edges where mask is uncovered (random small edges), merge nodes closer than 4 m (midpoint), snap node onto a nearby edge and split it (T-junction), collinear merge of degree-2 nodes with refit, prune short or thin edges.

## Input

road probability mask; `bev_raster`.

## Output

`bezier` graph.

## Training

none for the graph (per-image optimisation, `matching_free`).

## Differentiable?

yes inside the optimisation (rendering); the topology operators are discrete.

## Sensors and data

aerial; CityScale and SpaceNet.

## Reported results

SpaceNet TOPO-F1 / APLS 84.58 / 73.48 (SAM-Road++ 81.57 / 73.44), CityScale 80.59 / 70.24 (80.01 / 68.34); ablation SpaceNet: no TopoAdapt 82.80 / 69.45, no overlap loss 83.38 / 65.95, L2 only 70.94 / 37.10 (Tables 1, 3). Compact: 14.0 edges / km vs 52.2 for the GT.

## Compute

one RTX 4090; run time not stated.

## Code

none found (announced)

## Fit to our project

reuse_idea - effort M: node merge / T-junction split / collinear merge are a ready operator set for tree -> DAG (S7) and a matching-free coverage loss for E120-type refinement; DiffVG is an external CUDA dependency (a torch distance-field renderer would replace it).

## Needs

a mask or heat map as target; spatial grid for neighbour queries.

## Conflicts

test-time optimisation (300 iterations per image) is far from a feed-forward real-time block.

## Tried in our experiments

E120 (position refinement with point-to-polyline loss both ways, no matching): +0.012 F1@1, kept optional - same family of objective.
