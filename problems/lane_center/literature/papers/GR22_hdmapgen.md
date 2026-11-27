# GR22 - HDMapGen (hierarchical autoregressive graph generation of lane maps)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: . Pipeline stage: S4, X.*

## Papers

HDMapGen: A Hierarchical Graph Generative Model of High Definition Maps (Mi et al., CVPR 2021; arXiv 2106.14880) https://arxiv.org/abs/2106.14880

## Idea

Unconditional generation of lane maps: a coarse key-point graph (endpoints and intersections, about 30 % of the points) is grown node by node in DFS order from a random start; for each new node the model predicts its coordinates (Gaussian mixture) and, in one shot, its edges to all previous nodes (mixture of Bernoulli); a local decoder then fills each edge with a fixed-length coordinate sequence plus validity mask. A recurrent graph-attention network (GRAN style) with a GRU update is the backbone.

## Formulation

three global-decoder variants (coordinate-first, topology-first, independent); teacher forcing; NLL + BCE + MSE; temperature scaling of the mixture (tau about 0.2 bound for quality); Adam 1e-4, one V100. Plain per-point graph "fails completely" at this task.

## Sensors and data

Argoverse Miami (12,000 maps) and Pittsburgh (5,000) at 200 m x 200 m; in-house data.

## Reported results

connectivity Frechet 0.17 vs 0.50 (SketchRNN conditional); generation latency 0.20 s per 200 m map (PlainGen 0.89 s, SketchRNN 2.28 s); metrics are distributional (MMD of degree / spectrum, Frechet). Limitations: unsmooth local graphs, cut-offs and dead ends in the sequence baseline, no cycles discussed.

## Code

none found

## Fit to our project

reuse_idea - effort S: two-level (key points then curves) with one-shot edge prediction to all previous nodes; no unconditional generation need.

## Needs

DFS order. compat_conflicts: sampling; unconditional.

## Tried in our experiments

none. differentiable: no. training: `teacher_forcing`, `next_token`. compute: 0.20 s / map. verified: checked.
