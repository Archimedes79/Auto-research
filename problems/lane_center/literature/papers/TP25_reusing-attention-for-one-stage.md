# TP25 - Reusing attention for one-stage topology (attention-derived links) with SD-map distillation

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S6 T.*

## Papers

Reusing Attention for One-stage Lane Topology Understanding (Li et al., IROS 2025, arXiv 2507.17617) https://arxiv.org/abs/2507.17617

## Idea

Instead of a GNN, the pair feature for the topology MLP is built from the queries and keys of ALL decoder self-attention layers (stacked across layers, EGTR style), so the link head reads relations the decoder already computed. An SD-map teacher distils BEV features into a camera-only student.

## Formulation

per layer l: projected Q^l, K^l of the lane (CL) and traffic (TE) decoders, stacked to N x L x d/2; pair resource R_ij = concat[Q_i, K_j] over layers plus the last layer; lane-lane head reads R[CL, CL]; gated sums + MLPs. Distillation: L_BEV = ||F_student - F_teacher||^2 (teacher = SMERF converted, frozen; weighting not given). Matching and losses as TopoNet (not restated).

## Input

BEV (TopoNet's), 2 decoders.

## Output

N x N lane-lane and lane-traffic topology.

## Training

24 epochs, AdamW 1e-4, 4 GPUs batch 1 each.

## Differentiable?

partial - as host.

## Sensors and data

camera (student), + SD map (teacher); OpenLane-V2 A and B.

## Reported results

subset A student 40.6 OLS / TOP_ll 11.0 vs TopoNet 39.8 / 10.9; teacher 44.3 OLS / 17.1 vs SMERF 43.0 / 16.2; distillation +0.7 OLS (39.9 -> 40.6); extra feature interactions hurt (44.3 -> 42.9). Subset B no distillation 37.0 vs TopoNet 36.0. Speed +17 %, parameters 49.4 M (student) vs 62.6 M (TopoNet).

## Compute

see above.

## Code

https://github.com/Yang-Li-2000/one-stage.git

## Fit to our project

reuse_idea - effort S - a link head that reads the existing token attention instead of adding a GNN; gain is small and the paper's own interaction experiment is negative.

## Needs

multi-layer self-attention with queries / keys accessible.

## Conflicts

tied to a decoder with N-query self-attention.

## Tried in our experiments

E46 (node-conditioned head): no gain.
