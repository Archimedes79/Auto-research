# TP12 - RATopo (denser topology supervision through one-to-many assignment)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L S4 T.*

## Papers

RATopo: Improving Lane Topology Reasoning via Redundancy Assignment (arXiv 2508.15272) https://arxiv.org/abs/2508.15272

## Idea

In one-to-one matched training only the few matched queries carry topology labels, so the topology head sees very few valid pairs. Three changes (all but one training-only): reorder each decoder layer to cross-attention before self-attention, add parallel cross-attention blocks, and supervise topology also on an auxiliary one-to-many assignment (K = 3 predictions per GT lane).

## Formulation

layer CA -> SA -> FFN; M = 4 parallel cross-attention blocks with independent weights, merged by a linear layer; one-to-many matching per block (cost not described), K = 3. Label under redundant assignment: Z_rs = G_pq if r in the K predictions of GT lane p and s in those of lane q, else 0. L = L_det(o2o) + L_topo(o2o) + 2.0 L_topo(o2m); topology weights 5.0, focal.

## Input

BEV 200 x 100, 300 lane + 100 traffic queries, R50, image 1024 x 775.

## Output

unchanged host outputs (the o2m branch is dropped at inference).

## Training

24 epochs, AdamW 2e-4, batch 8, 8 A100.

## Differentiable?

partial - matching steps as in the host.

## Sensors and data

camera; OpenLane-V2 A and B; lane segments (LaneSegNet).

## Reported results

with TopoLogic (subset A, TOP_ll / TOP_lt / DET_l / OLS): original 23.9 / 25.4 / 29.9 / 44.1; valid-pairs-only supervision 27.2 / 28.2 / 30.1 / 46.4; +reorder 28.2 / 27.9 / 30.8 / 46.7; +lane o2m 30.5 / 31.5 / 30.3 / 48.0; full 32.2 / 33.9 / 31.8 / 49.0. Positives per lane 1 / 2 / 3 / 4 / 5: TOP_ll 30.4 / 32.2 / 32.2 / 31.7 / 30.8. Naive o2m gives nothing (27.0), group-DETR style 29.8. Hosts: TopoNet TOP_ll 10.9 -> 29.5 (A), 6.7 -> 33.3 (B); TopoLogic B 21.6 -> 37.3; LaneSegNet TOP_lsls +5.9 (baseline row ambiguous).

## Compute

reorder and o2m add no inference cost; the parallel cross-attention blocks add parameters (no numbers).

## Code

https://github.com/homothetic/RATopo

## Fit to our project

reuse_idea - effort S - confirms that link-head quality is limited by the NUMBER of labelled valid pairs; in our matching-free setting every predicted vertex can be paired with the ground truth through geometry, which is the dense end of this spectrum.

## Needs

any query-based host with a topology head.

## Conflicts

concept lives inside Hungarian assignment; nothing to port except the label-density argument.

## Tried in our experiments

none (E42 .. E56 used matching-free labels).
