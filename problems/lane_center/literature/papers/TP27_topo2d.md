# TP27 - Topo2D (2D lane priors initialise 3D lane queries)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S1.*

## Papers

Enhancing 3D Lane Detection and Topology Reasoning with 2D Lane Priors (Topo2D, arXiv 2406.03105) https://arxiv.org/abs/2406.03105

## Idea

A per-view 2D lane detector (hierarchical instance + point queries) provides the appearance and position (lifted along camera rays) for the 3D lane queries and 2D lane features enter the topology head.

## Formulation

20 instance + 11 point queries per view, 6 layers; PETR-style lifting of (u, v) over predefined depths, MLP to a 3D positional embedding; topology MLP over concatenated pooled 2D + 3D lane embeddings plus the embedded 3D coordinates; weights lane cls 2, reg 5, topology 5, GIoU 2.

## Input

perspective images only (no BEV encoder).

## Output

3D lanes + topology.

## Training

bipartite matching (named, not Hungarian explicitly); 24 epochs, AdamW 3e-4, batch 8.

## Differentiable?

partial - matching.

## Sensors and data

camera; OpenLane-V2 subset A only (B has no height).

## Reported results

subset A TOP_ll 22.3 vs TopoNet 10.9, OLS 44.5; ablation: 3D query init from 2D 28.8 / TOP_ll 22.4, random 24.7 / 19.1.

## Compute

not stated.

## Code

https://github.com/homothetic/Topo2D

## Fit to our project

not_suitable - needs perspective image features and per-view 2D detectors; the principle (good query initialisation) corresponds to our seeds.

## Needs

image features.

## Conflicts

cached BEV only.

## Tried in our experiments

E31 / E32.
