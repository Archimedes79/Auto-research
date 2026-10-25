# PL43 - SMERF (SD navigation-map polylines fused by cross-attention into lane perception / topology)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S9.*

## Papers

Augmenting Lane Perception and Topology Understanding with Standard Definition Navigation Maps (Luo et al., ICRA 2024, from memory) https://arxiv.org/abs/2311.04079

## Idea

OSM road polylines around the ego (GPS 3-DoF), each sampled at N = 11 points with sinusoidal embeddings (d = 32, T = 1000) + one-hot road type (6 + catch-all), linear layer, 6-layer transformer; cross-attention between BEV queries and SD features after each spatial cross-attention in the BEV encoder. All roads are used (road type labelled, no planned route distinguished).

## Formulation

backbones BEVFormer-DeTR baseline and TopoNet.

## Input

SD polylines + BEV features.

## Output

lane detection and topology.

## Training

not read.

## Differentiable?

yes.

## Sensors and data

OpenLane-V2 subset A (Argoverse 2); the abstract claims gains up to 60 %.

## Reported results

BEVFormer-DeTR baseline DET_l 17.0 -> 26.8, TOP_ll 2.3 -> 3.9, OLS 30.2 -> 34.8; TopoNet DET_l 28.2 -> 33.4, TOP_ll 4.1 -> 7.5, OLS 34.5 -> 39.4; gains largest at far range and at intersections. Noise robustness not evaluated.

## Compute

not read.

## Code

https://github.com/NVlabs/SMERF

## Fit to our project

reuse_idea - effort S - vector tokenisation of SD roads as an alternative to our SD raster.

## Needs

SD polylines and pose.

## Conflicts

no route channel.

## Tried in our experiments

none (our SD input is raster).
