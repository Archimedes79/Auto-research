# PL44 - OpenLane-V2-OSM: SD maps for online road-network perception and reasoning

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S9.*

## Papers

Enhancing Online Road Network Perception and Reasoning with Standard Definition Maps (arXiv 2408.01471; authors not read) project page https://henryzhangzhy.github.io/sdhdmap/

## Idea

Lane-centre-line detection and topology on OpenLane-V2 (Argoverse 2 subset A; the OLV2-OSM extension adds OSM for 1,000 maps). The SD map is fused as a raster (ResNet-18 features concatenated with BEV features + 2-layer ConvFuser for MapTR; stacked channel-wise for TopoNet) or as a graph (SD nodes augmented with BEV features at their positions, EdgeConv GNN, fused with centre-line queries by attention).

## Formulation

not read beyond the fusion description.

## Input

BEV features + SD raster or SD graph.

## Output

centre-lines with topology.

## Training

MapTR 110 epochs; TopoNet 24 epochs.

## Differentiable?

yes.

## Sensors and data

OpenLane-V2 / OLV2-OSM.

## Reported results

MapTR Chamfer mAP 40.0 -> 46.6 (OLV2 SD map) / 43.2 (OSM raster); the abstract's 30 % does not match the table (about 16.5 %). TopoNet OLS 34.8 -> 37.7 (+13 M parameters, raster) or 36.7 (+1.7 M, graph); TOP_ll 4.15 -> 7.71. Localisation-error robustness (performance loss, translation 0.25 / 0.5 / 1.0 / 2.0 m at 0 deg rotation): 0.45 / 0.90 / 3.07 / 10.82 %; at 10 deg rotation 13.64 ... 21.14 %. No route used.

## Compute

+13 M or +1.7 M parameters.

## Code

none (project page)

## Fit to our project

reuse_idea - effort S - graph-style SD fusion costs 1.7 M parameters vs 13 M for the raster fusion.

## Needs

an SD graph.

## Conflicts

perception range of OpenLane-V2.

## Tried in our experiments

E40 / E48 (misalignment sensitivity: +0.075 F1 with exact pose, -0.057 unaligned).
