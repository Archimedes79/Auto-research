# PL41 - SMART (scalable map prior from SD + satellite maps, pre-trained on HD maps)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S9, B.*

## Papers

SMART: Advancing Scalable Map Priors for Driving Topology Reasoning (authors not read; ICRA 2025 per the page) https://arxiv.org/abs/2502.04329 ; project page https://jay-ye.github.io/smart

## Idea

A map-prior encoder is pre-trained on geo-referenced HD maps (Argoverse 2: 200 k scenes and 4.4 M HD maps in total; main model 200 k scenes / 810 k HD maps) from SD maps (OSM polylines) and satellite tiles only, then frozen and plugged into online topology models. No navigation route is used (the page says a route could be used to pre-compute priors but this is not evaluated).

## Formulation

SD polylines resampled to 11 points, padded, sinusoidal embeddings, linear + 6-layer transformer with validity mask; satellite: Mapbox zoom 20 (about 0.11 m/px) resized to 500 x 250, ResNet-50 multi-scale features; prior = BEV grid 200 x 100 x 256 made by cross-attending SD then satellite features. TopoNet: the prior replaces the learnable BEV queries; TopoMLP: a cross-attention layer aligns the prior with perspective features.

## Input

OSM polylines + satellite tiles at the GPS location; BEV or perspective features.

## Output

BEV prior grid for the online model.

## Training

SMART 8 epochs, 8 x A100, about 2 days; online models 24 epochs; weights frozen online; pose noise not discussed.

## Differentiable?

yes (frozen prior).

## Sensors and data

Argoverse 2 HD maps for pre-training; OpenLane-V2 (tables below are the Argoverse 2-based subset).

## Reported results

OpenLane-V2 val DET_l / TOP_ll / OLS: TopoNet 28.6 / 10.9 / 39.8 -> SMART-OL 46.1 / 27.5 / 51.1; TopoMLP 28.5 / 21.7 / 44.1 -> 46.6 / 37.0 / 53.1; SD-only prior (online) 34.2 / 16.5 / 44.1; satellite-only 41.2 / 22.2 / 48.1; offline SMART alone 37.9 / 31.9; geographically disjoint split OLS 26.9 -> 33.6; data scaling 1x / 18x / 40x OLS 45.3 / 49.3 / 51.1.

## Compute

see training.

## Code

none listed (project page only)

## Fit to our project

reuse_idea - effort M - validity-masked polyline tokenisation of the SD map (N = 11 points).

## Needs

geo-referenced SD vectors and the pose.

## Conflicts

needs GPS-level alignment; route unused.

## Tried in our experiments

E40 (map prior at the grid) and E48 (registration).
