# TP17 - SMART (SD + satellite map prior, frozen) and SMERF

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M (SMART and SMERF)** (A abstract, M method, C code page). Verified: checked (SMART, SMERF; SMERF encoder width H and polyline count M not stated, BEV size not stated). Pipeline stage: S9 B.*

## Papers

SMART: Advancing Scalable Map Priors for Driving Topology Reasoning (Ye et al., ICRA 2025) https://arxiv.org/abs/2502.04329 ; SMERF: Augmenting Lane Perception and Topology Understanding with Standard Definition Navigation Maps (Luo et al., arXiv 2311.04079) https://arxiv.org/abs/2311.04079

## Idea

A map-prior network is trained offline to predict lane graphs from SD-map polylines and a satellite crop, supervised by HD maps of a large geo-referenced set; it is then frozen and its BEV features replace (or are cross-attended by) the learnable BEV queries of an online topology model.

## Formulation

SD map = m polylines of N = 11 points with attributes -> sinusoidal embedding + linear -> 6-layer transformer encoder with validity mask; satellite: Mapbox raster zoom 20 (about 0.11 m / pixel), rotated and cropped to ego, resized 500 x 250, ResNet-50 (last three stages). A learnable BEV map (200 x 100 x 256) cross-attends first to SD, then to satellite features. Pre-training: about 200 k scenes / 810 k HD maps from the Argoverse 2 motion-forecasting set (full set 199,908 scenes), 8 epochs, 8 A100 about 2 days. Injection: TopoNet (BEV queries replaced by prior features) and TopoMLP (extra cross-attention aligning prior with perspective features); decoder 200 queries + simplified GCN; focal + L1.

## Input

SD polylines, satellite raster, ego pose (GPS + heading; error handling not described).

## Output

BEV prior features (and offline lane graph).

## Training

two stages; prior frozen during online training; 24 epochs.

## Differentiable?

yes to the online model; the prior is frozen.

## Sensors and data

OpenLane-V2 (no A / B split reported), 100 x 50 m window.

## Reported results

online OLS / TOP_ll: TopoNet 39.8 / 10.9 -> 51.1 / 27.5; TopoMLP 44.1 / 21.7 -> 53.1 / 37.0. Modality OLS: SD only 44.1, satellite only 48.1, both 51.1. Data scale offline TOP_ll 16.2 / 28.9 / 31.9 (base, 18x, 40x). Geo-disjoint split OLS 26.9 -> 33.6. SMART alone offline DET_l 37.9, TOP_ll 31.9, 44.3 ms (TopoNet offline 172.6 ms). Sensor reduction: 40 % of sensor data with the prior comparable to full data without (figure only). SMERF (encoder reused by SCORE, MapVision, Reusing-Attention): OSM polylines per frame (M polylines, N = 11 points each, sinusoidal position encoding d = 32, T = 1000, one-hot road type), 6-layer transformer encoder trained jointly with the topology model (no extra loss), fused by cross-attention (BEV queries attend to SD tokens) after every spatial cross-attention of the BEV encoder; GPS pose given as a 3-DoF transform, noise not handled. Subset A (DET_l / TOP_ll / OLS): TopoNet 28.2 / 4.1 / 34.5 -> +SMERF 33.4 / 7.5 / 39.4; weaker baseline 17.0 / 2.3 / 30.2 -> 26.8 / 3.9 / 34.8. Encoder ablation (OLS): raster SD map 30.9, VectorNet-style 31.2, SMERF polylines 34.8 (xy position encoding is worth +2.3, normalisation +1.6); heads 4 / 8 / 16: 34.8 / 34.5 / 33.9; gain concentrated on far lanes (25 - 50 m: drop close to far 3.4 points without, 0.8 with SD map) and intersections; geo-disjoint split 16.9 -> 22.1.

## Compute

SMART alone 44.3 ms vs 172.6 / 328.2 ms for the online models (offline settings, see above).

## Code

SMART project page https://jay-ye.github.io/smart (no repository); SMERF https://github.com/NVlabs/SMERF (not opened)

## Fit to our project

reuse_idea - effort L - our route head already cross-attends vertex tokens to SD-map raster tokens; SMART adds large-scale offline pre-training of the prior and shows the benefit grows with data (TOP_ll 16.2 -> 31.9); no satellite data in our setting.

## Needs

ego pose / localisation; large HD-map corpus for pre-training.

## Conflicts

assumes GPS pose without error (our registration problem E40 / E48 is not addressed); satellite imagery.

## Tried in our experiments

E40 / E48 (map prior at the grid, learned registration): kept; E102 .. E104 (route head, cross-attention kept).
