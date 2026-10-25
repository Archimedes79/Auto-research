# C09 - SMERF: SD-map transformer encoder fused into BEV by cross-attention

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: Augmenting Lane Perception and Topology Understanding with SD Navigation Maps (Luo et al., ICRA 2024) https://arxiv.org/abs/2311.04079 ; code https://github.com/NVlabs/SMERF (mmdet3d 1.0.0rc6, Python 3.8, PyTorch 1.9.1 per README)
- depth: M+C (README)
- stage: S9
- idea: Polylines of the OSM roads around the GPS pose are sampled at N = 11 points, encoded with sinusoidal embeddings (d = 32, T = 1000, coordinates normalised to (0, 2 pi)) plus one-hot road type, passed through a 6-layer transformer encoder, and cross-attended after each spatial cross-attention step of a BEV lane-topology model; trained end-to-end without extra labels.
- results (OpenLane-V2 subset A, OLS): baseline 30.2 -> 34.8; TopoNet 34.5 -> 39.4 (DET_l 28.2 -> 33.4); gains largest far away and at intersections; on a geographically disjoint split smaller (TopoNet 21.7 -> 23.4). No pose-noise, no wrong-map experiment, latency not reported.
- differentiable: yes. sensors_data: camera, OpenLane-V2.
- fit_to_us: reuse_idea - effort S - confirms the cross-attention graph/BEV <-> SD tokens pattern (same as E104b); nothing new for matching.
- tried_before: cross-attention to SD raster tokens (E104b) kept.
- verified: partly
