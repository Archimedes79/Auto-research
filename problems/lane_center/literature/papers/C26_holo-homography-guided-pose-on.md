# C26 - HOLO: homography-guided pose on SD (OSM) maps with local correlation lookups

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: HOLO: Homography-Guided Pose Estimator Network for Fine-Grained Visual Localization on SD Maps (2026) https://arxiv.org/abs/2601.02730
- depth: M
- stage: S9
- idea: BEV and OSM raster (roads, buildings) features go to a dense correlation volume; each of 6 refinement iterations crops local patches around the current projection (no re-computation of attention); the decoder predicts corner displacements, converted to a homography by DLT; x, y from projecting the BEV centre, heading from a projected auxiliary point.
- formulation: loss BCE semantic (weight 1000) + L2 translation (1) + L1 yaw (10); AdamW + OneCycle, batch 16, 180 k iterations. Training perturbation +-30 m and +-30 deg of the map patch.
- results: nuScenes recall@1 m 36.41 % (with drivable-area supervision), 61.21 % @2 m, 73.74 % @1 deg; road-only SD variant 30.40 % @1 m; MapLocNet 20.10 / 45.54; stress test +-40 m still 18.24 % @1 m, 76.87 % @5 m. 25.6 FPS with 2 iterations, 19.9 FPS with 6 (RTX A6000). Differentiability of the DLT step not stated explicitly (supplement mentions a differentiable homography estimator). Code announced, not available.
- fit_to_us: reuse_idea - effort M - RAFT-style local correlation lookup around the current estimate is cheaper than a full volume.
- tried_before: E103 (correlation volume) was not better than cross-attention. verified: partly
