# C06 - Uni-PrevPredMap: tri-mode training (no prior / temporal prior / both) and perturbation table

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: Uni-PrevPredMap: Extending PrevPredMap to a Unified Framework of Prior-Informed Modeling (2025) https://arxiv.org/abs/2504.06647 ; code https://github.com/pnnnnnnn/Uni-PrevPredMap
- depth: M
- stage: T, S9, S8
- idea: One model that runs with no prior, a temporal prior, or a map prior plus a temporal prior; the training batch samples the three modes with fixed ratios so "missing prior" is a normal operating mode. Priors are rasterised heatmaps concatenated to BEV features and used to initialise queries.
- formulation: ratio non-prior : temporal : both = 0.5 : 0.3 : 0.2. Training perturbation: instance displacement +-6 m. Test table (nuScenes mAP): perfect map 85.6; instance displacement +-3 / 6 / 9 m 81.9 / 80.9 / 80.3; instance addition 85.2; deletion 82.5; frame displacement +-3 / 6 / 9 m 77.0 / 75.7 / 74.6; frame rotation +-15 deg 77.0; frame scale 0.8 - 1.2 77.0; empty map 74.0.
- training: standard detection loss; the "outdated prior" test mirrors a map about the trajectory start (qualitative only: predictions fall back toward the empty-prior behaviour).
- differentiable: yes. sensors_data: nuScenes (+ Argoverse2), camera.
- results: see formulation; FPS (A6000): no prior 14.2, temporal 12.2, map 13.1, both 11.5.
- code: https://github.com/pnnnnnnn/Uni-PrevPredMap (not opened)
- fit_to_us: reuse_idea - effort S - add a mode-sampling training scheme (no map / SD map / SD map shifted) for the route head and report "empty map" as the floor; their frame displacement is nearly the empty-map result, i.e. a frame-level shift of 3 - 9 m is almost as bad as no map.
- compat_conflicts: none. tried_before: none (explicit no-map mode).
- verified: partly
