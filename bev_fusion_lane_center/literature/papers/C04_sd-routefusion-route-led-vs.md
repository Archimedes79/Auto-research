# C04 - SD-RouteFusion: route-led vs perception-led trajectory with a learned gate

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: SD-RouteFusion: Ego-Trajectory Prediction with SD-Map Route Conditioning (Voloshyn et al., FUSION 2026) https://arxiv.org/abs/2607.01139 ; code https://github.com/zenseact/SD-RouteFusion
- depth: M
- stage: S10, S9
- idea: Encode the SD route (OSM route between start and end, sequence of 2D points, re-centred to the first point) with a small MLP to a 64-d vector; two mirrored cross-attention branches predict a route-led and an image-led trajectory; an MLP gate reads both embeddings and the difference of the two predictions and selects ONE branch (hard choice).
- formulation: gate logit trained with BCE against a soft target derived from the per-sample L2 errors of the two branches, scaled by a temperature (loss weights not stated). Route built by matching OSM links between GNSS start and end with a 20 m search radius growing in 50 m steps to 1 km; straight-line fallback for "no route".
- input: front camera, kinematics, route points.
- output: 8 s trajectory.
- training: supervised on logged trajectories; end-to-end; gate supervised by relative branch errors.
- differentiable: partial - the final selection is a hard branch choice (gate itself is trained with a soft target).
- sensors_data: front camera; 480 k private scenarios in 10 European countries + US (not public); public ZOD for reproduction.
- results: 8 s horizon ADE: image+kinematics 2.19, +route MLP fusion 1.96, SD-RouteFusion 1.82 (FDE 5.58 / 4.51 / 4.32). Image branch alone 2.59, route branch alone 1.91. Gate chose the route branch in 66 % of test scenarios. No controlled corruption experiment (natural corruption only: drift, outdated maps, connectivity errors), qualitative examples plus one failure case.
- compute: not stated.
- code: https://github.com/zenseact/SD-RouteFusion (released per paper; not opened)
- fit_to_us: reuse_idea - effort S - a gate that decides "trust route vs trust perception" is the closest published thing to abstention; supervise it with the error difference of two planner passes (with and without route).
- compat_needs: two planner passes (route masked / route given) - our planner has 93 k parameters so the cost is small.
- compat_conflicts: route is a global point list, not a lane attribute; hard branch choice blocks gradient through the selection.
- tried_before: none (planner loss into route head E119 no gain is a different thing).
- verified: partly (gate internals only as extracted)
