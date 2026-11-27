# C19 - Route as lane-union BEV channel: CaRL and direct flow-matching policy

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: CaRL: Learning Scalable Planning Policies with Simple Rewards (Jaeger et al., 2025) https://arxiv.org/abs/2504.17838 ; Learning Direct Control Policies with Flow Matching for Autonomous Driving (Ceresini et al., ITSC 2026) https://arxiv.org/abs/2605.14832
- depth: M (CaRL: first 100k of 153k characters)
- stage: S10, S9
- idea: Both render the ROUTE as a lane-level mask (A* route lanes in CaRL; union of the lanes of the planned route in the second paper, pixel value encodes ego speed) in a BEV raster next to obstacles / drivable area. CaRL reports that the policy largely ignores lane markings and road channels and "abuses" the route mask for steering (lane changes at fixed places); drawing the route only inside intersections cut these crashes but sometimes caused wrong turns there.
- results: CaRL longest6 v2 DS 64 / RC 82 (PDM-Lite 73 DS, PlanT 62); input ablation: wider FoV and route only outside junctions removed DS 41 -> 31, RC 94 -> 63; about 8 ms per step (CARLA), about 14 ms (nuPlan), nuPlan Val14 CLS-NR 91.3. Flow-matching policy: closed loop in-distribution collision 3.75 %, drivable area 93.75 %, route progress 98.12 % (NFE = 10, 25 ms on an A6000; NFE = 1 gives 9.25 % collisions); no route perturbation in either paper.
- differentiable: policy yes (RL / flow matching); route mask is an input.
- fit_to_us: reuse_idea - effort S - this is "route as lane attribute" in raster form; same shortcut risk as C18. Code CaRL: https://github.com/autonomousvision/CaRL (not opened).
- tried_before: none. verified: partly
