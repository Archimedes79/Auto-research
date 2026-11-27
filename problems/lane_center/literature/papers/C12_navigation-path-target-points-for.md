# C12 - Navigation-path target points for end-to-end planning (VAD-based)

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: Utilizing Navigation Paths to Generate Target Points for Enhanced End-to-End Autonomous Driving Planning (NTT, 2024) https://arxiv.org/abs/2406.08349
- depth: M
- stage: S10, S9
- idea: A rough Google Maps route is interpolated to 5 m spacing, the 11 points nearest the ego (about 50 m) are encoded as [d_i, cos h_i, sin h_i] by a 2-layer VectorNet-style encoder; a target-generation module scores dense candidates on reconstructed lane centre-lines and in front of the ego by cross-attention over scene tokens; the best candidate is added to the route feature as driving-intent query for a 6-waypoint decoder (every 0.5 s, 3 s).
- design for meter-level error: models mainly the DIRECTION of the route (difference vectors + heading), because the route may deviate by metres.
- results (nuScenes val): average L2 0.61 m / collision 0.20 % vs VAD-Base 0.72 / 0.22; turning scenes L2 1.40 vs 1.88 m (collision 0.47 vs 1.09 %); navigation command variant (target + command) 1.86 / 1.28 % worse than target + learned embedding; target module ablation 0.63 -> 0.61 L2, 0.71 -> 0.20 % collision. No noise test of the path.
- differentiable: partial - the target is the argmax of candidate scores (selection). Code: none found.
- fit_to_us: reuse_idea - effort S - confirms "candidate scoring on the lane centre-lines" as the consumer-side use of route information; our consumer reads the route probability per edge which is the lane-graph version of this.
- tried_before: planner consumer E117 (0.34 m at 20 m). verified: partly
