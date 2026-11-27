# C15 - Diffusion Planner: route lanes as an AdaLN condition

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: Diffusion-based Planning for Autonomous Driving with Flexible Guidance (Zheng et al., ICLR 2025) https://arxiv.org/abs/2501.15564 ; project page https://zhengyinan-air.github.io/Diffusion-Planner/ (no repo link in the content)
- depth: M
- stage: S10
- idea: Route = K lanes with P points each (coordinates only), encoded by an MLP-Mixer to Q_n, added to the diffusion timestep embedding and applied through adaptive layer norm to all tokens; lanes are 70 polylines x 20 points, 12 features per point.
- results: nuPlan Val14 89.87 / 82.80 (non-reactive / reactive, learning-based, no refinement), Test14-hard 75.99 / 69.22; with refinement Test14 94.80 / 91.75. About 20 Hz on an A6000 (0.04 s in one table, 0.05 s in the limitations section - the paper is inconsistent). 3 blocks, hidden 192, 6 heads, 10 denoising steps; params not stated. Wrong route not addressed.
- fit_to_us: reuse_idea - effort S - global route embedding via AdaLN is an alternative to per-edge route attributes; they differ in whether the planner can tell WHICH lane is the route.
- differentiable: yes. tried_before: none. verified: partly
