# C29 - EgoVM: histogram pose solver with covariance on vectorised lane maps

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: EgoVM: Achieving Precise Ego-Localization using Lightweight Vectorized Maps (2023, older anchor) https://arxiv.org/abs/2307.08991
- depth: M
- stage: S9
- idea: Grid-search offsets (+-3 m in x, y; +-3 deg yaw): map elements are projected into the BEV, BEV features are bilinearly sampled and scored against map embeddings by a shared MLP; softmax -> posterior; pose = posterior mean, covariance = posterior spread.
- training: covariance-weighted RMSE on the offset + KL loss on the grid + KL on random poses (mixture of t / von Mises / uniform) + focal segmentation loss; weights not stated.
- results: nuScenes full mode longitudinal / lateral MAE 10.9 / 3.4 cm, yaw 0.089 deg; 120 ms average (RTX 3080). Initial-pose noise range not stated.
- differentiable: yes. code: none found.
- fit_to_us: reuse_idea - effort M - "bilinear sampling of BEV features at projected map points, shared MLP score" is the cheapest volume (cost scales with candidates x map points); it is the same sampling primitive as our trace network (grid_sample on the pyramid).
- tried_before: E103-like volume dropped. verified: partly
