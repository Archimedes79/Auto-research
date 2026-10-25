# C21 - VADv2 and SparseDrive: discrete command / navigation embedding in the planner

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning (Chen et al.) https://arxiv.org/abs/2402.13243 ; SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation (Sun et al.) https://arxiv.org/abs/2405.19620 ; code https://github.com/hustvl/VAD , https://github.com/swc-17/SparseDrive
- depth: M
- stage: S10
- idea: VADv2: vocabulary of 4096 trajectories, navigation + ego state embedded by MLPs and added to the decoder output before a sigmoid per-action probability; CE against the closest demonstration + conflict loss. SparseDrive: three commands (left / right / straight), planning queries sized per command (N_cmd x K_p x C) from K-means intention points, 6 modes x 6 steps, winner-takes-all, selection filtered by the command and by a collision-aware rescore (collision 0.12 -> 0.08 %).
- results: VADv2 CARLA Town05 Long DS 85.1 / RC 98.4; 3DGS benchmark 125 ms, 0.40 B params (RTX 4090). SparseDrive-S: nuScenes L2 0.61 m, collision 0.08 %, 9.0 FPS, 85.9 M params; SparseDrive-B 0.58 / 0.06 %; VAD 0.72 / 0.21, UniAD 0.73 / 0.61.
- route: global command only; no route geometry, no SD map, no wrong-command test in either paper.
- fit_to_us: baseline - effort L - the contrast case: command-only interface; our lane-attribute interface carries strictly more information.
- differentiable: yes (except top-1). tried_before: none. verified: partly
