# C20 - DiffusionDrive: anchored truncated diffusion planner (command only)

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving (Liao et al., CVPR 2025) https://arxiv.org/abs/2411.15139 ; code https://github.com/hustvl/DiffusionDrive
- depth: M
- stage: S10, S4
- idea: K-means trajectory anchors (20 for NAVSIM, 18 for nuScenes) are diffused only 50/1000 steps; 2 denoising steps, cascade decoder with deformable BEV cross-attention and agent cross-attention, parameters shared across steps. The extracted page did not describe the use of the navigation command.
- results: NAVSIM 88.1 PDMS, 45 FPS on a 4090 (3.8 ms per step, 7.6 ms planning), about 60 M params; nuScenes L2 0.57 m, collision 0.08 %, 8.2 FPS.
- differentiable: yes except top-1 selection. fit_to_us: baseline - effort L - our consumer is 93 k parameters; take the idea of few-step anchored decoding only if we need multimodal paths.
- tried_before: none. verified: partly (command handling not seen)
