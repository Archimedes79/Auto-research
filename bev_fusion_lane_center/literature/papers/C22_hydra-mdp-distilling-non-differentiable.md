# C22 - Hydra-MDP: distilling non-differentiable rule scores into per-trajectory heads

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: Hydra-MDP: End-to-end Multimodal Planning with Multi-target Hydra-Distillation (Li et al., 2024) https://arxiv.org/abs/2406.06978 ; code (announced) https://github.com/woxihuanjiangguo/Hydra-MDP
- depth: M
- stage: L, S10
- idea: Every vocabulary trajectory (K-means of 700 k nuPlan trajectories, 4096 or 8192) is scored offline by the simulator metrics (no-at-fault collision NC, drivable area DAC, TTC, comfort, ego progress EP); one prediction head per metric learns these scores with BCE, so no post-processing is needed at inference.
- results: NAVSIM navtest PDMS best single model 86.5 (PDM-Closed with GT perception 89.1), scaled variant 91.0; 8192 better than 4096. The page did not mention navigation commands; latency not reported.
- differentiable: yes (scores are learned heads).
- fit_to_us: reuse_idea - effort M - add a "route compliance" score (fraction of the path inside the route lanes, computable from labels) as a distilled head or auxiliary loss on the consumer.
- tried_before: none. verified: partly
