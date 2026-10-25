# C13 - GoalFlow: goal point scoring and the shadow trajectory as a reliability check

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation (Xing et al., CVPR 2025) https://arxiv.org/abs/2503.05689 (HTML v6 read) ; code https://github.com/YvanYin/GoalFlow
- depth: M
- stage: S10, S4
- idea: A vocabulary of goal points (K-means of training end points x, y, heading; 4096 or 8192 centres) is scored by distance and a drivable-area (DAC) score; the best goal conditions a rectified-flow trajectory generator. A "shadow trajectory" is generated with the goal masked; if it deviates strongly from the goal-conditioned one, the goal is treated as unreliable and the shadow trajectory is output (no threshold given, no quantitative ablation).
- formulation: goal CE losses weights 1.0 (distance) and 0.005 (DAC); noise sigma 0.1 best; conditioning randomly masked in training (classifier-free style).
- results: NAVSIM PDMS 90.3; with ground-truth end point as goal 92.1; human 94.8; TransFuser 84.0. Inference per sample: 20 steps 177.8 ms (PDMS 89.9), 10 steps 92.4 ms, 5 steps 49.0 ms (90.3), 1 step 10.4 ms (88.9). No wrong-goal ablation.
- differentiable: partial - goal is an argmax from a scored vocabulary at inference.
- fit_to_us: reuse_idea - effort S - the shadow pass (condition dropped at training time, compare outputs at test time) is a label-free monitor of "does the route change the plan" for our planner; the GT-goal gap (90.3 -> 92.1) shows the value an oracle route could give.
- tried_before: oracle route 0.316 m path error (E119 note) - same kind of upper bound.
- verified: partly (no command usage in GoalFlow; goal chosen from scene features)
