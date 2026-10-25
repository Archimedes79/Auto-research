# MH57 - DTPP / TPP (pre-generated ego trajectory tree, conditional prediction, learned cost, joint differentiable training)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M (DTPP), A (TPP)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, S10, L.*

## Papers

DTPP: Differentiable Joint Conditional Prediction and Cost Evaluation for Tree Policy Planning in Autonomous Driving (Huang, Karkus, Ivanovic, Chen, Pavone, Lv, ICRA 2024) https://arxiv.org/abs/2310.05885 (arXiv HTML read); Tree-structured Policy Planning with Learned Behavior Models (Chen, Karkus, Ivanovic, Weng, Pavone, 2023) https://arxiv.org/abs/2301.11902 (abstract level)

## Idea

The ego tree of candidate trajectories is PRE-GENERATED (lane-centric target sampling), a conditional prediction model decodes all branches in one pass, a learned cost scores each branch, and prediction and cost are trained jointly; pruning to the top nodes halves the run time.

## Formulation

tree: two stages (0 - 3 s, 3 - 8 s); stage 1 from three reference paths with up to 30 sampled target states (third-order polynomial speed profiles), keep the top 5 nodes, expand each with 6 targets for stage 2; infeasible trajectories dropped, random discarding above the node limit M = 30. Cost per branch = horizon-averaged weighted sum of ego features, learned interaction features and a Gaussian RBF collision feature, weights from an MLP on the ego encoding (10 weights); P(s_m) = softmax(-c). Loss: smooth L1 for the prediction on the branch closest to the GT ego trajectory + cross-entropy of softmax-normalised plan scores against the GT label; weights beta1 = 1.0, beta2 = 0.2, IRL alpha = 0.1; the cost loss back-propagates into the prediction module; gradient through the branch selection not stated. - input / output: 10 agents, ego, 50 map elements (transformer, D = 256, 3 layers) -> branch trajectories, costs.

## Training

nuPlan, RTX 3080, batch 16, 20 epochs; regime `matching_free` (nearest branch as label).

## Differentiable?

partial - candidate generation and pruning are rule-based; cost and prediction are differentiable.

## Sensors and data

nuPlan (closed loop), vector inputs.

## Reported results

NR / R score: DTPP 0.8964 / 0.8978 (collision 0.025), TPP 0.7388 / 0.7699, PDM 0.9061 / 0.9150, IDM 0.6396 / 0.6168; tree vs single stage: single-3 s 0.7087 / 0.7044, single-8 s 0.7799 / 0.7558; joint vs separate training (no pruning) 0.8948 / 0.8933 vs 0.8134 / 0.8106.

## Compute

98.0 ms with pruning vs 199.6 ms without.

## Code

not checked

## Fit to our project

reuse_idea - effort M - the nearest published "pre-generated hypothesis tree + learned scoring + nearest-branch label + joint differentiable training"; pruning the top nodes per stage with a rule is accepted practice.

## Needs

a candidate generator; a nearest-branch label; a node-wise cost.

## Conflicts

the tree is not learned (rule-based), nodes are ego trajectories, not map elements.

## Tried in our experiments

none
