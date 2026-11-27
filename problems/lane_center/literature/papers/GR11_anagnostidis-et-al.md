# GR11 - Anagnostidis et al. (edge-sequence agent trained with MuZero-style RL, tree search at inference)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M** (A abstract, M method, C code page). Verified: partly (ablation and Table 2 columns garbled; ICCV PDF unreadable).. Pipeline stage: S4 (search), T.*

## Papers

Mastering Spatial Graph Prediction of Road Networks (Anagnostidis, Lucchi, Hofmann, ICCV 2023; arXiv 2210.00828, Oct 2022) https://arxiv.org/abs/2210.00828

## Idea

Given fixed candidate key points (from a segmentation detector), an agent adds edges by pointing at key points (two pointers = one edge, or EOS). Because the evaluation metrics (APLS, path / junction / sub-graph F1) are non-smooth, the agent is fine-tuned with a learned-model RL scheme (MuZero-style) whose reward is the *change in a weighted metric*; at inference a Monte-Carlo tree search over learned dynamics is run. The search tree is a DAG because the graph embedding is permutation invariant.

## Formulation

Policy = pointer network over N_V key points + EOS (dot product); representation = autoregressive transformer (two stacks of 16 self-attention layers, ResNet-18 / 50 features sampled at the key points); dynamics = LSTM predicting next hidden state and reward (up to 90 x fewer FLOPs than the decoder); value with a 601-bin support. Reward = delta of 0.35 APLS + 0.25 path-F1 + 0.25 junction-F1 + 0.15 sub-graph-F1; early training targets the graph from the post-processed segmentation mask. Pre-training: cross-entropy on a fixed autoregressive order (the ARM), RL with lower lr on pre-trained modules; 5-step unroll of the dynamics, Dirichlet root noise, re-analysis of old episodes, Ray workers. Search: PUCT-style selection with learned prior, simulation count at evaluation not stated (10 only in an illustration).

## Input

aerial image (300 x 300 resized; SpaceNet crops 400 x 400 with 100 px overlap) + key-point set; `image_feat` + `point_set`.

## Output

sequence of edges (pairs of key-point indices).

## Training

`next_token` pre-training then `rl` (reward from GT graph metrics); no matching.

## Differentiable?

no - sampling of actions, reward is a non-smooth metric; the learned dynamics model is differentiable internally.

## Sensors and data

aerial; SpaceNet, DeepGlobe, a synthetic city set (CityEngine).

## Reported results

GT key points, SpaceNet APLS: random 0.008, classifier 0.310, GCN 0.488, ARM 0.743, ours 0.778; fuzzy key points SpaceNet APLS 0.6523 vs best baseline 0.6422, path / junction / sub-graph F1 0.7431 / 0.7792 / 0.7922 (Tables 1 - 2). Order dependence of the supervised sequence model: perplexity 0.733 bits / edge in its training order but 30.84 in a random order, the RL model 4.636 / 4.565 (order-invariant graph embedding). Ablation (garbled in the fetched text): removing AR pre-training or visual key-point features costs on the order of 12 - 15 points, removing the tree search about 1 - 2, cross-attention to the image is not significant.

## Compute

trainable on a single GPU from the ARM start (stated); total compute not stated.

## Code

none found

## Fit to our project

reuse_idea - effort L (RL) / S (idea): the only search over graph-growth hypotheses in this cluster; its gain is the smallest effect of all components, which argues that for us a beam only pays if the per-step scorer is already good. The reward shaping by metric deltas is an alternative to a matching-free geometric loss.

## Needs

fixed key-point candidate set (our heat-map peaks); GT graph for rewards.

## Conflicts

RL / sampling excludes gradients through selections; cannot add or delete key points.

## Tried in our experiments

none.
