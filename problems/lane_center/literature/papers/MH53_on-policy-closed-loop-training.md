# MH53 - On-policy closed-loop training for multimodal prediction (execute one mode, detach across transitions)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: T, L.*

## Papers

Rectify, Don't Regret: On-Policy Closed-Loop Training for Multimodal Trajectory Prediction (Yadav, Bohn, Meisen, arXiv 2603.23393, v2 Sept 2026) https://arxiv.org/abs/2603.23393 (arXiv HTML read; code https://github.com/harshy105/CL-Traj). A companion paper (arXiv 2603.24155) is marked withdrawn and is not used.

## Idea

A multimodal predictor is rolled out in closed loop during training: each predicted mode is supervised only on states its own execution produced, and gradients are cut at the executed transitions so the model learns to recover from drift instead of "regretting" an earlier prediction.

## Formulation

receding horizon: each predicted mode is executed for T_sim (default 2.0 s, T_pred = 6 s), the model re-predicts from the resulting state; stop-gradient on the executed ego positions. The open-loop sample (n = 0) supervises classification (mode probabilities) and regression of all M modes; closed-loop samples (n >= 1) supervise only regression of the executed mode m*, with the original logged future as target; m* is chosen from the open-loop sample by L2 distance to the ground truth (WTA) and fixed for the rollout, so training unrolls ONE mode (cost scales with replanning count, not M x replanning). At test time all M modes are rolled out. An off-policy variant that re-selects the best mode each step was worse, especially at T_sim = 0.5 s. A fully differentiable closed loop ties the policy to the training replanning rate (shown in a figure only). - input / output: LMFormer-D (decoder-only) and an unmodified QCNet.

## Training

regime `free_rollout` (on-policy, detached) mixed with open-loop teacher signal; 5 seeds main, 3 ablations.

## Differentiable?

partial - detached by design.

## Sensors and data

nuScenes, Argoverse 2, DeepScenario.

## Reported results

closed-loop evaluation at T_sim = 0.5 s, collision % open-loop -> closed-loop training: nuScenes 3.95 -> 3.08, DeepScenario 4.46 -> 2.92, Argoverse 2 (QCNet) 2.84 -> 2.35; off-road % 5.65 -> 1.86, 22.28 -> 8.62, 16.67 -> 15.07; at T_sim = 6.0 s the two are comparable; abstract: up to 35 % fewer collisions, 67 % fewer off-road.

## Compute

closed-loop samples add cost proportional to replanning frequency; LMFormer-D has 45 % fewer parameters and 62 % fewer FLOPs than LMFormer.

## Code

https://github.com/harshy105/CL-Traj

## Fit to our project

reuse_idea - effort M - the only direct evidence for free roll-out with multiple hypotheses: execute the GT-nearest hypothesis (WTA from the teacher sample), supervise regression only for it, keep the classification (score) signal from the teacher-forced sample, detach across levels.

## Needs

a one-mode rollout path through the tree (the GT-nearest child continues), a separate teacher-forced pass for scores.

## Conflicts

detaching cuts the path from the downstream loss to the early children (our E121 finding that planner gradients through positions give no gain fits).

## Tried in our experiments

none
