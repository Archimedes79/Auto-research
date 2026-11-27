# MH18 - Learned anchor embeddings as discrete behaviour modes held fixed along an autoregressive rollout

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, T.*

## Papers

Tractable Joint Prediction and Planning over Discrete Behavior Modes for Urban Driving (Villaflor et al., ICRA 2024) https://arxiv.org/abs/2403.07232 (arXiv HTML)

## Idea

K = 8 learned anchor embeddings are the discrete behaviour modes of each agent; training is open-loop WTA with cross-entropy on the mode logits; at planning time the one-step decoder is rolled out autoregressively with the mode held fixed, so no retraining is needed.

## Formulation

z in {1..K}, anchor embeddings in R^{K x D} added to agent features (one set for ego, one for others); each mode predicts a trajectory with Gaussian waypoints (NLL), WTA: only the closest of the K predictions backpropagates the regression loss (per agent); cross-entropy on mode logits with a one-hot at the closest prediction; the decoder predicts H steps open-loop as an auxiliary task. Rollout: first predicted step only, mean output, z_0 = ... = z_{T-1}; planner evaluates only the K ego modes with N = 8 rollouts each (K^T joint sequences judged too expensive). - input / output: encoded agents and map -> K mode trajectories.

## Training

AdamW lr 2e-4 cosine, 100 epochs; regime `wta` (open loop), used `free_rollout` only at test time.

## Differentiable?

partial - WTA constant.

## Sensors and data

CARLA (merging, Longest6); ~1.9 M and ~7.4 M parameters.

## Reported results

merging 96.5 % success vs 85.5 % autopilot; Longest6 driving score 51.0 vs autopilot 44.6, PlanT 37.1. Mode collapse not discussed; discrete modes give more diverse proposals than a continuous latent CVAE.

## Compute

planning linear in agents and horizon; no runtime numbers.

## Code

not checked

## Fit to our project

reuse_idea - effort S - shows that WTA-trained mode slots can be unrolled for many steps with the slot held fixed (cheap consistency), relevant to a child carrying its slot identity down the tree.

## Needs

mode embedding conditioning; a one-step decoder.

## Conflicts

holding the mode fixed is a modelling choice (works "empirically"); no tree branching or pruning.

## Tried in our experiments

none
