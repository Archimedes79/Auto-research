# MH17 - PGP (policy over lane-graph edges, behaviour-cloning label, sampled traversals, K-means)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, S6, L.*

## Papers

Multimodal Trajectory Prediction Conditioned on Lane-Graph Traversals (Deo, Wolff, Beijbom, CoRL 2021) https://arxiv.org/abs/2106.15004 (read via ar5iv)

## Idea

A discrete policy over the successor edges of a lane graph is trained by behaviour cloning on the edges the future visits, sampled for many rollouts at inference, and each rollout conditions a trajectory decoder; the final modes are K-means clusters of many samples.

## Formulation

nodes = centreline snippets <= 20 m, 1 m poses; policy header = MLP scoring each outgoing edge from motion encoding and node encodings, softmax over outgoing edges plus an edge to an end state; label: each future pose assigned to the closest lane node whose direction is within a yaw threshold; edge visited if both endpoints visited; loss = sum of -log pi over visited GT edges (Eq. 3). Decoder: latent z ~ N(0, I), 5-d; regression loss = min over modes of average displacement error (WTA). 200 samples clustered to K = 10 (results at K = 5 and 10). Pre-train 100 epochs on GT traversals, fine-tune 100 epochs on policy-sampled paths (about 1 min vs 5 min per epoch). - input / output: HD-map lane graph + agent histories -> traversals -> trajectories.

## Training

Adam 1e-4, batch 32, 4 V100; regime `teacher_forcing` (pre-train) then `free_rollout`-like fine-tuning on sampled paths.

## Differentiable?

partial - sampling of traversals is discrete; WTA regression on the sampled modes.

## Sensors and data

nuScenes (HD map input).

## Reported results

nuScenes Table 1: MinADE5 1.30, MinADE10 1.00, MissRate5,2 0.61, MissRate10,2 0.37, off-road rate 0.03. Traversal + latent decoder reaches 1.41 distinct final lanes vs 1.22 latent-only. The paper notes sampling is often redundant; no collapse analysis, no greedy-vs-sampled comparison.

## Compute

many samples run in parallel; no latency numbers.

## Code

not checked

## Fit to our project

reuse_idea - effort M - a lane-graph fork is trained by behaviour cloning on edges; shows sampling + clustering as the standard way to get K diverse hypotheses from a branching policy; the sample-then-cluster step corresponds to our DAG merge.

## Needs

a lane graph as input (we produce it); node snippets.

## Conflicts

input is the HD map (ours is the output); sampling is not differentiable.

## Tried in our experiments

none
