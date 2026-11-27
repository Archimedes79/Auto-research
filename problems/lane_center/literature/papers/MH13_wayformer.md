# MH13 - Wayformer (64 learned queries, winner-take-all NLL, aggregation to 6)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, L.*

## Papers

Wayformer: Motion Forecasting via Simple & Efficient Attention Networks (Nayakanti et al., 2022) https://arxiv.org/abs/2207.05844 (arXiv HTML; venue not checked)

## Idea

A transformer decoder with k learned initial queries (one per mixture component) cross-attending to the scene encoding; the loss splits into a classification term for the winner and a regression term for the winner only. Benchmarks use 64 components reduced to 6 by the MultiPath++ aggregation.

## Formulation

winner = Gaussian whose mean is closest to the GT (distance not named); loss = -log Pr(winner | scene) - log Pr(GT | winner Gaussian) (hard WTA). Aggregation (appendix): greedy centroids that cover the most uncovered likelihood within distance D (2.3 Waymo, 2.9 Argoverse), then k-means style refinement (3 or 5 iterations), at most 6 trajectories; ensembles merge N = 3 decoders (Waymo) or 10 decoders x 15 replicas (Argoverse). - input / output: scene tokens (early / late / hierarchical fusion, latent-query bottleneck) -> 6 or 64 modes.

## Training

1 M steps, AdamW, lr 2e-4 linear decay, batch 256 (ablations; hyper-parameter ranges in text and appendix are inconsistent).

## Differentiable?

yes for heads; winner constant; aggregation outside the gradient path as read.

## Sensors and data

Waymo Open Motion, Argoverse; no raw sensors.

## Reported results

WOMD (early fusion; latent query vs factorised) minADE 0.545 / 0.545, minFDE 1.128 / 1.126, MR 0.123, mAP 0.419 / 0.412 (MultiPath++ 0.409); Argoverse Brier-minFDE 1.7408 (MultiPath++ 1.7932). No "6 modes directly vs 64 + aggregation" comparison and no collapse numbers on the page.

## Compute

ablation models 0.3 - 20 M parameters; latent queries 2x - 16x faster.

## Code

not checked

## Fit to our project

reuse_idea - effort S - confirms "train with many more hypotheses than needed, reduce by likelihood-aware aggregation".

## Needs

mode-wise distance and probability.

## Conflicts

needs a large mode pool; aggregation is inference-time clustering.

## Tried in our experiments

none
