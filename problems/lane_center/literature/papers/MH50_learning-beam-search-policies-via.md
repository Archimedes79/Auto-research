# MH50 - Learning beam search policies via imitation learning (beam-aware losses, continue / reset / stop)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M (partial)** (A abstract, M method, C code page). Verified: partly. Pipeline stage: T, L.*

## Papers

Learning Beam Search Policies via Imitation Learning (Negrinho, Gormley, Gordon, NeurIPS 2018) https://arxiv.org/abs/1811.00512 (read via ar5iv; experiments section not reached)

## Idea

Treat the beam as part of the model: a scoring function ranks the neighbours of the beam elements, an oracle cost-to-go defines which neighbour is best, and surrogate losses on the beam are minimised on trajectories collected by the learned policy.

## Formulation

beam = up to k nodes; candidates A_b = all neighbours; keep top-k by score s(v, theta); cost-to-go c*(v); beam cost = min over elements; cost increase when the best completion falls out of the beam. Losses: perceptron (first) = early update; perceptron / margin (last); cost-sensitive margin (last) close to BSO; upper bound on expected transition cost; log losses normalised over the beam or over all neighbours. Data collection: oracle, stop (first cost increase), reset (to the oracle's best node, as LaSO), continue (keep following the learned policy, as DAgger, the new strategy), mixtures with weight beta. Theory: no-regret reduction (DAgger-style), bounds for stop and reset depend on how often the roll-in deviates. - input / output: any search space with scores and an oracle.

## Training

regime `dagger` (continue) / `free_rollout`; Adam or SGD; losses differentiable w.r.t. the scores (how gradients pass the beam selection is not detailed).

## Differentiable?

partial - selection constant, loss on scores.

## Sensors and data

NLP tasks (not reached in the text read).

## Reported results

none seen.

## Compute

not stated.

## Code

not checked

## Fit to our project

reuse_idea - effort M - the clearest statement of training with the beam in the loop; continue (own beam) is our free roll-out, reset is gold-in-beam; the oracle cost-to-go can be a point-to-polyline residual of the subtree.

## Needs

an oracle cost per node (distance to the GT lane graph).

## Conflicts

needs the node scores to be comparable; loss choices matter (not tested for geometry).

## Tried in our experiments

none
