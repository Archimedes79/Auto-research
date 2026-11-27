# MH45 - Differentiable particle filtering without modifying the forward pass (stop-gradient reweighting)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, S8.*

## Papers

Differentiable Particle Filtering without Modifying the Forward Pass (Scibior, Wood, 2021) https://arxiv.org/abs/2106.10314 (read via ar5iv; last 28 k characters not read)

## Idea

Keep the hard resampling of the forward pass. Stop the gradient of the weights flowing into the resampling distribution and multiply each resampled particle's weight by v/stopgrad(v), which equals 1 in the forward pass; gradients then flow into the resulting weights.

## Formulation

automatic differentiation of the log-marginal-likelihood estimate recovers the score-function estimator of Poyiadjis et al.; unbiased for expectations under the unnormalised posterior (Theorem 1), consistent for the normalised one; the common practice of dropping resampling terms has a bias that does not vanish (citing Corenflos); marginal variant reduces variance from quadratic to linear in sequence length at quadratic cost in the particle number. - input / output: weighted particles.

## Training

linear Gaussian SSM (200 steps), VRNN on occluded 2-D images (40 steps).

## Differentiable?

partial - forward hard, gradient to weights through the stop-gradient trick.

## Sensors and data

synthetic and image sequences.

## Reported results

improves over standard PF with few particles and converges faster with many; on VRNN standard PF fits training data better but overfits, the new estimator is better on test; no numbers in the excerpt.

## Compute

small extra cost; the forward pass is unchanged.

## Code

not checked

## Fit to our project

reuse_idea - effort S - pattern for "hard top-M forward, multiplicative node weights carry the gradient of the downstream loss to the survivors' scores"; inconclusive for learning the proposal.

## Needs

multiplicative weights on survivors.

## Conflicts

dropped candidates still receive no gradient.

## Tried in our experiments

none
