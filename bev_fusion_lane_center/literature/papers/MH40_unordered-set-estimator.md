# MH40 - Unordered-set estimator (unbiased gradients from k samples without replacement)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, T.*

## Papers

Estimating Gradients for Discrete Random Variables by Sampling without Replacement (Kool, van Hoof, Welling, ICLR 2020) https://arxiv.org/abs/2002.06043 (read via ar5iv; last 29 k characters not read)

## Idea

An unbiased estimator of expectations (and policy gradients) over discrete variables from a set of k samples drawn without replacement (Gumbel-top-k / stochastic beam search); a Rao-Blackwellisation of single-sample and importance-weighted estimators.

## Formulation

sum over the sampled set of p(s) * R(S^k, s) * f(s), R = leave-one-out ratio (probability of the other samples under the distribution restricted to D minus {s}, divided by the probability of the full set); a second-order leave-one-out ratio gives a built-in baseline with no extra evaluation; for sequences use stochastic beam search; exact set probability in O(2^k) or by numerical integration (naive O(k!)). - input / output: a discrete policy -> unbiased gradient.

## Training

REINFORCE-type; combined with pathwise terms if f depends on theta.

## Differentiable?

yes in expectation (unbiased score-function gradient), not through the discrete choice.

## Sensors and data

Bernoulli toy, categorical VAE (10^2 and 10^20 latent states), TSP with 20 nodes.

## Reported results

lowest or comparable gradient variance in all three; categorical VAE on par with REINFORCE with replacement; TSP better than REINFORCE with replacement, stochastic sum-and-sample and a greedy rollout baseline, on par with a biased risk estimator.

## Compute

modest overhead for small k.

## Code

not checked

## Fit to our project

reuse_idea - effort M - the only unbiased route found for training a stochastic top-b / top-M selection, at the price of normalised sampling probabilities.

## Needs

normalised probabilities of the candidates; an episode-level reward.

## Conflicts

policy-gradient variance; not needed if local supervision of scores exists.

## Tried in our experiments

none
