# MH39 - Stochastic Beams and the Gumbel-top-k trick (sampling without replacement through a tree)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4.*

## Papers

Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement (Kool, van Hoof, Welling, ICML 2019) https://arxiv.org/abs/1903.06059 (read via ar5iv)

## Idea

Gumbel-perturbed log-probabilities, top-k = k samples without replacement. For tree models the same result is obtained inside beam search by keeping the k best perturbed partial scores per level; a child's perturbed score is sampled conditionally on its parent so the tree stays consistent.

## Formulation

G_i = phi_i + Gumbel(0); node G_S ~ Gumbel(phi_S); child score G~ = -log( exp(-G_parent) - exp(-Z) + exp(-G_child) ), Z = max over children; expand the k nodes with highest perturbed score per level (contains the ancestors of the top-k leaves). Estimators: weights p(y)/q(y) with inclusion probability q = 1 - exp(-exp(phi - kappa)), kappa = (k+1)-th largest perturbed score; self-normalised version biased but consistent and lower variance. - input / output: autoregressive model with normalised conditional probabilities; k sampled sequences.

## Training

none (inference sampler / estimator); no gradient estimator in this paper.

## Differentiable?

no (top-k of perturbed scores).

## Sensors and data

WMT14 en-fr newstest2014.

## Reported results

k = 5, 10, 20 and T 0.1 - 0.8: better mean and maximum BLEU at equal diversity than diverse beam search; estimation variance below Monte Carlo for T < 0.5.

## Compute

linear in k and sequence length.

## Code

not checked

## Fit to our project

reuse_idea - effort M - stochastic exploration of the beam in training (different nodes survive in different epochs).

## Needs

normalised conditional probabilities child | parent.

## Conflicts

our validity scores are independent survival products, not normalised conditionals.

## Tried in our experiments

none
