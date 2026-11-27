# MH35 - sparsemax and alpha-entmax (sparse normalised weights with exact zeros; exact beam search)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, L.*

## Papers

From Softmax to Sparsemax (Martins, Astudillo, ICML 2016) https://arxiv.org/abs/1602.02068; Sparse Sequence-to-Sequence Models (Peters, Niculae, Martins, ACL 2019) https://arxiv.org/abs/1905.05702 (both read via ar5iv)

## Idea

Replace softmax by the Euclidean projection onto the simplex (sparsemax) or its alpha-entmax generalisation: outputs have exact zeros, so a distribution over children has a small support; with a sparse output the set of positive-probability hypotheses can be enumerated and beam search becomes exact when the beam covers the support.

## Formulation

sparsemax p_i = max(0, z_i - tau(z)), tau from a sort (O(K log K)); Jacobian diag(s) - s s^T / |S(z)| (zero rows outside the support); sparsemax loss convex with gradient -q + sparsemax(z), zero once the margin is >= 1. alpha-entmax: p_i = [(alpha - 1) z_i - tau]_+^(1/(alpha - 1)); alpha = 2 is sparsemax, alpha -> 1 softmax; Fenchel-Young loss gradient p - e_y; exact sort algorithm for alpha = 1.5, bisection for general alpha; attention Jacobian diag(s) - s s^T / ||s||_1 with s_i = p_i^(2 - alpha). Output-loss alpha best near 1.33; alpha > 2 did not help. A loss for a distribution target y (two-hot for a fork) is the same Fenchel-Young form with p - y (from memory, not on the pages read). - input / output: score vector -> sparse probability vector.

## Training

replaces softmax cross-entropy; alpha is a hyper-parameter.

## Differentiable?

yes; entries outside the support get zero gradient.

## Sensors and data

SIGMORPHON 2018 inflection, six NMT pairs, SNLI attention, multi-label benchmarks.

## Reported results

beam search exact for 87 % (high-resource) and 79 % (medium) validation sequences in inflection; 1.5-entmax keeps about 16 of 17,993 target types in NMT; BLEU 1.5-entmax vs softmax de->en 26.17 vs 25.70, ro->en 30.15 vs 29.12; sparsemax below softmax on most NMT pairs; SNLI sparse attention 82.20 vs 82.08 soft vs 80.99 none.

## Compute

de->en about 10,500 / 13,000 / 14,500 target words per second (bisection / exact 1.5-entmax / softmax, GTX 1080).

## Code

not checked

## Fit to our project

reuse_idea - effort S - normalised competition over the b children of a tip with exact zeros for children that are not needed (a differentiable soft NMS).

## Needs

scores that compete (sum to 1) within a tip.

## Conflicts

one winner takes the mass whereas a fork has two true children (needs a two-hot target); an item that is wrongly zero gets no gradient and cannot recover; no demonstration of entmax inside a training-time beam.

## Tried in our experiments

none
