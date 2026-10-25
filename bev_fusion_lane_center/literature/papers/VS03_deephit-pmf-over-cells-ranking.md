# VS03 - DeepHit (single risk): PMF over (time) cells + ranking loss

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **A (paper abstract) + C (authors' `class_DeepHit.py` and pycox `loss.py` read)** (A abstract, M method, C code page). Verified: partly (paper not opened). Pipeline stage: S3, L.*

## Papers

DeepHit (Lee, Zame, Yoon, van der Schaar, AAAI 2018) https://ojs.aaai.org/index.php/AAAI/article/view/11842 ; authors' code https://github.com/chl8856/DeepHit ; re-implementation https://github.com/havakv/pycox

## Idea

Instead of a hazard per interval the network outputs ONE softmax over all (event, time) cells = the probability mass of the event time; a pairwise ranking loss orders individuals by risk.

## Formulation

(authors' code) shared sub-network with residual input concatenation, cause-specific sub-networks, dropout, linear layer to num_Event x num_Category logits, one softmax over all cells. LOSS_1 log-likelihood: uncensored = log probability of the observed cell; censored = log of the summed mass after the censoring time (= survival past it). LOSS_2 ranking: for subject pairs with t_i < t_j and event for i, penalty exp(-(F_i(t_i) - F_j(t_i)) / sigma1), sigma1 = 0.1 fixed in the file. LOSS_3 calibration (squared error of batch-summed predicted probability against the event indicator) exists in the code only; a comment says it is not in the paper version. Total = alpha LOSS_1 + beta LOSS_2 + gamma LOSS_3 + L2 (1e-4, hidden) + L1 (1e-4, output); alpha, beta, gamma are passed at training time, no defaults in the file. pycox single-risk version: nll_pmf + rank loss, alpha-weighted.

## Input

features. output: scores (PMF over cells + tail).

## Training

likelihood + ranking; matching_free.

## Differentiable?

yes.

## Sensors and data

medical tabular (paper not read).

## Reported results

none seen for the original paper; Kvamme-Borgan (VS01) report PMF slightly worse than logistic hazard (concordance and IBS above).

## Compute

not stated.

## Code

checked (two code pages above).

## Fit to our project

reuse_idea - S - a direct softmax over {endpoint at sample 0..K} is the "PMF" alternative to the hazard parametrisation: P(k) without a product, but no per-sample interpretation and, per Kvamme-Borgan, less stable across grids.

## Needs

none beyond VS01.

## Conflicts

the ranking loss needs comparable pairs (pairs of rays); not needed here.

## Tried in our experiments

none.
