# VS04 - PonderNet (halting hazard, expected loss under the halting law) and ACT

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M (PonderNet), A (ACT)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S3, L.*

## Papers

PonderNet: Learning to Ponder (Banino, Balaguer, Blundell, arXiv 2021) https://arxiv.org/abs/2107.05407 (HTML v2 read) ; Adaptive Computation Time (Graves, 2016, from memory) https://arxiv.org/abs/1603.08983 (not opened)

## Idea

A network run for a variable number of steps predicts at each step a halting probability lambda_n. The probability to stop at step n is p_n = lambda_n prod_{j<n}(1 - lambda_j) = the same hazard / survival algebra as our ray. The loss is the EXPECTED task loss under p_n plus a KL term to a geometric prior, so every candidate stop step receives a gradient weighted by its probability and no argmax is needed.

## Formulation

L = sum_n p_n * Loss(y, y_n) + beta * KL(p || geometric(lambda_p)); beta = 0.01 for parity, bAbI and PAI; lambda_p searched in (0,1] (parity, bAbI) and (0, 0.5] (PAI); lambda_p = 0.9 was the only parity setting where the method failed; N = 20 (parity), 10 (bAbI), search 7 - 10 (PAI), or N = smallest n with sum_{j<=n} p_j > 1 - eps, eps = 0.05. Mass left after N: renormalise p_n or give the remainder to the last step (p_N = 1 - sum_{n<N} p_n) = right censoring at N. Inference: sample Bernoulli(lambda_n) per step (or use the expectation); expected steps = sum n p_n. Versus ACT (authors' statement): ACT stops when the cumulative halting sum reaches 1 - eps, outputs a weighted average, has a biased cost gradient and a hard-to-set tau; PonderNet has unbiased low-variance gradients. How lambda_n is produced (sigmoid) and calibration of halting: not stated on the page.

## Input

state per step. output: scores (halting probability per step) + prediction per step.

## Training

expected loss + KL; matching_free (supervised).

## Differentiable?

yes in training (sum over all steps); inference by sampling is not.

## Sensors and data

parity, bAbI, Paired Associative Inference; nothing spatial.

## Reported results

task-specific, not copied.

## Compute

all N steps are evaluated in training.

## Code

none found officially (not checked).

## Fit to our project

reuse_idea - S - (a) endpoint loss as an expectation over P(k) of a per-endpoint regression loss; (b) a geometric-prior KL on P(k) as length prior against early stopping; (c) the leftover mass at the last ring = censoring at the range limit.

## Needs

a per-candidate loss that can be evaluated for every k (we can: distance of sample k to the GT lane).

## Conflicts

expected-loss training spreads mass over candidates; a sharp argmax endpoint at inference then needs calibration.

## Tried in our experiments

none.
