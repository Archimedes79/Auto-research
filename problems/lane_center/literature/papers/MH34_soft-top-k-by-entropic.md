# MH34 - SOFT top-k by entropic optimal transport (differentiable top-k, soft beam search)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4.*

## Papers

Differentiable Top-k with Optimal Transport (Xie et al., NeurIPS 2020) https://arxiv.org/abs/2002.06504 (read via ar5iv; last 22 k characters not read)

## Idea

Top-k is cast as transporting n scores to two anchors {0, 1} with masses k/n and (n - k)/n; the entropic plan gives a smooth indicator of the top-k set whose gradient is computed implicitly.

## Formulation

mu = 1/n over the scores; nu = (k/n, (n - k)/n) over {0, 1}; cost C_i1 = x_i^2, C_i2 = (x_i - 1)^2; Sinkhorn: G = exp(-C/eps), iterate p = mu / (G q), q = nu / (G^T p), plan = diag(p) G diag(q); smoothed indicator A = n * plan * [0, 1]^T. Backward: implicit differentiation of the KKT conditions (closed form in the appendix), no unrolling. Error bound ~ eps (ln n + ln 2) / (n * gap between the k-th and (k+1)-th score). eps used 1e-3 (MNIST kNN), 1e-5 (CIFAR-10 kNN), 1e-1 (beam search); iteration count not extracted. Cost O(n) per iteration (O(k n) for sorted top-k). - input / output: vector of n scores -> soft indicator (or soft sorted top-k).

## Training

inside kNN classifiers and inside beam search of machine translation (soft top-k replaces hard top-k in training, mixed with teacher forcing, rho = 0.8, beam 5).

## Differentiable?

yes (smooth, Jacobian has a nonzero entry for any input); bias from eps.

## Sensors and data

MNIST, CIFAR-10, WMT14 En-Fr, WMT16 En-De.

## Reported results

kNN CIFAR-10 92.6 % (k-times softmax 92.2, NeuralSort 90.7); MNIST 99.4 (NeuralSort 99.5); WMT14 En-Fr beam search 36.27 BLEU vs 35.38 reproduced baseline; WMT16 En-De top-k attention 37.30 vs 36.54 soft attention.

## Compute

O(n) per Sinkhorn iteration; implicit backward avoids storing iterations.

## Code

not checked

## Fit to our project

reimplement - effort S - ten lines on the M x b scores; gradient reaches all candidates including dropped ones; the soft indicator can weight node contributions in the next level.

## Needs

a scalar score per candidate; a loss that uses the soft weights (otherwise nothing is learned from them).

## Conflicts

dropped nodes stay partly in the graph during training, inference needs a hard top-M (train / test mismatch for eps > 0); eps trades bias against gradient smoothness.

## Tried in our experiments

none
