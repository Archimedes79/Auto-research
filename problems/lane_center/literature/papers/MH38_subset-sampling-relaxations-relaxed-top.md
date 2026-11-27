# MH38 - Subset-sampling relaxations: relaxed top-k (Xie-Ermon) and SIMPLE (exact k-subset forward, marginals backward)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4.*

## Papers

Reparameterizable Subset Sampling via Continuous Relaxations (Xie, Ermon, IJCAI 2019) https://arxiv.org/abs/1901.10517; SIMPLE: A Gradient Estimator for k-Subset Sampling (Ahmed, Zeng, Niepert, Van den Broeck, ICLR 2023) https://arxiv.org/abs/2210.01941 (both read via ar5iv)

## Idea

Draw k DISTINCT items with gradient. Xie-Ermon: Gumbel noise plus k successive softmaxes with the chosen item suppressed by log(1 - a). SIMPLE: sample an exact discrete k-subset in the forward pass (dynamic programming for the exactly-k distribution, O(n k)); in the backward pass use the Jacobian of the conditional marginals p(z_i = 1 | sum z = k).

## Formulation

relaxed top-k temperature t > 0; exact samples as t -> 0 (Prop. 1); ordering of weights preserved for t >= 1; complexity O(k m). SIMPLE: marginals are partial derivatives of the log-probability of the k-constraint, so autodiff computes them; for k = 1 it resembles straight-through Gumbel-softmax but differentiates through unperturbed marginals; vectorised O(log k log n) with divide and conquer. - input / output: n logits -> k-hot vector.

## Training

plugged into feature selection and VAEs.

## Differentiable?

partial - forward exact discrete (SIMPLE) / relaxed (Xie-Ermon); backward biased but low variance.

## Sensors and data

IMDB, MNIST / Fashion-MNIST / CIFAR-10 kNN; BeerAdvocate, Taste, discrete VAE MNIST.

## Reported results

Xie-Ermon IMDB-word post-hoc accuracy 90.7 -> 91.7 % (k = 10), kNN MNIST 99.3 % (NeuralSort 99.4), forward pass 0.110 s vs 3.694 s NeuralSort at m = 5000, k = 5. SIMPLE BeerAdvocate Aroma (k = 5) MSE 2.27 vs SoftSub 2.57, I-MLE 2.62, L2X 5.75; Taste (k = 10) 2.11 vs 2.18 / 2.38 / 6.92; lower bias and variance than score function, SoftSub, I-MLE, ST Gumbel-softmax on a 5-of-10 synthetic test.

## Compute

O(n k) for the DP.

## Code

not checked

## Fit to our project

reuse_idea - effort S - picking exactly M (or b) children with an unbiased-in-forward, low-variance gradient; both fit M x b <= 100 candidates.

## Needs

independent logits with an exactly-k constraint.

## Conflicts

the distribution is over subsets, it does not know geometric redundancy between picked nodes.

## Tried in our experiments

none
