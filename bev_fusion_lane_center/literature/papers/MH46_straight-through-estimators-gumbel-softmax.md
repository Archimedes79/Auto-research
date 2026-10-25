# MH46 - Straight-through estimators, Gumbel-softmax, ReinMax and the bias of the surrogate gradient

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M (three), A (Bengio)** (A abstract, M method, C code page). Verified: checked (Bengio 2013: from memory). Pipeline stage: S4, T.*

## Papers

Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation (Bengio, Leonard, Courville, 2013) https://arxiv.org/abs/1308.3432 (not opened); Categorical Reparameterization with Gumbel-Softmax (Jang, Gu, Poole, ICLR 2017) https://arxiv.org/abs/1611.01144; Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets (Yin et al., ICLR 2019) https://arxiv.org/abs/1903.05662; Bridging Discrete and Backpropagation: Straight-Through and Beyond (Liu et al., NeurIPS 2023, ReinMax) https://arxiv.org/abs/2304.08612 (last three read via ar5iv)

## Idea

A hard decision in the forward pass with a surrogate derivative in the backward pass. Gumbel-softmax: y_i = exp((log pi_i + g_i)/tau) / sum_j exp((log pi_j + g_j)/tau); the ST variant uses the one-hot sample forward and the soft sample backward. Yin et al.: coarse gradients with a vanilla or clipped ReLU surrogate are positively correlated with the true population gradient (descent direction), the identity surrogate is not. ReinMax: ST is a first-order (forward Euler) approximation of the gradient w.r.t. the logits; the Heun-type correction is second-order.

## Formulation

ST gradient = (df/dD) * d softmax/d theta. ReinMax: grad = 2 * grad^((pi+D)/2) - 0.5 * grad^ST, via pi_0 = softmax(theta), D sample, pi_1 = (D + softmax_tau(theta))/2 with stop-gradient, pi_2 = 2 pi_1 - 0.5 pi_0 then the ST trick. Gumbel-softmax schedule tau = max(0.5, exp(-r t)) every N in {500, 1000} steps, r in {1e-5, 1e-4}; small tau gives near one-hot samples with high gradient variance, large tau smooth but biased. ST and ReinMax work better with tau >= 1; ST Gumbel-softmax with tau <= 1. - input / output: logits -> hard one-hot / k-hot.

## Training

end to end; ReinMax costs about the same time and memory as ST (5.2 s per epoch on MNIST-VAE).

## Differentiable?

partial - the discrete choice has no gradient; the surrogate is biased (ST first-order, ST Gumbel-softmax biased).

## Sensors and data

SBN / VAE on MNIST, ListOps, polynomial programming, CIFAR ResNet-20 quantisation.

## Reported results

test NLL SBN (Cat.): ST 61.8, Gumbel-softmax 59.0, ST Gumbel-softmax 59.7; VAE (Cat.): ST 110.9, Gumbel-softmax 101.5, ST Gumbel-softmax 107.8. ResNet-20 4-bit activations: identity surrogate 54.16 %, vanilla ReLU 86.59 %, clipped ReLU 91.24 % (2-bit: 46.52 / 48.05 / 88.39). ListOps accuracy ReinMax 68.07 +- 1.18, STGS 67.30 +- 2.50, ST 66.26 +- 0.48; MNIST-VAE -ELBO ReinMax 103.21, ST 116.72. Yin et al.: a poor surrogate "generates coarse gradients incompatible with the energy landscape" (instability); proofs hold for the population loss (infinite samples) only.

## Compute

Gumbel-softmax / ST / ReinMax about equal cost.

## Code

not checked

## Fit to our project

reuse_idea - effort S - the generic fallback for any hard keep / pick decision in the tree; the choice of surrogate matters (identity can fail).

## Needs

a differentiable surrogate of the decision.

## Conflicts

bias; sensitivity to tau and surrogate shape.

## Tried in our experiments

none
