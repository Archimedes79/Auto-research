# MH44 - Differentiable particle filter via entropic optimal transport (resampling with gradient)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, S8.*

## Papers

Differentiable Particle Filtering via Entropy-Regularized Optimal Transport (Corenflos, Thornton, Deligiannidis, Doucet, ICML 2021) https://arxiv.org/abs/2102.07850 (read via ar5iv; last 45 k characters not read)

## Idea

Replace the ancestor draw by the entropic optimal-transport map from the weighted particle set to the uniform one: each new particle is a weighted average of the old ones, so gradients reach all particles.

## Formulation

X~_i = N * sum_k P^OT_{eps,i,k} X_k (barycentric projection, Eq. 13); stabilised Sinkhorn (Feydy et al.), cost rescaled so eps is scale-free; eps = 0.5 default; O(N^2) per iteration; gradients by gradient stitching on the converged potentials or autodiff through the loop. Ignoring the resampling gradient gives a biased ELBO gradient (Prop. 4.1); the OT filter is consistent for eps_N = o(1/log N) under compactness assumptions (Prop. 4.3); the likelihood estimate is biased (only affine functions preserved). Soft resampling (alpha = 0.1) performed worse than plain PF and OT filter. - input / output: weighted particle set -> equally weighted set.

## Training

end to end (ELBO, proposal learning, VRNN, localisation).

## Differentiable?

yes (bias controlled by eps).

## Sensors and data

linear Gaussian SSM, polyphonic music, maze localisation.

## Reported results

proposal learning (d = 25): parameter RMSE 0.11 for DPF (N = 25) vs 0.22 for PF (N = 500), ESS about 60 % vs 25 %; JSB test ELBO DPF -7.67, PF -7.92, soft PF -8.17; maze test MSE DPF 3.55 / 4.65 / 4.44 vs PF 10.71 / 11.86 / 12.88 and soft PF 9.14 / 10.12 / 11.42.

## Compute

O(N^2) per Sinkhorn iteration in training; plain O(N) filter afterwards.

## Code

not checked

## Fit to our project

reuse_idea - effort M - the beam cut with gradient to all nodes; barycentric merging averages positions, which blends different lanes unless the cost makes the transport local.

## Needs

weights and a cost in a vector space.

## Conflicts

averaged particles of different lanes are off-lane; bias; O(N^2).

## Tried in our experiments

none
