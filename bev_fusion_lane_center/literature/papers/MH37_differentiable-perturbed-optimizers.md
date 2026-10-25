# MH37 - Differentiable perturbed optimizers (noise-smoothed argmax / top-k with Fenchel-Young loss)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, L.*

## Papers

Learning with Differentiable Perturbed Optimizers (Berthet et al., NeurIPS 2020) https://arxiv.org/abs/2002.08676 (read via ar5iv)

## Idea

Replace a non-differentiable solver (argmax, top-k, shortest path) by its expectation under random perturbation of the input; the Jacobian is obtained by integration by parts and estimated with Monte Carlo; a Fenchel-Young loss needs only the perturbed maximiser.

## Formulation

y_eps(theta) = E[y*(theta + eps Z)], Z density proportional to exp(-nu(z)); Jacobian = E[y*(theta + eps Z) grad nu(Z)^T] / eps; Monte Carlo over M samples; L_eps(theta; y) = F_eps(theta) + eps Omega(y) - <theta, y>, gradient y_eps(theta) - y. Smooth, strictly convex F_eps, y_eps interior, gradient never zero; error bounds proportional to eps; M = 1, eps = 1 for shortest paths, M in {1, 1000} on CIFAR-10. Top-k appears only as an example polytope (Appendix B); no top-k experiments. - input / output: score vector -> smoothed indicator / path.

## Training

CIFAR-10 (600 epochs, batch 32), label ranking, Warcraft shortest paths (ResNet18, 50 epochs).

## Differentiable?

yes (smoothing bias eps).

## Sensors and data

CIFAR-10, Warcraft terrain, 21 label-ranking sets.

## Reported results

perturbed FY loss competitive with cross-entropy on CIFAR-10; shortest-path numbers not extracted.

## Compute

M solver calls per gradient (parallelisable).

## Code

not checked

## Fit to our project

reuse_idea - effort S - noise-based top-M with a task-agnostic gradient, no Sinkhorn; M samples of a size-30 top-k are cheap.

## Needs

a solver that can be called on perturbed scores.

## Conflicts

variance with small M; no published top-k experiment.

## Tried in our experiments

none
