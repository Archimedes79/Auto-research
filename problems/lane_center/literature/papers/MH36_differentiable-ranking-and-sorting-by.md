# MH36 - Differentiable ranking and sorting by optimal transport (soft rank, soft sort, soft quantile)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4.*

## Papers

Differentiable Ranking and Sorting using Optimal Transport (Cuturi, Teboul, Vert, NeurIPS 2019) https://arxiv.org/abs/1905.11885 (read via ar5iv)

## Idea

Sorting is an optimal assignment between the input values and a target measure on increasing anchor values; entropic regularisation gives a dense differentiable plan, from which soft rank and soft sort operators (and a soft quantile with a 3-point target measure) follow.

## Formulation

Sinkhorn on the n x m cost between inputs and m anchors; soft ranks / sorts from the plan; gradients by autodiff through the Sinkhorn iterations (not implicit); cost scales with iterations x n x m; top-k classification uses a ReLU-type loss on the soft rank, J_k(u) = max(0, u - k + 1), experiments only for k = 1; suggested eps 1e-2 or 1e-3 for inputs in [0, 1]; eps -> 0 recovers hard rank / sort; small eps gives vanishing gradients, large eps collapses outputs to averages. - input / output: vector of values -> soft ranks / sorted values.

## Training

CIFAR-10 / CIFAR-100 soft top-1 loss (CNN, ResNet18), MNIST sorting (on par with NeuralSort), least-quantile regression.

## Differentiable?

yes (bias from eps).

## Sensors and data

CIFAR, MNIST.

## Reported results

no absolute accuracy numbers in the text read (figures only); MNIST sorting precision slightly ahead of NeuralSort (Table 1).

## Compute

O(iterations * n * m).

## Code

not checked

## Fit to our project

baseline - effort S - alternative to MH34 for soft top-M of the candidate scores; same bias trade-off.

## Needs

scalar scores.

## Conflicts

no soft top-k defined by the paper itself (only k = 1 loss).

## Tried in our experiments

none
