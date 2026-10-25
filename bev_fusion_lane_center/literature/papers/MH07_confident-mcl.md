# MH07 - Confident MCL (non-winners pushed to uniform)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L.*

## Papers

Confident Multiple Choice Learning (Lee, Hwang, Park, Shin, ICML 2017) https://arxiv.org/abs/1706.03475 (read via ar5iv); code https://github.com/chhwang/cmcl (named in a search result; Python 2.7 / TensorFlow per the summary)

## Idea

MCL members stay confidently wrong on inputs they do not specialise in, so averaging them fails. The winner keeps the task loss; every other member gets a KL divergence to the uniform distribution (weight beta), which makes losers low-confidence.

## Formulation

confident oracle loss: for the member that best fits the example the usual task loss; for the others beta * KL(U || p_m); stochastic labeling replaces the exact KL gradient by a cross-entropy against sampled uniform labels; feature sharing: each member sees a random masked subset (probability lambda = 0.7) of the others' lower-layer features (before the first pooling); beta in {0.5 ... 1.5}, overlap K in {2, 3, 4}. - input / output: classifier ensemble (or FCN for segmentation); class posteriors.

## Training

regime `wta`; classification labels.

## Differentiable?

partial - winner choice constant; loser term differentiable.

## Sensors and data

CIFAR-10, SVHN, iCoseg.

## Reported results

CIFAR-10, 5 ResNet-20: MCL ensemble-average top-1 error 60.40 %, independent ensemble 15.34 %, CMCL 8.75 %; SVHN 3.68 %.

## Compute

not stated.

## Code

https://github.com/chhwang/cmcl

## Fit to our project

reuse_idea - effort S - the loser-side rule for scores: a child that is not responsible for a GT must be low-confident only when it is really off-lane; for on-lane losers the uniform-KL rule would be wrong (they are valid).

## Needs

per-member probabilities.

## Conflicts

designed for class posteriors; no analogue for geometry.

## Tried in our experiments

none
