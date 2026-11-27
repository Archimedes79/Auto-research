# MH03 - DiverseNet: WTA with several ground truths per input + "catch-up" term

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L.*

## Papers

DiverseNet: When One Right Answer is not Enough (Firman, Campbell, Agapito, Brostow; CVPR 2018, arXiv upload 2020) https://arxiv.org/abs/2008.10634 (read via ar5iv)

## Idea

Several valid labels per input are matched to hypotheses independently (each label to its best hypothesis, not one-to-one), and a second term always updates the hypothesis that is farthest from every label, so no hypothesis is starved ("one or more members fail to receive any training signal" is the named failure of stochastic MCL).

## Formulation

L_div = sum over (x, Y) sum_{y in Y} min_c l(f(c, x), y); L_catchup = (1/|C|) sum over data of max_c min_{y in Y} l(f(c, x), y); L = L_div + beta * L_catchup, beta = 1 everywhere. With one label per input only one term of L_div remains. Hypotheses come from one network with a control input c (one-hot or sampled vector) concatenated into the activations, not from separate heads; |C| = 8 (MNIST-OCC), 6 (ShapeNet). - input / output: image or volume -> |C| alternative outputs.

## Training

regime `wta` + catch-up; labels = the set of valid answers.

## Differentiable?

partial - argmin / argmax over hypotheses constant; catch-up gives gradient to the worst hypothesis only.

## Sensors and data

MNIST occlusion completion, ShapeNet volumes, traffic-flow prediction.

## Reported results

traffic flow, best endpoint error for k = 1..4 outputs: Lee et al. (sMCL) 0.313 / 0.215 / 0.172 / 0.158; L2 + dropout 0.209 / 0.194 / 0.187 / 0.184; DiverseNet 0.203 / 0.174 / 0.158 / 0.146. ShapeNet: about 12.6 M parameters versus 66.8 M for the tree-net baseline. Some sMCL members "failed to learn, giving null predictions" on MNIST-OCC (no count given).

## Compute

single network; not stated beyond parameters.

## Code

not checked

## Fit to our project

reimplement - effort S - this is the cleanest published form of "b hypotheses, 1 or 2 valid answers, no matching"; the catch-up term addresses the idle second child directly.

## Needs

distance between hypothesis and each valid label; beta weight.

## Conflicts

the catch-up term drags the worst child to the nearest label, so at normal tips both children land on the same lane point (redundant children; needs a separate redundancy step such as sibling attention, MH31); with a point-to-polyline loss the pull is to "any point of the lane".

## Tried in our experiments

none
