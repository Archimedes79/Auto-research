# MH05 - Evolving WTA (EWTA: k winners, k shrinking from M to 1) and sampling + fitting of mixtures

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L, T.*

## Papers

Overcoming Limitations of Mixture Density Networks: A Sampling and Fitting Framework for Multimodal Future Prediction (Makansi, Ilg, Cicek, Brox, CVPR 2019; authors from memory) https://arxiv.org/abs/1906.03631 (read via ar5iv)

## Idea

Name why mixture density networks fail (mode collapse, numerical instability in higher dimensions) and split the problem: first generate hypotheses with a WTA-style loss, then fit a mixture on top. To stop WTA hypotheses getting stuck between samples (and relaxed WTA leaving many at the centre) the k nearest hypotheses are updated per target, with k = M at the start, decreased to 1.

## Formulation

meta-loss sum_k w_k * l(h_k, y), w_k = 1 for the k closest hypotheses, 0 otherwise (selection by Euclidean distance); k decreases from M to 1 (explicit step schedule not stated). Fitting stage: each hypothesis gets an M_c-dim softmax assignment gamma, weights = mean assignment, means = assignment-weighted means, variances by the law of total variance; two FC layers + dropout 0.5; stages trained one after the other, then EWTA removed and everything fine-tuned with NLL. K = 40 hypotheses, 4 mixture components (SDD table: 20 hypotheses). - input / output: FlowNetS encoder + conv layers -> K position hypotheses; mixture of 4 Gaussians (Laplace on SDD).

## Training

EWTA stage 250 k iterations, fitting added at 350 k, joint fine-tuning after 550 k (MDN baseline 150 k + 200 k); regime `wta`.

## Differentiable?

partial - top-k selection constant; fitting stage fully differentiable.

## Sensors and data

synthetic CPI dataset (acronym expansion not on the page read), Stanford Drone Dataset.

## Reported results

CPI Table 1 (oracle error / EMD): WTA 6.96 / 3.94, relaxed WTA 7.94 / 2.82, EWTA 9.84 / 1.89 (lower EMD = hypotheses match the true distribution better although the oracle error is higher). Table 2 NLL / EMD: MDN 9.20 / 1.83, EWTAD-MDF 8.33 / 1.57. SDD NLL: MDN 9.71, EWTAD-MDF 9.33.

## Compute

single forward pass at inference; hardware not reported.

## Code

not checked

## Fit to our project

reuse_idea - effort S - for b = 2 the schedule k = 2 -> 1 is a two-stage form of annealing.

## Needs

ranking of the b hypotheses by distance.

## Conflicts

k winners per target put k children onto ONE target early; with b = 2 that is exactly the "both children on the same lane" state, wanted only early.

## Tried in our experiments

none
