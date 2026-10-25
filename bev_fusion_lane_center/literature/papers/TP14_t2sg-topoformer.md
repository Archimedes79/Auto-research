# TP14 - T2SG / TopoFormer (distance bias in self-attention; counterfactual layer)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: checked (digest: OLS equation as printed does not reproduce reported values; thresholds and weights not stated). Pipeline stage: S6.*

## Papers

T2SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving (Lv et al., CVPR 2025) https://arxiv.org/abs/2411.18894

## Idea

Lane features are mixed by self-attention whose logits get an additive spatial prior from the distance end_i -> start_j; a "counterfactual intervention" replaces the learned attention by zeros and trains on the difference of factual and counterfactual edge predictions.

## Formulation

A_SPM[i,j] = Norm(1 / (d(e_i, s_j) + eps)), d = L1, Norm divides by the mean inverse distance (asymmetric); A^l = softmax(X W_Q (X W_K)^T / sqrt(d) + A_SPM); GSA(X) = A^l X W_V; residual + FFN. CIL: counterfactual attention CSA = softmax(A_bar + A_SPM) X W_V with A_bar = zeros; E_TIE = E[E_A - E_A_bar]; edge loss = focal on E_TIE; inference uses the factual E_A. Edge head: two 3-layer MLPs for start / end embeddings, MLP_edge on the concatenation, sigmoid. Lane-traffic link: fixed rule (lane class equals traffic class).

## Input

BEVFormer-style BEV 200 x 100, 200 lane queries, 11 points per lane, R50, 1550 x 2048.

## Output

lane polylines and class (10 classes), adjacency.

## Training

two stages (detector, then T2SG), AdamW 2e-4 cosine, 16 V100, batch 16; epochs and loss weights not given; no matching description for topology.

## Differentiable?

yes for the bias; partial overall (DETR assignment).

## Sensors and data

camera; OpenLane-V2 A and B.

## Reported results

ablation subset A (DET_l / DET_t / TOP_ll / TOP_lt / OLS): no A_SPM 34.1 / 47.3 / 21.6 / 29.1 / 45.4; multiplicative 34.7 / 46.3 / 22.9 / 29.0 / 45.7; additive (final) 34.7 / 48.2 / 24.1 / 29.5 / 46.3. Counterfactual layer: none 32.2 / 47.0 / 22.2 / 28.6 / 44.9; mean / random / zero attention OLS 45.2 / 45.6 / 46.3. Main: A 46.3 OLS (TopoLogic 44.1), B 47.5 (42.3).

## Compute

no runtime reported.

## Code

https://github.com/MICLAB-BUPT/T2SG

## Fit to our project

reuse_idea - effort S - an inverse-distance bias on attention logits between vertex tokens (fork / merge candidates) is a one-line addition; the counterfactual layer shows only +0.1 .. 1.1 OLS over simple variants.

## Needs

start / end coordinates per token.

## Conflicts

none beyond the query-set host.

## Tried in our experiments

E45 (distance-based scoring) no gain over rules.
