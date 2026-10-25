# MH15 - DenseTNT (dense goal candidates, heat map, learned goal-set selection instead of NMS)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M (DenseTNT), A (TNT)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, L.*

## Papers

DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets (Gu, Sun, Zhao, ICCV 2021) https://arxiv.org/abs/2108.09640 (read via ar5iv; venue from memory); TNT: Target-driveN Trajectory Prediction (Zhao et al.) https://arxiv.org/abs/2008.08294 (A)

## Idea

Score many lane-aligned goal candidates with a softmax heat map (CE against the candidate nearest the GT endpoint), then replace NMS by a learned goal-set predictor that outputs K goals; because only one future is observed, set labels come from an offline optimiser that selects the K goals with minimal expected error under the heat map.

## Formulation

candidates within 3 m of centrelines within 50 m of the target, 1 m spacing; goal probability = softmax over candidates, BCE against a one-hot target at the candidate nearest the GT endpoint (Eqs. 3 - 4). Goal-set predictor: 12 heads, each with a one-layer self-attention encoder + max pool and a 2-layer MLP decoder, each outputting K goals + a confidence; losses: BCE on head confidences (best head = 1) + L1 to the pseudo-label. Pseudo-labels: offline hill-climbing minimising E[d(y_hat, Y)] under the heat map (perturbations L = 100 per step, accepts worse moves with probability 0.01). NMS fails because the threshold depends on the heat map and it is greedy. K not specified in the text read. - input / output: vector scene tokens -> dense goal probabilities -> K goals -> trajectories.

## Training

batch 64, hidden 128, 16 epochs stage 1 + 6 epochs stage 2, lr 1e-3, no augmentation; regime `wta`-like (nearest candidate) + pseudo-label regression.

## Differentiable?

partial - the pseudo-label optimiser is offline; the learned predictor is differentiable.

## Sensors and data

Argoverse (2 s history, 3 s future), Waymo 2021 challenge.

## Reported results

Argoverse validation: offline optimiser minADE 0.80, minFDE 1.27, MR 7.0 %; online goal-set predictor 0.82 / 1.37 / 7.0 %; Waymo 2021 challenge 1st place mAP 0.3281. Ablation (Table 2): dense heat maps beat sparse; variety loss gives MR 13.3 %. Offline optimisation about 100 ms per sample on 8 CPUs.

## Compute

online latency not reported.

## Code

not checked

## Fit to our project

reuse_idea - effort M - "dense candidate heat map + learned selection of a small set trained on pseudo-labels from an offline optimiser" is a learned replacement for our NMS with a matching-free label; our ray-ring samples are the dense candidates.

## Needs

dense candidates with a softmax score; an offline set optimiser for labels.

## Conflicts

the pseudo-label step is not end to end; 2-stage training.

## Tried in our experiments

none
