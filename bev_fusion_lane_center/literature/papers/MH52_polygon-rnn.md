# MH52 - Polygon-RNN++ (beam of candidate polygons ranked by a separately trained evaluator; RL against exposure bias)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, T.*

## Papers

Efficient Interactive Annotation of Segmentation Datasets with Polygon-RNN++ (Acuna, Ling, Kar, Fidler, CVPR 2018) https://arxiv.org/abs/1803.09693 (read via ar5iv)

## Idea

A ConvLSTM predicts polygon vertices one by one as a classification over a 28 x 28 grid (plus an end token); cross-entropy pre-training, then self-critical RL on IoU so the decoder trains on its own samples; at inference K = 5 first-vertex candidates each run a beam, and an evaluator network (trained with MSE to predict the IoU of a polygon) picks the best.

## Formulation

teacher-forced training with a distance-transform-smoothed target (Manhattan distance truncated at 2); RL: reward = IoU of predicted and GT masks, baseline = reward of the greedy polygon, temperature 0.6; evaluator candidates sampled at temperature 0.3 for diversity, trained separately after RL converged; beam width B not stated for the main results (fast variant B = K = 1, 295 ms per object). - input / output: image crop -> polygon vertices.

## Training

regime `teacher_forcing` then `rl`; selection by a learned verifier.

## Differentiable?

no for the RL path (reward); evaluator is a separate regressor.

## Sensors and data

Cityscapes (GT boxes, automatic mode).

## Reported results

cumulative ablation mean IoU: residual Polygon-RNN 62.16, + attention 65.43, + RL 67.17, + evaluator 70.21, + GGNN 71.38; evaluator with greedy sequences 69.72 vs beam search without evaluator 67.49.

## Compute

295 ms per object for B = K = 1.

## Code

not checked

## Fit to our project

reuse_idea - effort M - a learned verifier (IoU predictor) to choose among beam candidates and RL against exposure bias are both used; RL is outside our "no argmax on the training path" rule.

## Needs

IoU-like reward; evaluator network.

## Conflicts

RL / sampling; two-stage training.

## Tried in our experiments

none
