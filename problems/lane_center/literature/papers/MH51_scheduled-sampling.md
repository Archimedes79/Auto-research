# MH51 - Scheduled sampling (mix ground-truth and own predictions with a decaying probability)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: T.*

## Papers

Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks (Bengio, Vinyals, Jaitly, Shazeer, NeurIPS 2015) https://arxiv.org/abs/1506.03099 (read via ar5iv)

## Idea

Per token flip a coin: with probability eps_i feed the ground-truth previous token, else the model's own sample or argmax; eps_i decays over training (curriculum from easy to realistic).

## Formulation

schedules linear eps_i = max(eps, k - c i), exponential eps_i = k^i, inverse sigmoid eps_i = k/(k + exp(i/k)); the coin is per token (per sequence performed much worse); gradients are NOT propagated through the sampled token (left to future work); the term "exposure bias" is not used, the paper speaks of errors amplified early in the sequence. - input / output: RNN decoders.

## Training

regime `teacher_forcing` -> `free_rollout` curriculum.

## Differentiable?

partial - no gradient through the sampled token.

## Sensors and data

MSCOCO captioning, WSJ constituency parsing, speech frame prediction.

## Reported results

captioning (BLEU-4 / METEOR / CIDEr): baseline 28.8 / 24.2 / 89.5, scheduled sampling 30.6 / 24.3 / 92.1, always sampling 11.2 / 15.7 / 49.7; parsing F1: baseline 86.54, scheduled sampling 88.08, always sampling invalid trees; speech: always sampling 34.6 / 35.8 frame error rate vs baseline 15.0 (next-step) / 46.0 (decoding).

## Compute

no overhead beyond sampling.

## Code

not checked

## Fit to our project

reuse_idea - effort S - ramp the share of own children (instead of GT-nearest children) fed to the next level from 0 to 1; pure free roll-out from the start is documented as poor in captioning and parsing.

## Needs

a decoder that accepts either GT or own input at each level.

## Conflicts

the sampled input carries no gradient.

## Tried in our experiments

none
