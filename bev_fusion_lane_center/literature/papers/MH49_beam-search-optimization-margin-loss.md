# MH49 - Beam-Search Optimization (BSO): margin loss on the beam, reset when the gold prefix falls off

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: T, L.*

## Papers

Sequence-to-Sequence Learning as Beam-Search Optimization (Wiseman, Rush, EMNLP 2016) https://arxiv.org/abs/1606.02960 (read via ar5iv)

## Idea

Train a sequence model with the beam search it is decoded with: whenever the gold prefix is not within the beam (margin violation), a margin loss pushes the gold prefix above the K-th beam entry; the beam is then reset to contain the gold prefix (LaSO-style) and search continues.

## Formulation

L(f) = sum_t Delta(y_hat_{1:t}^(K)) * [ 1 - f(y_t, h_{t-1}) + f(y_hat_t^(K), h_hat_{t-1}^(K)) ]; Delta = 0/1 (word ordering, parsing) or 1 - smoothed sentence BLEU (MT); f is an unnormalised score (removes label bias); pre-training with word-level cross-entropy needed (random init failed); beam curriculum from 2, +1 every 2 epochs up to K_tr; constrained beam search (ConBSO) with hard constraints in the successor function; gradients flow through the gold and violating prefixes (not through the whole beam); Adagrad, batch 64. - input / output: sequence model with scores; beam of prefixes.

## Training

regime `free_rollout` with reset (gold-in-beam).

## Differentiable?

partial - top-K selection constant; gradient through the scores of gold and violating prefixes.

## Sensors and data

word ordering, dependency parsing, German-English MT.

## Reported results

word ordering BLEU (K_te = 10): seq2seq 31.0, BSO 34.3, ConBSO 34.5; dependency parsing UAS / LAS: 88.66 / 84.33 -> 91.17 / 87.41 (BSO), 91.57 / 87.26 (ConBSO); MT BLEU (K_te = 5): 24.03 -> 26.36.

## Compute

tokens/s 13,038 (seq2seq) vs 1,985 (K_tr = 2) and 1,458 (K_tr = 6); the stated 3.3x overhead is inconsistent with the K_tr = 2 figure (about 6.6x).

## Code

not checked

## Fit to our project

reuse_idea - effort M - "reset" = insert the GT-nearest node into the beam when it was cut; margin loss needs an unnormalised score (our validity).

## Needs

scores comparable across beam entries; a gold prefix (nearest-node sequence).

## Conflicts

the gold prefix must exist as a node sequence in the tree.

## Tried in our experiments

none
