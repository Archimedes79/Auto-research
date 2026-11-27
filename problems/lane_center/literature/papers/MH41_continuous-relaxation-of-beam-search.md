# MH41 - Continuous relaxation of beam search (soft top-k beam trained end to end)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, T.*

## Papers

A Continuous Relaxation of Beam Search for End-to-end Training of Neural Sequence Models (Goyal, Dyer, Berg-Kirkpatrick, AAAI 2018) https://arxiv.org/abs/1708.00111 (read via ar5iv)

## Idea

Replace the hard top-k of beam search by a soft top-k built from k peaked softmaxes; the beam state is a convex combination of previous states and expected input embeddings, so a task loss can be applied to the beam output. The temperature alpha is annealed up.

## Formulation

for rank i = 1..k: p_i = softmax over candidates of -alpha * (s - max_i)^2 with max_i the i-th largest score (iterated max operations); soft backpointer = row sums of p_i; new hidden state = backpointer-weighted average of previous hidden states; next input = expected embedding a_i^T E; cumulative score s_{t+1,i} = sum(s~_t * p_i). Loss: per-step Hamming loss accumulated along the soft beam, final loss = peaked softmax over final beam scores applied to the accumulated loss (soft max-margin variant worse). alpha annealed from non-peaked to peaked (constant 1.0 worse, large constant unstable in pilots); beam size 3, 50 epochs, warm start with cross-entropy. - input / output: sequence tagger (BiLSTM encoder, LSTM decoder, hidden 64 NER / 512 CCG); k label hypotheses.

## Training

regime `free_rollout` through the soft beam.

## Differentiable?

yes at finite alpha (all candidates get gradient); hard beam recovered for alpha -> inf if no ties.

## Sensors and data

German NER (CoNLL 2003), CCG supertagging (CCGbank); no NMT experiment.

## Reported results

CCG tag accuracy: cross-entropy greedy 80.35, cross-entropy hard beam 82.42, soft-beam training 85.78 (soft) / 85.82 (hard beam at test); NER macro F1: 54.92 greedy cross-entropy vs 56.38 / 56.15.

## Compute

about k times cross-entropy training; flat objective with huge gradients near change points at high alpha.

## Code

not checked

## Fit to our project

reuse_idea - effort M - template for "prune M nodes per level with gradient"; state mixing is only defined for recurrent states, a convex combination of two positions is not a lane point.

## Needs

scalar score per candidate; averageable states.

## Conflicts

averaging node positions or tokens blends different lanes; cost grows with the number of soft selections.

## Tried in our experiments

none
