# MH42 - Fully differentiable beam search decoder (beam as a computation graph, gold paths added to the normaliser)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4, T.*

## Papers

A Fully Differentiable Beam Search Decoder (Collobert, Hannun, Synnaeve, ICML 2019) https://arxiv.org/abs/1902.06022 (read via ar5iv)

## Idea

Express the decoder (lexicon, transition and word-LM scores, merging of equal hypotheses by logadd, pruning to a beam) as differentiable operations and train the acoustic model through it with a sequence-level criterion; the gold alignments are always included in the normaliser.

## Formulation

per hypothesis (lexicon state, LM state, score); equal states merged with logadd; beam keeps the top scores (discrete; no gradient through the top-K decision, gradient through the scores of the kept hypotheses); loss = -( logadd over gold alignments (Forward) - log( beam sum - beam-gold sum + gold-alignment sum ) ) (Eq. 8); too small a beam diverges; beam 500 mostly (1000, 2000 tested). - input / output: frame token scores -> word sequence.

## Training

beam-level criterion; regime `free_rollout` through the beam output.

## Differentiable?

partial - pruning hard; gradient through kept scores only.

## Sensors and data

Wall Street Journal speech.

## Reported results

WSJ test WER: ASG 10 M model beam 8000 5.6; DBD 10 M beam 500 5.9; DBD 7.5 M beam 500 5.3, beam 1000 5.1; acoustic-only ASG 4-gram 11.3 vs DBD 4-gram 9.9.

## Compute

epoch 35 / 90 / 180 min at beam 500 / 1000 / 2000; CPU-only criterion; custom implementation about 10x faster than autograd.

## Code

not checked

## Fit to our project

reuse_idea - effort L - portable lesson: put the labelled hypothesis into the normaliser so that the gold path never leaves the beam during training (gold-in-beam).

## Needs

additive path scores, logadd merging of equal states.

## Conflicts

continuous positions need a clustering rule to define "equal states" (our DAG conversion).

## Tried in our experiments

none
