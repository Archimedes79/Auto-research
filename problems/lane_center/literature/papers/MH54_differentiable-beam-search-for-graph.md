# MH54 - Differentiable beam search for graph generation (decoding-aware training for information extraction)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **A** (A abstract, M method, C code page). Verified: partly. Pipeline stage: T.*

## Papers

Information Extraction with Differentiable Beam Search on Graph RNNs (El Khbir, Tomeh, Charnois, LREC-COLING 2024) https://aclanthology.org/2024.lrec-main.796/ (abstract only)

## Idea

Graph generation cast as autoregressive sequence labelling; training is made decoding-aware with a differentiable version of beam search to reduce exposure bias; a new way to quantify exposure bias is proposed.

## Formulation

not read (the page does not say which relaxation or loss). - input / output: not read.

## Training

decoding-aware; regime `free_rollout`.

## Differentiable?

yes (abstract).

## Sensors and data

ACE05, CoNLL04.

## Reported results

gains over a version without decoding-aware training (abstract); numbers not seen.

## Compute

not stated.

## Code

none found

## Fit to our project

reuse_idea - effort M - recent evidence that differentiable-beam training is used for graph-structured outputs, but only in text.

## Needs

not read.

## Conflicts

not read.

## Tried in our experiments

none
