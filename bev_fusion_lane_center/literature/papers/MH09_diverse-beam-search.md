# MH09 - Diverse Beam Search (group-wise beam with a dissimilarity penalty)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S4.*

## Papers

Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models (Vijayakumar et al., 2016) https://arxiv.org/abs/1610.02424 (read via ar5iv)

## Idea

Split the B beams into G groups; each group runs ordinary beam search, but its scores get a penalty for candidates already chosen by earlier groups. Inference-time only; no retraining.

## Formulation

score = log-prob(candidate) - lambda * Delta(candidate, tokens chosen by previous groups at this step); Hamming diversity is the best simple form; G = B gives the most exploration; lambda in 0.2 - 0.8 worked; G = 1 is standard beam search. About T + G steps if groups are staggered in parallel. - input / output: any sequence model; B diverse sequences.

## Training

none.

## Differentiable?

no (selection top-k; inference-time).

## Sensors and data

COCO / PASCAL-50S captioning, VQG, WMT en-fr (B = 20).

## Reported results

COCO captioning oracle SPICE@20 28.09 vs 27.34 (standard beam); distinct 4-grams 5.67 -> 13.44; MT oracle BLEU-4@20 18.77 vs 18.44.

## Compute

little over standard beam search.

## Code

not checked

## Fit to our project

reuse_idea - effort S - a cheap rule to avoid M beam nodes collapsing onto one lane: penalise a candidate by its closeness to already selected nodes (a hand-made NMS inside the beam).

## Needs

a dissimilarity function between candidates (distance in metres).

## Conflicts

hand rule, no gradient; competes with learned sibling attention (MH31).

## Tried in our experiments

none
