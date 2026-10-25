# MH27 - DN-DETR (query denoising: noised GT queries with known assignment)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L, T.*

## Papers

DN-DETR: Accelerate DETR Training by Introducing Query DeNoising (Li et al., CVPR 2022) https://arxiv.org/abs/2203.01305 (read via ar5iv)

## Idea

Hungarian matching is unstable early in training (the same query is matched to different objects across epochs). Add extra decoder queries built from noised GT boxes and labels; each reconstructs its own GT, so no matching is needed; they are removed at inference.

## Formulation

box noise: centre shift |dx| < lambda1 * w/2, |dy| < lambda1 * h/2; scale in [(1 - lambda2) w, (1 + lambda2) w]; label flip ratio gamma; defaults lambda1 = lambda2 = 0.4, gamma = 0.2, 5 denoising groups; attention mask: matching queries cannot see denoising queries and denoising groups cannot see each other (without it performance degrades); instability index = number of query-to-target assignment changes between consecutive epochs. - input / output: DAB-DETR / Deformable DETR decoders.

## Training

50 epochs; regime `denoising` + `hungarian_set`.

## Differentiable?

yes.

## Sensors and data

COCO.

## Reported results

DAB-DETR R50 50 ep 42.2 -> 44.1 (groups 0 / 1 / 5: 42.2 / 43.4 / 44.1); DC5 12 ep 38.0 -> 41.7; DN-Deformable 48.6 vs 46.9; DINO +1.4 at 12 ep.

## Compute

GFLOPs 94.4 -> 94.5 - 94.6; epoch time 51.1 -> 57.7 min.

## Code

not checked

## Fit to our project

reuse_idea - effort S - "noised GT as hypotheses with a known target" is a matching-free training signal for the refine / rescore stages and for a recursion that starts from the noised previous set (cf. stream query denoising in the earlier review).

## Needs

a mask separating denoising queries from hypothesis queries.

## Conflicts

trains refinement and scoring, not generation of new hypotheses.

## Tried in our experiments

none
