# VM25 - AMap (future-prior distillation into a single-frame student)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: T, S8.*

## Papers

AMap: Distilling Future Priors for Ahead-Aware Online HD Map Construction (arXiv 2512.19150; authors not read) https://arxiv.org/abs/2512.19150

## Idea

A MapTracker-based teacher gets FUTURE context (4 cached future BEV features, future-frame queries aligned by an MLP); a student without any temporal path (R18, learnable queries only) is trained to match the teacher's BEV features (masked L2 with a GT-derived spatial mask) and query logits (Hungarian + KL). Targets better prediction of the road ahead.

## Formulation

total = basic (BEV + vector + transformation) + distill (feature + logits), all weights 1.0; teacher R50 frozen, 4 frames sampled from 10; AdamW 5e-4, wd 0.01, cosine to 1.5e-6, batch 16, 8 H20. BEV range / epochs not stated.

## Input

teacher: past + future frames; student: current frame.

## Output

polylines.

## Training

teacher-student; Hungarian for logits.

## Differentiable?

partial.

## Sensors and data

camera; nuScenes, AV2 (MapTracker partitioning).

## Reported results

nuScenes R18 student mAP 64.49 (A-mAP 66.28, R-mAP 65.11, 31.5 FPS) vs non-temporal baseline 62.81; R18 temporal MapTracker 69.34; AMap R50 non-temporal 69.26 (20.1 FPS); MapTR-based R18 baseline 32.35 -> 47.84.

## Compute

31.5 FPS (R18 student).

## Code

none on the page.

## Fit to our project

reuse_idea - M - privileged-information distillation: a recursion that sees past AND future frames can serve as teacher for a causal student; gains were measured on the "ahead" region.

## Needs

future frames in training (we have the sequences). Cached-BEV: yes.

## Conflicts

logit distillation uses Hungarian.

## Tried in our experiments

none.
