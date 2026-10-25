# PL18 - AMap: forward-region errors matter more (ahead-aware map metric A-mAP)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: E.*

## Papers

AMap: Distilling Future Priors for Ahead-Aware Online HD Map Construction (arXiv 2512.19150; authors not read) https://arxiv.org/abs/2512.19150

## Idea

Split the region of interest into ahead and rear halves relative to ego heading and report AP per half (A-mAP, R-mAP), because the road ahead matters more downstream. Motivation: MapTR output masked in front of or behind the ego and fed to HiVT (a predictor, not a planner). The method distils a teacher with future frames into a current-frame student.

## Formulation

AP per half like global mAP; teacher with 5 future frames; multi-level BEV feature distillation with a GT-derived spatial mask; asymmetric query distillation (Hungarian matching of 100 student queries to 116 teacher queries, KL on matched logits).

## Input

online map predictions (for the metric).

## Output

A-mAP, R-mAP.

## Training

teacher-student distillation (not needed for the metric).

## Differentiable?

n/a for the metric.

## Sensors and data

nuScenes, Argoverse 2.

## Reported results

forward mask 100 %: minADE +9.73 %, minFDE +13.95 %, MR +20.96 % (reference 0.3856 / 0.7919 / 0.0854); backward mask 100 %: +0.67 %, +0.66 %, -4.45 %. nuScenes (R50, no temporal): mAP 69.26, A-mAP 70.19, R-mAP 69.61. No planning experiment (only a qualitative figure).

## Compute

not read.

## Code

not found on the page.

## Fit to our project

reuse_idea - effort S - report graph metrics for the forward / route-relevant half.

## Needs

heading-aligned ROI split.

## Conflicts

the split is directional, not distance-weighted.

## Tried in our experiments

none.
