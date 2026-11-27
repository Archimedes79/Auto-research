# SN34 - Out-of-fold (held-out) features for second-stage training

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **A** (A abstract, M method, C code page). Verified: partly (search summary). Pipeline stage: T.*

## Papers

A Robust Stacking Framework for Training Deep Graph Models with Multifaceted Node Features (arXiv 2022) https://arxiv.org/abs/2206.08473 ; classical stacked generalisation (Wolpert 1992; from memory)

## Idea

If a first-stage model has memorised its training data, a second stage trained on its outputs for the same data sees features that are cleaner than at test time; the fix is to train the second stage only on predictions for data the first stage never saw (k-fold partition, k copies of the first model).

## Formulation

not read.

## Input

first-stage outputs.

## Output

n/a.

## Training

cross-fitting; regime none.

## Differentiable?

yes.

## Sensors and data

graph benchmarks (not driving).

## Reported results

none seen (search-summary statement of the principle).

## Compute

k copies of the first stage.

## Code

not checked.

## Fit to our project

reuse_idea - effort L for the foundation (retraining k copies is impossible with released weights) - but scene-level held-out features can be approximated by weighting the heads' training towards cached features of scenes outside the foundation's training split; this explains the 0.93 vs 0.82 gap qualitatively.

## Needs

knowing which scenes the foundation trained on (the nuScenes train list).

## Conflicts

we do not control the foundation's training.

## Tried in our experiments

none.
