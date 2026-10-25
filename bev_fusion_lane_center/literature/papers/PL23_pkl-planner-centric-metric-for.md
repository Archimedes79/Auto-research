# PL23 - PKL: planner-centric metric for perception

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **A** (A abstract, M method, C code page). Verified: partly. Pipeline stage: E.*

## Papers

Learning to Evaluate Perception Models Using Planner-Centric Metrics (Philion et al., CVPR 2020) https://arxiv.org/abs/2004.08745

## Idea

Judge detections by how much they change a planner's output (a KL divergence between planner distributions on predicted vs GT inputs; the name and formula are from memory); abstract: downweights distant and slow objects; humans sided with it in 79 % of cases where it disagreed with standard metrics. Planner and dataset (nuScenes) not on the page.

## Formulation

not read.

## Input

not read.

## Output

not read.

## Training

not read.

## Differentiable?

not read.

## Sensors and data

not read on the page.

## Reported results

human study 79 % (abstract).

## Compute

none seen.

## Code

not checked.

## Fit to our project

reuse_idea - effort S - compare the consumer's output on predicted vs GT graph instead of comparing geometry.

## Needs

a deterministic or probabilistic consumer.

## Conflicts

none.

## Tried in our experiments

none (the oracle-vs-predicted swap of E119 / E117 is the same principle).
