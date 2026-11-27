# PL22 - nuPlan benchmark and its evaluation protocols

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **A** (A abstract, M method, C code page). Verified: partly. Pipeline stage: E, D.*

## Papers

nuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles (Caesar et al., arXiv 2106.11810) https://arxiv.org/abs/2106.11810

## Idea

About 1,500 h of human driving in four cities, a lightweight closed-loop simulator (reactive and non-reactive agents) and planning-specific metrics. Route = roadblock-id sequence and the map = lane graph (both seen in planner code, PL03-PL06). Protocol splits used by the papers above: Val14 (1,118 scenarios), Test14-random, Test14-hard; scores OLS, CLS-NR, CLS-R.

## Formulation

not read (abstract page only).

## Input

not read.

## Output

not read.

## Training

not read.

## Differentiable?

n/a.

## Sensors and data

nuPlan (Boston, Pittsburgh, Las Vegas, Singapore).

## Reported results

see PL02-PL07.

## Compute

not read.

## Code

devkit https://github.com/motional/nuplan-devkit (named in the READMEs of PL03-PL06)

## Fit to our project

not_suitable - effort L - nuPlan logs and maps are not our nuScenes frames; its metric families are the template.

## Needs

nuPlan maps and logs.

## Conflicts

dataset.

## Tried in our experiments

none.
