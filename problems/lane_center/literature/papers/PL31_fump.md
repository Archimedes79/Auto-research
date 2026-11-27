# PL31 - FUMP (planning as motion prediction with map nodes)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M (tail cut)** (A abstract, M method, C code page). Verified: partly. Pipeline stage: S10.*

## Papers

Fully Unified Motion Planning for End-to-End Autonomous Driving (arXiv 2504.12667; authors not read) https://arxiv.org/abs/2504.12667

## Idea

Ego and other vehicles share a two-stage decoder (state-agnostic multimodal generation, then ego-state-aware refinement). Agent and map features from the SparseDrive perception become graph nodes, grouped into four zone subgraphs plus a global graph (PointNet, set attention); motion and plan queries cross-attend to these nodes.

## Formulation

not read in detail.

## Input

SparseDrive instance features.

## Output

ego plan and agent motion.

## Training

not read.

## Differentiable?

yes (end-to-end), details not read.

## Sensors and data

nuScenes, NAVSIM, Bench2Drive.

## Reported results

nuScenes open loop avg L2 0.39 m vs SparseDrive 0.61, collision 0.06 vs 0.08 %; NAVSIM PDMS 87.8 vs TransFuser 84.0 (85.1 reproduced; the abstract's +3.7 % does not match the table); Bench2Drive DS 45.67 vs VAD 42.35.

## Compute

not read.

## Code

not found on the page.

## Fit to our project

reuse_idea - effort M - map nodes in zone subgraphs.

## Needs

instance nodes.

## Conflicts

needs agents.

## Tried in our experiments

none.
