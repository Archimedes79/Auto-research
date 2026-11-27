# PL27 - VADv2 (probabilistic planning over a vocabulary with map tokens)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10, L.*

## Papers

VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning (Chen et al., 2024) https://arxiv.org/abs/2402.13243

## Idea

Planning vocabulary of N = 4096 trajectories (furthest-trajectory sampling over demonstrations; 6 waypoints / 3 s); planning tokens (embeddings of candidates) are queries in a cascaded transformer over scene tokens (vectorised map tokens predicting centre-lines, dividers, boundaries, crossings, supervised as in MapTRv2; agent tokens; traffic-element tokens; front-view image tokens); a navigation embedding (MLP, form not stated) and ego state are added; output = probability per action.

## Formulation

loss = distribution loss (KL to the data distribution = CE to the nearest vocabulary entry) + conflict loss (vocabulary entries that conflict with other agents' GT futures or road boundaries are pushed down; map and agents enter only through these labels) + token losses.

## Input

scene tokens, navigation embedding, ego state.

## Output

probability over the vocabulary.

## Training

loss above; regime: imitation with conflict labels.

## Differentiable?

planning gradient to the map head not stated.

## Sensors and data

CARLA cameras; vocabulary statistics from nuScenes / NAVSIM.

## Reported results

CARLA Town05 Long closed loop DS 85.1, RC 98.4, IS 0.87; NAVSIM navtest PDMS 89.3 and Bench2Drive DS 76.15 quoted for reference; no nuScenes planning numbers.

## Compute

not read.

## Code

https://github.com/hustvl/VAD (link given in the VADv2 abstract)

## Fit to our project

reuse_idea - effort S - conflict labels from the map: vocabulary entries leaving the drivable area get a negative label.

## Needs

boundary / drivable-area test per vocabulary entry.

## Conflicts

needs a vocabulary of trajectories.

## Tried in our experiments

none.
