# PL24 - Input-swap evaluation protocols (GT vs predicted vs masked inputs into one fixed consumer)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M (composed from the cards)** (A abstract, M method, C code page). Verified: partly (synthesis). Pipeline stage: E.*

## Papers

composed from PL14 (Plug-and-Play TE), PL15 (Gu CVPR 2024), PL18 (AMap), PL19 (BEV-Planner), PL28 (GoalFlow oracle goal), PL23 (PKL)

## Idea

Evaluate a map or graph by swapping the input of one fixed consumer and comparing outputs or metrics. Variants seen: GT vs predicted topology into VAD; GT HD map vs online centre-line map into DenseTNT; forward / backward masks of the predicted map into HiVT; oracle vs predicted goal in GoalFlow; zeroed ego velocity or blank images in VAD.

## Formulation

(my synthesis) fixed consumer; input families {GT, predicted, masked by region, perturbed (shift, jitter, element drop), shuffled}; metrics sensitive to the swapped factor (boundary violation, direction, miss rate) and subsets (turning, forward half); report the swap gap; add an ego-status-removed variant so the consumer cannot ignore the graph.

## Input

consumer + input variants.

## Output

swap-gap metrics.

## Training

none (frozen consumer) or staged.

## Differentiable?

n/a.

## Sensors and data

nuScenes (PL14, PL15, PL18, PL19), NAVSIM (GoalFlow).

## Reported results

VAD with GT vs TopoMLP topology: L2 0.62 / 0.62 m, collision 0.16 / 0.16 % (no gap); DenseTNT: online MapTRv2-centre-line 0.8466 / 1.345 / 0.1520 vs GT HD map 0.8809 / 1.489 / 0.1903 (online better); HiVT forward mask 100 %: minADE +9.73 %, backward +0.67 %; GoalFlow oracle goal 92.1 vs predicted 90.3 PDMS; VAD with zeroed ego velocity L2 6.16 m, blank images 0.46 m.

## Compute

n/a.

## Code

https://github.com/alfredgu001324/MapUncertaintyPrediction (prediction consumers)

## Fit to our project

reuse_idea - effort S - the project already did a partial swap (oracle route 0.316 m vs 0.34 m in E119; graph A 0.36 m vs B 0.34 m in E117), small gaps like the literature.

## Needs

a consumer that accepts GT and predicted graphs in one format.

## Conflicts

consumers that exploit ego status hide the swap gap.

## Tried in our experiments

partly - oracle-route and A-vs-B swaps with E117 (E119).
