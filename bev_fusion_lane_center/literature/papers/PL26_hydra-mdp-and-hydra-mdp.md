# PL26 - Hydra-MDP and Hydra-MDP++ (map enters only through distilled rule-based sub-scores)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10, L.*

## Papers

Hydra-MDP: End-to-end Multimodal Planning with Multi-target Hydra-Distillation (Li et al., 2024) https://arxiv.org/abs/2406.06978 ; Hydra-MDP++ https://arxiv.org/abs/2503.12820

## Idea

A planning vocabulary of 4096 / 8192 K-means trajectories (700 K nuPlan samples, 40 waypoints, 4 s) is scored by heads that imitate the sub-scores of the rule-based PDM simulation (NC, DAC, TTC, C, EP; ++ adds TLC, DDC, LK, EC) plus an imitation head. The map is used offline to compute the teacher scores, not as a network input; BEV segmentation / lane heads are not used by the student (++: optional lane and traffic-light supervision hurts planning).

## Formulation

imitation = softmax CE against the distance to the log-replay; distillation = per-metric BCE against simulated sub-scores; inference cost = -(w1 log S_im + w2 log S_NC + w3 log S_DAC + w4 log(5 S_TTC + 2 S_C + 5 S_EP)) with grid-searched weights; ensembling of encoders and of sub-scores.

## Input

image / lidar tokens + ego status (++: command as well).

## Output

scored vocabulary entries; the lowest-cost trajectory.

## Training

CE + BCE; vocabulary simulated offline.

## Differentiable?

yes for the heads; the teachers are rules.

## Sensors and data

NAVSIM cameras + lidar.

## Reported results

navtest PDMS: Hydra-MDP-V8192 83.0; weighted EP 86.5; scaled (V2-99 / ViT-L) 91.0; distilling only the overall PDM score gives 80.2; PDM-Closed with GT perception 89.1 (EPDMS 82.8 in ++). ++: ResNet-34 PDMS 86.6 / EPDMS 80.6, V2-99 91.0 / 84.1. The paper attributes the low DAC of neural planners to not knowing the drivable-area extent.

## Compute

~206 ms (ResNet-34) and ~271 ms (V2-99) latency (++).

## Code

https://github.com/NVlabs/Hydra-MDP

## Fit to our project

reuse_idea - effort M - score candidate paths by graph-derived compliance sub-scores, to distil or to use directly.

## Needs

a simulator-style scorer from the lane graph and drivable area.

## Conflicts

needs a rule simulation per vocabulary entry.

## Tried in our experiments

none.
