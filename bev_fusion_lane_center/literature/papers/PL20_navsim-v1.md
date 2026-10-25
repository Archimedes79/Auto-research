# PL20 - NAVSIM v1 (non-reactive simulation, PDM score)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: E.*

## Papers

NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking (Dauner et al., NeurIPS D&B 2024) https://arxiv.org/abs/2406.15349

## Idea

4 s planning from a recorded frame; the ego is rolled out by an LQR controller on a kinematic bicycle model at 10 Hz while other agents follow their log. PDMS = (NC x DAC) x (5 EP + 5 TTC + 2 C) / 12 with NC no at-fault collision (static-object contact 0.5), DAC drivable-area compliance (map-based), EP progress, TTC time-to-collision, C comfort.

## Formulation

agent input = 8 cameras, merged lidar, ego status (pose, velocity, acceleration), one-hot driving command (left / straight / right, plus unknown); 2 s history at 2 Hz. The map is NOT an agent input (maps, tracks and occupancy are available for training targets only, not released for test frames; the scorer uses the map). Data OpenScene (a redistribution of nuPlan, 120 h at 2 Hz); navtrain about 103 k samples, navtest about 12 k; challenging-scene filter removes frames where constant velocity scores above 0.8 and frames where the human scores below 0.8 (constant velocity falls to 22 %, humans 95 %). Human-violation filter documented for v2.

## Input

sensors + ego status + command.

## Output

4 s trajectory at 10 Hz (x, y, heading).

## Training

n/a (benchmark).

## Differentiable?

n/a.

## Sensors and data

cameras + lidar, nuPlan locations.

## Reported results

navtest PDMS (NC / DAC / TTC / Comf / EP): constant velocity 20.6 (68.0 / 57.8 / 50.0 / 100 / 19.4); ego-status MLP 65.6 (93.0 / 77.3 / 83.6 / 100 / 62.8); TransFuser 84.0; LTF 83.8; UniAD 83.4; PARA-Drive 84.0; human 94.8. PDMS correlates better with the nuPlan closed-loop score than open-loop scores (qualitative on the page). Numbers differ across papers (Hydra-MDP lists TransFuser 78.0, PDM-Closed with GT perception 89.1).

## Compute

cheap per scene (4 s rollout); no figures.

## Code

https://github.com/autonomousvision/navsim

## Fit to our project

baseline - effort M - no radar, no nuScenes data, no map input for agents; DAC / DDC / LK are the map-sensitive terms and can be re-implemented.

## Needs

drivable-area polygons and lane geometry for scoring.

## Conflicts

nuPlan / OpenScene data only.

## Tried in our experiments

none.
