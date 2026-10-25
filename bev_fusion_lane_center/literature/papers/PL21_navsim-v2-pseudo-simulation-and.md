# PL21 - NAVSIM v2 pseudo-simulation (EPDMS) and the map-sensitive sub-scores

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: E.*

## Papers

Pseudo-Simulation for Autonomous Driving (Cao et al., CoRL 2025) https://arxiv.org/abs/2506.04218 ; NAVSIM docs (metrics) https://github.com/autonomousvision/navsim

## Idea

Stage 1 rolls out the plan on a real frame; stage 2 scores the plan on pre-rendered synthetic observations (Gaussian-splat views) whose start points lie around the expert's endpoint (lateral steps of 0.5 m up to +-2.0 m, longitudinal steps of 5.0 m), weighted by a Gaussian kernel (sigma^2 = 0.1) of the distance to the planner's stage-1 endpoint; combined score = stage-1 score x weighted stage-2 average.

## Formulation

EPDMS = (NC x DAC x DDC x TLC) x weighted average of TTC (5), EP (5), LK (2), HC (2), EC (2). DDC driving-direction compliance in {0, 1/2, 1}; LK lane keeping penalises staying far from the centre-line for long and is disabled at intersections; TLC traffic-light compliance; a penalty is ignored if the human commits the same violation.

## Input

plan from any agent (4 s).

## Output

EPDMS.

## Training

n/a.

## Differentiable?

n/a.

## Sensors and data

camera + lidar (nuPlan / OpenScene).

## Reported results

correlation with nuPlan closed loop r = 0.89 (R^2 = 0.8, 83 planners; the text says 40 + 46 planners, the caption 37 + 46); single-stage open loop r = 0.83; ADE vs nuPlan open-loop score almost uncorrelated; navhard 450 stage-1 and 5,462 stage-2 observations. navhard EPDMS (snapshot 03/2026): PDM-Closed (privileged) 56.6; DrivoR 54.5; SimScale 53.2; GuideFlow 51.5; ZTRS 48.1; RAP 39.6; LTF 25.1; ego MLP 14.1; constant velocity 11.4.

## Compute

about 6 x less than closed-loop simulation (README claim).

## Code

https://github.com/autonomousvision/navsim

## Fit to our project

reuse_idea - effort M - DDC / LK / DAC as graph-sensitive sub-scores, and the idea of scoring a plan from perturbed start states.

## Needs

lane direction, centre-line distance, drivable area.

## Conflicts

nuPlan data and a rendering pipeline.

## Tried in our experiments

none.
