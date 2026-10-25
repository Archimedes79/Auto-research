# PL02 - PDM-Closed / PDM-Open / PDM-Hybrid (rule-based centre-line planner, closed-loop test bed)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M+C (partial)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10, E.*

## Papers

Parting with Misconceptions about Learning-based Vehicle Motion Planning (Dauner, Hallgarten, Geiger, Chitta; CoRL 2023) https://arxiv.org/abs/2306.07962

## Idea

nuPlan planner that needs only the route's centre-line, forecasts of other agents and the drivable-area map for scoring: IDM proposals along the centre-line, rolled out by an LQR tracker on a kinematic bicycle model, best chosen by a nuPlan-like proxy score. PDM-Open (MLP on centre-line + ego history) reaches the best open-loop score without agent or map input beyond the centre-line; PDM-Hybrid adds a learned offset. Core message: open-loop and closed-loop scores are misaligned.

## Formulation

centre-line from a graph search over the lanes of the route (algorithm name not confirmed on the pages read). Route = roadblock-id sequence, repaired in code by BFS (start off-route: backward depth 15, else forward depth 30; gaps between consecutive roadblocks bridged by forward BFS depth 30; loops cut). Proposals: 5 target speeds (20 ... 100 % of the speed limit) x 3 lateral offsets (-1, 0, +1 m) = 15, 4 s at 10 Hz; others forecast with constant velocity over 8 s; best proposal extended to 8 s by IDM; emergency brake if a collision is expected within 2 s; score weights in the supplement (not read). PDM-Open: centre-line at 1 m up to 120 m + 2 s ego history at 5 Hz, each projected to 512, MLP 2 x 512, 8 s output at 0.5 s.

## Input

ego pose, lane graph (nuPlan map API), route roadblock ids, tracked objects, drivable-area map (scorer).

## Output

8 s trajectory.

## Training

none for Closed; L1 imitation on about 177 k scenarios for Open / Hybrid.

## Differentiable?

no (Closed); the learned parts yes.

## Sensors and data

nuPlan with privileged GT map and GT objects (paper: localisation errors, map imperfections, misdetections are not considered); Val14 = 1,118 scenarios.

## Reported results

Val14 CLS-R / CLS-NR / OLS: PDM-Closed 92 / 93 / 42; PDM-Hybrid 92 / 93 / 84; PDM-Open 54 / 50 / 86; IDM 77 / 76 / 38; GC-PGP (lane-graph GNN planner) 55 / 59 / 83; Urban Driver 50 / 53 / 82; PlanCNN 72 / 73 / 64; log replay 80 / 94 / 100. 2023 nuPlan challenge: PDM-Hybrid 90, rank 1 of 25.

## Compute

PDM-Closed 91 ms, PDM-Open 7 ms, PDM-Hybrid 96 ms (hardware not stated).

## Code

https://github.com/autonomousvision/tuplan_garage

## Fit to our project

reimplement - effort M - the original needs nuPlan devkit and nuPlan maps; the principle (best route path through the graph by dynamic programming over route probability, IDM, proxy score) is small.

## Needs

lane graph with successors; route (roadblock ids or our per-vertex route probability); drivable-area raster; centre-line at 1 m up to about 120 m (our BEV reaches 50 m forward); speed limit.

## Conflicts

privileged-map assumption; the roadblock BFS repair has no counterpart for a predicted graph with gaps; centre-line quality is the single point of failure.

## Tried in our experiments

none (E117 is a learned consumer; no rule-based consumer in the project).
