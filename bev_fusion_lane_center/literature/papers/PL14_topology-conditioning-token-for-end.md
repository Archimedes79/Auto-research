# PL14 - Topology-conditioning token for end-to-end planners (GT vs predicted topology)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M (page cut at 100k characters)** (A abstract, M method, C code page). Verified: partly. Pipeline stage: S10, E.*

## Papers

Plug-and-Play Traffic Element Awareness for End-to-End Autonomous Driving (arXiv 2608.18035; authors not read) https://arxiv.org/abs/2608.18035

## Idea

Traffic elements (4 light states, 9 sign classes lifted to 3-D) and the ego-centric topology (centre-line-to-centre-line and centre-line-to-element relations of OpenLane-V2) are rendered as a sentence, encoded by a frozen BERT-base (about 110 M parameters) into queries and concatenated with the planner's BEV queries.

## Formulation

ego lane found from the ego position; LCLC gives ego-related centre-lines, LCTE the elements governing the ego lane; Q_plan = [Q_bev; Q_topo]; traffic-element 3-D centres from 2-D detection + UniDepthv2 + lidar.

## Input

lane subgraph (as text) + BEV queries of the base planner.

## Output

planner trajectory (base planners VAD, Orion on nuScenes; LTF, DiffusionDrive, DrivoR on NAVSIM; VAD, DriveTransformer-Large on Bench2Drive).

## Training

planners fine-tuned with the extra tokens (details not read).

## Differentiable?

no through the text rendering and the frozen BERT; gradient reaches the planner only.

## Sensors and data

nuScenes (OpenLane-V2 annotations), NAVSIM (traffic elements only), Bench2Drive (GT topology).

## Reported results

nuScenes with VAD (ego scope, BERT encoder): GT topology avg L2 0.62 m / collision 0.16 %; topology predicted by TopoMLP 0.62 / 0.16 (identical); baseline VAD 0.72 / 0.22; full method 0.60 / 0.17; Orion 0.34 / 0.37 -> 0.26 / 0.23. Bench2Drive driving score VAD 42.3 -> 56.4, DriveTransformer-Large 63.46 -> 68.29 (traffic elements and topology combined).

## Compute

BERT 110 M parameters (frozen); rest not read.

## Code

none (project page https://zzongzheng0918.github.io/TE-Aware-E2E-AD/)

## Fit to our project

reuse_idea - effort S - a graph summary as one conditioning token is cheap; on nuScenes open loop the planner cannot tell GT from predicted topology, so the metric does not resolve graph quality.

## Needs

a text or vector summary of the ego-relevant subgraph.

## Conflicts

needs a text encoder; topology comes from OpenLane-V2 annotations, not from nuScenes map-expansion lanes.

## Tried in our experiments

none.
