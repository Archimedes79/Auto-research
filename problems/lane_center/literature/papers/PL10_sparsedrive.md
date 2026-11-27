# PL10 - SparseDrive (sparse map instances as planner keys / values)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10.*

## Papers

SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation (Sun et al., ICRA 2025) https://arxiv.org/abs/2405.19620

## Idea

Symmetric sparse perception (detection / tracking and online mapping share one decoder design) feeds a parallel motion planner. Map instances: N_m = 100 polylines x N_p = 20 points with anchors from K-means. The ego instance (initialised from the pooled smallest-scale front-camera feature) attends agents and map instances and plans in command-specific modes (K_p = 6, T_p = 6). Hierarchical selection: command filter, then collision-aware rescoring with the two most confident agent trajectories.

## Formulation

ego anchor from kinematics, velocity from the previous prediction (against ego-status leakage); ego status decoded as an auxiliary output; agent-temporal cross-attention, agent-agent self-attention, agent-map cross-attention; memory queue of 3 frames; mapping range 60 x 30 m; tracking id once confidence exceeds 0.2.

## Input

multi-view image features.

## Output

ego trajectories per command + scores; map and agent instances.

## Training

stage 1 perception from scratch (S: 100 epochs batch 8 lr 4e-4; B: 80 epochs), stage 2 joint 10 epochs with no frozen weights (no detach mentioned); regime: Hungarian matching for instances (from memory), imitation for planning.

## Differentiable?

partial - stage-2 gradients reach perception queries; instance matching, top-k and rescoring are discrete.

## Sensors and data

nuScenes cameras.

## Reported results

nuScenes val avg L2 / collision: SparseDrive-S 0.61 m / 0.08 %, -B 0.58 / 0.06 (VAD-style metrics); UniAD 0.73 / 0.61 and VAD 0.72 / 0.21 in the same table; collision-aware rescoring 0.12 -> 0.08 %; no ablation on map features or ego status.

## Compute

9.0 FPS (S), 7.3 FPS (B) on an RTX 4090; 85.9 M / 104.7 M parameters; training 18 + 2 h (S).

## Code

https://github.com/swc-17/SparseDrive

## Fit to our project

reuse_idea - effort M - instance features + anchors as keys / values and per-command modes are portable, the mmcv-based code is not.

## Needs

map instance tensor (feature + 20 x 2 anchors) and detections for the rescoring.

## Conflicts

Hungarian matching; mmdet3d stack; K-means anchors over polylines assume a fixed point order.

## Tried in our experiments

E31 / E32 (matched sparse set decoder dropped) concern the perception side only.
