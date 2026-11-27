# PL38 - MTR (768 polylines x 20 points, local k-NN attention, intention queries)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10.*

## Papers

Motion Transformer with Global Intention Localization and Local Movement Refinement (Shi et al., NeurIPS 2022) https://arxiv.org/abs/2209.13508

## Idea

Map = up to 768 nearest polylines with up to 20 points (about 10 m in WOMD); per point position, direction and type; agent-centric coordinates; PointNet-like encoder (5-layer MLP width 64, max-pool, projection to 256). 6-layer transformer encoder with attention restricted to the 16 nearest polylines (4 to 64 tested). Decoder with 64 static intention queries (k-means of GT endpoints) and dynamic searching queries refined over 6 layers, cross-attending a dynamic collection of 128 polylines near the current predicted trajectory; NMS (2.5 m) picks 6 of 64.

## Formulation

Gaussian regression per decoder layer (NLL on the hard-assigned positive component + CE on its probability) + auxiliary dense future prediction (L1); equal weights.

## Input

polyline set + agent histories.

## Output

6 trajectories (GMM) per target.

## Training

AdamW lr 1e-4, batch 80, 30 epochs, 8 GPUs, no augmentation; regime: wta (hard intention assignment).

## Differentiable?

yes except NMS.

## Sensors and data

Waymo (GT map).

## Reported results

WOMD test marginal mAP 0.4129, minADE 0.6050, minFDE 1.2207, MR 0.1351; joint mAP 0.2037.

## Compute

parameter count not reported.

## Code

https://github.com/sshaoshuai/MTR

## Fit to our project

reuse_idea - effort M - the dynamic map collection along the current predicted path is "sample map features around the plan"; k-NN attention.

## Needs

fixed-size polyline set with positions.

## Conflicts

no topology; NMS.

## Tried in our experiments

none.
