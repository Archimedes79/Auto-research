# SN29 - UDGA: adapters for multi-view BEV detectors under camera / dataset shift

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (adapter section, Tables 2, 9 - 11)** (A abstract, M method, C code page). Verified: partly (reader could not map the head-only table row). Pipeline stage: T.*

## Papers

Unified Domain Generalization and Adaptation for Multi-View 3D Object Detection (arXiv 2024) https://arxiv.org/abs/2410.22461

## Idea

Bottleneck adapters (down projection, activation, up projection, BatchNorm on input, parallel to the pretrained blocks, near-identity init) can be inserted into backbone, view transform, BEV encoder and head; the text reports that adapters only at the detection head decrease Lyft -> nuScenes performance.

## Formulation

y = B(x) + A(x); best setup conv down, linear up, 8.8 M parameters (full fine-tuning 51.7 M; BEVFormer variant 12.2 M of 33.5 M).

## Input

BEV detector features.

## Output

boxes.

## Training

parameter-efficient fine-tuning with target-domain labels at 1 - 100 % fractions; regime none (supervised).

## Differentiable?

yes.

## Sensors and data

camera, nuScenes <-> Lyft, CARLA shifts.

## Reported results

Lyft -> nuScenes NDS / mAP at 1 % target data: full FT 0.476 / 0.369, theirs 0.526 / 0.404, SSF 0.245 / 0.079; extrinsic-height shift in CARLA: NDS 0.666 -> 0.313 (Table 7).

## Compute

8.8 M trainable parameters.

## Code

not checked.

## Fit to our project

reuse_idea - effort M - indicates that adapters inside the backbone matter more than head-only adaptation when the shift is geometric; ours is a scene / location shift on the same rig, so the effect may differ.

## Needs

access to inner layers of the foundation (not in the cache).

## Conflicts

the cache holds final features only.

## Tried in our experiments

none.
