# SN40 - Other camera-radar detectors 2023 - 2025 (RCM-Fusion, SpaRC, CRT-Fusion, CR3DT): abstract level

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **A (search-result summaries; not opened)** (A abstract, M method, C code page). Verified: partly (abstract-level search summaries). Pipeline stage: B.*

## Papers

RCM-Fusion: Radar-Camera Multi-Level Fusion for 3D Object Detection (Kim et al., ICRA 2024) https://arxiv.org/abs/2307.10249 ; SpaRC: Sparse Radar-Camera Fusion for 3D Object Detection https://arxiv.org/abs/2411.19860 ; CRT-Fusion: Camera, Radar, Temporal Fusion Using Motion Information for 3D Object Detection https://arxiv.org/abs/2411.03013 ; CR3DT: Camera-RADAR Fusion for 3D Detection and Tracking https://arxiv.org/abs/2403.15313

## Idea

RCM-Fusion fuses at two levels (a radar-guided BEV encoder in which radar BEV features steer the camera-to-BEV transform, plus a radar grid-point refinement of instance proposals). SpaRC replaces dense BEV with a sparse fusion transformer (sparse frustum fusion, range-adaptive radar refinement) to avoid depth-estimation errors of dense BEV methods. CRT-Fusion models object-level dynamics to align moving objects across frames but still aggregates densely in BEV.

## Formulation

not read.

## Input

images + radar.

## Output

boxes (CR3DT: tracks).

## Training

not read.

## Differentiable?

yes (assumed from the architecture class; not read).

## Sensors and data

camera + radar, nuScenes.

## Reported results

SpaRC abstract: +2.9 NDS and +2.6 mAP on nuScenes over its baseline (reference method not seen); RCM-Fusion: state of the art among single-frame radar-camera methods at its time (no numbers seen).

## Compute

not stated.

## Code

not checked.

## Fit to our project

reuse_idea - effort M - none gives map or lane outputs; they show the two design poles (dense radar-guided BEV vs sparse query fusion) next to the GaussianCaR Gaussian-set pole.

## Needs

images.

## Conflicts

detection only; object-level temporal alignment does not apply to static lanes.

## Tried in our experiments

none.
