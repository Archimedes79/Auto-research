# SN25 - RCBEVDet: radar point + transformer backbone, RCS-aware BEV, cross-attention fusion

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B.*

## Papers

RCBEVDet: Radar-camera Fusion in Bird's Eye View for 3D Object Detection (Lin et al., CVPR 2024; from memory) https://arxiv.org/abs/2403.16440 ; code https://github.com/VDIGPKU/RCBEVDet (URL from memory, not opened)

## Idea

RadarBEVNet = dual-stream radar backbone (PointNet-style branch + transformer branch with distance-modulated self-attention, linked by injection / extraction cross-attention) and an RCS-aware BEV encoder that scatters each point to nearby BEV pixels within an RCS-scaled radius; CAMF fuses with the camera BEV by deformable cross-attention in both directions plus conv blocks.

## Formulation

radar features RCS + Doppler, multiple sweeps (number and ego compensation not stated); camera stream = BEVDepth with temporal fusion, BEVPoolv2, trained first then frozen; 12 epochs AdamW, CBGS.

## Input

images + radar.

## Output

boxes.

## Training

staged (camera stream frozen in stage 2); regime matching_free (heatmap head).

## Differentiable?

yes.

## Sensors and data

camera + radar, nuScenes, View-of-Delft.

## Reported results

val R50 NDS 56.8 / mAP 45.3, 21.3 FPS; test V2-99 NDS 63.9 / mAP 55.0; ablation: + RadarBEVNet 55.7 NDS, + CAMF 56.4; radar dropout robustness better than CRN (Table 7).

## Compute

21.3 FPS (RTX 3090, FP16, batch 1).

## Code

see papers line.

## Fit to our project

reuse_idea - effort M - RCS-radius splatting is a cheap radar densifier; no segmentation or map task in the paper (CTAB, SN36, adds one).

## Needs

images.

## Conflicts

detection only.

## Tried in our experiments

none.
