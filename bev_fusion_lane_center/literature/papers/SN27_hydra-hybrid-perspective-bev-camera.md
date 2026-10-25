# SN27 - HyDRa (and RaCFormer): hybrid perspective / BEV camera-radar fusion

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (HyDRa), A (RaCFormer)** (A abstract, M method, C code page). Verified: checked (HyDRa), partly (RaCFormer: abstract level). Pipeline stage: B.*

## Papers

Unleashing HyDRa: Hybrid Fusion, Depth Consistency and Radar for Unified 3D Perception (Wolters et al., ICRA 2025 per arXiv v4) https://arxiv.org/abs/2403.07746 ; RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion (Chu et al., CVPR 2025) https://arxiv.org/abs/2412.12725 (abstract level only)

## Idea

HyDRa fuses radar pillars with image columns in the perspective view (a height-association transformer) and refines in BEV with a radar-weighted depth-consistency attention; BEV features concatenated with radar-BEV and an SE block; an occupancy head reuses the BEV features. RaCFormer samples instance features from both BEV and the image view with polar queries and a Doppler-based implicit dynamic catcher.

## Formulation

BEV 128 x 128 at 0.8 m (R50, 256 x 704), 256 x 256 for larger models; 118 depth bins; last eight history frames; trained 20 epochs per task.

## Input

images + radar pillars (sweeps / ego compensation not specified).

## Output

boxes, tracks, Occ3D occupancy.

## Training

per task; regime matching_free (heatmap head).

## Differentiable?

yes.

## Sensors and data

camera + radar, nuScenes, View-of-Delft, Occ3D.

## Reported results

HyDRa val R50 NDS 58.5 / mAP 49.4, R101 61.7 / 53.6; test 64.2 NDS / 57.4 mAP; Occ3D mIoU 44.4 vs 40.7 (FB-OCC), night 30.1 vs 25.8; RaCFormer (abstract): 64.9 mAP / 70.2 NDS on nuScenes.

## Compute

not stated.

## Code

https://github.com/phi-wol/hydra and https://github.com/cxmomo/RaCFormer (both named on the pages read; not opened).

## Fit to our project

reuse_idea - effort M - the occupancy output on a camera-radar BEV shows radar helps dark / unobserved regions; both need images.

## Needs

images.

## Conflicts

perspective-view fusion inside the trained encoder.

## Tried in our experiments

none.
