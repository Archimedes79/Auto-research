# VS43 - Visibility / observability masks for censoring (Occ3D, ray-cast occlusion, per-point visibility heads)

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M (Occ3D, PanopticBEV), A (the others)** (A abstract, M method, C code page). Verified: partly. Pipeline stage: S3, D, E.*

## Papers

Occ3D (Tian et al., NeurIPS 2023 datasets track, from memory) https://arxiv.org/abs/2304.14365 ; Bird's-eye-view panoptic segmentation using monocular frontal view images (PanopticBEV, Gosala, Valada, RA-L 2022) https://arxiv.org/abs/2108.03227 ; Parametric depth based BEV feature learning (Yang et al., arXiv 2023, abstract only) https://arxiv.org/abs/2307.04106 ; GaussianCaR (arXiv 2602.08784, ICRA 2026 per the search summary; the paper itself was not opened) https://arxiv.org/abs/2602.08784 ; per-point visibility heads in Anchor3DLane (VS36) and LATR (VS38)

## Idea

How to say "the model could not see this" so it is not counted as "the lane ended". (1) Occ3D: LiDAR visibility = a voxel is visible if it reflects a point or a ray passes through it; free = traversed only; unobserved = the rest; camera visibility = a ray from each occupied voxel to the camera origin, voxels behind the first occupied voxel are unobserved, voxels reached by no camera ray unobserved; evaluation counts only observed voxels. (2) PanopticBEV: a BEV pixel is occluded if it is lower than a previously seen pixel along a 2-D ray cast from the camera across the BEV image (built from the predicted height map); occlusion becomes an explicit stuff CLASS to predict ("extremely difficult to hallucinate labels behind occlusions"), pixels outside the field of view are zeroed and cropped; the page does not say whether occlusion pixels are ignored in the loss or evaluation. (3) The parametric-depth paper proposes a visibility-aware evaluation metric (abstract; how the visibility is obtained is not stated there). (4) Trained per-point visibility: Anchor3DLane (L1, weight 1) and LATR (BCE, weight 1.0); OpenLane labels carry a visibility attribute (from memory). (5) The foundation: GaussianCaR is reported on nuScenes with IoU 57.3 % vehicles, 82.9 % roads, 50.1 % lane dividers (search summary), so a vehicle BEV output exists in the published model; whether the cached 7 map-layer logits contain a vehicle layer is not known to this cluster.

## Formulation

Occ3D grid 200 x 200 x 16 at 0.4 m, +-40 m, z from -1 to 5.4 m; virtual camera points at depth 1e3; PanopticBEV BEV image 768 x 704 (KITTI-360) / 896 x 768 (nuScenes) at 7.4 / 7.6 cm per pixel; no further thresholds on the pages.

## Input

occupancy labels (from lidar) or a height map + camera origin. output: binary masks or an occlusion class.

## Training

labels / evaluation masks; PanopticBEV learns the occlusion class.

## Differentiable?

not applicable.

## Sensors and data

nuScenes lidar + cameras (Occ3D); camera only (PanopticBEV: KITTI-360, nuScenes); camera + radar (GaussianCaR).

## Reported results

none copied.

## Compute

not stated.

## Code

URL from memory, not opened: https://github.com/Tsinghua-MARS-Lab/Occ3D

## Fit to our project

reuse_idea - S - (a) a static camera-coverage mask (field of view, minimum distance) needs only the calibrations; (b) an occluder mask from a vehicle / object layer (if the foundation or a small head provides one) combined with a ray rule from the cameras or the ego position, in the PanopticBEV style, gives a "shadow" mask without lidar; (c) lidar-derived visibility could build censoring LABELS offline, but that brings lidar into the label pipeline (a project decision); (d) a learned observability head with BCE as in LATR, or an explicit "occluded" class as in PanopticBEV.

## Needs

a mask or class per sample position.

## Conflicts

camera + radar only at inference - the mask must be predicted or purely geometric.

## Tried in our experiments

none.
