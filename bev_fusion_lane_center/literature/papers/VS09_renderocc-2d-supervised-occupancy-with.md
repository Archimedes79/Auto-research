# VS09 - RenderOcc: 2D-supervised occupancy with volume rendering, auxiliary rays, weighted ray sampling

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S3, L.*

## Papers

RenderOcc (Pan et al., ICRA 2024) https://arxiv.org/abs/2309.09502

## Idea

Semantics and depth are rendered from a 3D volume along camera rays and compared with 2D labels, so no 3D occupancy label is needed; rays from adjacent frames add views; a weighting scheme chooses which rays enter the loss.

## Formulation

sigma = softplus(MLP_d(V)), S = MLP_s(V); alpha(z_k) = 1 - exp(-sigma(z_k) beta_k); T(z_k) = exp(-sum_{t<k} sigma(z_t) beta_t); semantics = sum_k T alpha S; depth = sum_k T alpha z_k. Loss = CE (semantics) + SILog (depth) + L_reg (distortion loss and total-variation loss on the density; weights not stated). One ray per pixel, 38,400 rays per batch on nuScenes (25,600 on SemanticKITTI); points per ray K not stated for the main model (ablation: unified, hierarchical 64 -> 64 or 64 -> 128, mip360-style step 0.5 voxel). Auxiliary rays: 6 adjacent frames on nuScenes (4 on SemanticKITTI) moved by ego pose. Weighted ray sampling: class-balance W_b = exp(lambda_s (max(M) / N(C(r)) - 1)) and a temporal weight (dynamic-object rays get a coefficient < 1), W = W_b W_t as sampling probability, fixed ray count per batch. Rays leaving the volume: not stated.

## Input

volume feature V. output: occupancy / semantic volume.

## Training

supervised from 2D labels (projected lidar with segmentation); matching_free.

## Differentiable?

yes (ray selection only chooses loss terms).

## Sensors and data

nuScenes (Occ3D labels for evaluation), SemanticKITTI; cameras.

## Reported results

Occ3D-nuScenes mIoU: RenderOcc 2D-only 23.93, with 3D labels too 26.11; BEVStereo 24.51, BEVFormer 23.67 (3D-supervised); depth from structure-from-motion instead of lidar 21.11 vs 23.93.

## Compute

not stated.

## Code

URL from memory, not opened: https://github.com/pmj110119/RenderOcc

## Fit to our project

reuse_idea - S - (a) supervise a per-sample ray accumulator with 1-D labels; (b) importance weights on which rays enter the loss (rare / hard rays); (c) the distortion regulariser (VS07).

## Needs

camera rays and a 3D density field.

## Conflicts

the cached BEV rasters carry no depth axis.

## Tried in our experiments

none.
