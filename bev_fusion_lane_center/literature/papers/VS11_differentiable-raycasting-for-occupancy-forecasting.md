# VS11 - Differentiable raycasting for occupancy forecasting (Khurana et al.; formula read in the successor Agro et al.)

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **A + C for Khurana et al. (abstract and code README; their method section NOT read); M for Agro et al. (ar5iv page), which extends the same raycaster to 3-D and states the equations itself** (A abstract, M method, C code page). Verified: partly (Khurana et al. method section not opened; the formulas come from the successor paper). Pipeline stage: S3, S10, L.*

## Papers

Differentiable raycasting for self-supervised occupancy forecasting (Khurana, Hu, Dave, Ziglar, Held, Ramanan, ECCV 2022) https://arxiv.org/abs/2210.01917 ; Point cloud forecasting as a proxy for 4D occupancy forecasting (Agro et al., CVPR 2023, venue from memory) https://arxiv.org/abs/2302.13130

## Idea

Predicted future occupancy is "rendered" into future lidar sweeps by differentiable raycasting; comparing with real sweeps trains the forecaster without occupancy labels; the emergent occupancy then drives a planner. In the successor the renderer is a hit-probability chain along each ray.

## Formulation

(Agro et al., Eqs. 7 - 10; not attributed to Khurana by the page) probability that the ray stops at voxel i: p_i = prod_{j<i}(1 - z_j) * z_i with z the predicted occupancy in [0,1]; expected depth = sum_i p_i * lambda_i, lambda_i = distance at the voxel boundary where a ray stops; rays are traversed voxel by voxel (Amanatides-Woo), stops only at voxel boundaries or infinity; RAYS LEAVING THE GRID: in training a virtual stop at the observed depth beyond the grid is added (Eq. 9), whereas the appendix says the mass that does not terminate is placed at the grid boundary (the two descriptions differ), at evaluation rays are clamped to the volume; loss = L1 between rendered and measured depth summed over rays, gradients reach every voxel the ray crosses; grid = 4-D occupancy (X x Y x Z x T, Z and T folded into channels for 2-D convolutions), nuScenes +-70 m in x and y, +-4.5 m in z. Khurana et al.: README only - `rasterize.py` builds BEV object occupancy and "shadow" maps, `precast.py` future visible free-space maps and contour points; training arguments include sampled trajectories (clothoid or data-driven) and a weight on a free-space binary cross-entropy; test scripts compute ray-distance error and planning losses.

## Input

past lidar sweeps. output: future occupancy / free space; rendered depth; planner cost (Khurana et al.).

## Training

self_supervised from lidar; planner cost on occupancy.

## Differentiable?

yes (the point of the papers).

## Sensors and data

lidar; nuScenes (with CAN bus) and ONCE named in the Khurana README; nuScenes, KITTI-Odometry, Argoverse 2.0 in Agro et al.

## Reported results

Khurana abstract: raycasted-lidar forecasting up to 15 F1 points better; planner collisions up to 17 % fewer relative to freespace-centric planners. Agro et al. (their own numbers): nuScenes 1 s L1 1.40 m, AbsRel 10.37 %, 3 s L1 1.71 m, against a ray-tracing baseline 1.50 m / 2.44 m.

## Compute

not stated.

## Code

checked https://github.com/tarashakhurana/emergent-occ-forecasting (README: PyTorch, MIT licence stated on the page, C++ ground segmentation built with CMake, based on an earlier "ff" planner repo).

## Fit to our project

not_suitable (lidar); reuse_idea - S - the hit-probability chain p_i = prod(1 - z_j) z_i is again the survival algebra, and "virtual stop at the observed depth outside the grid" is a published way to give a ray that leaves the volume a defined endpoint (a censoring-like convention).

## Needs

lidar sweeps, voxel traversal. compat_conflicts: no lidar in our pipeline.

## Tried in our experiments

none.
