# PL17 - SUPER-AD (BEV-segmentation uncertainty re-weights and filters diffusion proposals)

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10, L.*

## Papers

SUPER-AD: Semantic Uncertainty-aware Planning for End-to-End Robust Autonomous Driving (arXiv 2511.22865; authors not read) https://arxiv.org/abs/2511.22865

## Idea

Aleatoric uncertainty on the BEV segmentation head: per pixel and class a Gaussian logit (mean, std); the Monte-Carlo mean over T reparameterised samples gives probabilities, drivable classes are summed and a binary entropy is computed. A DiffusionDrive decoder proposes trajectories; each is scored by the minimum pixel safety along its path, which re-weights selection; trajectories crossing non-drivable pixels are discarded. A lane-following regulariser adds supervision.

## Formulation

loss = -log of the average softmax over T samples of mu + sigma * eps; S_safe = p_pos (1 - H) + 0.5 H; lane loss = intent loss (L1 between a predicted per-point "follow lane" mask and an expert mask) + centre loss (distance of flagged points to the nearest lane-centre pixel of the BEV segmentation).

## Input

BEV segmentation logits with std, DiffusionDrive proposals.

## Output

selected trajectory.

## Training

end-to-end with DiffusionDrive; weights not read.

## Differentiable?

partial - proposal weighting is soft, the exclusion of non-drivable crossings is hard.

## Sensors and data

NAVSIM, camera only.

## Reported results

navtest PDMS 87.7 (NC 98.1, DAC 97.0, EP 81.5, TTC 93.9); EPDMS navtest 84.26 vs 83.96 (DiffusionDrive camera + lidar); navhard + navsafe EPDMS 69.74 vs 66.24; ablation: BEV encoder 59.70, vector-map uncertainty 59.02, segmentation uncertainty 69.19, full 69.74; lane regularisation 66.55 -> 67.78 (text and table disagree on which uncertainty type is better; the table is used here).

## Compute

not read.

## Code

none given.

## Fit to our project

reuse_idea - effort S - a graph validity could weight path safety, and the drivable probability raster exists in our cached map-layer logits.

## Needs

raster of drivable probability along candidate paths.

## Conflicts

relies on BEV segmentation, not on a graph.

## Tried in our experiments

none.
