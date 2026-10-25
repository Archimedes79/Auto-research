# PL15 - Producing and Leveraging Online Map Uncertainty in Trajectory Prediction

*Pass: loop 16 (2026-10-23), cluster `planner_nav` (planner interface, navigation, map matching, end-to-end driving). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S10, E.*

## Papers

Producing and Leveraging Online Map Uncertainty in Trajectory Prediction (Gu et al., CVPR 2024) https://arxiv.org/abs/2403.16439

## Idea

Online mapping heads (MapTR, MapTRv2 incl. a centre-line variant, StreamMapNet) output a Laplace location and scale per vertex; predictors (HiVT, DenseTNT) take (mu, b, class probabilities) as vertex features. It provides a ready nuScenes pipeline that scores a predicted centre-line map by downstream prediction.

## Formulation

vertex (x, y) = two independent Laplace; loss sum of log(2b) + |v - mu| / b (regression weight 0.03 for MapTR / MapTRv2, 2 for StreamMapNet); std = sqrt 2 b; coordinates ego-relative, scene rotated to heading "up"; DenseTNT (VectorNet lane encoder) concatenates uncertainty to the vertex vector; HiVT changes from a vector set to a point set and doubles the node dimension; no resampling or ordering described.

## Input

online vector map (vertices with scale and class probabilities), agent histories.

## Output

agent trajectories (3 s from 2 s of history).

## Training

staged; no end-to-end mapping + prediction (the "up to 50 %" claim concerns faster convergence of the predictor, e.g. DenseTNT 88 -> 44 epochs; the table label -22 % for the centre-line row does not match 99 -> 7 epochs).

## Differentiable?

yes inside each stage; no gradient from prediction to mapping.

## Sensors and data

nuScenes (500 / 200 / 150 scenes), 2 Hz upsampled to 10 Hz; mAP at Chamfer 0.5 / 1.0 / 1.5 m.

## Reported results

minADE / minFDE / MR without -> with uncertainty: HiVT + MapTR 0.4015 / 0.8418 / 0.0981 -> 0.3854 / 0.7909 / 0.0834; HiVT + MapTRv2-centre-line 0.3790 / 0.7822 / 0.0853 -> 0.3727 / 0.7492 / 0.0726; DenseTNT + MapTRv2-centre-line 0.8466 / 1.345 / 0.1520 -> 0.8135 / 1.311 / 0.1593 (MR slightly worse); DenseTNT with the GT HD map 0.8809 / 1.489 / 0.1903, worse than the online centre-line map without uncertainty. Largest MR gain -15 %; several DenseTNT cells get worse. Calibration not measured (future work).

## Compute

not stated.

## Code

https://github.com/alfredgu001324/MapUncertaintyPrediction

## Fit to our project

baseline - effort M - our graph pieces can be written as centre-line instances and scored by HiVT / DenseTNT on nuScenes, but the stack (trajdata, mmdet3d-based mapping code) is heavy.

## Needs

polylines with class probabilities and per-vertex scale, ego-relative frame.

## Conflicts

fixed point count per instance in the original models; HiVT / DenseTNT use no topology edges.

## Tried in our experiments

none.
