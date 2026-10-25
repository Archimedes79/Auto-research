# SN14 - nuScenes radar facts relevant to lane evidence (sparsity, range, resolution, fields, filters, compensation)

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (paper radar statements; devkit class; review passages on radar limits, first ~100k of ~255k characters)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: D.*

## Papers

nuScenes: A multimodal dataset for autonomous driving (Caesar et al., CVPR 2020; from memory) https://arxiv.org/abs/1903.11027 ; devkit `RadarPointCloud` (data_classes.py) https://github.com/nutonomy/nuscenes-devkit ; Radar-Camera Fusion for Object Detection and Semantic Segmentation in Autonomous Driving: A Comprehensive Review (Yao et al., IEEE T-IV 2023) https://arxiv.org/abs/2304.10410 ; ARS 408-21 datasheet on a reseller page https://platformselect.autonomoustuff.com/product/continental-ars-408-21

## Idea

Five 77 GHz FMCW radars at 13 Hz, up to 250 m range, return sparse, imprecise points from reflecting objects with Doppler velocity; the devkit offers ego-compensated velocities and cluster-quality flags; multi-sweep loading moves past sweeps into a reference frame using ego poses.

## Formulation

18 fields (x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp, is_quality_valid, ambig_state, x_rms, y_rms, invalid_state, pdh0, vx_rms, vy_rms); default filters invalid_state = 0, dyn_prop 0-6, ambig_state = 3; `from_file_multisweep` defaults nsweeps 5, min_distance 1 m, applies ref_from_car * car_from_global * global_from_car * car_from_current to each past sweep and stores a time lag; vx_comp / vy_comp are compensated for ego motion (only ego motion: moving objects stay displaced between sweeps).

## Input

radar sweeps.

## Output

point sets.

## Training

none.

## Differentiable?

no - data.

## Sensors and data

nuScenes; range "<= 250 m" in the sensor table; the paper says radar returns are even sparser than lidar and less precise in localisation, annotated objects hold up to 40 radar returns at 10 m and 10 at 50 m; FoV and angular resolution not stated on the nuScenes page. The unit is named ARS 408-21 by a third-party paper seen in a search result (probably arXiv 2203.08038); the nuScenes pages read do not name the model. Datasheet of that unit as reproduced on a reseller page: far mode to 250 m, azimuth +-9 deg, range resolution up to 1.79 m, azimuth resolution 1.6 deg; near mode to 70 m (100 m within +-45 deg, 20 m at +-60 deg), azimuth +-60 deg, range resolution 0.39 m, azimuth resolution 3.2 deg at 0, 4.5 deg at +-45, 12.3 deg at +-60. Derived here: lateral resolution = range x azimuth resolution is about 1.3 m at 24 m and 2.8 m at 50 m in near mode (1.4 m at 50 m in far mode, but only inside +-9 deg) - a lane is typically about 3.5 m wide (from memory), so single radar points cannot place a centre line. Review (Yao et al.): 3D radar has no height, clutter and multipath noise, "a dozen points on a car", weak on stationary obstacles, only radial Doppler.

## Reported results

none (data facts). From other cards: radar has no usable elevation (CRN pillar height 1); one sweep is too sparse (Simple-BEV 53.1 vs 55.7 IoU with 3 sweeps); GaussianCaR 1 / 4 / 7 sweeps 54.9 / 56.2 / 57.3 vehicle IoU.

## Compute

n/a.

## Code

devkit (licence not checked).

## Fit to our project

reuse_idea - effort S - radar returns come from reflecting structure (vehicles, guardrails, signs), not from road paint; evidence for lane geometry is indirect (road-edge reflectors, vehicle motion), so rays that run 24 m ahead along unmarked or empty road get little radar support; the camera path carries far lanes.

## Needs

raw radar is not in the cache (only fused features).

## Conflicts

moving targets smear across sweeps; clutter and ghost returns (CRN limitations).

## Tried in our experiments

none.
