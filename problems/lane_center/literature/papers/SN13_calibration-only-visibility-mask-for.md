# SN13 - Calibration-only visibility mask for censoring rays (derived card, not a paper)

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M (ingredients); the recipe itself is not from a paper** (A abstract, M method, C code page). Verified: partly (ingredients checked, recipe derived here; calibration values from a third-party tutorial page). Pipeline stage: D, L, S3.*

## Papers

ingredients seen in: BEVFormer (BEV pillar points projected to cameras, "hit" views) https://arxiv.org/abs/2203.17270 ; Simple-BEV ("valid" mask of voxels inside each camera frustum) https://arxiv.org/abs/2206.07959 ; BEVCar (voxel assigned to one or two cameras by field of view) https://arxiv.org/abs/2403.11761 ; PON (FoV AND lidar ray) SN10 ; Occ3D SN11 ; GaussianCaR config (depth 1 - 61 m, image crop) SN01 ; nuScenes paper (camera FoV). The recipe is derived here.

## Idea

A BEV cell (or ray sample) is "observed by the camera path" if its ground point projects into at least one image with usable depth and resolution. Samples that fail the test are right-censored in the survival product: no lane-end claim may be made beyond the last observed sample.

## Formulation

(derived here) for each cell centre p = (x, y, z0) in the ego frame and camera c: p_c = T(cam<-ego) p; require depth z_c in [d_min, d_max] (GaussianCaR lifting range [1, 61] m); u, v = K p_c / z_c with the SAME resize / crop as the foundation saw (ida matrix times K; top crop), require inside the image with a margin; require a ground sampling distance (metres of ground per pixel along the viewing ray) below a limit g_max: for a level camera at height h above flat ground and focal length f in pixels, GSD(d) = d^2 / (f h) (derived). With CAM_FRONT values printed in a third-party nuScenes tutorial (fx = fy = 1266.4 px, cx 816.3, cy 491.5 on 1600 x 900 images, camera 1.54 m above the ego origin) this gives about 0.30 m at 24 m, 1.3 m at 50 m, 1.9 m at 61 m at full resolution, twice as large at half resolution (0.6 / 2.6 / 3.8 m); near-field blind ring: ground closer than h / tan(vFOV / 2) = about 4.3 m is outside a level camera (vFOV = 2 atan(450 / 1266) = 39 deg; pitch ignored; derived). The ego-origin height equals the ground height only approximately. Horizontal FoV: five cameras 70 deg offset by 55 deg, rear 110 deg (nuScenes paper), so no azimuth gaps beyond the near field. Soft form: V(x, y) = max over cameras of a product of smooth border and GSD weights. Occlusion by vehicles is NOT in this mask: at training time add shadow polygons cast from the camera centre over the footprints of annotated 3-D boxes, or use a lidar-ray mask as PON does; at test time a learned visibility head could predict it (PON-style entropy term on the rest).

## Input

intrinsics, extrinsics, image crop, ego frame; optional GT boxes.

## Output

V(x, y) in [0, 1] on the 200 x 200 grid (fixed, precomputable per frame from calibration).

## Training

none (fixed mask); used as a multiplier / censoring flag in the survival loss.

## Differentiable?

no - geometric rule, a fixed tensor on the loss path (no gradient needed).

## Sensors and data

any calibrated camera rig; radar adds no ground observation (SN14).

## Reported results

none (not tested). The foundation's own coverage can be read off by rendering its Gaussians with all-ones features (SN01).

## Compute

negligible (precomputable per frame).

## Code

none.

## Fit to our project

reimplement - effort S - sample the mask with the same `grid_sample` as the features; choose g_max on VAL by checking where the foundation's lane-logit reliability breaks down against GSD.

## Needs

the foundation's exact crop / resize parameters; boxes only for the occlusion extension.

## Conflicts

flat-ground assumption; slopes and bridges break the GSD formula.

## Tried in our experiments

none.
