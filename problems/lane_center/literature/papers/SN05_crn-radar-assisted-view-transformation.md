# SN05 - CRN: radar-assisted view transformation + multi-modal deformable aggregation

*Pass: loop 16 (2026-10-23), cluster `sensors_bev` (camera-radar BEV foundations, sensors, data sets, robustness). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: B.*

## Papers

CRN: Camera Radar Net for Accurate, Robust, Efficient 3D Perception (Kim et al., ICCV 2023; authors / venue from memory) https://arxiv.org/abs/2304.00670

## Idea

Radar pillars projected into the camera frustum give a radar occupancy map that multiplies the image depth distribution before BEV pooling (radar-assisted view transformation, RVT); camera and radar BEV maps are then fused by multi-modal deformable cross-attention (MFA); previous fused BEV maps are aligned and concatenated.

## Formulation

6 past radar sweeps accumulated (normalised RCS, Doppler speed per point; ego-motion handling not described), pillar height 1 (no radar elevation), PointNet + sparse conv; image features at 16x, depth bins 2-58 m at 0.5 m (D = 112; long range 2-116 m); radar occupancy via sigmoid, image context x depth distribution x occupancy; MFA: 6 layers, 8 heads, 4 sampling points, optional top-4096 sparse queries; detection range +-51.2 m, pillar 0.2 m, down-sampling 4 -> 128 x 128; segmentation +-50 m at 0.5 m -> 200 x 200 with a CVT-style conv decoder, focal loss.

## Input

6 images + radar sweeps + (training) lidar for depth supervision.

## Output

CenterPoint-style 3D boxes; BEV vehicle and drivable-area maps.

## Training

regime matching_free for segmentation, heatmap targets for detection; 24 epochs AdamW (2e-4 / 1e-4); depth supervised by projected lidar points; radar sweeps and points randomly dropped as augmentation.

## Differentiable?

yes (top-k sparse query selection is the one hard operation, optional).

## Sensors and data

camera + radar (lidar only for training supervision), nuScenes.

## Reported results

BEV segmentation val (R50, 256 x 704): vehicle 58.8, drivable 82.1, 24.8 FPS. Detection val R50: NDS 56.0 / mAP 49.0 (CRAFT 51.7 / 41.1); frames 1 -> 5: mAP 42.9 -> 48.4 (Table 12; orientation error unstable); radar removed entirely: -5.6 % mAP relative. Failure modes: rare classes with few radar returns, clutter, no elevation.

## Compute

20.4 FPS (R50, detection).

## Code

https://github.com/youngskkim/CRN (URL from memory, not opened); GaussianCaR notes that CRN has no released segmentation code.

## Fit to our project

baseline - effort L - already compared as a detection-pretrained foundation (E27: F1@1 0.624 vs 0.867); radar occupancy times image depth is the cleanest "radar steers the lift" idea.

## Needs

lidar-supervised depth, image features, radar pillars.

## Conflicts

detection-pretrained features carry object evidence, not lane geometry (matches E27); cannot be plugged onto a cached raster.

## Tried in our experiments

E27 (CRN detection-pretrained vs GaussianCaR: 0.624 vs 0.867 F1@1 -> GaussianCaR kept).
