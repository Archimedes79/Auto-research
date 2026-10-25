# RB05 - RAF: reliability map for the camera branch, supervised by pseudo labels from other sensors

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: RAF: Reliability-Aware Fusion of Camera, LiDAR, and 4D RADAR for Robust 3D Object Detection in Adverse Weather (Park, Jeong, Yoon, arXiv 2607.04587, July 2026) https://arxiv.org/abs/2607.04587 (opened HTML)
- depth: M
- stage: B, L
- idea: A camera branch is added to a frozen LiDAR + 4D-radar detector. A small conv estimator predicts a per-pixel reliability map that gates camera features before view transformation (keeping a fraction alpha = 0.2 of unreliable features). The map is trained with pseudo labels from feature similarity between projected LiDAR-radar voxel features and image features; a calibration-aware local match (max over 5 x 5 window) tolerates calibration error.
- formulation: total loss = detection + 0.25 similarity + 0.25 reliability; pseudo labels are sparse (projected pixels only); image-level Clean / Noisy labels supervise the projectors.
- input: camera features, projected LiDAR-radar voxel features (training only).
- output: sigmoid reliability map in the camera feature space.
- training: pseudo-label supervision, no matching.
- differentiable: yes (the gate); the pseudo-label generation is training-only.
- sensors_data: K-Radar (Clean 21 / Mixed 28 / Noisy 9 sequences by camera occlusion), VoD in the supplement.
- results: Sedan APBEV: 3D-LRF 64.5 -> 71.0, L4DR 77.5 -> 82.0; fully occluded camera ("Noisy") equals the baseline 59.0 APBEV while naive camera fusion drops. Missing sensors are NOT handled (stated limitation).
- compute: negligible; L4DR 4.29 -> 4.19 FPS, 3D-LRF 6.95 FPS unchanged.
- code: https://github.com/parkie0517/RAF (not opened).
- fit_to_us: reuse_idea - effort M - a cell-wise reliability map is an exact home for our label-free heat monitor; the pseudo-label idea (agreement between two sensors defines reliability) could be used with radar vs camera-derived heat.
- compat_needs: two separable sensor streams for the labels.
- compat_conflicts: our fused raster has no sensor-separate features; labels would need the foundation's intermediate features.
- tried_before: label-free heat monitor (similar goal, not cell-wise supervised).
- verified: checked
