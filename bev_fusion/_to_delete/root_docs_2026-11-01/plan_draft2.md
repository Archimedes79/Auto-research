# Front-camera + radar BEV: plan for discussion (draft 2, 2026-10-25)

Status: DRAFT for the discussion with Mirko; nothing is decided. Once agreed, each stage of section 10 becomes an experiment card (experiments/E<id>_*/experiment.md) and the open points become DECISIONS entries.
Literature: the F4 scan of 2026-10-25 (60 papers, literature/papers.csv axis A1 / A5); ids below are arXiv ids, notes in literature/papers/.
Numbers marked (calib) come from `temp/loop19/cam_front_geometry.py` on the nuScenes calibration; (lit) from a paper note; (est) are my estimates, to be measured.

## 0. Goal and boundaries
- A small, very fast BEV system from the FRONT camera (CAM_FRONT) plus radar: semantic segmentation (map layers, lane markings, objects) on a 0.5 m grid, temporal fusion that separates the stationary environment from moving targets, later detection.
- Separate from the lane-graph product (GaussianCaR foundation untouched); its BEV could later feed the lane-graph heads.
- One RTX 3060 12 GB shared with the main project; numbers through evaluate.py and run.py as in the main project.
- Design rules: fully differentiable training path; no learned view transform where geometry is known; everything fixed is precomputed at startup; every uncertainty that geometry gives is computed, not learned.

## 1. Data
| item | status |
|---|---|
| CAM_FRONT keyframes (2 Hz, annotated) | on disk, 20,418 frames, 510 scenes (parts 1-6) |
| camera sweeps, all six cameras (12 Hz) | download started 2026-10-25 15:34: parts 1-6, 97.9 GB from `motional-nuscenes` (`temp/loop19/get_camera_blobs.sh`, ETag check, unpack to `~/data/nuscenes_full`) |
| radar samples + sweeps (~13 Hz, 5 radars) | on disk, parts 1-6 |
| LIDAR_TOP | not on disk (parts 1-6: 80.2 GB) - open decision, only for depth supervision |
| parts 7-10 (340 more scenes) | not on disk: camera 79.4 GB + radar 0.8 GB - open decision |
| map expansion, ego poses, calibration, boxes | on disk |

CAM_FRONT geometry (calib): fx 1253-1266 px at 1600 x 900, mounting height 1.49-1.54 m, optical-axis elevation -0.32 .. +0.78 deg; 850 calibrations (one per scene), so the mounting pitch varies by about 1 deg between scenes.

## 2. Vision front end
- Backbone, ImageNet-pretrained, fine-tuned with a low learning rate and frozen BN: ResNet-34 as the default; ResNet-18, RegNetY-800MF, EfficientNet-B0 as speed variants; DINOv2-S as a quality variant (2501.08118: +7.4 vehicle IoU in LSS (lit), latency unknown). Choice by our own batch-1 timing on the 3060.
- Neck: FPN to one output map, C = 64-96 channels.
- Resolution is set by the ground geometry, not by habit (calib):

| range | ground metres per image row (full res) | image px per 0.5 m laterally (full res) |
|---|---|---|
| 10 m | 0.05 | 63 |
| 20 m | 0.21 | 31.5 |
| 35 m | 0.65 | 18.0 |
| 50 m | 1.32 | 12.6 |
| 60 m | 1.90 | 10.5 |

  - Laterally, a half-resolution image (800 px wide) with stride-8 features gives about one feature column per 0.5 m cell at 50 m - enough.
  - Along the range, the image is the bottleneck: at stride 8 on half resolution one feature row covers 16 full-res rows = 3.4 m of ground at 20 m and 21 m at 50 m. A stride-8 map alone cannot place lane markings beyond about 25 m.
  - Proposal (to discuss): a two-band input - the full image at half resolution (stride 8) plus a horizon band (the rows that see 20-60 m, about 60-80 full-res rows (est)) at full resolution with stride 4 vertically; or one anisotropic neck (stride 8 across, stride 4 down) over the lower image half.
- Geometry-aware features: append the ray direction of each feature pixel in the virtual camera (azimuth, elevation) as two channels, so the depth head knows where it looks.
- Augmentation: photometric, horizontal flip (with the LUT mirrored), small random pitch / roll / yaw applied through the virtual-camera homography (section 3.1), so augmentation and geometry stay consistent.

## 3. BEV transform (fixed geometry, learned distribution, explicit uncertainty)

### 3.1 Virtual upright camera
- A rotation about the camera centre is an exact homography: H_t = K_v R_t K^-1 with R_t = R_dyn(t) R_mount (per scene from the calibration, per frame from the dynamic pitch / roll). It needs no depth.
- The virtual camera has zero roll, a fixed nominal pitch and fixed intrinsics K_v for ALL scenes. Every image is warped once into it (grid_sample on the image before the backbone, about 0.1 ms (est)).
- Consequences: one lookup table for the whole dataset (instead of 850 calibrations); in the virtual camera every column is a vertical plane, so BEV position depends only on (column, horizontal range); roll and pitch, static and dynamic, are handled exactly.
- Dynamic pitch / roll source (to discuss): the ego pose of the camera timestamp relative to a road-plane estimate, a learned per-frame horizon offset (BEV-LaneDet 2210.06006 learns its virtual camera), or both; the pitch residual sigma_theta that remains enters 3.3.

### 3.2 Column-polar lookup table (built once at startup)
- Column c of the virtual camera <-> azimuth phi_c; range bin k <-> horizontal range r_k; cell = r_k (cos phi_c, sin phi_c) + camera offset.
- Range bins: K = 64-96, uniform in inverse range (matches the image, whose range resolution falls with r^2) between 2 and 60 m.
- For every (c, k) the table holds the BEV cells and weights of its footprint: laterally r_k * dphi (several cells at far range, one at near range), along the range the bin width. As a sparse matrix M of size (C_cols * K) x (H * W); BEV = M^T X with X the (C_cols * K) x C feature-mass tensor. Fast-BEV measured a table lift at 0.006 ms (lit, 2301.12511); MatrixVT does the same as one matmul (2211.10593).
- Normalisation: each cell also receives the summed weight (mass channel), so the BEV feature is a weighted mean plus its support.

### 3.3 Along-ray distribution per pixel, collapsed per column
- Two modes per pixel (v, c), mixed with a learned ground probability p_g(v, c):
  - ground mode: a pixel below the horizon hits the road at r_g(v) = h / tan(elevation of row v) - exact for a flat road. Its range spread is computed: sigma_g^2 = (r_g^2 sigma_theta / h)^2 + (r_g^2 / (f h) * dv)^2 (pitch error plus row quantisation). It carries road surface and lane markings.
  - free mode: a categorical over the K range bins (softmax per pixel), for everything above the ground (vehicles, poles, walls).
- Because position depends only on (c, r), the row is summed out before the splat: X[c, k] = sum_v f(v, c) * ( p_g N(r_k; r_g(v), sigma_g) + (1 - p_g) p_k(v, c) ). This is one batched contraction over v, then the fixed sparse splat - no per-pixel scatter.
- Height (optional): split the sum into 2-3 height slices (ground / 0-2 m / above, from the row's height at range r_k) and keep them as separate channel groups.
- Explicit distance uncertainty: the mass channel, the per-column expected range and its variance (from the mixture) are splatted with the features, so the BEV knows how sharp each cell's evidence is.
- Gaussian-footprint variant: quantise the predicted variance to 3-4 levels and precompute one table per level; the soft level weights choose the footprint width - Gaussian-primitive behaviour (GaussianLSS 2504.01957: variance sets the splat footprint, 11.2 ms for six cameras on an RTX 4090 (lit)) without a rasteriser.
- Ablation arms: L0 uniform along the ray (Fast-BEV, null), L1 categorical only (LSS / MatrixVT), L2 ground + categorical (ours), L3 L2 with variance-level footprints.

## 4. Radar: point encoding and a learned polar distribution per sensor
Block diagrams of the whole system, the camera lift and this radar branch: shown in the chat of 2026-10-25 (to be saved as the architecture drawing of the first card).
- Input per detection (nuScenes radar point cloud): position (x, y), range r, azimuth phi (in the sensor and in the ego frame), sensor id and mounting yaw (sin / cos), RCS, Doppler: radial velocity v_r (ego-compensated, `vx_comp`, `vy_comp` projected on the line of sight), `dyn_prop`, `ambig_state`, `invalid_state` (as features, not filters - to discuss), measured spreads `x_rms`, `y_rms`, `vx_rms`, `vy_rms` (meaning to be confirmed on the data), age dt = t_ref - t_meas (section 6).
- Derived features (fixed, not learned): static residual r_s = v_r - v_static(u) with v_static = -(v_ego + omega_ego x p) . u (wrapped into the unambiguous interval, or as sin / cos of the phase), unit line-of-sight vector u.
- Point MLP (shared, about 3 layers, 64 channels) to an embedding e_i; optional local attention among points within 3 m of the same window (a few hundred points per frame (est), cheap).
- Measurement covariance in the sensor's polar frame: sigma_r (range) and sigma_phi (azimuth, from x_rms / y_rms or a learned per-sensor constant), so the Cartesian spread grows laterally with r.
- Learned polar distribution, centred on the sensor (direction proposed by Mirko 2026-10-25, replaces the sigma points of draft 2): every radar has its own polar grid (range r, azimuth phi in the sensor frame; resolution to discuss, e.g. 0.25-0.5 m x 0.5-1 deg). Per detection i a head predicts a categorical p_i over a local window of that grid (e.g. 9 range x 9 azimuth bins, to discuss) centred on the measured (r_i, phi_i), from the embedding e_i. The window lives in sensor polar coordinates, so the learned spread is a range / azimuth spread by construction and its Cartesian width grows with range.
  - prior: a KL term towards the sensor's measurement-noise Gaussian in polar (from the rms fields or a learned per-sensor sigma_r, sigma_phi), weight annealed, so p_i starts at the physical uncertainty and moves away only where the data asks (radar hits the front of a car, the car extends behind it)
  - supervision: end to end through the BEV losses; optionally an auxiliary likelihood of the GT object footprint under p_i for detections inside or near a box
  - accumulation: polar feature grid F_s[r, phi] += p_i e_i per sensor s; a fixed table per sensor, built at startup, maps the polar grid to BEV (the radar analogue of the camera table in 3.2); the summed weight is kept as a support channel
  - parametric alternative: a mixture of 2-3 Gaussians in (r, phi) per detection, evaluated on the window (fewer outputs, smoother)
  - ablation: R0 point at mu (pillar, RC-BEVFusion 2305.15883), R1 fixed measurement Gaussian in polar (equivalently unscented sigma points), R2 learned categorical polar window, R3 learned polar mixture
- Doppler channels, fixed: A = sum p u u^T (3 values), b = sum p v_r u (2 values), accumulated in a polar Doppler grid with the same weights p_i and mapped by the same table; v = A^-1 b where A is well conditioned, the eigenvalues of A as the observability. A single return measures only v . u; A and b add up the constraints from several returns and sweeps.
- Counts and RCS statistics per cell (sum of weights, mean / max RCS) as extra channels.

## 5. Fusion (radar must not be dominated by vision)
- Per cell: camera BEV features + camera confidence (mass, range variance) and radar BEV features + radar confidence (weight sum, A eigenvalues, age); gate g = sigmoid(conv([F_c, F_r, conf_c, conf_r])); fused = conv([g F_c, (1 - g) F_r]) or two gates (not forced to sum to 1).
- Early fusion variant (second stage): radar depth as a prior for the camera's free mode (radar points projected into the virtual camera, a depth hint per column), as in CRN / RaCFormer 2412.12725 (lit: RCS embedding in the depth head +0.8 NDS).
- Training against vision dominance: radar-only auxiliary heads on the radar BEV (same targets, training only); modality dropout (camera off / radar off each with p = 0.15-0.2 (to discuss)); stage-wise training (radar branch pre-trained alone, section 10); camera-off and radar-off drops as standing metrics.

## 6. Sensor timing
- nuScenes timing: every sample_data (each camera image, each radar sweep) has its own timestamp and its own ego pose; cameras run at 12 Hz, the five radars at about 13 Hz, unsynchronised; keyframe annotations are at the lidar time, and the camera of a keyframe is triggered when the lidar sweeps across it, so even at keyframes camera time and annotation time differ by tens of ms (est).
- Reference time: t_ref = the timestamp of the current front-camera image; the output BEV is the ego frame at t_ref.
- Static content: exact compensation T_ego(t_ref)^-1 T_ego(t_meas) for every radar sweep and for the previous BEV state.
- Moving content: the radial part of the motion is known from Doppler, the tangential part is not. Per radar point: shift by v_r u dt (radial) and inflate the tangential covariance by (v_t,max dt)^2 weighted by the moving probability from r_s - static returns are not inflated, moving ones grow with their age. In the polar distribution of section 4 this inflation widens the azimuth prior (the tangential direction) of moving returns with dt. dt is also an input feature.
- Radar window per camera frame: all sweeps of the five radars since the previous camera frame (about one per radar at 12 Hz camera rate (est)), plus the radar memory (section 8).
- Labels at the camera time: map labels are exact at any time (static map, ego pose of the image); boxes are interpolated between the annotated keyframes by instance (position, yaw, size) to the camera timestamp; interpolated labels for training only, evaluation only on keyframes at the matched time.
- Inference: event-driven - radar sweeps update the radar memory when they arrive, the camera frame triggers the fused update and the output.

## 7. BEV network (convolutions)
- Grid (to discuss): x 0..60 m, y -26..+26 m, 0.5 m -> 120 x 104 cells.
- Coordinate channels appended to the fused input: x, y, range, azimuth, in-FOV mask (uncertainty and resolution depend on range, the network must know where it is).
- Trunk: ResNet-style 2D encoder 0.5 m (C 64) -> 1 m (128) -> 2 m (256), FPN back to 0.5 m; about 3-5 M parameters; the grid has only 12.5 k cells, so the trunk should cost a few ms (est, to measure).
- The temporal state (section 8) sits between fusion and trunk at 0.5 m.

## 8. Temporal fusion: ego motion, stationary environment, moving targets
- Ego motion: the previous state is warped into the ego frame at t_ref with SE(2) from the ego poses (grid_sample, differentiable); cells that enter the FOV are marked as new (age channel 0); odometry-noise augmentation in training (pose jitter).
- Moving probability per cell m(x): from the radar static residual and learned from the state; supervised by the GT (cells of boxes with speed > 0.5 m/s).
- Stationary memory S_s (long, seconds): S_s,t = ConvGRU(warp(S_s,t-1), obs_t) with an update gate that sees the observation confidence (camera mass and variance, radar static weight) and (1 - m); not observed or occluded cells keep their memory with a slow decay. Lane markings and road edges accumulate here; far-range lanes become sharp because each cell is seen several times at shrinking range as the car approaches - this is where temporal fusion fixes the range-resolution problem of section 2.
- Moving memory S_d (short, about 0.5 s): before the update, warp by ego motion AND by a per-cell flow v dt (flow from A^-1 b where observable, otherwise from a flow head on the state, supervised by GT box velocities); S_d,t = ConvGRU(flowwarp(warp(S_d,t-1)), obs_t). Control: the same memory without the flow warp (BEVDet4D-style, 2203.17054).
- Readout: [S_s, S_d, m] into the trunk; the heads see both.
- Training: sequences of T = 4-8 frames at 12 Hz (or stride 2-3), truncated backprop through time (older state detached), loss on every frame with labels (map labels at every frame, boxes interpolated); a time-gap embedding (VideoBEV 2303.05970) so variable frame gaps are allowed.
- Literature caveats (lit): ego-align alone can be no better than no history (4DR360: 47.12 vs 47.45 mAP), a learned state adds more; learned moving-object warps gave little so far (RLG-TPV +0.004 NDS) - so the flow warp must earn its place against the control.

## 9. Semantic head and ground truth
- Head (pixel-based, multi-label sigmoid, 0.5 m), classes (to discuss):
  - areas: drivable area, ped crossing, walkway, stop line, car park
  - lines: lane divider, road divider / boundary, lane centre-line (link to the main project)
  - objects: vehicle (car, truck, bus, trailer, construction), pedestrian, two-wheeler
  - auxiliary: moving mask m, flow (vx, vy) of moving cells
- Losses: focal / BCE for areas and objects; for lines BCE + Dice plus a 2D signed (truncated, 2 m) distance field target - a cheap BEV SDF that suits thin structures (instead of the 3D SDF, which the scan did not support for lanes); uncertainty weighting between groups (1705.07115) with a clamped log-variance.
- Ground truth (CPU, one cache per frame, keyframes now, sweeps after the download):
  - areas and lines from the map expansion with the ego pose at the camera time (`NuScenesMap.get_map_mask` for polygons; dividers rasterised as 1-cell polylines plus the distance field; centre-lines from the discretised lane arcs)
  - objects: box footprints (rotated rectangles) at the camera time (interpolated, section 6), with velocity
  - masks: front FOV from the virtual camera; range mask; occlusion by ray casting from the camera through the GT box footprints in BEV (cells behind a box are "occluded": ignored for objects, flagged for map classes); object visibility token (> 40 %, as CVT)
  - self-test: re-project the labels into the (virtual) image and overlay; one hand-checked patch
- Metrics: IoU per class in the FOV mask, per range bin (0-20, 20-35, 35-60 m), visible vs occluded; line classes also by a distance-tolerant F1 (0.5 / 1.0 m); ms per frame by module; camera-off / radar-off drops.

## 10. Stages (each one card with prediction, rule, kill)
| stage | content | control | kill |
|---|---|---|---|
| S0 | GT cache, masks, metric code, timing harness, virtual camera + LUT with self-tests | overlay self-test, null (per-cell training mean) | - |
| S1 | camera only, single frame: front end + lift L0-L3 + seg head | L0, L1 | L2 not above L1 by the band |
| S2 | radar only: encoder R0-R2 + aux heads, single window vs radar memory | R0 | R2 not above R1; memory not above single window |
| S3 | fusion, modality dropout | S1, S2 | fusion not above S1 |
| S4 | temporal: ego warp, then static memory, then moving memory with flow | S3 single frame; memory without flow | no gain beyond the band |
| S5 | detection head (CenterPoint 2006.11275, 1 m grid) on the shared trunk | S4 seg only | seg drop beyond the band |
| S6 | optional: height head (FlashOcc 2311.12058) + depth rendering loss (needs lidar) | S5 | no gain |

## 11. Budget (est, to be measured in S0)
- S1 arm: ResNet-34 at 800 x 288 (+ horizon band), batch 8, 0.5 m grid, about 20 epochs on ~17 k frames - several hours on the 3060; with 12 Hz sweeps the data grows six-fold, so epochs are defined by keyframes and sweeps are sampled.
- Temporal training multiplies memory by T; truncated BPTT and a frozen or cached backbone for the first temporal experiments.
- Speed target (to discuss): < 15 ms per frame end to end, batch 1, idle GPU.

## 12. Open questions for the discussion
1. Two-band / anisotropic front end, or plain half resolution first?
2. Grid extent (60 m x +-26 m?) and classes.
3. Dynamic pitch: ego pose, learned horizon, or both?
4. Radar: all five radars or those that see the front cone; nuScenes state flags as features or filters; polar grid resolution and window size of the learned distribution (categorical window or Gaussian mixture)?
5. Fusion: late gate only, or also radar depth into the camera free mode?
6. Temporal: frame rate in training (12 Hz vs 4-6 Hz), sequence length, static memory decay.
7. Lidar (80 GB) for depth supervision, parts 7-10 (80 GB) for more scenes?
8. Speed target and the split (255 or 510 scenes).
