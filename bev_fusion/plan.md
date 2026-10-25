# Camera + radar BEV system - plan for discussion (draft 3, 2026-10-25)

Status: DRAFT for the discussion with Mirko; nothing is decided. Draft 2 (front camera, virtual upright camera, sigma points) is in `plan_draft2.md`; this draft replaces it after Mirko's comments of 2026-10-25:
- the problem is the real world, nuScenes is only one dataset: the design takes the sensor set, the calibration and the timestamps as inputs, never as constants of one dataset
- radar: a learned distribution per detection, f(x, y; detection, r(x, y; calibration), phi(x, y; calibration), time), not sigma points
- camera: frustums (rays of pixel groups) in the perspective space instead of a virtual upright camera
- every block founded on papers and their code; the ground truth observable; a test for every step

Paper ids are arXiv ids; "(lit)" = from a scout note in `literature/papers/` (all ids checked on their abstract pages, 2026-10-25). Code licences from `literature/code/front_bev_code_sources.md` (scout, read on the repository pages).

## 0. The problem, not the dataset
- Product: a BEV environment model for a vehicle with any number of cameras and radars at any mounting: semantic map layers (road, markings, lanes), objects, later detection, at 0.5 m, fast.
- Input contract (the only thing the system knows about a sensor): its model (pinhole / fisheye intrinsics, radar field of view and resolution), its pose in vehicle coordinates (calibration), the timestamp of each measurement, and the ego motion. Nothing is tuned to one calibration.
- nuScenes is the first training and validation set (front camera + five radars as the first rig; all six cameras are on disk once the download finishes). A second dataset or a changed rig is the test that the design does not depend on nuScenes.

## 1. One principle for every sensor
Every measurement becomes a probability mass over world positions, written in the sensor's own measurement space (where its uncertainty is simple), mapped by the sensor's model and calibration into vehicle coordinates, evaluated at the BEV cells it can reach, and added into one BEV grid. The measurement spaces differ (Mirko, 2026-10-25: a camera projection is not a polar system):
- camera: a perspective (projective) space - image coordinates (u, v) plus depth z along the optical axis; the model is the projection (pinhole: x = K [R|t] X; fisheye: its own unprojection). A pixel group is a rectangle in the image, so its preimage is a rectangular frustum (pyramid) from the camera centre, not a circular cone, and its angular size changes across the image (smaller towards the edges for a pinhole).
- radar: a polar space - range r, azimuth phi (elevation unknown for 2D radars), plus Doppler; the model is the polar-to-Cartesian map of the sensor pose.

| | measurement space | what the sensor knows well | what it does not know | what is learned |
|---|---|---|---|---|
| camera pixel group | perspective: (u, v) region x depth z | the image region (a narrow frustum) | the depth along it | the distribution over depth |
| radar detection | polar: (r, phi) + Doppler | the range (and radial velocity) | the azimuth (wide), the height | the spread around the detection |

Time is part of the position: each measurement is moved to the output time t_ref before it is evaluated (exact for static content, by Doppler for the radial motion of moving content, otherwise as a wider spread).

```
cameras ──> backbone + FPN ──> pixel-group frustums ──> learned depth distribution ──> frustum-slab splat ──┐
                                                                                                           ├──> fusion ──> temporal state ──> BEV trunk ──> heads
radars ──> time alignment ──> point encoder ──> learned density f(cell; detection, calib, dt) ──> splat ───┘
```
Block diagrams of draft 2 were shown in the chat; the drawing will be redrawn for this draft after the discussion.

## 2. Camera: image backbone + lift through lookup tables, height kept as channels (Mirko, 2026-10-25)

Mirko's proposals are the two standard lift families of the literature; both are built, on the same backbone and the same output interface, and compared.

### 2.1 Common parts
- Image backbone, ImageNet-pretrained (ResNet-34 default; ResNet-18 / EfficientNet-B0 as speed variants), FPN to one feature map at stride 8 of the half-resolution image (decided: 60 m, no horizon strip).
- Output interface of every lift: a voxel feature volume V[x_car, y_car, z_car, C] in vehicle coordinates - 0.5 m cells, x 0..60 m, y -26..26 m, z_car from -1 to 3 m in 8 slices of 0.5 m - then height into channels: reshape to BEV[x, y, Z * C] and a 1 x 1 conv to C ("feat[z_car]"; Simple-BEV 2206.07959 and Fast-BEV 2301.12511 do this; FlashOcc 2311.12058 is the same reshape in the other direction).
- All geometry is a lookup table computed once from the calibration (intrinsics + extrinsics in vehicle coordinates), rebuilt in milliseconds when the calibration changes; the network only reads the table.

### 2.2 Lift P ("pull", backward): world points at elevations, projected into the image
1. For every BEV cell (x, y) and every elevation z_k: the 3D point (x, y, z_k) in vehicle coordinates is projected into the camera with the calibration -> image position (u, v) on the feature map, plus a valid flag (in front of the camera and inside the image).
2. Lookup table: for every voxel (x, y, k) its (u, v) - a sampling grid of size 120 x 104 x 8.
3. Run time: bilinear sampling of the feature map at those positions (grid_sample) -> V[x, y, k, C]; then height into channels.
4. Learned elevations (Mirko): the 8 heights z_k are learnable parameters (initialised at the slice centres); the projection is then recomputed in every forward pass (a 100 k-point matrix product, negligible), so the gradient moves the heights. A later variant: per-cell height offsets predicted by the network (the deformable reference points of BEVFormer 2203.17270, without its attention stack).
- What it does not know: depth - a feature is copied to every voxel on its ray; the stack of heights and the trunk must sort out where along the ray the content is (radar helps most here).
- Basis: OFT 1811.08188 (orthographic feature transform, the first pull lift), Simple-BEV 2206.07959 (parameter-free bilinear sampling at multiple heights, height reshaped into channels; argues that resolution and batch size matter more than the lift; camera + radar 55.8 vs camera 47.6 mean IoU on its README (digest)), Fast-BEV 2301.12511 (the table as a gather: view transform 0.006 ms on CUDA at 256 x 704; uniform fill along the ray), BEVFormer (learned sampling offsets).
- Code: aharley/simple_bev (MIT) - the closest match, readable plain PyTorch; Fast-BEV (licence not stated: ideas only).

### 2.3 Lift S ("splat", forward): each feature pixel along its ray, with Mirko's intermediate coordinates
1. For every feature pixel (u, v) and every depth bin d: the point in camera coordinates is P_cam = z_d * K^-1 [u, v, 1] = (x_n z_d, y_n z_d, z_d) - the "stretched" coordinates (x_cam * z_cam, y_cam * z_cam, z_cam); then P_car = R P_cam + t with the calibration, and its voxel index (ix, iy, iz) in the volume above, plus a valid flag.
2. Lookup table: for every (u, v, d) its voxel index.
3. Run time: the network predicts per feature pixel a feature c (C channels) and a depth distribution alpha over the D bins; each (u, v, d) adds alpha_d * c into its voxel (scatter-add with the precomputed index, no frustum tensor kept: the BEVPoolV2 idea in plain PyTorch); then height into channels.
- Depth bins: D bins over 1-60 m (linear-increasing width, CaDDN 2103.01100); depth supervision possible later (lidar, not downloaded; BEVDepth 2206.10092: the depth loss alone +2.2 mAP) and a camera-aware gate on the depth head (BEVDepth: intrinsics / extrinsics through an MLP as a squeeze-excitation gate - needed for "any rig").
- Basis: LSS 2008.05711 (frustum points, categorical depth, sum pooling; nuScenes: vehicles 32.07 vs OFT 30.05 IoU, drivable 72.94 vs 71.69, lane boundary 19.96 vs 18.07), BEVDepth (camera-aware depth, depth loss), BEVPoolV2 2211.17111 (precomputed indices, no frustum tensor).
- Code: BEVDepth (MIT) for the depth head and its camera-aware gate; LSS (NVIDIA research-only licence: re-implemented, never copied); our own scatter-add with the table.

### 2.4 Lift PD ("pull with depth weight"): keeps the uncertainty and is a pure gather (recommended, Mirko 2026-10-25: "what maintains uncertainty and is fast")
| lift | keeps the distance uncertainty | inference cost | gaps at far range |
|---|---|---|---|
| P (pull, Simple-BEV) | no: the feature is copied to every voxel on its ray | one gather per voxel | none (every voxel samples) |
| S (splat, LSS) | yes: a depth distribution per pixel | scatter-add over pixels x bins (atomics); fast only with a custom kernel (BEVPoolV2 0.82 ms at 640 x 1600, GPU not named) | yes: one voxel per (pixel, bin), far voxels between rays stay empty |
| PD (pull with depth weight) | yes: the same depth distribution as S | two gathers per voxel and a product | none |
1. The image head predicts per feature pixel a feature c (C channels) and a depth distribution over the D bins (categorical, as S; it can be supervised like BEVDepth).
2. Lookup table per voxel (x, y, z_k): its image position (u, v) AND its depth along the optical axis z_cam, expressed as a fractional bin index - pure geometry from the calibration.
3. Run time: V(x, y, z_k) = c(u, v) * p(z_cam | u, v) - bilinear gather of the feature and linear interpolation of the depth probability at the voxel's own depth. Then height into channels.
- The depth distribution is the explicit distance uncertainty: a voxel receives the feature only as far as the pixel believes its content lies at that depth. A multi-modal distribution (road behind a car top) is kept.
- Cost for one camera: 120 x 104 x 8 = 100 k voxels x 64 channels, about 6 M multiply-adds plus the gathers - well below a millisecond (estimate, to be measured), no scatter, no custom CUDA, easy to export (TensorRT has grid sample).
- Learned elevations (Mirko's proposal) work unchanged: the table is then recomputed per forward pass from the learned z_k.
- Parametric variant PD-G, in INVERSE depth (Mirko, 2026-10-25): the depth distribution per pixel as a mixture of 2-3 Gaussians in u = 1/z_cam (e.g. ground, object, far / sky with mu_u near 0). Why 1/z: monocular cues are linear in 1/z (a ground point's row offset below the horizon is f h / z, an object's image size f S / z), so their noise is about constant in 1/z and grows as z^2 in z. Measured on the loaded DualBEV depth head (`temp/setup/check_inverse_depth_std.py`, 40 nuScenes mini front images, median per depth group): std_z 0.19 m at 2-5 m to 9.95 m at 30-40 m (a factor 52 over the groups), std_u 0.0057-0.0116 1/m (a factor 2.0). The 40-60 m group is truncated by the head's last bin at 60 m.
  - voxel weight: the exact mass of the mixture over the voxel's depth extent, Phi((u_hi - mu)/sigma) - Phi((u_lo - mu)/sigma) with u_lo, u_hi = 1/(z + d/2), 1/(z - d/2) precomputed in the table (no Jacobian to handle, no aliasing between bins, cheap: erf per voxel and component)
  - not uniform bins in 1/z over 1-60 m: 64 such bins would spend most of them at 1-3 m and make the last bin about 55 m wide; the mixture avoids the choice; if bins are needed, log-spacing or 1/z-uniform from about 4 m
  - first version needs no training: the moments of the pre-trained categorical give mu_u and sigma_u per cell (unimodal); a small head then learns the mixture (multi-modal: road behind a car top)
  - gains over the categorical: continuous (no 0.5 m quantisation, no cut at 60 m: far mass stays a tail), 3-9 numbers instead of 118, a smooth gradient for learned elevations, and an explicit sigma for the fusion: sigma_z = z^2 sigma_u of the camera against the radar's about constant sigma_r, so camera-near / radar-far follows from the uncertainties
  - basis (from memory, to verify): inverse-depth parametrisation of monocular SLAM (Civera, Davison, Montiel, IEEE T-RO 2008), disparity outputs of self-supervised depth (Monodepth / Monodepth2), inverse-depth hypothesis sampling in multi-view stereo; GaussianLSS 2504.01957 for variance as the footprint.
- Basis (scout of 2026-10-25, ids checked; all published numbers are 3D detection on nuScenes val, none is a segmentation or map result):
  - DualBEV 2403.05402 (ECCV 2024), HeightTrans: fixed heights per cell projected, feature sampled bilinearly and multiplied by the depth probability (trilinear sample) - the closest published form of PD. Sampling operators on a BEVDet base, one RTX 3090: depth-weighted sampling 30.6 mAP / 39.0 NDS at 0.32 ms vs unweighted pull 30.5 / 37.6 at 7.15 ms vs FB-BEV's attention version 29.5 / 36.6 at 12.5 ms. Code Apache-2.0.
  - FastBEV++ 2512.08237: the pure-gather form, Gather(feature) x Gather(depth prob), TensorRT-native without a plugin; R50 0.478 -> 0.488 NDS with the depth term; 134 FPS INT8 on a Tesla T4 (whole network). Code Apache-2.0 (built on BEVDet). The page is partly inconsistent.
  - FB-BEV 2308.02236 (ICCV 2023): the exact weight at a fractional depth by interpolating the two adjacent bins; depth-aware vs not +0.8 to +1.1 NDS; view transform 2.6 ms at 128 x 128 BEV (GPU not named). NVIDIA licence, non-commercial: re-implement, never copy.
  - DFA3D 2307.12972: the depth-weighted aggregation without a frustum tensor, 29 MB and 5.2 ms vs 3204 MB and 161.9 ms for the naive volume (GPU not named); with lidar depth the lift gains +15 mAP (supports the oracle test of 2.6). IDEA licence, non-commercial: re-implement.
  - PointBeV 2312.00703: the depth-free sparse pull on map classes (lane 49.6 IoU), MIT, plain PyTorch - a reference for lift P and for per-camera visibility masks.
  - the pieces: LSS / CaDDN / BEVDepth (depth head), Simple-BEV / Fast-BEV (gather table), GaussianLSS and FisheyeGaussianLift 2511.17210 (variance as the footprint, ray tables for any camera model).
- Gap: no published depth-weighted pull is evaluated on map layers or with a single front camera - our S1 is the first such number.

### 2.5 Which one, and what the literature says
- In the literature the families are close for segmentation (LSS vs OFT: +2.0 vehicles, +1.3 drivable, +1.9 lane boundary for the splat); Simple-BEV argues that the parameter-free pull is enough once resolution, batch size and radar are right.
- Plan: S1 compares PD (recommended), P (no depth: the null of the depth weight) and S (splat: the classic) on the same backbone, trunk, head and budget; PD-G and learned elevations as variants of PD; the cheaper wins a tie.
- Shelved for later (draft 3): the frustum-slab footprint (exact area weights instead of one voxel per point) and the ground hypothesis; they are refinements of S and come back only if S wins and far range is the weak point.

### 2.6 Tests (what it must do -> how we prove it)
| requirement | test |
|---|---|
| the tables are geometrically right | synthetic: a point at a known (x, y, z) in vehicle coordinates; P: its voxel's (u, v) must equal the projection of the point; S: the pixel of its projection at its true depth bin must index its voxel; for every calibration and for random synthetic rigs (roll, pitch, yaw, height, other intrinsics) |
| P, PD and S agree on geometry | a one-hot depth and a feature map that is 1 at one pixel: S's voxel must be the voxel where PD is non-zero, and one of those where P samples that pixel |
| the depth weight is right | PD with a uniform depth distribution must equal P / D; with a one-hot depth at a known bin only the voxels at that depth receive the feature |
| rig independence | move the camera 1 m forward in the calibration and shift the scene: the BEV moves 2 cells, otherwise identical; rotate the camera with a synthetic image rendered for that pose: same BEV |
| the lift can carry the truth (oracle) | S with one-hot depth from a known depth (lidar if downloaded, else rendered synthetic scenes) and the GT semantics of the image: the lifted BEV must reproduce the GT layers; this is the upper bound of the lift |
| learned heights learn | the gradient reaches z_k (non-zero, finite); on a synthetic scene with content at one height the heights move towards it |
| differentiable | torch.autograd.gradcheck on small tables; the gradient reaches the depth head and the backbone |
| fast | ms per frame per lift, batch 1, idle GPU; Fast-BEV's 0.006 ms gather as the reference for P |

## 3. Radar: a learned density per detection (variant; the first version is the simple block of section 12 "Decided": hard raster + properties + 2 convs)

### 3.1 What it does, in plain words (your formulation)
For each detection d with position (x_d, y_d) in vehicle coordinates, its own sensor calibration c_d (radar position and direction in vehicle coordinates) and its time t_d:
1. Time alignment: the sensor pose is expressed at the output time t_ref (ego motion between t_d and t_ref applied to c_d); this moves static detections exactly to where they are at t_ref. dt = t_ref - t_d is kept.
2. Support: the BEV cells around the detection (a window in the sensor's range / azimuth, e.g. +-3 sigma of the sensor's nominal accuracy).
3. For every support cell (x, y): its range r(x, y; c_d) and azimuth phi(x, y; c_d) relative to the (moved) sensor - pure geometry from the calibration.
4. A small network gives the weight of that cell: w_d(x, y) = softmax over the support of g(r(x, y; c_d) - r_d, phi(x, y; c_d) - phi_d, r_d, e_d, dt), with e_d the detection's embedding (RCS, Doppler, static residual, rms fields, sensor type). The detection's feature, possibly modulated by the same inputs, is added to those cells with these weights; the total mass of a detection is 1 (or a learned confidence).
5. Doppler: the radial part of the motion is shifted by v_r dt along the line of sight; the tangential part is unknown, so g gets dt and the moving probability and can widen in azimuth for moving returns. The Doppler constraints (u u^T, v_r u) are added with the same weights into the A, b channels.

The function g works in the sensor's polar frame, so it learns range-like and azimuth-like spreads; because the polar coordinates are computed from the calibration, the same g serves every radar at any mounting.

### 3.2 Paper and code basis
| piece | paper (what we take) | code (licence) |
|---|---|---|
| a learned function of continuous geometric offsets as the kernel that places point features into a grid | Deep Parametric Continuous Convolution 2101.06742 (lit): kernel g = MLP(relative offset), output points may be a dense BEV grid - the formal template; we change the offset to polar coordinates in the sensor frame and add r_d, the features and dt. ContFuse 2012.10992 (lit): the offset must be an MLP input (removing it hurts); one neighbour was as good as three | no code for either: our own implementation |
| compact-support learned kernel (low-parameter alternative to a free MLP) | KPConv 1904.08889 (lit): a few learned lobes with linear fall-off, optionally deformable | KPConv-PyTorch (MIT) |
| dt as a point channel | PointPillars 1812.05784 (lit; its nuScenes config carries a timestamp channel over 10 sweeps), TransCAR 2305.00397 (lit: radar sweeps 0/1/3/5/10 give NDS 42.2/44.3/44.9/46.4/46.3, dt not ablated) | OpenPCDet (Apache-2.0) |
| novelty | no paper found with the sensor pose / mounting as an input or a dt-conditioned density (2026-10-25): the radar density g is our own | - |
| learned footprint beats a fixed Gaussian for radar | Deep radar ISM 2305.12409 (lit: geometric mixture 15.3 vs learned 38.7 % P(occ|occ)), Sless et al. 1904.00415 (lit, nuScenes: Gaussian 0.223 vs U-Net 0.439 mIoU), Weston et al. 1810.08151 (lit: polar variance + KL) | - |
| RCS- / uncertainty-scaled spread, point encoders | RCBEVDet++ 2409.04979 (lit: +1.4 NDS from the RCS-scaled scatter), RC-BEVFusion 2305.15883 (lit: pillar encoder, +17 ms), PointPillars 1812.05784 (lit, the R0 baseline), GaussianCaR's radar branch (lit: radar Gaussians 56.9 vs scatter 56.1 IoU) | CRN, RaCFormer, CenterFusion (MIT); OpenPCDet PointPillars (Apache-2.0); GaussianCaR (Apache-2.0) |
| Doppler as a constraint, static / moving | RadarNet 2007.14366 (lit: radial velocity back-projection), radar static-moving 2511.20003 (lit: sine law of v_r over azimuth) | - |

### 3.3 Tests
| requirement | test |
|---|---|
| geometry right | synthetic radar at a random pose: a detection at (r_d, phi_d) must put its mass centre at the right cell; same detection seen by two radars at different poses -> same cell |
| time alignment right | a static world point measured at several times while the ego moves: after alignment all land in one cell; a target moving radially at known v: shifted by v dt |
| the encoding keeps the information (reconstruction) | from the radar BEV alone, a small decoder must reconstruct the detection cloud: per cell occupancy (IoU against the rasterised cloud), positions (mass centres within the sensor accuracy), Doppler and RCS (linear probe, R^2); compared with a plain rasterisation of the points (R0). If the encoder cannot give its input back, it loses information |
| the learned spread is sane | mass centre stays near the detection, entropy and KL to the sensor prior bounded; plots of g over (dr, dphi) per sensor and per dt |
| rig independence | move a radar in the calibration and transform its detections consistently: the radar BEV is unchanged; drop a radar: only its area changes |
| it helps | radar-only aux heads above the null; fusion above camera-only (section 4) |

## 4. Fusion (after the temporal fusion of each branch, at camera time; Mirko, 2026-10-25)
- Main: concatenation of [camera planes (t_c); radar planes moved to t_c; their mass and dt channels] -> 3 x 3 conv (the BEVFusion ConvFuser style); modality dropout; radar-only auxiliary heads so the radar branch must learn on its own.
- Fallback 1 (proven camera-radar fusion, GaussianCaR, Apache-2.0, `literature/code/gaussiancar_fusion.md`): its FeatureRectifyModule - per-channel and per-cell sigmoid cross-gates between the camera and radar maps (about 30 lines of plain torch) - instead of the plain concat; our mass / dt channels as an extra input of the gate as the ablation.
- Fallback 2 (strong reference, CRN, MIT, `literature/code/crn_fusion.md`): deformable cross-attention between camera and radar BEV (multi-modal feature aggregation), ported with grid_sample.
- Basis: BEVFusion-style concat + conv (lit, SN20 of the lane project), GaussianCaR CMX fusion (lane divider 50.1 / drivable 82.9 IoU on nuScenes val, 75.6 ms whole network on an RTX 4090), CRN (fusion 6.8 of 44 ms); comparison table `literature/code/fusion_blocks_comparison.md`.
- Tests: camera off -> radar-only result, radar off -> camera-only result (no collapse); shuffled radar (from another frame) -> the gain over camera-only disappears (null control); gate maps plotted over range (radar should dominate far, camera near).

## 5. Sensor timing
- Every measurement has its own timestamp and ego pose (nuScenes: per sample_data; a real vehicle: per message). Output time t_ref = the newest camera image (or a fixed output clock).
- Static: exact pose change between t_meas and t_ref. Moving: radial part by Doppler, the rest as uncertainty (3.1). Camera images from several cameras with different times: each camera's frustums use its own pose at its own time.
- Basis: BEVDet4D 2203.17054 (lit: alignment matters, unaligned concat is worse than no history), RadarNet (lit).
- Tests: the time-alignment tests of 3.3; a camera test with a synthetic static scene and a time offset; latency budget per block.

## 6. BEV network
- Grid around the vehicle (front-only first: x 0..60 m, y -26..26 m, 120 x 104 cells; full surround, also to 60 m, once all cameras are used), 0.5 m; coordinate channels (x, y, range, azimuth per sensor coverage) appended.
- Main trunk after the fusion (standard, as the best camera(-radar) BEV papers - BEVDet / BEVDepth / CRN / DualBEV): a ResNet-style BEV encoder with 3 stages (strides 2, 4, 8: 0.5 m -> 1, 2, 4 m; channels 2C, 4C, 8C, BasicBlocks) and an FPN_LSS-style decoder (the two deepest scales upsampled and concatenated, convs, upsampled back to 0.5 m), then the semantic head (section 8). The grid divides cleanly: 120 x 104 -> 60 x 52 -> 30 x 26 -> 15 x 13. Option: initialise it from DualBEV's pre-trained `img_bev_encoder_backbone.` / `img_bev_encoder_neck.` (in our checkpoint, 80 input channels: a 1 x 1 projection of the fused map to 80 channels in front; pre-trained on detection, so an ablation, not a given).
- Fallback (proven camera-radar trunk, GaussianCaR): its 4-stage two-stream fuser (FeatureRectify + cross-attention fusion at each of 4 scales, stride-2 convs + ResBlocks between them) and its DPT-style decoder over the 4 fused scales - fusion inside the pyramid instead of once before it; pure torch except its rasteriser, which we do not need (our inputs are already BEV maps).
- Basis: BEVDet / CRN / DualBEV BEV encoders (CustomResNet + FPN_LSS; `literature/code/dualbev.md`, `crn_fusion.md`), GaussianCaR fuser + DPT head (`gaussiancar_fusion.md`), BEVFusion / CVT decoders (lit), CenterPoint trunk (lit, MIT).
- Tests: overfit one scene to near-perfect IoU (wiring); ms per frame.

## 7. Temporal fusion (ego motion, stationary environment, moving targets)
### 7.1 Main design T2: temporal fusion per branch, one splat function (Mirko, 2026-10-25)
- camera branch, own time axis (frames t_k, 12 Hz): past camera BEVs -> current camera time t_c; radar branch, own time axis (sweeps t_j of all radars): past detections / radar BEVs -> radar time t_r (the newest sweep), radar memory and radar-only aux heads there; then the radar BEV hops from t_r into t_c; finally [camera planes (t_c); radar planes (t_c); mass and dt channels] are concatenated -> fusion convs
- three prediction steps, all with the same splat (ego speed v and yaw rate omega as inputs: a vehicle reports them, for nuScenes `ego_speed_yawrate()` derives them from consecutive poses):
  1. vision: the camera temporal state of the previous camera frame -> t_c, dt_v = t_c - t_c,prev (about 1/12 s)
  2. radar: the radar temporal state of the previous sweep -> t_r, dt_r = t_r - t_r,prev (per radar sweep; or per detection at point level)
  3. radar to vision: the radar temporal state at the newest radar time -> t_c, dt = t_c - t_r,last (either sign)
  each branch then fuses its own current planes with its predicted state (concat + conv; ConvGRU as the variant), the result is the new state of that branch and feeds the next frame (recurrent, frame by frame)
- block diagrams: the splat function and this chain were shown in the chat of 2026-10-25 (to be saved as the architecture drawing of the S4 card)
- fallback for the temporal part (proven): FIERY's ego warp (pull by the SE(2) of the poses, `literature/code/fiery_ego_warp.md`) + a ConvGRU per branch = arm A
- one function for all three moves (`training/splat.py`): v_hat = f_theta(F(x, y)) - a per-cell MLP as 1 x 1 convs with 2 outputs (object velocity, m/s, source axes; last layer zero-initialised, so training starts as pure ego compensation; bounded by v_max tanh); position p' = R(-omega dt) (p + v_hat dt - d), d = (v dt, 1/2 v omega dt^2) - ego speed v and yaw rate omega (what a vehicle reports), translation first order in omega (no division by omega), rotation exact (one cos / sin per frame); dt either sign
  - accuracy (`temp/setup/check_ctrv_taylor.py`, static points up to 60 m): a first-order ROTATION would cost distance x (omega dt)^2 / 2 - chained over camera frames 0.30 m in a city turn (8 m/s, 0.5 rad/s, 0.5 s), 1.24 m in a tight turn; with the exact rotation the rest is of order v omega^2 dt^3 (millimetres per frame); long histories are carried frame by frame
- the splat: forward warp of each source (grid cell or radar detection) to the 4 cells around p', bilinear, normalised, optional importance z (softmax splatting: the more important source wins a collision); mass = sum of the kernel weights is the confidence channel (0 where nothing landed)
- v_hat is supervised with the GT velocity per cell (CRT-Fusion's recipe: box footprints, ground velocity in the source axes, 0 below 1 m/s, loss on object cells) and fine-tuned end to end through the splat (the bilinear splat alone only gives a position gradient within one cell); the radar head sees Doppler and the A, b channels, so it mainly learns the tangential part
- self-test of splat.py: exact 2-cell shift for 1 m of ego motion, landing point = formula (centroid exact) and within v omega^2 dt^3 of the exact constant-turn-rate motion, object velocity, forward-then-back, mass conservation, collision by importance, gradcheck, the velocity head learns, a moving blob lands at x + v dt (without v_hat it lags)
- arms: T2 (main) vs B (7.2, one joint state with pulls) vs A (ego warp only + ConvGRU, the control)

### 7.2 Earlier design B: one joint state, pulls into the camera time (now a comparison arm)
Design agreed with Mirko (2026-10-25): three BEV maps with their own times are fused in the camera time - the camera BEV (t_cam, the reference), the radar BEV (t_radar; per cell its mean age) and the past state (t_past); each first gets its own 3 x 3 conv. The past yields the pull positions:
- per cell of the reference grid and per source s: x_s = T_ego(t_s <- t_cam) (x - v_c(x) dt_s), dt_s = t_cam - t_s, with v_c the cell's velocity in camera-time axes (equivalently T x - R v_c dt; "T x - v dt" holds only if v is expressed in the t_s axes); grid_sample pulls the source at x_s (a gather like lift PD: fast, differentiable). At 10-30 m/s and dt = 0.5 s the displacement is 5-15 m, so v is read at the pulled position once more (one fixed-point step) before the final pull
- ego motion is exact (poses); the object motion v (2 channels per cell, plus a moving probability and its uncertainty) comes from a motion estimator on the ego-aligned past state and the current sensors (incl. the radar's Doppler channels); order: ego-align the past -> estimate v -> pull
- no double counting: the radar encoder already moves each detection to t_cam by the ego motion and its radial Doppler part; the BEV pull adds only the remaining object motion, with the cell's mean radar age as its dt
- static memory: ego warp only, long, confidence-gated update (lane markings sharpen as the car approaches); moving memory: ego + flow, short
- every pulled source carries a confidence (age, moving probability, flow uncertainty); the fusion gates by it (holes and ghosts where content moves)
- training signal for v: GT box velocities rasterised per cell; a consistency term between the pulled past and the current BEV
- arms: B (this design, main) against A (ego warp only + ConvGRU, the control); later options: C (learned deformable pull offsets, BEVFormer temporal self-attention) if an explicit v does not learn, D (event-driven filter: the state predicted to each sensor's own time and updated there) if the time alignment is the weak point; E (static memory + tracked objects) not planned
- Basis: CRT-Fusion 2411.03013 (camera + radar + time: per-cell velocity and occupancy from a motion feature estimator, past features shifted by v dt above 1 m/s, recurrent; R50 nuScenes val 57.2 NDS vs CRN 56.0, night 33.0 vs 30.4 mAP; 14.5 FPS on an RTX 3090). Its code (Apache-2.0, `literature/code/crt_fusion.md`): a forward PUSH (scatter to 4 neighbours, colliding sources overwrite), ego motion separate (velocity offset first, then one ego matrix), fixed dt 0.5 s, 6 history frames stacked in channels and sparsified by predicted occupancy (would drop lanes); the velocity shift is assigned through `grid[mask][:, 0:2] = ...`, which in PyTorch writes into a copy (checked: `temp/setup/check_crt_mask_assign.py`), so - if the released line is as read - the shift does nothing, and the paper has no isolated ablation of it (only motion estimator + temporal fusion together, +1.1 NDS): the gain of a velocity warp is unproven, our arm B vs A is the test. We take its velocity GT recipe (box footprints rasterised, ground velocity in ego axes, zero below 1 m/s, loss on foreground cells) and head shapes, and re-write the pull, FIERY 2104.10490 (exact ego warp, ego-motion channels, spatial GRU; MIT, plain PyTorch), BEVDet4D 2203.17054 (alignment matters), VideoBEV 2303.05970, MemFusionMap 2409.18737 (lit).
- Tests: pure ego motion over a static scene: the pulled past must align exactly with the current map (warp test against the GT map in world coordinates); a target moving at known v must land at x + v dt (the control A must lag); radar with a known age: pulled to t_cam once, not twice; holes and ghosts carry low confidence; gain per range bin (far range should gain most).

## 8. Semantic head
- Multi-label at 0.5 m: areas (drivable, ped crossing, walkway, stop line), lines (lane divider, road divider, lane centre-line) with a truncated 2D distance-field target (a cheap BEV SDF for thin lines), objects (vehicle, pedestrian, two-wheeler), auxiliary moving mask and flow; masked losses; uncertainty weighting 1705.07115 (lit).

## 9. Ground truth: how it is made and how you can look at it

### 9.1 Generation (CPU, cached per frame)
1. For every camera frame (keyframes and, after the download, 12 Hz sweeps): the vehicle pose at the image time t_ref.
2. Map layers from the map expansion, transformed into the vehicle frame at t_ref: polygons (drivable area, ped crossing, walkway, stop line) rasterised with fractional coverage per cell; lines (dividers) drawn 1 cell wide plus a truncated distance field; lane centre-lines from the discretised lane arcs.
3. Objects: the annotated boxes, interpolated per instance to t_ref (position, yaw, size) between the 2 Hz keyframes; footprints rasterised; velocity per box. Interpolated labels are flagged (training only; evaluation on keyframes).
4. Masks: coverage = cells reached by any camera frustum (from the calibration, generic for any rig); occlusion = ray casting from each camera centre through the GT box footprints (cells behind a box are occluded); range bins.
5. Stored per frame (compressed arrays) with an index of provenance (map version, interpolated or annotated, timestamps).

### 9.2 Observing it (a viewer tool, local files, not in git)
- Per frame a three-panel picture: (1) the camera image with the GT projected into it (dividers and centre-lines as lines, areas translucent, boxes as 3D wireframes, radar detections as dots with Doppler arrows) - if the projected lines do not sit on the painted markings, the GT or the transforms are wrong; (2) the BEV raster with each layer in its colour, coverage and occlusion masks hatched, the camera frustums and radar fields of view outlined, the ego vehicle drawn; (3) a text strip with the timestamps, dt of each sensor, the interpolation flag and the class cell counts.
- Per scene a 12 Hz video of those pictures, and an interactive page (scene / frame slider, layer toggles) to browse a scene; key frames sent to you.

### 9.3 Checks of the GT itself
- Reprojection: image edge strength along the projected dividers must be higher than along the same lines shifted by 0.5 m (catches calibration, pose and time errors automatically).
- World consistency: the map layers of frame t warped by the ego motion to frame t+1 must equal the layers of t+1 (IoU near 1 except at the coverage edge).
- Interpolation error: leave one keyframe out, interpolate it from its neighbours, compare with its annotation (position / yaw error per class and speed) - an upper bound for the 12 Hz labels.
- Statistics: cells per class and range bin, per city; empty or broken scenes listed.

## 10. Test levels for the whole chain (cheap first)
| level | what | example |
|---|---|---|
| 0 unit, known answer | geometry, splat, metric code on synthetic input | point-to-cell tests of 2.5 / 3.3, IoU on hand-made masks |
| 1 GT checks | 9.3 | reprojection, world consistency, interpolation error |
| 2 sanity runs | overfit one scene; null (per-cell training mean); oracle (GT depth into the lift); shuffled radar | wiring, upper and lower bounds |
| 3 information tests | reconstruction of each sensor's input from its BEV | radar cloud from the radar BEV; image-plane semantics re-projected from the camera BEV |
| 4 invariance tests | rig changes, sensor drop-out, time offsets | moved / rotated sensor, one radar off, shifted timestamps |
| 5 comparison on VAL | each stage against its control with two seeds and the noise band | S1 lift arms, S2 radar arms, ... |
| 6 speed | ms per block, batch 1, idle GPU | |

## 11. Stages
| stage | content | control |
|---|---|---|
| S0 | GT generation + viewer + GT checks; geometry tables with their unit tests | - |
| S1 | camera only: lift PD (pull with depth weight; variants PD-G, learned elevations), seg head | P (pull without depth, Simple-BEV-like), S (splat, LSS-like) |
| S2 | radar only: learned density g, reconstruction test, aux heads | point / pillar (R0), fixed polar Gaussian (R1) |
| S3 | fusion | S1, S2 |
| S4 | temporal: ego warp, stationary memory, moving memory with flow | single frame; memory without flow |
| S5 | detection head | S4 seg only |

## 12. Decided and open questions

### Decided (Mirko, 2026-10-25)
- range: the system is built for 60 m (grid and depth bins end at 60 m)
- no horizon strip for now: one image stream, half resolution, stride 8 (pixel groups of 16 x 16 full-resolution pixels); the full-resolution horizon strip (stride 4 over the rows whose ground lies at 15-100 m) stays a later option
- what that means (geometry for a camera like CAM_FRONT, f ~ 1266 px, h ~ 1.5 m): across the frustum is 0.25 m wide at 20 m and 0.76 m at 60 m (fine for 0.5 m cells); along the range a group covers 3.4 m of road at 20 m, 10 m at 35 m, 21 m at 50 m - beyond about 20 m single-frame road positions come from the learned depth distribution inside the frustum and from temporal fusion (each far cell is seen again closer), which section 7 must therefore deliver and the per-range evaluation must show

- camera lift: image backbone + lookup tables, height kept as channels (feat[z_car]); lifts P / S / PD compared in S1, PD recommended because it keeps the depth uncertainty and is a pure gather (section 2)
- first building block FIXED (Mirko, 2026-10-25): the camera block produces the world voxel map with lift PD (pull with depth weight); P and S stay in S1 only as controls; the next design topic is the radar
- radar block FIXED, simple (Mirko, 2026-10-25): every detection goes into the BEV cell where it resides (hard raster, 0.5 m, same grid and index order as the camera BEV) with its properties - distance from the sensor, direction from the sensor in the vehicle frame (circular), Doppler as a circular value (sin / cos of k pi v_r / v_max, k = 1, 2, 4), RCS, age - and sub-cell offset; then two 3 x 3 convolutions (`training/radar_bev.py`, self-test passes). This replaces the learned density g of section 3 for the first version; section 3 and `radar_encoding.md` stay as the variants (polar image + soft splat + pull, Cartesian soft footprint, learned sigma, height profile), each to be justified against this simple block in the radar stage S2 (hard raster = R0).
- camera and radar blocks are settled: both deliver a BEV on the same grid (camera 128 channels, radar 32) for the fusion; next design topics: fusion, temporal state, heads
- kept for later (Mirko, 2026-10-25): PD-G in inverse depth (mixture in 1/z, exact voxel mass, section 2.4); the categorical lift PD on the pre-trained head stays the working version until then

### Proposed, not yet objected to
- depth bins: 80, 1-60 m, linear-increasing spacing (CaDDN LID: a few cm near, about 1.5 m at 60 m)
- shelved for now (Mirko chose the lift families of section 2): the frustum-slab footprint and the ground hypothesis as an S1 arm; previously proposed as: fixed per-calibration interval from the frustum's edge rays on the road plane z = 0, widened by sigma_d ~ d^2 sigma_theta / h with one learned global sigma_theta; one learned ground probability per group mixes it with the free depth distribution; a per-frame pitch / road-plane input later, if the slope errors show up

### Open
3. Radar support window and the form of g (MLP on offsets, or a categorical over a polar window as in draft 2); the sensor prior.
4. What "time" enters g: dt only, or dt and the ego speed / moving probability.
5. Front-only first or all six cameras from the start (the download brings all six).
6. Lidar (80 GB) for the oracle and depth supervision?
7. A second dataset or rig for the invariance tests (which one is available)?
