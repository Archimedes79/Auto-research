# Camera + radar BEV system - plan for discussion (draft 3, 2026-10-25)

Status: DRAFT for the discussion with Mirko; nothing is decided. Draft 2 (front camera, virtual upright camera, sigma points) is in `plan_draft2.md`; this draft replaces it after Mirko's comments of 2026-10-25:
- the problem is the real world, nuScenes is only one dataset: the design takes the sensor set, the calibration and the timestamps as inputs, never as constants of one dataset
- radar: a learned distribution per detection, f(x, y; detection, r(x, y; calibration), phi(x, y; calibration), time), not sigma points
- camera: rays (cones) from pixel groups instead of a virtual upright camera
- every block founded on papers and their code; the ground truth observable; a test for every step

Paper ids are arXiv ids; "(lit)" = from a scout note in `literature/papers/` (all ids checked on their abstract pages, 2026-10-25). Code licences from `literature/code/front_bev_code_sources.md` (scout, read on the repository pages).

## 0. The problem, not the dataset
- Product: a BEV environment model for a vehicle with any number of cameras and radars at any mounting: semantic map layers (road, markings, lanes), objects, later detection, at 0.5 m, fast.
- Input contract (the only thing the system knows about a sensor): its model (pinhole / fisheye intrinsics, radar field of view and resolution), its pose in vehicle coordinates (calibration), the timestamp of each measurement, and the ego motion. Nothing is tuned to one calibration.
- nuScenes is the first training and validation set (front camera + five radars as the first rig; all six cameras are on disk once the download finishes). A second dataset or a changed rig is the test that the design does not depend on nuScenes.

## 1. One principle for every sensor
Every measurement becomes a probability mass over world positions, written in the sensor's own polar coordinates around its calibrated pose, evaluated at the BEV cells it can reach, and added into one BEV grid.

| | what the sensor knows well | what it does not know | what is learned |
|---|---|---|---|
| camera pixel group | the direction (a narrow cone) | the range along the cone | the distribution over range |
| radar detection | the range (and Doppler) | the azimuth (wide), the height | the spread around the detection |

Time is part of the position: each measurement is moved to the output time t_ref before it is evaluated (exact for static content, by Doppler for the radial motion of moving content, otherwise as a wider spread).

```
cameras ──> backbone + FPN ──> pixel-group cones ──> learned range distribution ──> cone-footprint splat ──┐
                                                                                                           ├──> fusion ──> temporal state ──> BEV trunk ──> heads
radars ──> time alignment ──> point encoder ──> learned density f(cell; detection, calib, dt) ──> splat ───┘
```
Block diagrams of draft 2 were shown in the chat; the drawing will be redrawn for this draft after the discussion.

## 2. Camera: cones from pixel groups

### 2.1 What it does, in plain words
1. The backbone turns the image into a feature map; each feature cell sees a group of pixels (e.g. 8 x 8 or 16 x 16 pixels at the backbone stride).
2. From the camera model and the calibration, each pixel group is a cone in vehicle coordinates: apex at the camera centre, opening = the angular size of the group. This works for any camera, any mounting (roll, pitch, yaw included), and for fisheye by swapping the ray model; nothing is warped, the image stays as it is.
3. The network predicts, per pixel group, a distribution over range bins along the cone (plus a ground hypothesis, 2.3).
4. At each range bin the cone's cross-section covers an area on the ground (narrow near, wide far); the feature times the bin's probability is spread over the BEV cells inside that area, weighted by overlap.
5. Steps 2 and 4 are pure geometry: for a given calibration they are computed once into a sparse table (pixel group, range bin) -> (BEV cells, weights). At run time the lift is one sparse matrix product. When the calibration changes (another vehicle, an online re-calibration), the table is rebuilt (milliseconds).

### 2.2 Why cones and not rays
A single ray through each pixel-group centre leaves gaps between rays at range (at 50 m two neighbouring stride-16 columns at 1600 px width are about 0.6 m apart, more than a cell) and hits cells unevenly; the cone covers the ground without gaps and splits mass by area, so far cells get the right share. It also gives the lateral uncertainty for free: the cone's width is the angular resolution of the feature map.

### 2.3 Ground hypothesis
For a cone that points below the horizon, the intersection of its axis with the road plane (z = 0 in vehicle coordinates) gives a range r_g; its spread follows from the cone's vertical opening and the pitch uncertainty (sigma_r ~ r_g^2 sigma_theta / h). A learned ground probability p_g mixes this sharp hypothesis with the learned range distribution: road surface and markings take the ground hypothesis, raised objects the learned bins. Dynamic pitch (braking, bumps) and slope changes enter only r_g and sigma_theta.

### 2.4 Paper and code basis
| piece | paper (what we take) | code (licence) |
|---|---|---|
| lift by a learned depth distribution along camera rays, splat to BEV | LSS 2008.05711 (lit): per-pixel categorical depth times feature, sum into BEV | nv-tlabs/lift-splat-shoot (NVIDIA Source Code License, research only: re-implement, never copy) |
| categorical depth with depth supervision | CaDDN 2103.01100 (lit: LID bins, focal depth loss), BEVDepth 2206.10092 (lit: camera-aware depth gate, lidar depth loss) | OpenPCDet / CaDDN (Apache-2.0), BEVDepth (MIT) |
| parameter-free point sampling (a control) | Simple-BEV 2206.07959 (lit) | aharley/simple_bev (MIT) |
| precomputed indices / lift as one sparse matrix | BEVPoolV2 2211.17111 (lit), MatrixVT 2211.10593 (lit), Fast-BEV 2301.12511 (lit: table lift 0.006 ms) | BEVDet `bev_pool_v2` (Apache-2.0), MatrixVT (MIT); our own sparse matrix in plain PyTorch |
| cone instead of ray, area-weighted footprint | Mip-NeRF 2103.13415 (lit): the pixel cone (radius = pixel width x 2/sqrt(12)) between two depths t0, t1 has a closed-form Gaussian (mean, covariance); its xy block is the BEV footprint of one (pixel group, range bin). Zip-NeRF 2304.06706 (lit): the mass of a Gaussian inside one cell by erf - the per-cell weight of the table. Mip-Splatting 2311.16493 (abstract): a cell-sized filter against aliasing of sub-cell splats | google/mipnerf (Apache-2.0, JAX, archived: formulas only); Zip-NeRF: no official code |
| depth variance sets the footprint width; one splat interface for camera and radar | GaussianLSS 2504.01957 (lit), GaussianCaR 2602.08784 (lit: camera and radar Gaussians in one rasteriser, 0.5 m, 75.6 ms on an RTX 4090; its Gaussian size is depth std x 0.5 - we replace it by the cone covariance) | GaussianLSS (MIT), GaussianCaR (Apache-2.0; already cloned in WSL `~/work/gaussiancar`) |
| novelty | the scout found no published BEV lift that uses a cone / pixel-footprint integral (three searches, 2026-10-25): the cone lift is our own combination of Mip-NeRF geometry with an LSS / CaDDN depth head | - |
| ground prior | OFT 1811.08188 (lit), Cam2BEV 2005.04078 (lit: IPM breaks on raised objects - hence the mixture) | Gen-LaneNet homography utils (Apache-2.0) for cross-checks |

### 2.5 Tests (what it must do -> how we prove it)
| requirement | test |
|---|---|
| the table is geometrically right | synthetic: a point at a known (x, y, z) projected into the image; its pixel group's cone at the true range bin must cover the cell of (x, y); for every calibration of the dataset and for random synthetic rigs |
| mass is conserved | the weights of each (group, bin) inside the grid sum to its covered area fraction; a uniform image gives the analytic coverage map |
| rig independence | move the camera 1 m forward in the calibration and shift the scene accordingly: the lifted BEV moves by 2 cells and is otherwise identical; rotate the camera (roll / pitch) with a synthetic image rendered for that pose: same BEV |
| the lift can carry the truth (oracle) | feed one-hot depth from a known depth (lidar if downloaded, else rendered synthetic scenes) and the GT semantics of the image: the lifted BEV must reproduce the GT BEV layers up to the cone resolution; this gives the upper bound of the lift |
| it is differentiable | torch.autograd.gradcheck on a small table; gradient reaches the depth head and the backbone |
| it is fast | ms per frame, batch 1, idle GPU, against Fast-BEV's table lift as the reference |

## 3. Radar: a learned density per detection

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

## 4. Fusion
- Per cell gate from both BEVs and their confidence channels (camera: mass, range variance; radar: support weight, Doppler observability, age); modality dropout; radar-only auxiliary heads so the radar branch must learn on its own.
- Basis: BEVFusion-style concat + conv (lit, SN20 of the lane project), gated / confidence-aware fusion in CRN and RCBEVDet (lit); modality dropout as in robust fusion papers (to collect).
- Tests: camera off -> radar-only result, radar off -> camera-only result (no collapse); shuffled radar (from another frame) -> the gain over camera-only disappears (null control); gate maps plotted over range (radar should dominate far, camera near).

## 5. Sensor timing
- Every measurement has its own timestamp and ego pose (nuScenes: per sample_data; a real vehicle: per message). Output time t_ref = the newest camera image (or a fixed output clock).
- Static: exact pose change between t_meas and t_ref. Moving: radial part by Doppler, the rest as uncertainty (3.1). Camera images from several cameras with different times: each camera's cones use its own pose at its own time.
- Basis: BEVDet4D 2203.17054 (lit: alignment matters, unaligned concat is worse than no history), RadarNet (lit).
- Tests: the time-alignment tests of 3.3; a camera test with a synthetic static scene and a time offset; latency budget per block.

## 6. BEV network
- Grid around the vehicle (front-only first: x 0..60 m, y -26..26 m; full surround once all cameras are used), 0.5 m; coordinate channels (x, y, range, azimuth per sensor coverage) appended; ResNet / FPN trunk.
- Basis: BEVFusion / CVT decoders (lit), CenterPoint trunk (lit, MIT).
- Tests: overfit one scene to near-perfect IoU (wiring); ms per frame.

## 7. Temporal fusion (ego motion, stationary environment, moving targets)
- Ego-motion warp of the previous state; a long stationary memory with a confidence-gated update (lane markings sharpen as the car approaches); a short moving memory with a flow warp (flow from Doppler where observable, else a flow head); a learned moving mask supervised by box velocities.
- Basis: BEVDet4D, VideoBEV 2303.05970, FIERY 2104.10490 (temporal block in plain PyTorch, MIT), MemFusionMap 2409.18737 (lit).
- Tests: a static scene driven through: the memory must not smear (warp test against the GT map in world coordinates); a moving target: the flow warp must keep it in place, the control without flow must lag; gain per range bin (far range should gain most).

## 8. Semantic head
- Multi-label at 0.5 m: areas (drivable, ped crossing, walkway, stop line), lines (lane divider, road divider, lane centre-line) with a truncated 2D distance-field target (a cheap BEV SDF for thin lines), objects (vehicle, pedestrian, two-wheeler), auxiliary moving mask and flow; masked losses; uncertainty weighting 1705.07115 (lit).

## 9. Ground truth: how it is made and how you can look at it

### 9.1 Generation (CPU, cached per frame)
1. For every camera frame (keyframes and, after the download, 12 Hz sweeps): the vehicle pose at the image time t_ref.
2. Map layers from the map expansion, transformed into the vehicle frame at t_ref: polygons (drivable area, ped crossing, walkway, stop line) rasterised with fractional coverage per cell; lines (dividers) drawn 1 cell wide plus a truncated distance field; lane centre-lines from the discretised lane arcs.
3. Objects: the annotated boxes, interpolated per instance to t_ref (position, yaw, size) between the 2 Hz keyframes; footprints rasterised; velocity per box. Interpolated labels are flagged (training only; evaluation on keyframes).
4. Masks: coverage = cells reached by any camera cone (from the calibration, generic for any rig); occlusion = ray casting from each camera centre through the GT box footprints (cells behind a box are occluded); range bins.
5. Stored per frame (compressed arrays) with an index of provenance (map version, interpolated or annotated, timestamps).

### 9.2 Observing it (a viewer tool, local files, not in git)
- Per frame a three-panel picture: (1) the camera image with the GT projected into it (dividers and centre-lines as lines, areas translucent, boxes as 3D wireframes, radar detections as dots with Doppler arrows) - if the projected lines do not sit on the painted markings, the GT or the transforms are wrong; (2) the BEV raster with each layer in its colour, coverage and occlusion masks hatched, the camera cones and radar fields of view outlined, the ego vehicle drawn; (3) a text strip with the timestamps, dt of each sensor, the interpolation flag and the class cell counts.
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
| S1 | camera only: cones + learned range distribution (+ ground hypothesis), seg head | ray-point lift (LSS-like), uniform along the cone (Fast-BEV-like) |
| S2 | radar only: learned density g, reconstruction test, aux heads | point / pillar (R0), fixed polar Gaussian (R1) |
| S3 | fusion | S1, S2 |
| S4 | temporal: ego warp, stationary memory, moving memory with flow | single frame; memory without flow |
| S5 | detection head | S4 seg only |

## 12. Open questions for the discussion
1. Pixel-group size (stride 8 or 16) and whether a finer vertical stride is needed for far range.
2. Range bins: how many, spacing (uniform in inverse range?), the ground hypothesis in or out of S1.
3. Radar support window and the form of g (MLP on offsets, or a categorical over a polar window as in draft 2); the sensor prior.
4. What "time" enters g: dt only, or dt and the ego speed / moving probability.
5. Front-only first or all six cameras from the start (the download brings all six).
6. Lidar (80 GB) for the oracle and depth supervision?
7. A second dataset or rig for the invariance tests (which one is available)?
