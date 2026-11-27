# PL46 - OrienterNet: full-text extraction

| field | value |
|---|---|
| paper | OrienterNet: Visual Localization in 2D Public Maps with Neural Matching. P.-E. Sarlin, D. DeTone, T.-Y. Yang, A. Avetisyan, J. Straub, T. Malisiewicz, S. Rota Bulo, R. Newcombe, P. Kontschieder, V. Balntas. CVPR 2023. arXiv 2304.02009 (v1 only, submitted 2023-04-04). Links opened: https://arxiv.org/abs/2304.02009 and https://arxiv.org/html/2304.02009v1 (not opened: project page, OpenReview, supplementary PDF) |
| axis, tier, K ids | A6 (assignment to a map graph; the "no pose given" step), tier 1 core, K99 (differentiable localisation of a BEV raster against a navigation map), used in E48, E74, E82, E103e |
| level-1 note(s) | [PL46](../papers/PL46_orienternet.md) (home note), [C28](../papers/C28_orienternet-family-exhaustive-probability-volume.md) (OrienterNet family: OSMLoc, hierarchical re-localisation 2603.01613, DiffVL) |
| code extraction | [orienternet](../code/orienternet.md) (repository verified: https://github.com/facebookresearch/OrienterNet, licence CC BY-NC style, read only) |
| sources read | arXiv abs page (metadata, versions); arXiv HTML v1, four queries: Sections 1 and 3 (problem, map encoding, BEV inference, matching, fusion), Sections 5 - 6 and appendices (MGL, training details, Tables 2 - 4, ablations, runtime, limitations), evaluation priors and rotations, Section 6 sub-sections. Repository files read through raw.githubusercontent.com (listed in the code extraction). All on 2026-10-24. |
| reading marks | **No section is marked "checked": every page came through the summarising fetch tool.** Paper text, equations and tables: **digest** (numbers copied from the digest; equation typesetting as returned by the tool). Repository facts used for the paper-versus-code part: **digest**. Local project facts (E ids, K99, HANDOVER-level notes): read from local files, **checked**. Anything labelled "estimate" is my own arithmetic, not a paper number. Figures, appendix B.1 text and supplementary material were not seen (marked "not found" where needed); nothing in this file is from **memory** except the general-knowledge lane width in section 5.2, which is labelled. |
| extracted by, date | extraction sub-agent (Claude Sonnet 5.5), 2026-10-24 |

## 1. Problem and claim
Given one gravity-aligned calibrated image (or a sequence) and a rasterised OpenStreetMap (OSM) tile centred on a coarse location prior, estimate the 3-DoF camera pose (x, y, heading) in the map. OSM is a planimetric map: classes and 2D shapes of objects, no appearance, no height. The paper's claim: a network that lifts the image into a metric neural bird's-eye view (BEV) and matches it, by exhaustive template matching over every (x, y, rotation) in the Fourier domain, against a learned neural encoding of the OSM raster gives pose probabilities that can be trained from camera poses alone (negative log-likelihood of the true pose) and fused over frames by multiplying warped volumes. For us the relevant part is the second half: a differentiable pose volume between a ground-aligned BEV raster and a map raster, with an explicit uncertainty.

## 2. Method as built

### 2.1 Pipeline and shapes (digest; code numbers from `orienternet.yaml`, see section 4)
```
image I (gravity-rectified: roll = pitch = 0, known focal length f, known camera height)
  | Phi_image: U-Net, ResNet-101 encoder (paper); latent 128 ch, plus per-pixel scale scores S_uv (33 bins)
  v
polar lift (E1-E3): one ray per image column u, depth planes d * Delta (Delta = 0.5 m), weights alpha from the scale scores
  v
Cartesian resample (linear, along the lateral axis) -> BEV grid on the camera frustum, Delta = 0.5 m
  v
Phi_BEV: 4 residual bottleneck blocks -> T (L x D x N, N = 8) and confidence C in [0,1] (sigmoid)

OSM elements (7 area, 10 line, 33 point classes) -> 3-channel integer raster at Delta = 0.5 m (W x H)
  v  learned class embedding per channel (16-d in the released config), concatenated
  v  Phi_map: VGG-style encoder + decoder -> F (W x H x N, N = 8, not normalised); paper also names a unary location prior Omega (W x H)

M[xi] = correlation of F with the rotated, confidence-masked BEV (T * C)  for every (x, y) and K rotations (batched FFT)
P = softmax(M + Omega) over (W, H, K)                        [log_softmax over the flattened volume]
```
- Pose volume size (arithmetic from the paper's numbers): training map tile 128 x 128 m at 0.5 m is 256 x 256 = 65,536 positions; with K = 64 (training) that is 4.19 M cells, with K = 512 (test) 33.6 M cells. Rotation step 360/K: 5.6 deg (K = 64), 0.70 deg (K = 512).
- BEV extent: the paper text digest says "32 x 32 m at 0.5 m". The released config has x_max = 32 and z_max = 32, and the projection code builds a lateral span of 2 * x_max, so the code BEV is about 64 m wide and 32 m deep (about 129 x 64 cells). The two statements differ; which one is the trained model is not settled by what I read (see section 4).

### 2.2 Inputs, outputs
- **Input:** image I with pinhole calibration; gravity direction from the inertial unit (the image is rectified by a homography so that roll and tilt are zero); camera height estimated from the distance to the ground in a local SLAM reconstruction (assumed mostly planar world); OSM data from a square area centred on a coarse location prior xi_prior. Heading is not part of the prior in the paper (full circle searched).
- **Prior (what "no pose given" means in practice):** the prior is a noisy GPS position or a previous estimate, "can be off by over 20 meters" (paper digest). The OSM square is sized by how noisy the prior is. Training: maps of 128 x 128 m centred on points sampled randomly within 32 m of the ground-truth pose. KITTI evaluation: ±20 m and ±10 deg around ground truth (restricted pose space for a fair comparison; OrienterNet itself needs only the position prior), maps 64 x 64 m. Aria: tile centred on the noisy GPS; predictions constrained to within 64 m (Seattle) and 24 m (Detroit) of the GPS fix. MGL validation: the paper gives no evaluation prior; the repository evaluation uses max_init_error = 32 m (code).
- **Output:** probability volume P over (x, y, theta) on the tile; the pose is the maximum-likelihood estimate argmax P. When P is mostly unimodal the uncertainty can be read from the covariance of P around the maximum (paper). Multi-frame: fused volume.

### 2.3 Equations (numbered as returned by the digest; symbols as in the paper)
- **E1** (polar feature): X_bar[u,d] = sum_v alpha[u,d,v] * X[u,v]; X is the U x V x N image feature map, d a depth plane, v the image row, u the column.
- **E2** (log-distributed scales): sigma = { sigma_min * (sigma_max / sigma_min)^(i/S) : i = 0..S }, S = 32 (so 33 values), sigma_min = 2^1, sigma_max = 2^9 (appendix C); this gives a depth interval [0.5 m, 128 m] for f = 256 px.
- **E3** (weights): alpha[u,d,v] = softmax over v of ( S[u,v] interpolated at f / (d * Delta) ). The network predicts scale scores (independent of calibration) and the focal length enters explicitly, so other cameras work at test time.
- **E4** (posterior): P = softmax(M + Omega); M is the image-conditioned matching score, Omega the learned unary prior broadcast over rotations.
- **E5** (matching score): M[xi] = 1/(U Z) * sum over BEV cells p of F[xi(p)]^T (T * C)[p]; xi(p) maps a BEV cell to map coordinates under pose xi; * is the element-wise confidence mask. The K rotations of T are computed together and the correlation is a single convolution implemented as a batched multiplication in the Fourier domain. (The code divides by the number of valid template pixels instead of the constant, see section 4.)
- **E6** (sequence / multi-camera fusion): P(xi_i | {I_j}, map) = product over views of P(xi_i composed with the relative pose xi_ij | I_j, map). Each single-view volume is warped into the reference frame and the log volumes are added. Continuous streams: iterative warp-and-normalise, as in Markov localisation.
- **Loss (unnumbered):** L = -log P[xi_gt]; P is interpolated trilinearly at the continuous ground-truth pose (sub-pixel supervision).

### 2.4 Losses
Only the negative log-likelihood of the ground-truth pose. No auxiliary depth, segmentation or map-class loss; the paper says the model is "trained only with camera poses" (abstract digest). No Hungarian matching or assignment of any kind. Weighting: none (single term). The released code has an optional Gaussian label-smoothing variant (default off), see section 4.

### 2.5 Labels / ground truth
Camera poses only. MGL: Mapillary images with calibration, noisy GPS and 6-DoF poses in a global frame from SfM fused with GPS; sequences from 2017 onwards; frames subsampled to at least 4 m apart; fisheye undistorted to pinhole, 360-degree panoramas split into four 90-degree views; each location split into disjoint train and validation sets. 12 cities (San Francisco, Amsterdam, Berlin, Vilnius, Helsinki, Milan, Paris, Montrouge, Le Mans, Nantes, Avignon, Toulouse). Size is inconsistent inside the paper: Section 5 says 760k images, appendix B.2 says 826k training and 2k validation views, Table 6 per-city counts sum to about 855k. KITTI: RTK ground truth, "Test2" split not overlapping KITTI or MGL training (construction and frame count not seen). Aria: pseudo ground truth, accuracy "lower than 1 m" after visual checks (text), 5 locations, 2153 (Seattle) and 2725 (Detroit) frames.

### 2.6 Training recipe
- Architecture detail (paper): U-Nets with ResNet-101 (image) and VGG-16 (map) encoders; 4 residual blocks in Phi_BEV; S = 32; N = 8; K = 64 in training (GPU memory) and 512 at test; images 512 x 512; replicate padding in the map CNN and in the matching.
- Batch 9 on 3 V100 (16 GB); augmentation: random flip and rotation of the raster maps; model selection by early stopping on validation loss. **Not stated in the paper:** optimiser, learning rate, iterations or epochs, parameter count (code gives Adam, lr 1e-4, see section 4).
- Pre-training on MGL and fine-tuning on KITTI (Table 3, rows c and d).

### 2.7 Inference
Exhaustive scoring of the whole volume, softmax, argmax (or expectation, in the code). Sequence mode: per-frame volumes warped by the known relative poses and multiplied. Optional GPS fusion (code): a hard window or Gaussian prior on the volume. Prior-dependent restriction of the volume to a mask (code, `map_mask`) and optional yaw prior (code, `yaw_prior`). There is no learned post-processing; the only hidden rule step is the argmax readout.

## 3. Numbers

**Table 2 (MGL validation, ablation of the design; position recall at 1 / 3 / 5 m and orientation recall at 1 / 3 / 5 deg).**

| model | position R@1/3/5 m | orientation R@1/3/5 deg |
|---|---|---|
| (a) Retrieval | 2.02 / 15.21 / 24.21 | 4.50 / 18.61 / 32.48 |
| (b) Refinement | 8.09 / 26.02 / 35.31 | 14.92 / 36.87 / 45.19 |
| (c) OrienterNet planar | 14.28 / 44.59 / 56.08 | 20.43 / 50.34 / 64.30 |
| OrienterNet full | 15.78 / 47.75 / 58.98 | 22.14 / 52.56 / 66.32 |

Retrieval replaces BEV inference and matching by a global image embedding; Refinement needs an initial orientation (within 45 deg of truth in this test) and warps a map view to the image; "planar" replaces the occupancy-based lifting by a homography-based warp.

**Table 3 (KITTI Test2; lateral and longitudinal recall relative to the viewing axis, R@1/3/5 m; orientation R@1/3/5 deg), OSM rows only.**

| approach | training | lateral | longitudinal | orientation |
|---|---|---|---|---|
| Retrieval | MGL | 37.47 / 66.24 / 72.89 | 5.94 / 16.88 / 26.97 | 2.97 / 12.32 / 23.27 |
| Refinement | MGL | 50.83 / 78.10 / 82.22 | 17.75 / 40.32 / 52.40 | 31.03 / 66.76 / 76.07 |
| OrienterNet (a) | MGL | 53.51 / 88.85 / 94.47 | 26.25 / 59.84 / 70.76 | 34.26 / 73.51 / 89.45 |
| OrienterNet + sequence (b), 20 s | MGL | 79.71 / 97.44 / 98.67 | 55.21 / 95.27 / 99.51 | 77.87 / 97.76 / 100 |
| OrienterNet (c) | KITTI | 51.26 / 84.77 / 91.81 | 22.39 / 46.79 / 57.81 | 20.41 / 52.24 / 73.53 |
| OrienterNet (d) | MGL + KITTI | 65.91 / 92.76 / 96.54 | 33.07 / 65.18 / 75.15 | 35.72 / 77.49 / 91.51 |

Satellite-map rows (DSM, VIGOR, Refinement on KITTI) are in the paper's Table 3 and not copied here. Caption: 20 s sequences saturate recall at 3 m / 3 deg above 95 %. Note the row label of (b) is garbled in the source page.

**Table 4 (Aria AR data; position R@1/3/5 m).** Seattle single frame: GPS 1.25 / 8.82 / 18.44, OrienterNet 3.39 / 14.49 / 23.92; Seattle multi-frame (10 keyframes): GPS 1.76 / 9.20 / 20.48, OrienterNet 21.88 / 61.26 / 72.92. Detroit single: GPS 3.96 / 27.75 / 51.33, OrienterNet 6.26 / 32.41 / 51.76; Detroit multi: GPS 4.09 / 31.36 / 53.41, OrienterNet 17.18 / 68.77 / 89.26. Fusion "more than doubles" OrienterNet accuracy and only marginally helps GPS.

**Ablations that decide transfer.**
- Map classes (Fig. 10, no numbers in the text): buildings, roads, footpaths and cycleways matter most, likely because they are most frequent. Our SD raster has far fewer classes than OSM's 50.
- Field of view (Fig. 11, no numbers): a 50 % smaller horizontal FoV roughly halves accuracy. The size of the BEV evidence is decisive; our BEV (100 x 100 m) is much larger than the 32 m deep BEV here.
- Architecture (Table 2): exhaustive matching beats retrieval and refinement by a wide margin; the occupancy-style lifting is worth +1.5 points of R@1 m over the planar warp.
- No ablation on rotation count K, on map or BEV resolution, or on the location prior size is reported.

**Compute.** 94 ms per localisation on an RTX 2080: 37 ms BEV inference, 51 ms matching (the parts sum to 88 ms, about 6 ms unexplained). Parameter count not reported. Training: 3 x V100 (paper) or 3 x RTX 3090 for 340k iterations, 3 - 4 days (README, code extraction). Test matching uses K = 512 rotations on a 256 x 256 tile; my arithmetic: the zero-padded kernel tensor for the FFT has K x N x H x W = 512 x 8 x 65,536 = 268 M floats (about 1.07 GB in fp32), which is the memory driver.

**Failure modes (Fig. 9 and Section 6.3).** Fails when the scene lacks distinctive elements, when elements are missing from OSM, and when elements repeat (ambiguous pose); OSM may be spatially inaccurate and its detail varies by city (trees are mapped in some cities and not in others). Aria (head-mounted, AR conditions) is far below driving. Sequence fusion is needed to resolve ambiguity (Figs. 5, 6).

**Caveats.** Recalls are on different priors and maps per data set (MGL validation prior not stated, KITTI ±20 m / ±10 deg, Aria 24 - 64 m). Single-image position recall at 1 m is only 15.8 % on MGL: sub-metre accuracy is the property of the sequence product, not of one frame. The released model reproduces slightly different numbers (section 4). Do not compare with our numbers.

## 4. Paper versus code
Read through a summarising tool, so each item is a digest, not a line-level check; details and file names are in [../code/orienternet.md](../code/orienternet.md).

| item | paper | released code / config |
|---|---|---|
| BEV extent | 32 x 32 m at 0.5 m (text) | x_max = 32, z_max = 32 and a lateral span of 2 * x_max in the projection: about 64 m wide x 32 m deep |
| number of scale bins | S = 32 | num_scale_bins = 33 (consistent with E2, i = 0..S) |
| rotations | K = 64 training, 512 test | 64 training; evaluation default 256 (128 if out of memory); pretrained entry overrides to 256 |
| map encoder backbone | VGG-16 | vgg19 (not pretrained), decoder 128 / 64 / 64, replicate padding, 3 downsamplings |
| unary location prior Omega | part of the model (E4) | `unary_prior: false` in the released config, so Omega is not used by the released model; the code adds a map log-prior only if enabled |
| matching normalisation | 1/(U Z) | scores divided by the number of valid template pixels per rotation |
| training prior range | 32 m | max_init_error = 48 in `mapillary.yaml`; 32 in the KITTI config and in the MGL evaluation |
| batch / hardware | 9 on 3 x V100 | 12 on 3 x RTX 3090, Adam lr 1e-4, max_steps 500,000 in the config, 340k iterations in the README |
| image encoder | ResNet-101 U-Net | config sets resnet101 over the FPN decoder; the class default is resnet50 |
| released numbers | Table 2: 15.78 / 47.75 / 58.98 | README: xy 14.37 / 48.69 / 61.7 and yaw 20.95 / 54.96 / 70.17 (1 / 3 / 5 m or deg, 256 rotations); sequential (10 frames) 29.73 / 73.25 / 91.17 |
| readout | argmax | argmax and a circular-mean expectation, both computed without gradient |

## 5. Transfer to our pipeline

| question | answer |
|---|---|
| what it needs from us | An ego-centred, gravity-aligned metric BEV (we have it: the cached 39-channel 200 x 200 grid at 0.5 m, so OrienterNet's whole image branch and polar lift are not needed), an SD-map raster in a global or local metric frame, a coarse location prior (GNSS-level, tens of metres), optionally a heading prior, and ground-truth ego poses for supervision (we have exact nuScenes poses, so unlimited synthetic prior offsets). No image features, no lidar. |
| what it gives us | The "no pose given" front end of A6: a posterior over the ego pose in the SD frame with explicit uncertainty (entropy, peak mass, ridge shape), a temporal product of such volumes (our time recursion applied to the pose), and a natural trust / abstain signal. It does not give lane assignment by itself. |
| conflicts | argmax and soft readout only at inference (training path is log-softmax plus NLL, differentiable, no argmax, no matching: no conflict); the exhaustive volume is memory-hungry (section 5.3); released code and weights are CC BY-NC style with a non-standard licence file: read only, re-implement; no mmcv; the original also needs a camera branch we do not need. Risk: the map encoder can memorise the few nuScenes locations (location leakage, 85 % overlap of validation locations with training per OVERVIEW section 2, A9). |
| reuse mode | Re-implement from description (about 300 lines of pure PyTorch: rotated templates, FFT correlation, log-softmax, trilinear NLL, belief warping). Idea-level reuse for the rest. |
| effort | M (days). Blocker: choose the BEV input (frozen 39 ch through a small trainable CNN versus our own dense lane-head output), the SD raster with route / lane-count channels, and a fast enough coarse-to-fine variant; a location-disjoint evaluation is needed to trust the number. |
| expected gain and risk | Gain is on the open item 2 (route attribute that degrades to road level with abstention) and on temporal fusion, not on route success when the pose is already good: E103e showed that a correlation-volume localisation works as a module (pose error 2.5 to 0.8 m at 2 m map error with a fixed temperature, 1.1 m inside the trained head) but did not improve the route success of the registration-free head. Risk: single-frame accuracy is the weak point (R@1 m of 15.8 % on MGL); needs the sequence product; repetitive straight roads give a longitudinal ridge by construction. |
| feeds which experiment / K id | K99 exists (pose volume, soft-argmax). Proposed new techniques, to be registered before use: (a) route head marginalised over the pose posterior (top-k poses weighted by P, no argmax), (b) coarse FFT volume at 1 m plus a fine Frenet-lateral volume, (c) temporal product of pose volumes with ego-motion warping and held volume under sensor failure, (d) peak mass / entropy of P as the abstain input. |

### 5.1 How the pose-volume matching would be wired to our BEV
1. **BEV side (template).** Two options. (i) The frozen 39-channel BEV goes through a small trainable CNN (Phi_BEV: a few 3 x 3 convs or 2 - 4 bottleneck blocks, 8 output channels plus a sigmoid confidence), exactly the role of OrienterNet's Phi_BEV but without the lifting. Gradient flows only into this small head (the foundation stays frozen). (ii) Use our own dense lane head output (heat map plus direction as sin / cos, 3 channels), optionally blurred: the template then lives in the same semantic space as the SD road raster, needs almost no extra parameters, and the pose NLL back-propagates into the dense head (this satisfies "gradient reaches every module"). Option (ii) is the cheaper first experiment; option (i) can use road edges and drivable area information that the lane field lacks.
2. **Map side.** Rasterise the SD route graph (centre-lines of all roads in range, not only the route) into class channels: road class, one-way flag, lane count, junction nodes, and a route flag. Learned class embeddings (16-d as in the paper) and a small CNN to N = 8 channels; keep the route flag out of the registration features so registration does not depend on the thing we want to infer (use it only downstream). Blur the road lines (sigma of 1.5 - 3 m) or let the map CNN learn the width, since the SD centre-line sits in the middle of the carriageway while the perceived lane centres are offset by up to about one lane width.
3. **Resolution and size (proposal, not tested).** Coarse stage at 1 m cells (average-pool the 0.5 m BEV by 2: 100 x 100 cells; the 0.75 m sigma heat-map lines survive at 1 m), tile about 192 m (template diameter 100 m plus twice the search radius of 20 m plus margin), search window ±20 m (GNSS-level prior), heading: ±12 deg in 3 deg bins if a compass / IMU yaw prior exists (9 bins), otherwise the full circle in 6 deg bins (60 bins). Fine stage at 0.5 m on a 128 m crop around the coarse peak, ±8 m and ±6 deg in 1.5 deg bins. Volume sizes (my arithmetic): coarse 192 x 192 x 60 = 2.2 M cells, FFT kernel tensor K x N x H x W = 60 x 8 x 36,864 = 17.7 M floats (71 MB), which is 15 times smaller than OrienterNet's test setting of 268 M floats; fine stage 15 x 8 x 65,536 = 7.9 M floats (31 MB). Runtime not measured; the chain must stay near its 31 ms budget, so run registration at a lower rate (the pose between runs follows from ego motion) or only when the volume's peak mass drops. The paper itself spends 51 ms on K = 512 on an RTX 2080. The code trick of sampling only one quadrant of the rotations with `grid_sample` and obtaining the other three exactly by 90-degree rotations cuts the sampling cost four-fold at no loss.
4. **Learnable parts.** BEV CNN (option i) or none (option ii); class embeddings and map CNN; optional temperature on the scores (code has it, default off); optional unary prior Omega (the released model switched it off; the paper uses it). Everything else (rotation, FFT correlation, softmax, NLL) is parameter-free and differentiable.
5. **Loss and labels.** NLL of the true ego pose in the SD frame: the nuScenes ego pose in the (simulated) SD map frame is exact, so apply random synthetic prior offsets (as in the paper's 32 m disc / the code's uniform square), random heading offsets, and the map-error models already used in E21 / E22, E59 and E74 (translation, drift). Trilinear interpolation of the log-volume at the true pose, with circular padding on the heading axis. Frames available: 8,414 TRAIN frames against about 760k+ for OrienterNet, so expect overfitting to the map; keep the map encoder small and evaluate on a location-disjoint subset of VAL.

### 5.2 What accuracy is needed for a lane-level assignment
- **Lateral.** A lane is about 3.0 - 3.75 m wide (general knowledge, not checked against the nuScenes map expansion). To place a perceived lane against the right side of the SD centre-line the lateral error must stay below half a lane, so about 1.75 m at the 95 % level (about 0.9 m standard deviation if Gaussian). Our own results agree: with graph-ICP registration route success is 0.923 / 0.883 / 0.782 at map errors of 1 / 2 / 3 m, without registration 0.640 / 0.305 / 0.153 (E59). So the target for a registration module is median error at or below 1 m and 90 % of frames inside 1.5 m laterally.
- **Heading.** A heading error eps displaces the map by eps times the look-ahead: 1 deg is 0.87 m at 50 m, 2 deg is 1.75 m at 50 m (arithmetic). The planner and the route flag look ahead 50 - 80 m, so heading must be good to about 1 - 2 deg. This is finer than OrienterNet's single-image numbers (orientation R@1 deg 22 % on MGL, 34 % on KITTI) but within reach after the sequence product (77.9 % at 1 deg on KITTI with 20 s).
- **Longitudinal.** Matters only near junctions and forks (where the route turns); along a straight road it is unobservable from road geometry, and the volume correctly shows a ridge. This is a reason to keep the volume and marginalise instead of taking an argmax: the route head sees the whole ridge. Single-image OrienterNet longitudinal R@1 m is 26 % on KITTI; with sequences 55 %.
- **What registration cannot fix.** The SD map's own geometric error (OSM-type centre-lines are a few metres off the true lane centre) and the offset between road centre-line and lane centre are not removable by a rigid pose; at lane level the perceived lane graph has to supply the lane structure, and the registration only has to place the perceived graph on the right road branch (road level, which is the fallback), then the route attribute is carried along the perceived lanes.
- **Existing evidence in our experiments:** E82 (lane-count-aware template, ICP) brought pose error from 1.11 to 0.62 m yet route success moved by at most 0.016; E48 refined a correlation search from a median 0.54 m to 0.43 m at a 1 m map error. So beyond roughly 1 m the registration accuracy is not the bottleneck of route success; the missing piece is the "no pose given" robustness (large offsets, wrong road, abstention), which is exactly what a pose volume offers.

### 5.3 Differentiability and what to do with the volume
- Differentiable: the softmax volume, the NLL at the true pose (trilinear sampling), the FFT correlation, the rotated-template sampling (bilinear `grid_sample`). Not differentiable and used only for readout: argmax over (x, y, theta); the code also computes a probability-weighted mean (circular mean for heading) with covariance, without gradient.
- For our training path, avoid the argmax: (a) warp the SD raster by the expected pose (mean from the volume) and feed it to the existing route head, as UMPE does with a regressed pose (note C02); only safe when the volume is unimodal; (b) evaluate the route head for the top-k poses of P and weight the route probabilities by P (selection with gradient through the weights), which handles ridges and two parallel roads; (c) pass entropy, peak mass inside a neighbourhood and the covariance to the trust / abstain output. (b) is the closest to the project rule and the most expensive; use k = 3 - 5.
- Sequences: the paper's E6 is a product of volumes under known relative poses. We already warp vertices and tokens with ego motion; warping the log-volume with the same transform and adding the new frame's log-volume is a second, cheap recursion on the pose, giving a held registration during short sensor outages (compare E67 - E72).

### 5.4 Scope warnings
- The paper never runs on 2D lane-line or centre-line maps; the matching signal for roads comes from OSM roads, footways, buildings and points. Our BEV carries lane structure but fewer map classes (no buildings in SD), so on roads without junctions the posterior will be long along the road: expected, not a bug.
- The volume cost scales with the tile and rotation count, not with the BEV size; keep the search window small by using the GNSS prior and a yaw prior (the code has a yaw-prior mask and a hard-window GPS fusion that we can copy conceptually).

## 6. Not found / open
- Optimiser, learning rate and iteration count in the paper (the code says Adam, lr 1e-4); parameter count; any ablation on K, on map or BEV resolution, or on prior size.
- Numeric values behind Figures 10 and 11; text of appendix B.1 (class list, OSM data version: the README says Geofabrik, November 2021); supplementary material.
- Which BEV extent the trained model really used (32 x 32 m per text, about 64 x 32 m per config).
- Exact construction and frame count of the KITTI Test2 split; the evaluation prior on MGL validation; the unexplained 6 ms in the runtime.
- Whether the released unary-prior-off model is the one behind the paper's tables.
- For our use: no study of matching a lane-centre heat map against a road-centre-line raster; no abstain rule from the volume (the follow-up 2603.01613 uses the peak mass only to size the fine search window, per note C28, not read here).

## 7. Our experiments that touch it
- **E48** (T3): learned registration of the map prior, a fine matching after a correlation search: median error 0.54 to 0.43 m at a 1 m map error (kept).
- **E59** (T3): graph-ICP registration plus geometric route transfer, no parameters: route success 0.923 / 0.883 / 0.782 at 1 / 2 / 3 m against 0.640 / 0.305 / 0.153 without.
- **E74** (T3): graph-ICP registration in front of the end-to-end head: route success nearly flat in the map error (+0.06 over the geometric transfer at 3 m) (kept).
- **E82** (T4): rough localisation with a lane-count-aware template (ICP), held through a failure: pose error 1.11 to 0.62 m, route success at most +0.016.
- **E103 / E103e** (T5): navigation attribute inside the network without registration (step-token head, graph transformer, learned localisation) 0.78 - 0.71 against 0.86 - 0.84 for rule graph plus registration; the correlation-volume localisation with soft-argmax (E103e) worked as a module (2.5 to 0.8 m at 2 m error, 1.1 m inside the trained head) but did not help route success; dropped in favour of the cross-attention route head (E104 / E104b, 0.853 - 0.820 at map errors 0 - 3 m).
- Source of these lines: `experiments/INDEX.csv`, `experiments/EXPERIMENT_TABLE.csv` and the loop notes `literature/sources/lit_extra10.md`; E103e details were not re-checked against the run records.
