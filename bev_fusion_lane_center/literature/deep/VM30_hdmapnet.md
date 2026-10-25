# VM30 - HDMapNet: full-text extraction

| field | value |
|---|---|
| paper | HDMapNet: An Online HD Map Construction and Evaluation Framework. Qi Li, Yue Wang, Yilun Wang, Hang Zhao. ICRA 2022 (the arXiv page itself lists no venue). arXiv 2107.06307, cs.CV / cs.AI; v1 2021-07-13, v2 2021-07-15, v3 2021-10-24, v4 2022-03-18 (current). Links opened: arxiv.org/abs/2107.06307, arxiv.org/html/2107.06307v4, ar5iv.labs.arxiv.org/html/2107.06307, github.com/Tsinghua-MARS-Lab/HDMapNet (front page, folder listings, raw files), arxiv.org/html/2208.14437 (MapTR, only for its speed table) |
| axis, tier, K ids | A2 (dense lane-centre field), tier 1 core; no K id of its own; the dense head of our chain is K62 (U-Net), HDMapNet is its published ancestor |
| level-1 note(s) | [VM30](../papers/VM30_hdmapnet.md) (home note), [SN15](../papers/SN15_hdmapnet-dense-bev-decoder-vectorisation.md) |
| code extraction | none written (decision of the session owner: code extraction only for the most meaningful papers). Repository https://github.com/Tsinghua-MARS-Lab/HDMapNet, GPL-3.0, read only; section 4 says where a later code extraction should start |
| sources read | arXiv abstract page (v4); arXiv HTML v4 and ar5iv, in about twelve separate queries (structure, section III-A, III-B, IV-A, IV-B, IV-C, tables I - IV by cell lookup, introduction and conclusion, vectorisation paragraph); GitHub front page, folder listings of `postprocess/`, `model/`, `data/`, `evaluation/`, raw files `LICENSE`, `postprocess/vectorize.py`, `postprocess/cluster.py`, `postprocess/connect.py`, `loss.py`, `train.py`, `model/hdmapnet.py`, `model/base.py`, `data/const.py`, `data/rasterize.py`, `data/dataset.py`, `data/vector_map.py`; MapTR arXiv HTML (Table 1 lookup). All read 2026-10-25. The PDF was not tried. Local files read for sections 5 and 7: `literature/OVERVIEW.md`, the two level-1 notes, `experiment_execution/plans/Experiment_Table_EN.md` (rows E17, E18, E31, E32, E33), `literature/register.md` (K62) |
| reading marks | every statement about the paper and about the code is **digest**: all pages came through a summarising fetch tool (no page was seen raw; the tool declined to reproduce text or whole tables, so tables were read by cell lookup and are per-cell digests). Line numbers and constants quoted for code are the tool's counts and readings; no code was copied and none is quoted. Statements about our own files and experiments are **checked** (files read). Section 5 is my analysis, not the paper's content. Anything marked "memory" was not seen |
| extracted by, date | extraction agent (Claude Sonnet 5.5, sub-agent of the literature session), 2026-10-25 |

## 1. Problem and claim
- Predict vectorised local map elements around the ego vehicle (lane dividers, road/lane boundaries, pedestrian crossings; **not** lane centre-lines) from six surround cameras, from lidar, or from both, in one frame, for nuScenes; the paper frames it as "HD semantic map learning": local maps built online as a complement to, not a replacement of, global HD map reconstruction.
- Gap closed: until then BEV map models were either segmentation only (no instances) or had no benchmark. The paper adds a dense three-head BEV decoder (semantic, instance embedding, direction), a rule-based vectoriser, and semantic (IoU, Chamfer distance) and instance (AP at Chamfer thresholds) metrics.
- The claim that matters for us: **dense first, vectorise second works on nuScenes and a matching-free dense loss is enough to train it**; the vectors themselves are never trained. The cost is a chain of hard rules after the network (argmax, clustering, thinning, greedy tracing) and, as measured by a later paper, under one frame per second.

## 2. Method as built

### Pipeline (digest; shapes marked "code" are from the repository defaults, the paper text gives none)
```
6 images (code default 128 x 352) -> EfficientNet-B0 (ImageNet) -> perspective features, 64 ch, stride 16 (code)
   -> neural view transformer: per camera an MLP over all perspective positions (paper Eq. 1;
      code: one two-layer MLP per view over the flattened spatial grid, shared across channels)
   -> geometric projection with camera extrinsics into the ego BEV, average over the 6 cameras   -> image BEV, 64 ch (code)
lidar points -> PointPillars variant, dynamic voxelisation, PointNet with a 64-d layer (paper), 128 ch out (code) -> pillar BEV
   -> concatenate (64 + 128 = 192 ch in code) when both are used
   -> BEV decoder: fully convolutional, ResNet "with three blocks" (code: ResNet-18 stem, layer1 - layer3, no max-pool,
      no layer4, then two up-sampling blocks to full BEV resolution), three branches:
        semantic   (code: 3 classes + background = 4 ch)
        instance embedding (code: 16 ch)
        direction  (code: 36 bins + 1 background = 37 ch)
   -> post-processing (rules, section "Inference" below) -> polylines + confidence + class
```
Variants: HDMapNet(Surr), HDMapNet(LiDAR), HDMapNet(Fusion). The sensor-fusion variant is plain concatenation of the two BEV feature maps (paper and code agree). A short-term temporal variant pastes earlier-frame feature maps into the current frame by ego pose and merges them with max-pooling before the decoder (paper, Surr only, Table IV); a long-term variant accumulates segmentation probabilities over many frames and is shown only in a figure (Fig. 6), with no function given and no number.

### Inputs, outputs
- Input: images (code default 128 x 352) with camera calibration and ego pose, and/or lidar points. Output in the ego frame, BEV extent and resolution **not stated in the paper text** (asked in three separate queries); code defaults: x in [-30, 30] m, y in [-15, 15] m at 0.15 m, i.e. a 60 x 30 m patch on a 400 x 200 pixel canvas (axis convention not checked). Three classes: divider (road divider and lane divider merged), pedestrian crossing, boundary; the paper calls the third "lane boundary", the tables "Boundary".
- Final output (code): per instance an integer pixel polyline, one confidence, one type id (0 divider, 1 pedestrian crossing, 2 boundary). The vectors are not trained and no loss touches them.

### Equations (numbered as in the paper; digest, plain-text rewrite)
- E1 view transformer: the camera-frame feature at (h, w) is an MLP of all perspective-view features of that image, F_c[h][w] = phi_V^{hw}(F_pv[1][1], ..., F_pv[H_pv][W_pv]).
- E2 pillar feature: f_j = PointNet({f_p : p in pillar j}), each point has 3 coordinates plus K extra features.
- E3 variance term of the discriminative loss: L_var = (1/C) sum_c (1/N_c) sum_j [ ||mu_c - f_j|| - delta_v ]_+^2; C = number of ground-truth instances, N_c = pixels of instance c, mu_c = its mean embedding, [x]_+ = max(0, x).
- E4 distance term: L_dist = 1 / (C (C - 1)) sum over ordered pairs cA != cB of [ 2 delta_d - ||mu_cA - mu_cB|| ]_+^2.
- E5 L = alpha L_var + beta L_dist, alpha = beta = 1, delta_v = 0.5, delta_d = 3.0.
- Direction step (no number in the paper): C_next = C_now + Delta_step * D, D the unit vector of the predicted bin; Delta_step "predefined".
- E6 IoU = |P and G| / |P or G| on an H x W x D grid (D classes), curves rasterised first.
- E7 directional Chamfer: CD_Dir(S1, S2) = mean over p in S1 of min over q in S2 of ||p - q||. E8 CD = CD_Dir(S1, S2) + CD_Dir(S2, S1) (a sum). The Table I caption says CD is "the average of the two"; see section 3 for the conflict.
- E9 AP = (1/10) sum over recall r in {0.1, ..., 1.0} of AP_r, AP_r = precision at recall r.

### Losses (what the paper says; the code differs in places, section 4)
- Semantic: cross-entropy (paper). Instance embedding: discriminative loss E3 - E5 (De Brabandere et al.), alpha = beta = 1, delta_v = 0.5, delta_d = 3.0, no regularisation term in the code. Direction: classification into N_d classes spread uniformly over the unit circle, softmax; **each lane pixel has two positive entries** (the travel direction is unknown, so the tangent and its opposite are both labelled 1); non-lane pixels carry a zero vector and are excluded from back-propagation. Loss weights are not given in the paper (code: seg 1.0, var 1.0, dist 1.0, direction 0.2).
- Assignment rule: **none**. Everything is a dense per-pixel loss on rasterised labels; there is no Hungarian matching and no vector-level loss. The discriminative loss forms its clusters from the ground-truth instance map, not from a matching.

### Labels / ground truth (paper: not described; from code, digest)
- Map layers (nuScenes map expansion): road dividers and lane dividers (one class), pedestrian crossings (the two longest edges of each crossing polygon become lines), and "contours": the boundary rings of the union of road-segment and lane polygons, clipped to the patch (no set difference, no centre-line anywhere). Lines are resampled at 1 m in the vector step.
- Rasterisation (`preprocess_map` and helpers): lines drawn with OpenCV polylines at thickness 5 px (0.75 m at 0.15 m); each polyline gets a unique instance id running across all classes (stored as uint8, so more than 255 polylines would wrap); where thickened lines of different classes overlap the higher class index wins; the semantic target is a boolean map per class plus a background channel (4 channels).
- Direction label: per segment, the angle of the segment vector (counter-clockwise from +x in canvas coordinates, no y flip) is divided by 10 degrees, rounded to the nearest bin and shifted by one, so the label is 1 .. 36 and 0 is background. A forward pass and a backward pass (the point order reversed, so +180 degrees = +18 bins) are drawn and each pixel keeps the label of the **last segment drawn** over it. The two one-hot maps (37 channels) are added and divided by their per-pixel sum, so a lane pixel carries 0.5 at two bins (not 1 at two bins as the paper says) and a background pixel carries 1 in channel 0. Label augmentation: none visible.

### Training recipe (paper: Adam, learning rate 1e-3, nothing else; rest from code defaults, digest)
- Code defaults: Adam lr 1e-3, weight decay 1e-7, StepLR factor 0.1 every 10 epochs, 30 epochs, batch size 4, gradient clipping at max norm 5.0, semantic positive weight 2.13, instance dimension 16, 36 angle classes, line thickness 5. Hardware and training time: not found.

### Inference (the hidden rule steps; paper has three sentences, the rest is code, digest)
Paper: cluster the instance embeddings with DBSCAN, apply NMS to remove redundant instances, greedily link the pixels of each instance along the predicted directions. Code (`postprocess/`), per image and per class (skipping background):
1. Class probabilities by softmax over the classes; the class mask is the **hard argmax one-hot** (non-differentiable).
2. Mask clean-up: morphological closing with a 5 x 5 elliptical kernel, then every 8-connected component of at most 100 pixels (2.25 m2 at 0.15 m) is deleted.
3. DBSCAN on the 16-d embedding of the remaining mask pixels, **after per-dimension standardisation**, eps = 1.5 and min_samples = 50 (the class defaults 0.35 and 200 come from the LaneNet code this file is derived from and are overridden); noise (label -1) is dropped; each cluster is one instance. Position is not a clustering feature.
4. Confidence of an instance = mean class probability over all of its mask pixels. No score threshold is applied in the vectoriser; the confidence is returned for the AP ranking.
5. "NMS" is a thinning of the class-probability map to its ridge: a pixel survives if it equals the local maximum in a 1 x 5 window or in a 5 x 1 window (tolerance 1e-4); which window applies is decided per pixel by comparing 9 x 5 and 5 x 9 average pools (vertical or horizontal orientation of the line).
6. Ordering: the ridge pixels of an instance are sorted by x or y (whichever span is larger) and then by a greedy nearest-neighbour chain with two ad-hoc rules (skip a candidate more than 3 px away that lies within 5 px of an already chained point; stop when a candidate more than 10 px away appears after 90 % of the points); the pairwise distance matrix is N x N for N ridge pixels.
7. Direction-guided tracing (`connect_by_direction`, step 7 px = 1.05 m, bins of 10 degrees): the direction map is first reduced to the **top-2 bins** per pixel (softmax, take the argmax, zero a window of 5 bins around it, take the next argmax). A **random** ridge pixel is the start. Repeatedly: take the first unused direction slot at the current end point; target = end point + 7 px along that bin's angle; delete every remaining candidate pixel within 6 px of the end point (second NMS); take the candidate nearest to the target if it is closer than 50 px, otherwise stop; append it; mark at the new point the slot nearest to the reverse angle as used (so the walk does not return). Grow forward from the start, then reverse the list and grow from the other end. Stops: no unused slot, no candidate left, or the 50 px gate fails.
8. Output: int32 pixel polylines with one vertex about every 7 px, plus confidence and type. No simplification, smoothing, resampling in metres, or merging of fragments is applied in this file (the return variable is named "simplified" but nothing simplifies).
What is non-differentiable and untrained: steps 1 - 7, all of them; they run on CPU (OpenCV, scikit-learn, NumPy loops). Hard-coded pixel constants tie them to the 0.15 m grid.

## 3. Numbers (digest; nuScenes validation; protocol of this paper only)

Table I, semantic segmentation, IoU in % (higher is better) and Chamfer distances in m (lower is better), cell lookups; per class four values IoU / CDP / CDL / CD, "All" = mean over the three classes (my arithmetic: the per-class values average to the All values in every row I tried).

| method | Divider | Ped crossing | Boundary | All classes |
|---|---|---|---|---|
| IPM* | 14.4 / 1.149 / 2.232 / 2.193 | 9.5 / 1.232 / 3.432 / 2.482 | 18.4 / 1.502 / 2.569 / 1.849 | 14.1 / 1.294 / 2.744 / 2.175 |
| IPM(B) | 25.5 / 1.091 / 1.730 / 1.226 | 12.1 / 0.918 / 2.947 / 1.628 | 27.1 / 0.710 / 1.670 / 0.918 | 21.6 / 0.906 / 2.116 / 1.257 |
| IPM(CB) | 38.6 / 0.743 / 1.106 / 0.802 | 19.3 / 0.741 / 2.154 / 1.081 | 39.3 / 0.563 / 1.000 / 0.633 | 32.4 / 0.682 / 1.42 / 0.839 |
| Lift-Splat-Shoot | 38.3 / 0.872 / 1.144 / 0.916 | 14.9 / 0.680 / 2.691 / 1.313 | 39.3 / 0.580 / 1.137 / 0.676 | 30.8 / 0.711 / 1.657 / 0.968 |
| VPN | 36.5 / 0.534 / 1.197 / 0.919 | 15.8 / 0.491 / 2.824 / 2.245 | 35.6 / 0.283 / 1.234 / 0.848 | 29.3 / 0.436 / 1.752 / 1.337 |
| HDMapNet(Surr) | 40.6 / 0.761 / 0.979 / 0.779 | 18.7 / 0.855 / 1.997 / 1.101 | 39.5 / 0.608 / 0.825 / 0.624 | 32.9 / 0.741 / 1.267 / 0.834 |
| HDMapNet(LiDAR) | 26.7 / 1.134 / 1.508 / 1.219 | 17.3 / 1.038 / 2.573 / 1.524 | 44.6 / 0.501 / 0.843 / 0.561 | 29.5 / 0.891 / 1.641 / 1.101 |
| HDMapNet(Fusion) | 46.1 / 0.625 / 0.893 / 0.667 | 31.4 / 0.535 / 1.715 / 0.790 | 56.0 / 0.461 / 0.443 / 0.459 | 44.5 / 0.540 / 1.017 / 0.639 |

(* = labels projected from the 2D HD map into the perspective view.) CDP = Chamfer distance label to prediction ("precision"), CDL = prediction to label ("recall"). **Inconsistency in the paper itself:** Eq. 8 defines CD as the sum of the two directions, the caption says "the average of them", and the printed CD (for example 0.667 for the Fusion divider from CDP 0.625 and CDL 0.893) equals neither (mean 0.759, sum 1.518). I could not resolve it (code not read for the metric); treat the CD column as unverified and prefer IoU, CDP, CDL.

Table II, instance detection, AP in % at Chamfer thresholds 0.2 / 0.5 / 1.0 m, last value = mean over the three (mAP); All = mean over classes.

| method | Divider | Ped crossing | Boundary | All classes |
|---|---|---|---|---|
| IPM(B) | 2.6 / 9.8 / 19.6 / 10.7 | 1.6 / 4.8 / 7.8 / 4.7 | 2.2 / 9.2 / 23.7 / 11.7 | 2.1 / 7.9 / 17.0 / 9.0 |
| IPM(CB) | 10.2 / 25.0 / 36.8 / 24.0 | 2.0 / 7.8 / 12.2 / 7.3 | 10.1 / 27.9 / 45.5 / 27.8 | 7.4 / 20.2 / 31.5 / 19.7 |
| Lift-Splat-Shoot | 9.1 / 23.8 / 35.9 / 22.9 | 0.9 / 5.4 / 8.9 / 5.1 | 8.5 / 22.9 / 41.2 / 24.2 | 6.2 / 17.4 / 28.7 / 17.4 |
| VPN | 8.8 / 22.7 / 34.9 / 22.1 | 1.2 / 5.3 / 9.0 / 5.2 | 9.2 / 24.1 / 42.7 / 25.3 | 6.4 / 17.4 / 28.9 / 17.5 |
| HDMapNet(Surr) | 13.7 / 30.7 / 40.6 / 28.3 | 1.9 / 7.4 / 12.1 / 7.1 | 13.7 / 33.9 / 50.1 / 32.6 | 9.8 / 24.0 / 34.3 / 22.7 |
| HDMapNet(LiDAR) | 1.0 / 5.7 / 15.1 / 7.3 | 1.9 / 5.8 / 9.0 / 5.6 | 5.8 / 20.2 / 39.6 / 21.9 | 2.9 / 10.6 / 21.2 / 11.6 |
| HDMapNet(Fusion) | 15.0 / 32.6 / 46.0 / 31.2 | 6.4 / 13.6 / 17.8 / 12.6 | 26.7 / 51.5 / 65.6 / 47.9 | 16.0 / 32.6 / 43.1 / 30.6 |

Table III (Fusion, IoU over all classes by weather): normal 44.9, night 39.3, rainy 38.7. Table IV (Surr, short-term temporal fusion, IoU): 1 frame 32.9, 2 frames 35.8, 4 frames 36.4.

Camera against lidar against fusion (text and tables): camera-only is better on dividers (IoU 40.6 against 26.7) and marginally on crossings (18.7 against 17.3, instance AP 7.1 against 5.6); lidar is better on boundaries (44.6 against 39.5), which the authors attribute to elevation changes at boundaries; fusion is best everywhere, and its largest gains over camera are on crossings (IoU 31.4) and boundaries (instance AP 47.9). Radar: I saw no statement (not checked in the paper text). No weather result exists for camera-only or lidar-only.

- Ablations that decide transfer: the paper has no ablation of the dense heads, the loss terms, the direction bins, or any post-processing parameter. The baselines (IPM, IPM(B), IPM(CB), LSS, VPN) share the encoder and decoder and differ only in the view transformation, so Tables I and II isolate the view transformer. The paper never says how the baselines are turned into instances (embedding, DBSCAN, tracing or something else), so Table II may mix pipelines.
- Intro claims against the tables (my arithmetic on the digest): fusion minus the best Table I baseline (IPM(CB), 32.4) is +12.1 IoU points, which matches the stated 12.1 but is +37 % relative, not the "more than 50 % in all metrics" of the abstract; the stated +13.1 mAP matches fusion minus VPN (30.6 - 17.5), not minus the best baseline IPM(CB) (30.6 - 19.7 = +10.9, +55 %, which matches the "55.4 % over IPM" in the text). I could not reconcile the abstract's wording.
- Compute: **the paper gives no parameter count, no FPS and no post-processing time** (asked in the introduction, results and conclusion queries; none reported) although it calls the map immediately applicable to real-time planning. Second-hand: MapTR (arXiv 2208.14437) Table 1 lists HDMapNet at 0.8 FPS (camera, EfficientNet-B0), 1.0 FPS (lidar, PointPillars) and 0.5 FPS (camera + lidar); the caption says the FPS values were measured on one machine with an RTX 3090; the paper does not say whether HDMapNet's post-processing is included, and calls its grouping "heuristic and time-consuming". My own inference, not a measurement: a P-MapNet table cited in our project notes (`literature/sources/lit17/C_map_matching_planner.md`, second-hand, not re-checked) lists 21.2 FPS for HDMapNet, about 26 times more; the gap is consistent with the CPU post-processing dominating when it is counted, but GPU, input size and protocol differ, so it is not proof.
- Caveats: the AP thresholds here are 0.2 / 0.5 / 1.0 m; MapTR and its successors use 0.5 / 1.0 / 1.5 m, and MapTR reports 23.0 / 24.1 / 31.0 mAP for HDMapNet camera / lidar / fusion under that protocol (per-class AP_ped 14.4 / 10.4 / 16.3, divider 21.7 / 24.1 / 29.6, boundary 33.0 / 37.9 / 46.7, digest). The close agreement of 22.7 and 23.0 is a coincidence of two different protocols, not a reproduction. The standard nuScenes train / validation scene split is used (code), which is location-overlapping; the project's notes on VM46 quote a drop of HDMapNet IoU from 38.8 to 21.2 on a geographically disjoint split (second-hand, not re-checked). Classes are dividers, boundaries and crossings: none of these numbers is a centre-line number.

## 4. Paper versus code
Repository: https://github.com/Tsinghua-MARS-Lab/HDMapNet (owner Tsinghua-MARS-Lab; 770 stars, 130 forks, 13 commits, `main`; commit hash and date not shown on the page, so not recorded; README: ICRA 2022, "CVPR 2021 Workshop best paper nominee"; no pretrained model mentioned in the README). **Licence: `LICENSE` is the GNU General Public License, Version 3, 29 June 2007 (header lines read).** GPL-3.0 means read only: describe, re-implement from your own description, copy nothing. No clone, no download was made. Stack seen: Python, PyTorch, torch-scatter, OpenCV, Shapely, scikit-learn, EfficientNet; no mmcv.

Differences between the paper text and the code (all digest, to be confirmed by a later code extraction):
- Semantic loss: the paper says cross-entropy; the code uses sigmoid binary cross-entropy with logits and a positive weight of 2.13 on four channels (background included), while inference uses softmax over the classes and an argmax, so training and decoding use different output functions.
- Direction loss: the paper says two entries set to 1 and softmax; the code builds a target of 0.5 at two bins, applies softmax, then a **binary** cross-entropy on the probabilities (all 37 channels, so each of the 35 zero bins is also penalised), masked to lane pixels, divided by (lane pixels x 37), weight 0.2. The optimum is therefore a 0.5 / 0.5 distribution on the two bins.
- Discriminative loss: the paper's distance hinge uses 2 delta_d; the code (as read) uses delta_d and an extra division by 2 per unordered pair, and a regularisation term exists but is zero and not used. The weights alpha = beta = 1 are applied as separate scales in the training loop.
- Not in the paper at all: DBSCAN eps 1.5 / min_samples 50 on standardised embeddings, the mask closing, the 100-pixel component filter, the ridge thinning, the sort heuristics, the random tracing start, the top-2 bins with a 5-bin suppression window, Delta_step = 7 px, the 50 px gate, N_d = 36, label thickness 5 px, the BEV range 60 x 30 m at 0.15 m, the image size 128 x 352, the 16-d embedding, the optimiser schedule (30 epochs, batch 4, StepLR).
- Possible defect flagged by the reading tool, unverified: in `model/base.py` the direction branch seems to call the embedding branch's first up-sampling block instead of its own, so the two heads would share it. Other tool-flagged issues are NumPy-version breakages (`np.int`, `np.bool`) in the post-processing.
- Reproducibility: the tracing starts from a random ridge pixel, so repeated runs on the same output can give different polylines.

Start here for a later code extraction (not written; the session owner limited code extractions to the most meaningful papers):
- `postprocess/vectorize.py` (`vectorize`, `get_pred_top2_direction`) and `postprocess/cluster.py` (`LaneNetPostProcessor`: closing, component filter, DBSCAN) for the decoding chain and every constant of section 2.
- `postprocess/connect.py` (`sort_points_by_dist`, `connect_by_step`, `connect_by_direction`) for the tracing rule: direction slots, 7 px step, 6 px NMS, 50 px gate, random start.
- `loss.py` (`DiscriminativeLoss`) and `train.py` (inner loop, about lines 85 - 105) for the exact loss terms, margins, masks and denominators.
- `data/rasterize.py` (`preprocess_map`, `mask_for_lines`, `get_discrete_degree`, `overlap_filter`) and `data/dataset.py` (`get_semantic_map`) for the label construction, in particular the overlap rule at crossings.
- `evaluation/` (`chamfer_distance.py`, `AP.py`, `iou.py`, `dataset.py`) was only listed, never read: the sampling of curve points, the matching and the CD definition behind the Table I discrepancy are open.

## 5. Transfer to our pipeline

| question | answer |
|---|---|
| what it needs from us | Only a BEV raster: the heads and the vectoriser are independent of the encoder, so the frozen cached 200 x 200 grid at 0.5 m (39 ch) is enough; no ego poses (single frame), no SD map, no image features, no lidar for the dense part. The post-processing constants are in pixels of a 0.15 m grid and would have to be re-derived in metres: 50 DBSCAN samples are 1.1 m2 at 0.15 m but 12.5 m2 at 0.5 m, the 7-px step (1.05 m) is 2 cells of ours, and a 5-px line is 0.75 m against our heat-map sigma of 0.75 m |
| what it gives us | Axis A2 and the dense-to-sparse bridge of A3: (a) the **published rule baseline** for converting dense maps to vectors on nuScenes, with measurable weaknesses (vertex density, false fragments, no junctions, speed); (b) direction as a **classification over bins** with a soft multi-hot target, which can be multi-modal; (c) a matching-free **instance embedding** loss that splits a mask into polylines; (d) metric definitions (mask IoU, bidirectional Chamfer, AP over recall levels) |
| conflicts | Every step after the heads is a rule and non-differentiable (hard argmax, morphology, connected components, DBSCAN, ridge test by equality, greedy chain, random start, greedy tracing): no gradient reaches the heads from the vectors, which is exactly open item 6 of the overview, not a solution to it. Classes are dividers, boundaries and crossings, not centre-lines: the two-positive direction target exists because a divider has no travel direction. The embedding head gives one instance id per pixel, which cannot express shared trunks of a DAG. GPL-3.0 (read only). Fixed pixel constants. No fork or merge handling (the paper text, as queried, never mentions junctions, forks, merges or topology) |
| reuse mode | Idea only for the direction bins and the discriminative loss (re-implement from the description, a few lines each); re-implement the vectoriser from the description in metres as a **comparison baseline**; adapt no code (GPL) |
| effort | S (hours) for a rule-baseline vectoriser from the description applied to our dense head output and scored with our S, Q, vertices per 100 m; M (days) for a direction-bin head with a multi-modal target at fork and merge cells and its read-out by the ray fan, blocker: label construction of the bin target from our GT graph and a clean junction evaluation (J at 6 m) |
| expected gain and risk | Baseline: low gain, high information (it fixes a number for "dense then rule" on our own scores, but our own E31 / E33 tracer is already a stronger descendant). Direction bins: possible gain on J (champion 0.265 against 0.408 at equal sparsity) if the 2-channel regressed direction collapses to a mean direction at forks and merges, which is my hypothesis and has not been measured; risk: a tie in S on clean frames as with most head changes, and extra channels on the dense head. Embedding head: low expected gain, parallel lanes are separated by position and direction already |
| feeds which experiment / K id | Proposed techniques, to be registered in `register.md` before use: (1) a multi-modal direction-bin side head (softmax over N_d bins, soft multi-hot target with one mode per outgoing branch at forks and per incoming branch at merges, masked normalised BCE or KL); (2) the HDMapNet-style vectoriser as a documented rule baseline (R-HD, re-implemented in metres). No new K id is assigned by this extraction |

### 5.1 What exactly is the dense-to-sparse conversion (answer to the main question)
It is a chain of seven rule steps with three learned rasters as input: class mask by hard argmax; morphological closing and a small-component filter; DBSCAN of the mask pixels on the standardised embedding (this is where instances are made); ridge thinning of the class probability by local-maximum tests; ordering of the ridge pixels; direction-guided tracing that walks from a random ridge pixel in both directions in 7-pixel steps and snaps each step to the nearest remaining ridge pixel; polylines, confidence and type out. The embedding says which pixels belong together, the thinned probability says where the line is, the direction says in which order to walk. Nothing in the chain is trained on its own output, no step has a parameter that a loss can reach, and the final polyline is a set of ridge pixels, not a fitted curve.

### 5.2 How junction and fork cells are handled by the direction head
Not at all by design. The label of a pixel is a single polyline's forward / backward pair: within a class the last segment drawn over a pixel wins, across classes the higher index wins, and the instance map likewise gives each pixel one id. So at a crossing the head is trained on one polyline's direction (the others are overwritten), and the embedding is one vector per pixel. At inference the top-2 bins are meant as "forward and backward" of one line (bins at least 3 apart because of the 5-bin suppression window), not as two outgoing branches; the tracer marks the reverse slot as used, so it walks one line at a time and never branches. A fork is therefore either two independent polylines that happen to touch, or a gap. The paper does not discuss the case (none of the queries on the text returned a mention of junctions, forks or merges). The useful detail for us: because the head is a **classification over bins** it can in principle represent two modes at a fork cell where a regressed unit vector (our 2-channel direction) can only give their mean; HDMapNet uses this only for the divider ambiguity, but a target with one mode per outgoing branch is a small change.

### 5.3 Comparison with our dense head and verified-ray builder
| aspect | HDMapNet | ours (loop 16 champion) |
|---|---|---|
| dense output | 3-class semantic + 16-d instance embedding + 36-bin undirected direction | lane centre heat map (Gaussian sigma 0.75 m) + 2-channel travel direction |
| target | dividers, boundaries, crossings; direction ambiguous by construction | centre-lines with a known travel direction |
| grid | 60 x 30 m at 0.15 m (400 x 200) | 100 x 100 m at 0.5 m (200 x 200), cached |
| conversion | hard argmax -> morphology -> DBSCAN -> ridge thinning -> chain -> random-start tracing, all on CPU | tip with position and heading -> teardrop of 12 rays read by bilinear grid_sample -> hazard / survival validity -> b children -> DAG with merge radius 1.5 m, snap 1.0 m -> junction head -> fork relocation, on the training path |
| differentiable after the dense head | no | yes |
| vertices | one per about 1.05 m, roughly 95 per 100 m (my arithmetic from the 7 px step; no simplification step seen) | 13.5 per 100 m (control 21.8, Douglas-Peucker oracle 14.3) |
| forks / merges | none | junction head, J = fork / merge F1 at 6 m |
| false-lane guard | component area >= 100 px (2.25 m2), DBSCAN noise, confidence = mean pixel probability, no threshold in the vectoriser | verification along each ray, hazard, sparsity penalties |
| speed | not stated in the paper; 0.5 - 1.0 FPS in MapTR's table | 31 ms per frame |

### 5.4 What in its post-processing is a baseline we should beat
- **Vertex density:** about 95 vertices per 100 m against our 13.5. That is about seven times the champion's vertex density, which our sparsity-aware score S would punish heavily; the fair baseline is HDMapNet followed by a Douglas-Peucker pass, which the oracle value 14.3 already represents.
- **False and fragmented lanes:** the only guards are the 100-pixel component filter, DBSCAN noise and min_samples; one embedding cluster per instance means a lane can split into several clusters when the embedding drifts and two parallel lanes can merge when it does not; the confidence is a mean over mask pixels and is not a calibrated validity. This is the "false lane" failure mode the project wants to avoid.
- **Determinism and speed:** a random start and an N x N distance matrix per instance; no timing is published.
- **Topology:** none; the output is a set of disjoint polylines, not a graph.
- A rule baseline R-HD can be built in an afternoon from the description above; note it needs an embedding head to be faithful, otherwise connected components replace DBSCAN and it becomes a variant of our own E31 tracer.

## 6. Not found / open
- Paper text: BEV extent and resolution, image size, K, MLP widths, N_d, Delta_step, DBSCAN and NMS parameters, epochs, batch size, hardware, training time, line thickness for the evaluation raster, number of sampled points per curve in the Chamfer metric, how the baselines are vectorised, the function of the long-term fusion (all absent from the HTML text I could query).
- Speed: no HDMapNet-side timing of the network or of the post-processing; MapTR's FPS does not say whether post-processing is included.
- Table I: the CD column conflicts with both Eq. 8 and the caption; the evaluation code behind it was not read.
- Radar: no statement seen. Pretrained weights: none mentioned in the README. Commit hash of the code state: not shown on the page.
- The PDF was not tried, so no appendix or supplement was seen. Everything here is digest level; numbers must be checked against the paper before they enter a report.

## 7. Our experiments that touch it
- E17 (CRN heads scaled to 8,414 training frames: FCN against U-Net, FCN 0.264 to 0.493, U-Net +0.12 to 0.613) and E18 (ConvNeXt-V2 + transformer bottleneck as dense decoder, 0.507, 0.11 below the U-Net): our dense head is a U-Net (K62) instead of HDMapNet's ResNet decoder; no embedding or direction-bin head was ever tried.
- E31 / E33 (heat map + direction field to a directed graph with the tracer): F1 0.879 / 0.851 (VAL / TEST), route F 0.765 / 0.702; even a perfect field gives route F 0.838. This is our counterpart of HDMapNet's tracing, with a single-valued travel direction.
- E32 (sparse DETR-style LaneFormer instead of the grid): 0.058 F1 and 0.178 route F below the grid approach, the evidence for "dense first".
- Not tried: a bin-classification direction head, a multi-modal target at forks and merges, an instance-embedding loss, the HDMapNet vectoriser as a documented baseline scored with S, Q and vertices per 100 m.

## Corrections to the level-1 notes (VM30, SN15)
- VM30 "softmax CE on lane pixels (direction)": the code (digest) uses a binary cross-entropy on softmax probabilities against a 0.5 / 0.5 target, masked to lane pixels, normalised by lane pixels x 37, weight 0.2. VM30 "CE (semantic)": the code uses sigmoid BCE with positive weight 2.13 over four channels (the paper text says cross-entropy). The paper's "two entries set to 1" is not what the code builds.
- VM30 fit line "direction as a classification with both directions positive is the published way to avoid a direction ambiguity": the two-positive target exists because dividers and boundaries have no travel direction (an undirected tangent); it does not apply to centre-lines with a known travel direction, and it is not a fork or merge mechanism. The transferable part is the classification over bins (multi-modal), not the two-positive trick. SN15 "the multi-class direction loss with two positive labels is the one detail we do not use": it should not be used for centre-lines.
- VM30 "0.8 FPS camera (MapTR's table)": confirmed for MapTR's Table 1 (digest), which also lists lidar 1.0 and fusion 0.5 FPS on an RTX 3090 per its caption; the paper itself reports no speed, and whether post-processing is included is not stated. SN15 "Compute: not stated": confirmed.
- VM30 canvas and range (60 x 30 m, 0.15 m, 200 x 400, thickness 5, 36 bins, DBSCAN 1.5 / 50, step 7): consistent with the code defaults I read; the paper text states none of them (SN15 already says so). The note's "no resampling interval": consistent (nothing resamples the output); the ground truth is resampled at 1 m.
- VM30 "Differentiable? no after the heads" and "no fork / merge handling": confirmed.
- SN15 "Tried in our experiments: E17 / E18 (U-Net dense head kept; dense head + tracer beats sparse)": E17 / E18 compare dense decoders; the dense-head-plus-tracer result against sparse is E31 / E33 against E32.
- SN15 "radar only mentioned as a possible extension": not checked here; I saw no radar statement.
