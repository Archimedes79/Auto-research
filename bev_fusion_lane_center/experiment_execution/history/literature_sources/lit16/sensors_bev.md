# Cluster SN - BEV encoders, camera-radar fusion, the frozen foundation, what a head can read from it (loop 16 literature research)

Date: 2026-10-23. Prefix SN. Schema: `SCHEMA.md`; project facts: `PROJECT_CONTEXT.md`. Version 2 (complete pass; extendable).

## 1. Coverage statement

- Sources opened (reading only): arXiv HTML / ar5iv pages, GitHub raw files and listing pages, one Hugging Face model page, search-result summaries of CVF / ICLR / proceedings pages, a nuScenes devkit source file, a third-party nuScenes tutorial page, a reseller page with the radar datasheet. Every page read went through a reader model that returns a summary, so numbers were cross-checked where two pages overlap; conflicts are written into the cards. I downloaded nothing; one PDF fetch (BEVFusion) returned binary and the harness stored it in its own tool-results folder (not used).
- Depth C+M: **GaussianCaR** (paper HTML, README, LICENSE, pyproject, configs `task/map.yaml`, `module/gaussiancar.yaml`, `data/nuscenes.yaml`, source files `model.py`, `radar_encoder.py`, `fuser.py`, `dpt.py`, `render.py`, `loading.py`, `labels.py`, `nuscenes_map_api.py`, HF page, vendored rasteriser licence, issues page).
- Depth M (method + experiments read): GaussianBeV, BEVCar, Simple-BEV, CRN, RESAR-BEV, FIN, CTAB, RCBEVDet, CRAFT, HyDRa, Horizon3D, BEVFusion (MIT and PKU), BEVFormer, LSS, BEVDepth, BEVDet, TopoBDA (radar rows + supplement), TopoNet, HDMapNet, MapTR, MapTRv2, LaneSegNet, StreamMapNet, RNTR, CenterLineDet, Lilja et al. (location leakage), PON, Occ3D, parametric-depth BEV, radar BEV observability mask (Sebastian et al.), CRKD, BEVDistill, OccFeat, UDGA, UFCE benchmark, nuScenes paper radar statements, devkit radar class, radar-camera review.
- Depth A only: LP-FT (fine-tuning vs linear probing), RaCFormer, RCM-Fusion, SpaRC, CRT-Fusion, CR3DT, Gu et al. (map uncertainty; direct BEV feature attention), a stacking paper, BEVGuide, CaR1. Not read: SOLOFusion / BEVDet4D, StreamPETR, GaussianLSS, MapPrior.
- Not found: any camera-radar lane-graph or HD-map paper other than the radar rows of TopoBDA; any paper that trains lane heads on frozen cached camera-radar BEV rasters; any paper that measures the train-versus-unseen feature gap of a frozen BEV encoder; a published range-binned (near / mid / far) IoU for lane or divider layers (FIN has it only in figures). A calibration-only visibility mask has no dedicated paper; card SN13 is a derivation from checked ingredients and says so.
- Caveat: several papers have text / table inconsistencies (flagged by the reader); I give the table number and say when the text disagrees. Cross-paper tables are not protocol-matched (see SN07).

## 2. Cards (ordered by relevance to our design)

### CARD SN01 - GaussianCaR (our frozen camera-radar foundation)
- papers: GaussianCaR: Gaussian Splatting for Efficient Camera-Radar Fusion (Montiel-Marin, Antunes-Garcia, Sanchez-Garcia, Llamazares, Caesar, Bergasa; ICRA 2026) https://arxiv.org/abs/2602.08784 ; code https://github.com/santimontiel/gaussiancar ; project page https://www.santimontiel.eu/projects/gaussiancar ; weights https://huggingface.co/santimontieleu/gaussiancar
- depth: M+C
- stage: B
- idea: Every camera feature-map cell (6 cameras, EfficientViT-L2 at 1/8 resolution) and every radar point (7 accumulated sweeps of 5 radars) becomes a 3D Gaussian carrying a 128-d feature. Both sets are splatted orthographically (alpha blending) onto a 200 x 200 BEV grid, fused by a 4-stage CMX-style module, decoded by a DPT-style conv decoder and read by segmentation heads. The released map checkpoint predicts seven binary map layers.
- formulation: Camera: D = 64 uniform depth bins in [1, 61] m (config `depth_num` 64, `depth_max` 61.0); Gaussian mean = softmax-weighted bin depth back-projected with K and extrinsics + learned 3-D offset; covariance from the spread of the depth distribution scaled by 0.5 (6 numbers); opacity sigmoid, Gaussians with alpha < 0.01 dropped (the renderer drops <= 0.05). Radar: per point 3 coordinates + 51 feature columns (one-hots of dynprop / ambig_state / invalid_state, pdh0 ordinal, rcs, vx, vy, vx_comp, vy_comp, rms values, time lag, 2 flags); only |x|,|y| < 50 m and |z| < 10 m kept; past sweeps moved to the reference frame by ego poses (devkit `from_file_multisweep`, 7 sweeps, min distance 1 m); PTv3 U-Net (encoder depths 1 each, channels 64/64/128/128/256, patch 64, grid 2.0), four MLP heads (2 x 512, LN, GELU, dropout 0.1) give offset, 6-number covariance (softplus diagonal, bounded correlation), opacity, 128-d feature; one Gaussian per point, max 3500 points. Splat: F = sum_i f_i a_i prod_{j<i}(1 - a_j). Fuse: 4 stages at 200 / 100 / 50 / 25 cells, 128 channels (config `embed_dims` 128), rectify + fuse modules. Decoder: DPT-style (channels 2 x 128 = 256), main head `dim_last` 256, auxiliary head 128 on the stage-1 fused map. Map task: layers lane, road_segment, road_divider, lane_divider, ped_crossing, walkway, carpark_area, one BCE-type loss each (weight 1.0), `min_visibility` 0.
- input: 6 surround images (paper: 448 x 800; the repo config `scale_0_5` shows h 224, w 480, top_crop 46 - not reconciled) + radar point set (<= 3500 points x 54 columns) in the nuScenes LiDAR_TOP flat ego frame; no lidar at inference, no depth supervision stated.
- output: 7 x 200 x 200 map logits (0.5 m cells, +-50 m, heading-aligned ego frame). `forward` also returns `features_cam`, `features_radar` (per-modality BEV maps), `features_fused` (4-scale list), `aux_output`, Gaussian counts. Accumulated alpha / coverage is NOT returned (the renderer returns only the rendered feature channels). Meaning of the layers (code read): labels come from the nuScenes devkit `get_map_mask` on a 100 x 100 m patch into 200 x 200; in the devkit the "lane" mask uses only `lane` records - `lane_connector` is a lookup-only layer and not a mask layer, so junction connector areas have no own supervision; `road_divider` / `lane_divider` are polylines drawn 2 pixels thick (= 1 m at 0.5 m cells); the other layers are polygons.
- training: regime matching_free (per-pixel segmentation). L = L_sem + auxiliary L_sem (BCE + Dice + centerness + offset in the paper; weights not stated); AdamW, peak LR 3e-4 (warm-up, linear decay), weight decay 1e-7, 40 epochs, effective batch 16, 4 x A100 80 GB, FP32 only (README: PTv3 part has no backward under mixed precision); BEV augmentation (config [30,20,0,20,0,0]) + image flip / zoom / rotation. nuScenes train split (paper: 700 / 150 / 150 scenes, results on validation). Image backbone `efficientvit_l2.r384_in1k`, `pretrained: true` (the paper does not state pretraining). Map layers are trained without visibility masking (`min_visibility` 0; the paper: visibility filtering does not apply to map evaluation).
- differentiable: yes (differentiable Gaussian rasteriser); hard operation: Gaussians below the opacity threshold are dropped. In our use the network is frozen.
- sensors_data: 6 cameras + 5 radars (7 sweeps, ~0.5 s at 13 Hz), nuScenes only (Boston, Singapore); BEV 100 x 100 m, 0.5 m, 200 x 200; camera Gaussians exist only for depth 1 - 61 m (+ offset); radar points cropped to +-50 m.
- results: nuScenes val IoU (paper Tables I, II): vehicle 57.3 (visibility >= 40%), drivable area 82.9, lane divider 50.1 (BEVCar 58.4 / 83.3 / 45.3; CRN 58.8 vehicle; GaussianBeV 82.6 / 47.4). Radar sweeps 1 / 4 / 7: 54.9 / 56.2 / 57.3 vehicle IoU. PTv3 radar branch: 48.0 -> 55.0 vehicle IoU (Table III, XYZ scatter). Runtime 75.6 ms vehicle / 81.1 ms map (RTX 4090, FP32, batch 1). Only drivable area and lane divider are published for the map task; the other five layers (incl. the lane polygon layer) have no published numbers. No failure-case or limitation section; the qualitative figure shows day, rain and night frames without discussion.
- compute: parameter count not stated; Gaussians converge to ~14k (vehicle) / ~24k (map) per frame.
- code: https://github.com/santimontiel/gaussiancar - README says the project "is released under the Apache-2.0 License"; LICENSE file = Apache License 2.0 (no copyright holder filled in); HF page licence field Apache-2.0; project page states no licence; `pyproject.toml` has no licence field. CAUTION: the vendored CUDA rasteriser `gaussiancar/ops/diff-gaussian-rasterization/LICENSE.md` is the Gaussian-Splatting licence (Inria / MPII; non-commercial research and evaluation only); the vendored `ops/ptv3` folder shows no licence file (the original PTv3 repo shows MIT). Stack: Python >= 3.10 < 3.13, torch 2.8.0, Lightning 2.5, Hydra, spconv-cu126, torch-scatter, flash-attn, timm, peft, nuscenes-devkit; no mmcv / mmdet. Checkpoints `nuscenes_vehicle.ckpt`, `nuscenes_map.ckpt` on HF (model card empty). GitHub: 47 stars, 11 commits, no open issue (open view only).
- fit_to_us: adapt_code - effort S - it IS the frozen foundation; extra outputs need a small code change and no retraining: return per-modality BEV maps, the fused pyramid, and an alpha-coverage map (render the same Gaussians with all-ones features: F = sum a_i prod(1 - a_j) is the accumulated alpha; derived here from the paper's rendering formula, not tested).
- compat_needs: calibration + 6 images + radar sweeps at cache time; our cache = 32 PCA channels of the 256-ch decoder feature + 7 logits (+ a 32-ch fused feature).
- compat_conflicts: frozen (no gradient to sensors); features are cleaner on its own training scenes than on unseen scenes (project: F1@1 0.93 vs 0.82); logits in unobserved cells are learned priors (no visibility masking in training); single-frame camera, no temporal state; camera depth cap 61 m; FP32 only; non-commercial rasteriser licence; 1 m wide divider bands and the missing connector layer limit what the logits say about thin lines and junction interiors (lane_divider IoU 50.1 is the best published value in its table, still only half overlap).
- tried_before: E27 (detection-pretrained CRN vs GaussianCaR: F1@1 0.624 vs 0.867 -> kept); E17 / E18 decoders on the cache (U-Net kept); E39 / E41 temporal feature fusion on the cache (+0.001 / +0.008 F1); E40 / E48 map prior.
- verified: checked

### CARD SN02 - RNTR: frozen BEV encoder + sequence head (closest precedent for a frozen interface)
- papers: RNTR, road network from surround cameras as an integer sequence (Lu et al., ICCV 2023; title from memory) https://arxiv.org/abs/2402.08207
- depth: M (first ~100k of ~117k characters read)
- stage: B, T, D
- idea: A camera BEV encoder (LSS-style) is first pretrained for centerline segmentation, then frozen while a sequence transformer (autoregressive, semi-AR or non-AR) is trained on its features. A later "topology-inherited training" (TIT) trains the transformer on clean lidar-BEV features, distils the camera encoder to those features (L1) and fine-tunes everything jointly. Unfreezing the encoder barely matters; the split choice matters a lot.
- formulation: BEV x -48..48 m, y -32..32 m (cell size written "11m", probably a typo for 1 m; not confirmed); images 128 x 352; ResNet-50 (Cityscapes DeepLabV3+ pretrained) or VoVNetV2; TIT: stage 1 transformer on lidar-BEV features (300 epochs), stage 2 camera encoder distilled by L1 to the frozen lidar-BEV features (24 epochs), stage 3 joint fine-tune (100 epochs). Token layout and losses not read in detail.
- input: surround images (+ optional OpenStreetMap SD-map sequence as prompt); BEV channel count not stated.
- output: token sequence (vertices, parent index, Bezier control points).
- training: regime next_token (AR) / parallel decoding (NAR); encoder frozen in the default setup (they "load and freeze the parameters of the BEV encoder").
- differentiable: partial - token sampling / argmax at inference; encoder frozen.
- sensors_data: camera (lidar only in TIT stage 1, SD map optional), nuScenes, 96 x 64 m BEV.
- results: default (overlapping) split landmark F / reachability F: AR 54.1 / 61.3, semi-AR 56.0 / 65.7, NAR 55.1 / 65.1 (Table V). Non-overlapping split: 36.5 / 40.9, 36.9 / 43.0, 36.2 / 42.3; AR + TIT 39.1 / 49.8; SD-map variant + TIT 54.7 / 62.6. Frozen vs unfrozen BEV encoder (assembled stage, Table X): landmark F 38.6 vs 39.1, reachability F 48.6 vs 49.8 (the page calls the difference not significant).
- compute: NAR 5.5 FPS vs AR 0.1 FPS (hardware not read).
- code: not checked.
- fit_to_us: reuse_idea - effort S - same regime as ours (frozen BEV encoder, head trained on its features, evaluation on an overlap-free split); the token head itself is out of scope (sampling, argmax).
- compat_needs: a BEV feature map from a segmentation-pretrained encoder; an overlap-free split for honest numbers.
- compat_conflicts: token sequences need argmax / sampling (not differentiable); the lidar-BEV teacher in TIT needs lidar (not available to us).
- tried_before: none for the sequence head; the frozen-foundation regime itself is our system B.
- verified: partly (reader summary; part of the page not read)

### CARD SN03 - Localization is all you evaluate: location leakage in nuScenes / Argoverse 2 splits
- papers: Localization Is All You Evaluate: Data Leakage in Online Mapping Datasets and How to Fix It (Lilja, Fu, Stenborg, Hammarstrand; CVPR 2024) https://arxiv.org/abs/2312.06420 ; splits https://github.com/LiljaAdam/geographical-splits (named in a search result, not opened)
- depth: M
- stage: D, E
- idea: On the official splits most validation / test samples lie within 5 m of a training sample, so online-mapping models can score well by recalling a memorised implicit map. Geographically disjoint splits (near and far extrapolation) lower every method sharply and change which design choices look useful.
- formulation: leakage = share of val / test samples with a training sample within 5 m. Near-extrapolation split: regions defined manually from map attributes, 70 / 15 / 15 train / val / test, original test set kept; far-extrapolation: city folds for cross-validation.
- input: sample locations; models re-run with their original configs.
- output: new splits + re-evaluation.
- training: unchanged regimes (HDMapNet, VectorMapNet, MapTR, MapTRv2, StreamMapNet).
- differentiable: no - evaluation protocol.
- sensors_data: nuScenes, Argoverse 2; camera (one MapTRv2 camera + lidar row).
- results: within 5 m of a training sample, original splits: nuScenes val 79.4 % / test 85.5 %, AV2 45.0 % / 41.9 %; near-extrapolation 0.9 % / 1.1 % and 0 % (Table 2). nuScenes mAP val / test original -> near split: MapTR R50 48.8 / 59.9 -> 19.0 / 19.7; MapTRv2 61.5 / 72.7 -> 26.7 / 26.2; StreamMapNet 62.6 / 74.4 -> 26.1 / 27.9; MapTRv2 camera + lidar 54.0 / 65.8 -> 20.0 / 20.1; HDMapNet IoU 31.3 / 38.8 -> 17.1 / 21.2. Average drop > 35 mAP on nuScenes; far-extrapolation CV mean for MapTRv2 21.4. Lifter ranking flips (GKT 19.0, BEVFormer 20.6, LSS 21.0); the gain from depth supervision shrinks. The authors show a model drawing dividers in occluded areas that no image shows.
- compute: not stated.
- code: splits repository named above (not opened).
- fit_to_us: reuse_idea - effort S - our TRAIN / VAL / TEST are scene-disjoint, not necessarily location-disjoint, and the foundation saw its own train scenes: report a location-distance-binned or location-disjoint TEST result next to the scene-disjoint one.
- compat_needs: sample ego positions in the global frame (nuScenes ego poses).
- compat_conflicts: none; it is an evaluation protocol.
- tried_before: none (our splits are scene-disjoint only).
- verified: checked

### CARD SN04 - BEVCar: camera-radar fusion for BEV map and object segmentation
- papers: BEVCar: Camera-Radar Fusion for BEV Map and Object Segmentation (Schramm et al., IROS 2024) https://arxiv.org/abs/2403.11761
- depth: M
- stage: B
- idea: A frozen DINOv2 ViT-B/14 with a learnable adapter gives multi-scale image features; a voxel volume centred on the front camera is filled by projecting image features along camera rays; sparse radar points initialise deformable-attention queries so that radar guides the image lifting ("radar-aided lifting"); six transformer blocks and a ResNet-18 BEV encoder produce the BEV map.
- formulation: images 448 x 896; radar: 5 sweeps (following the Simple-BEV code), features x, y, z, uncompensated vx, vy, RCS; points grouped in voxels matching the BEV grid, FCN + max pooling per voxel, height compression -> f_rad; fusion Q^F = f_rad + position + BEV query, deformable attention onto the lifted image features; grid 100 x 100 m, 200 x 200 cells, 8 height bins up to 10 m, reference frame = front camera; head = two conv layers + 1x1 conv with (M + 1) channels. Map loss: alpha-balanced focal (alpha 0.25, gamma 3); vehicle: BCE.
- input: 6 images (448 x 896) + radar sweeps.
- output: (1 + 7) x 200 x 200 maps (vehicle + drivable area, carpark, ped crossing, walkway, stop line, road divider, lane divider).
- training: regime matching_free; nuScenes official split (28,130 train / 6,019 val samples); epochs, optimiser, LR, batch not stated; backbone frozen (DINOv2) + trained adapter.
- differentiable: yes.
- sensors_data: camera + radar, nuScenes, 100 x 100 m at 0.5 m.
- results: val IoU (Table I), camera-only ViT-B/14 -> camera+radar ViT-B/14: vehicle 48.8 -> 58.4, drivable area 81.1 -> 83.3, lane divider 40.6 -> 45.3, map mean 50.5 -> 54.3; Simple-BEV++ (ViT) 54.5 / 81.2 / 40.4. Range (Table III, vehicle IoU): 0-20 m 74.0 vs 68.7 camera-only, 20-35 m 56.5 vs 45.8, 35-50 m 41.0 vs 29.1. Map IoU day / rain / night (Table IV): 57.3 / 48.8 / 42.4 (camera-only 53.3 / 45.1 / 39.2). Ablation (Table II): radar-driven lifting and fusion +4.2 map mIoU over parameter-free lifting. Runtime A100 forward: 382 ms (ViT-B/14), 137 ms (ResNet-101). Authors: radar-only not evaluated, sensor failure not evaluated.
- compute: parameters not stated; 2.6 FPS (ViT variant).
- code: https://github.com/robot-learning-freiburg/BEVCar (URL from memory, not opened).
- fit_to_us: baseline - effort L - cannot run on a cache (needs images, deformable attention, DINOv2); evidence that radar adds about +4.7 lane-divider IoU and that map IoU drops in rain / night even with radar.
- compat_needs: image features (frozen DINOv2 + adapter), radar voxel encoder.
- compat_conflicts: queries and lifting tied to the image path; not usable on a cached BEV raster.
- tried_before: none (foundation choice E27 compared CRN and GaussianCaR only).
- verified: checked

### CARD SN05 - CRN: radar-assisted view transformation + multi-modal deformable aggregation
- papers: CRN: Camera Radar Net for Accurate, Robust, Efficient 3D Perception (Kim et al., ICCV 2023; authors / venue from memory) https://arxiv.org/abs/2304.00670
- depth: M
- stage: B
- idea: Radar pillars projected into the camera frustum give a radar occupancy map that multiplies the image depth distribution before BEV pooling (radar-assisted view transformation, RVT); camera and radar BEV maps are then fused by multi-modal deformable cross-attention (MFA); previous fused BEV maps are aligned and concatenated.
- formulation: 6 past radar sweeps accumulated (normalised RCS, Doppler speed per point; ego-motion handling not described), pillar height 1 (no radar elevation), PointNet + sparse conv; image features at 16x, depth bins 2-58 m at 0.5 m (D = 112; long range 2-116 m); radar occupancy via sigmoid, image context x depth distribution x occupancy; MFA: 6 layers, 8 heads, 4 sampling points, optional top-4096 sparse queries; detection range +-51.2 m, pillar 0.2 m, down-sampling 4 -> 128 x 128; segmentation +-50 m at 0.5 m -> 200 x 200 with a CVT-style conv decoder, focal loss.
- input: 6 images + radar sweeps + (training) lidar for depth supervision.
- output: CenterPoint-style 3D boxes; BEV vehicle and drivable-area maps.
- training: regime matching_free for segmentation, heatmap targets for detection; 24 epochs AdamW (2e-4 / 1e-4); depth supervised by projected lidar points; radar sweeps and points randomly dropped as augmentation.
- differentiable: yes (top-k sparse query selection is the one hard operation, optional).
- sensors_data: camera + radar (lidar only for training supervision), nuScenes.
- results: BEV segmentation val (R50, 256 x 704): vehicle 58.8, drivable 82.1, 24.8 FPS. Detection val R50: NDS 56.0 / mAP 49.0 (CRAFT 51.7 / 41.1); frames 1 -> 5: mAP 42.9 -> 48.4 (Table 12; orientation error unstable); radar removed entirely: -5.6 % mAP relative. Failure modes: rare classes with few radar returns, clutter, no elevation.
- compute: 20.4 FPS (R50, detection).
- code: https://github.com/youngskkim/CRN (URL from memory, not opened); GaussianCaR notes that CRN has no released segmentation code.
- fit_to_us: baseline - effort L - already compared as a detection-pretrained foundation (E27: F1@1 0.624 vs 0.867); radar occupancy times image depth is the cleanest "radar steers the lift" idea.
- compat_needs: lidar-supervised depth, image features, radar pillars.
- compat_conflicts: detection-pretrained features carry object evidence, not lane geometry (matches E27); cannot be plugged onto a cached raster.
- tried_before: E27 (CRN detection-pretrained vs GaussianCaR: 0.624 vs 0.867 F1@1 -> GaussianCaR kept).
- verified: checked

### CARD SN06 - Simple-BEV: parameter-free bilinear lifting + rasterised radar
- papers: Simple-BEV: What Really Matters for Multi-Sensor BEV Perception? (Harley et al., ICRA 2023; authors / venue from memory) https://arxiv.org/abs/2206.07959
- depth: M
- stage: B
- idea: A fixed 3D voxel grid is projected into every camera and bilinearly samples ResNet-101 features (no learned depth), the volume is folded into BEV channels, radar returns are rasterised into the same grid, and a small ResNet-18 style decoder predicts vehicle segmentation. Message: batch size, resolution and radar matter more than the lifting method.
- formulation: grid 100 x 100 m x 10 m, 200 x 8 x 200 voxels (0.5 x 1.25 x 0.5 m), reference = front camera; radar: each point to the nearest X-Z cell, 15 non-position fields as channels (R = 15), concatenated with RGB BEV features and reduced by a 3x3 conv; sweeps aligned to the current frame (3 sweeps 55.7 IoU, 1 sweep 53.1; nuScenes outlier filter off 55.7 vs 53.7; binary occupancy only 55.0); decoder 3 ResNet-18 blocks, segmentation CE + centerness + offset L1 with uncertainty weights; 25,000 iterations AdamW 5e-4 one-cycle.
- input: 6 images (448 x 800) + radar sweeps.
- output: BEV vehicle map (200 x 200).
- training: regime matching_free; larger batch gives about +14 IoU (2 -> 40); crop / resize augmentation 45.8 -> 47.4.
- differentiable: yes.
- sensors_data: camera, radar, optionally lidar occupancy; nuScenes vehicle superclass.
- results: val vehicle IoU: C 47.4, C+R 55.7, C+L 60.8; lifting: bilinear 47.4, deformable attention 46.5, depth splatting 44.4. Radar called "sparse and noisy", single sweep too sparse.
- compute: ~42 M parameters (bilinear), 83 ms at 448 x 800.
- code: https://github.com/aharley/simple_bev (URL from memory, not opened).
- fit_to_us: reuse_idea - effort S - confirms that the radar value comes from several aligned sweeps and from keeping the metadata fields (GaussianCaR does both); batch-size and resolution findings explain why heads on frozen features cannot repair a weak encoder.
- compat_needs: radar fields incl. metadata; several sweeps aligned to the current frame.
- compat_conflicts: vehicle only, no map layers evaluated.
- tried_before: none.
- verified: checked

### CARD SN07 - RESAR-BEV: camera-radar map segmentation with progressive residual refinement
- papers: RESAR-BEV: An Explainable Progressive Residual Autoregressive Approach for Camera-Radar Fusion in BEV Segmentation (arXiv 2025) https://arxiv.org/abs/2505.06515
- depth: M
- stage: B
- idea: A coarse BEV map from a "Driver" transformer is refined by "Modifier" transformers that predict residuals at several semantic scales (road topology -> lane boundaries) against a ground truth decomposed offline into multi-scale token maps.
- formulation: 6 cameras 448 x 672, ResNet-101 (1/4, 1/8, 1/16); radar voxel features of 6 values (x, y, z, vx, vy, RCS), up to 10 points per voxel, 6 sweeps aligned by sensor -> global -> ego transforms; BEV +-50 m, 200 x 200, 0.5 m; vertical +-5 m in 50 bins -> 10; losses: residual token-map losses weighted 2 / 3 / 4 / 5 + Dice weighted 10.
- input: 6 images + radar sweeps.
- output: 7 classes trained (drivable, ped crossing, walkway, stop line, road divider, lane divider, vehicle); Table I reports three.
- training: regime matching_free; nuScenes.
- differentiable: yes.
- sensors_data: camera + radar, nuScenes, 100 x 100 m at 0.5 m.
- results: Table I (their re-evaluation) drivable / vehicle / lane divider IoU: RESAR-Standard 83.53 / 56.87 / 44.43; BEVCar 79.40 / 54.17 / 43.07; CRN 81.34 / 54.80 / 39.69 (BEVCar's own paper has 83.3 / 58.4 / 45.3 - cross-paper tables are not protocol-matched). Lane divider sunny 45.9, rainy 44.1. Limitations named: multi-lane intersections, rain ground reflections, long range, sensor failure. 31.9 M parameters, 14.6 FPS (hardware not stated).
- compute: 31.9 M parameters.
- code: not checked.
- fit_to_us: reuse_idea - effort M - multi-scale residual decomposition of the target is an idea for refining thin structures, but it needs the image path.
- compat_needs: images, radar voxels.
- compat_conflicts: end-to-end encoder, autoregressive refinement stages.
- tried_before: none.
- verified: partly (reader flagged text-table mismatches; table values used)

### CARD SN08 - GaussianBeV / GaussianLSS family (camera-only Gaussian view transform)
- papers: GaussianBeV: 3D Gaussian Representation meets Perception Models for BeV Segmentation (Chabot et al., WACV 2025) https://arxiv.org/abs/2407.14108 ; GaussianLSS (only seen as a table row and as the rasteriser source in GaussianCaR; not read)
- depth: M (GaussianBeV), not found (GaussianLSS)
- stage: B
- idea: One Gaussian per image pixel per camera; a depth head and a 3D offset head place it, an allocentric quaternion gives its rotation; a feature-rendering rasteriser with orthographic projection splats them to a 200 x 200 BEV that goes into an LSS or Simple-BEV backbone and segmentation heads.
- formulation: EfficientNet-b4 + Simple-BEV neck, 128 channels; z = (f_n / f)(1/d - 1) with d a disparity in [0, 1]; scale abs, opacity sigmoid; loss = segmentation (BCE, centerness, offset) + early loss on the rasteriser output + log-depth loss; AdamW 3e-4, wd 1e-7, up to 100 epochs; BEV 100 x 100 m at 0.5 m.
- input: 6 images (224 x 448 or 448 x 800).
- output: vehicle, pedestrian, drivable area, lane boundary maps.
- training: regime matching_free; nuScenes (28,130 / 6,019 images stated; the text says 750 sequences).
- differentiable: yes.
- sensors_data: camera only, nuScenes, 200 x 200 at 0.5 m.
- results: val IoU 448 x 800: drivable 82.6, lane boundaries 47.4 (MatrixVT drivable 83.5, lane 44.8); vehicles (visibility > 40 %) 47.5 at 224 x 448, 50.3 at 448 x 800; without a BEV backbone 45.9.
- compute: 24 fps and 13 fps (A100) for the two resolutions.
- code: not checked.
- fit_to_us: reuse_idea - effort S - GaussianCaR's camera branch is this design with a coarse-to-fine depth; GaussianCaR's map protocol follows this paper and BEVCar (evaluation at 448 x 800).
- compat_needs: images.
- compat_conflicts: none new.
- tried_before: none.
- verified: checked

### CARD SN09 - TopoBDA: the only camera-radar lane-topology result found (radar rows)
- papers: TopoBDA: Towards Bezier Deformable Attention for Road Topology Understanding (Kalfaoglu et al., Neurocomputing 2026 per the earlier review) https://arxiv.org/abs/2412.18951
- depth: M (radar and supplement passages; the first 200k of 223k characters)
- stage: B, S4
- idea: A Bezier-control-point deformable-attention decoder for centerlines; the paper also fuses SD map, lidar and radar to show headroom. Radar (subset B = nuScenes) is "fused in voxel space"; the supplement says only that radar points are assigned to their nearest voxel and concatenated with the camera and lidar voxel features before height flattening and a 2D conv.
- formulation: radar p = (x, y, z, F_radar), nearest-voxel assignment to H_bev x W_bev x Z x C_radar (200 x 104 x 20, 0.5 m, z in [-10, 10] at 1 m -> about 100 x 52 m); no radar encoder, sweep count, RCS / velocity use stated. Decoder: 4 Bezier control points, iterative refinement, one-to-many auxiliary matching R = 4 (training only).
- input: 6 images + radar (+ lidar, SD map in other rows).
- output: Bezier centerlines + topology scores.
- training: regime hungarian_set (one-to-one) + one-to-many auxiliary, end to end.
- differentiable: partial - Hungarian matching at training.
- sensors_data: OpenLane-V2 subset B (nuScenes) has radar; subset A (Argoverse 2) has none; evaluation +-50 m forward, +-25 m lateral.
- results: OpenLane-V2 subset B, ResNet-50 (Table 3): camera DET_l 45.1 / TOP_ll 35.6 / OLS 49.9; camera + radar 49.4 / 38.6 / 54.8; camera + lidar 57.5 / 46.0 / 61.6. Text: radar + lidar gives no gain over camera + lidar (row not seen).
- compute: not stated.
- code: not checked.
- fit_to_us: reuse_idea - effort S - evidence that radar adds about +4.3 DET_l / +3.0 TOP_ll to a camera BEV lane model without a dedicated radar encoder (a coarse point-to-voxel scatter).
- compat_needs: voxel-space fusion inside the encoder.
- compat_conflicts: decoder uses Hungarian; radar is fused inside the trained encoder.
- tried_before: E31 / E32 (matched sparse decoder dropped).
- verified: checked

### CARD SN10 - PON: visibility mask = field of view AND lidar ray passes; entropy loss on invisible cells
- papers: Predicting Semantic Map Representations from Images using Pyramid Occupancy Networks (Roddick, Cipolla; CVPR 2020; authors / venue from memory) https://arxiv.org/abs/2003.13402 ; code https://github.com/tom-roddick/mono-semantic-maps (URL from memory, not opened)
- depth: M (visibility, loss, extent, split)
- stage: D, L, E
- idea: BEV ground truth is rasterised from map and box annotations; a cell counts as visible only if it lies in the camera field of view and at least one lidar ray passes through it. Invisible cells are ignored by the cross-entropy and by the evaluation, and get an entropy-maximising loss so the network is not pushed to a confident answer there.
- formulation: balanced BCE (weights = sqrt of inverse class frequency); entropy-maximising uncertainty loss with weight 0.001 applied only to non-visible cells; IoU at 0.5 probability with non-visible cells ignored; BEV 50 m ahead, 25 m to each side, 25 cm per pixel.
- input: monocular image per camera.
- output: BEV occupancy per class + high entropy where unobserved.
- training: regime matching_free; train / val sequences redistributed because the default splits share road segments.
- differentiable: yes (loss terms; the mask is a fixed label-side tensor).
- sensors_data: camera (lidar only to build the mask), nuScenes, Argoverse.
- results: none needed here (mask design).
- compute: not stated.
- code: see papers line.
- fit_to_us: reuse_idea - effort S - template for censoring: no loss on invisible samples plus an optional entropy term; the lidar-ray condition is not available from calibration alone (see SN13).
- compat_needs: a per-cell visibility mask at training time.
- compat_conflicts: lidar rays for the second condition.
- tried_before: none.
- verified: checked

### CARD SN11 - Occ3D: camera and lidar visibility masks by ray casting
- papers: Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving (Tian et al., NeurIPS D&B 2023; authors / venue from memory) https://arxiv.org/abs/2304.14365 ; code https://github.com/Tsinghua-MARS-Lab/Occ3D (URL from memory, not opened)
- depth: M (Sec. 3.3.2 and appendix algorithm summary)
- stage: D, E
- idea: A voxel is "observed" for the lidar mask if it holds a return or a lidar ray passes through it; for the camera mask each occupied voxel is joined to the camera origin and the first occupied voxel along each pixel ray is observed, everything behind it and every voxel no pixel ray scans is unobserved. Evaluation counts only observed voxels.
- formulation: camera mask: for each pixel a virtual point at a large depth, ray from the camera origin to it, traverse voxels, copy the lidar-derived state (occupied / free) else "not observed"; no explicit frustum culling (unscanned voxels stay unobserved); Waymo has a blind spot behind the car. Grid nuScenes [-40, 40] m, 0.4 m, 200 x 200 x 16.
- input: aggregated lidar + camera poses.
- output: binary mask per camera view.
- training: none (label tool); the paper gives no training procedure for the masks (a search summary of another paper says training with visible masks can make a model ignore regions beyond the visible area).
- differentiable: no - label tool.
- sensors_data: lidar + camera calibration, nuScenes / Waymo.
- results: none used.
- compute: not stated.
- code: see papers line.
- fit_to_us: reuse_idea - effort M - same logic for lane rays, but it needs a 3D occupancy (lidar) to know occluders; the camera-frustum part can be taken alone.
- compat_needs: 3D occupancy or boxes for occluders.
- compat_conflicts: lidar-derived (project: no lidar input).
- tried_before: none.
- verified: checked

### CARD SN12 - Visible versus occluded BEV cells: how much of the BEV is unobserved and what happens there
- papers: Parametric Depth Based Feature Representation Learning for Object Detection and Segmentation in Bird's Eye View (arXiv 2307.04106; authors / venue not read) https://arxiv.org/abs/2307.04106
- depth: M (visibility sections and Table 4)
- stage: D, E, X
- idea: Per-voxel visibility is derived from a predicted (or lidar-completed) parametric depth distribution along each camera ray: occlusion B(d) = F(d) - 0.5 exp(-mu / b) for a Laplacian depth model, visibility V = 1 - B, max over z for the BEV; thresholds tau_vis / tau_occ (values not given) split the BEV into visible and occluded regions for evaluation.
- formulation: ground-truth visibility from lidar points projected to the images and a depth-completion network; BEV 200 x 200 at 0.5 m (intermediate volume 400 x 400 x 12).
- input: depth distribution per ray.
- output: soft visibility map in [0, 1].
- training: used as feature weighting; the evaluation split is what matters here.
- differentiable: yes (soft visibility); the GT version is offline.
- sensors_data: camera (+ lidar for GT), nuScenes.
- results: nuScenes val (Table 4): visible cells 66.9 %, occluded 33.1 %. Lane IoU visible / occluded: LSS 23.1 / 6.5, M2BEV 39.8 / 12.8, proposed 41.9 / 12.4; road IoU visible / occluded: 79.4 / 47.1, 82.9 / 48.9, 84.8 / 48.9; overall lane 19.9 / 36.1 / 36.5.
- compute: not stated.
- code: not checked.
- fit_to_us: reuse_idea - effort S - quantifies how weak lane evidence is on unobserved cells (about a third of the BEV); a soft visibility V(x, y) in [0, 1] is the natural multiplier for survival terms.
- compat_needs: a depth distribution per camera ray or a lidar depth map (for GT).
- compat_conflicts: depth-based, not calibration-only.
- tried_before: none.
- verified: checked

### CARD SN13 - Calibration-only visibility mask for censoring rays (derived card, not a paper)
- papers: ingredients seen in: BEVFormer (BEV pillar points projected to cameras, "hit" views) https://arxiv.org/abs/2203.17270 ; Simple-BEV ("valid" mask of voxels inside each camera frustum) https://arxiv.org/abs/2206.07959 ; BEVCar (voxel assigned to one or two cameras by field of view) https://arxiv.org/abs/2403.11761 ; PON (FoV AND lidar ray) SN10 ; Occ3D SN11 ; GaussianCaR config (depth 1 - 61 m, image crop) SN01 ; nuScenes paper (camera FoV). The recipe is derived here.
- depth: M (ingredients); the recipe itself is not from a paper
- stage: D, L, S3
- idea: A BEV cell (or ray sample) is "observed by the camera path" if its ground point projects into at least one image with usable depth and resolution. Samples that fail the test are right-censored in the survival product: no lane-end claim may be made beyond the last observed sample.
- formulation: (derived here) for each cell centre p = (x, y, z0) in the ego frame and camera c: p_c = T(cam<-ego) p; require depth z_c in [d_min, d_max] (GaussianCaR lifting range [1, 61] m); u, v = K p_c / z_c with the SAME resize / crop as the foundation saw (ida matrix times K; top crop), require inside the image with a margin; require a ground sampling distance (metres of ground per pixel along the viewing ray) below a limit g_max: for a level camera at height h above flat ground and focal length f in pixels, GSD(d) = d^2 / (f h) (derived). With CAM_FRONT values printed in a third-party nuScenes tutorial (fx = fy = 1266.4 px, cx 816.3, cy 491.5 on 1600 x 900 images, camera 1.54 m above the ego origin) this gives about 0.30 m at 24 m, 1.3 m at 50 m, 1.9 m at 61 m at full resolution, twice as large at half resolution (0.6 / 2.6 / 3.8 m); near-field blind ring: ground closer than h / tan(vFOV / 2) = about 4.3 m is outside a level camera (vFOV = 2 atan(450 / 1266) = 39 deg; pitch ignored; derived). The ego-origin height equals the ground height only approximately. Horizontal FoV: five cameras 70 deg offset by 55 deg, rear 110 deg (nuScenes paper), so no azimuth gaps beyond the near field. Soft form: V(x, y) = max over cameras of a product of smooth border and GSD weights. Occlusion by vehicles is NOT in this mask: at training time add shadow polygons cast from the camera centre over the footprints of annotated 3-D boxes, or use a lidar-ray mask as PON does; at test time a learned visibility head could predict it (PON-style entropy term on the rest).
- input: intrinsics, extrinsics, image crop, ego frame; optional GT boxes.
- output: V(x, y) in [0, 1] on the 200 x 200 grid (fixed, precomputable per frame from calibration).
- training: none (fixed mask); used as a multiplier / censoring flag in the survival loss.
- differentiable: no - geometric rule, a fixed tensor on the loss path (no gradient needed).
- sensors_data: any calibrated camera rig; radar adds no ground observation (SN14).
- results: none (not tested). The foundation's own coverage can be read off by rendering its Gaussians with all-ones features (SN01).
- compute: negligible (precomputable per frame).
- code: none.
- fit_to_us: reimplement - effort S - sample the mask with the same `grid_sample` as the features; choose g_max on VAL by checking where the foundation's lane-logit reliability breaks down against GSD.
- compat_needs: the foundation's exact crop / resize parameters; boxes only for the occlusion extension.
- compat_conflicts: flat-ground assumption; slopes and bridges break the GSD formula.
- tried_before: none.
- verified: partly (ingredients checked, recipe derived here; calibration values from a third-party tutorial page)

### CARD SN14 - nuScenes radar facts relevant to lane evidence (sparsity, range, resolution, fields, filters, compensation)
- papers: nuScenes: A multimodal dataset for autonomous driving (Caesar et al., CVPR 2020; from memory) https://arxiv.org/abs/1903.11027 ; devkit `RadarPointCloud` (data_classes.py) https://github.com/nutonomy/nuscenes-devkit ; Radar-Camera Fusion for Object Detection and Semantic Segmentation in Autonomous Driving: A Comprehensive Review (Yao et al., IEEE T-IV 2023) https://arxiv.org/abs/2304.10410 ; ARS 408-21 datasheet on a reseller page https://platformselect.autonomoustuff.com/product/continental-ars-408-21
- depth: M (paper radar statements; devkit class; review passages on radar limits, first ~100k of ~255k characters)
- stage: D
- idea: Five 77 GHz FMCW radars at 13 Hz, up to 250 m range, return sparse, imprecise points from reflecting objects with Doppler velocity; the devkit offers ego-compensated velocities and cluster-quality flags; multi-sweep loading moves past sweeps into a reference frame using ego poses.
- formulation: 18 fields (x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp, is_quality_valid, ambig_state, x_rms, y_rms, invalid_state, pdh0, vx_rms, vy_rms); default filters invalid_state = 0, dyn_prop 0-6, ambig_state = 3; `from_file_multisweep` defaults nsweeps 5, min_distance 1 m, applies ref_from_car * car_from_global * global_from_car * car_from_current to each past sweep and stores a time lag; vx_comp / vy_comp are compensated for ego motion (only ego motion: moving objects stay displaced between sweeps).
- input: radar sweeps.
- output: point sets.
- training: none.
- differentiable: no - data.
- sensors_data: nuScenes; range "<= 250 m" in the sensor table; the paper says radar returns are even sparser than lidar and less precise in localisation, annotated objects hold up to 40 radar returns at 10 m and 10 at 50 m; FoV and angular resolution not stated on the nuScenes page. The unit is named ARS 408-21 by a third-party paper seen in a search result (probably arXiv 2203.08038); the nuScenes pages read do not name the model. Datasheet of that unit as reproduced on a reseller page: far mode to 250 m, azimuth +-9 deg, range resolution up to 1.79 m, azimuth resolution 1.6 deg; near mode to 70 m (100 m within +-45 deg, 20 m at +-60 deg), azimuth +-60 deg, range resolution 0.39 m, azimuth resolution 3.2 deg at 0, 4.5 deg at +-45, 12.3 deg at +-60. Derived here: lateral resolution = range x azimuth resolution is about 1.3 m at 24 m and 2.8 m at 50 m in near mode (1.4 m at 50 m in far mode, but only inside +-9 deg) - a lane is typically about 3.5 m wide (from memory), so single radar points cannot place a centre line. Review (Yao et al.): 3D radar has no height, clutter and multipath noise, "a dozen points on a car", weak on stationary obstacles, only radial Doppler.
- results: none (data facts). From other cards: radar has no usable elevation (CRN pillar height 1); one sweep is too sparse (Simple-BEV 53.1 vs 55.7 IoU with 3 sweeps); GaussianCaR 1 / 4 / 7 sweeps 54.9 / 56.2 / 57.3 vehicle IoU.
- compute: n/a.
- code: devkit (licence not checked).
- fit_to_us: reuse_idea - effort S - radar returns come from reflecting structure (vehicles, guardrails, signs), not from road paint; evidence for lane geometry is indirect (road-edge reflectors, vehicle motion), so rays that run 24 m ahead along unmarked or empty road get little radar support; the camera path carries far lanes.
- compat_needs: raw radar is not in the cache (only fused features).
- compat_conflicts: moving targets smear across sweeps; clutter and ghost returns (CRN limitations).
- tried_before: none.
- verified: checked

### CARD SN15 - HDMapNet: dense BEV decoder (semantic, instance embedding, direction) + vectorisation
- papers: HDMapNet: An Online HD Map Construction and Evaluation Framework (Li et al., ICRA 2022; from memory) https://arxiv.org/abs/2107.06307 ; code https://github.com/Tsinghua-MARS-Lab/HDMapNet (URL from memory, not opened)
- depth: M
- stage: B, S6
- idea: A fully convolutional BEV decoder with three heads (semantic segmentation, instance embedding, per-pixel direction) is applied to BEV features from cameras (MLP view transform), lidar (PointPillars) or both; vectorisation clusters embeddings, applies NMS and links pixels by following the predicted directions.
- formulation: decoder = ResNet-style, 3 blocks; direction head has N_d classes on the unit circle with two positive labels per lane pixel (forward and backward unknown), non-lane pixels excluded from the loss; instance loss L = L_var + L_dist (alpha = beta = 1, delta_v 0.5, delta_d 3.0); Adam 1e-3; post-processing: DBSCAN on embeddings, NMS, greedy step C_next = C_now + step * D. BEV extent: not in the ar5iv text; the 60 x 30 m at 0.15 m (400 x 200) convention is documented by MGMap and MV-Map (search summaries).
- input: BEV features from any encoder.
- output: seg + embedding + direction rasters -> polylines.
- training: regime matching_free (CE, discriminative loss, direction CE).
- differentiable: partial - the network is differentiable; DBSCAN / NMS / greedy linking are not.
- sensors_data: camera, lidar, fusion; nuScenes (lane boundary, divider, ped crossing); radar only mentioned as a possible extension.
- results: nuScenes "All classes": Surr IoU 32.9, mAP 22.7; LiDAR 29.5 / 11.6; Fusion 44.5 / 30.6; temporal fusion (Surr) IoU 32.9 / 35.8 / 36.4 for 1 / 2 / 4 frames; fusion normal / night / rainy IoU 44.9 / 39.3 / 38.7.
- compute: not stated in the text read.
- code: see papers line.
- fit_to_us: reuse_idea - effort S - the direction head is what our dense head (heat map + 2-channel direction) already does; the multi-class direction loss with two positive labels is the one detail we do not use.
- compat_needs: a BEV feature raster only.
- compat_conflicts: DBSCAN / greedy linking are not differentiable.
- tried_before: E17 / E18 (U-Net dense head kept; dense head + tracer beats sparse).
- verified: checked

### CARD SN16 - MapTR: DETR-style map decoder reading a BEV feature map
- papers: MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction (Liao et al., ICLR 2023; from memory) https://arxiv.org/abs/2208.14437 ; code https://github.com/hustvl/MapTR (URL from memory, not opened)
- depth: M
- stage: B, S4
- idea: Hierarchical queries (instance query + point query) with deformable attention over the BEV feature map predict each map element as an ordered point set; a permutation-equivalent Hungarian matching handles the start-point ambiguity. The BEV encoder (GKT by default) is a pluggable block.
- formulation: q_ij = q_i(ins) + q_j(pt); MapTR-tiny 50 instance x 20 point queries, 6 decoder layers; loss = 2 L_cls + 5 L_p2p + 5e-3 L_dir; point-to-point cost 50.3 mAP vs Chamfer cost 47.5; range x in [-15, 15] m, y in [-30, 30] m (30 x 60 m), BEV cell 0.3 m (tiny) / 0.75 m (nano), about 100 x 200 cells (derived).
- input: BEV features (any encoder: IPM, LSS, deformable attention, GKT).
- output: classified point sets.
- training: regime hungarian_set (instance matching + permutation-minimising point matching), end to end with the encoder.
- differentiable: partial - Hungarian matching at training.
- sensors_data: camera, lidar, both; nuScenes.
- results: val mAP: tiny C R50 24 ep 50.3, 110 ep 58.7; tiny L 55.6; tiny C&L 62.5; lifters: IPM 46.2, LSS 49.5, deformable 49.7, GKT 50.3. The paper says it is "compatible with LiDAR and RADAR" (extension "straightforward"), but it does not claim the decoder works on any BEV features.
- compute: tiny 35.9 M parameters, 11.2 FPS.
- code: see papers line.
- fit_to_us: baseline - effort M - decoder is separable from the encoder (needs a 39 -> 256 channel adapter and normalised-coordinate reference points), but the Hungarian loss conflicts with our matching-free losses (E31 / E32 dropped this family).
- compat_needs: BEV feature map B in R^(H x W x C), C = 256 in practice (from memory).
- compat_conflicts: Hungarian matching; ordered-point-set labels.
- tried_before: E31 / E32 (matched sparse decoder dropped).
- verified: checked

### CARD SN17 - LaneSegNet: lane-segment queries with lane attention on a 200 x 100 BEV at 0.5 m
- papers: LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving (Li et al., ICLR 2024; from memory) https://arxiv.org/abs/2312.16108 ; code https://github.com/OpenDriveLab/LaneSegNet (URL from memory, not opened)
- depth: M
- stage: B, S4, S6
- idea: Lane segments (centerline + left / right boundary + type) are predicted by DETR-style queries; "lane attention" lets each head sample a local region around reference points placed along the predicted segment; the BEV comes from three BEVFormer-style encoder layers.
- formulation: BEV grid 200 x 100 cells for +-50 x +-25 m (0.5 m per cell, derived); 6 decoder layers, 200 queries, lane attention 8 heads x 32 sampling points per reference point (4 reference points per segment); instance-mask head by dot product with BEV features (mask supervision +2.0 APls); topology from predecessor / successor MLPs; Hungarian assignment; loss weights vector 0.025, seg 3.0, cls 1.5, line type 0.01, topology 5.0; AdamW 2e-4, 24 epochs, batch 8.
- input: BEV features from ResNet-50 + FPN + 3 BEVFormer layers.
- output: centerline + offsets (10 x 3 points), masks, adjacency.
- training: regime hungarian_set, end to end.
- differentiable: partial - Hungarian.
- sensors_data: camera, OpenLane-V2.
- results: lane segment mAP 32.6 vs MapTRv2 28.5 (Table 1); lane attention ablation: 28.7 -> 32.6 mAP, TOP_lsls 6.9 -> 8.1.
- compute: decoder 6.74 M parameters, 20.67 ms.
- code: see papers line.
- fit_to_us: reuse_idea - effort M - its BEV grid is a crop of our raster (same 0.5 m cells: take 200 x 100 of 200 x 200); lane attention = reference points along the segment, close to our rays; Hungarian conflicts.
- compat_needs: BEV features of 256 channels in a 200 x 100 grid.
- compat_conflicts: Hungarian one-to-one assignment.
- tried_before: E31 / E32 (matched sparse decoder dropped); E42 / E45 (learned link heads within 0.01 of rules).
- verified: checked

### CARD SN18 - StreamMapNet: BEV-level GRU fusion and a geographically new split
- papers: StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction (Yuan et al., WACV 2024; from memory) https://arxiv.org/abs/2308.12570 ; code https://github.com/yuantianyuan01/StreamMapNet (URL from memory, not opened)
- depth: M
- stage: B, S8
- idea: The previous BEV feature is warped by the ego-pose transform and fused with the current one by a GRU; top-k queries (k = 33) are propagated with their reference points transformed; multi-point attention keeps cost O(N_p). The paper also introduces new disjoint splits.
- formulation: ResNet50 + FPN + one BEVFormer layer; N_q = 100, N_p = 20, N_off = 1; loss = 50 L_line + 5 L_focal + L_trans (weights as printed); first 4 epochs single frame; "100 x 50 m" = 50 m front / back and 25 m lateral, "60 x 30 m" = 30 / 15.
- input: BEV features of consecutive frames.
- output: polylines.
- training: regime hungarian_set, streaming (no gradient into previous frames).
- differentiable: partial - Hungarian.
- sensors_data: camera; nuScenes, Argoverse 2.
- results: nuScenes new split mAP 33.9 (60 x 30 m), 23.0 (100 x 50 m); original split 62.9 at 30 m; overlap of the old split 85 % (nuScenes) / 54 % (AV2), new split 11 % / 0 %; AV2 ablation (50 m): direct prediction 41.7 -> + query propagation 42.8 (+1.1) -> + transformation loss 43.7 (+0.9) -> + BEV fusion 46.1 (+2.4).
- compute: 13.2 FPS (nuScenes), 8 x GTX3090 training.
- code: see papers line.
- fit_to_us: reuse_idea - effort S - BEV-level temporal fusion gave +2.4 mAP, our cache analogue E41 gave +0.008 F1; the 30 m -> 50 m drop (33.9 -> 23.0) shows range sensitivity.
- compat_needs: ego poses between frames.
- compat_conflicts: needs a trainable BEV encoder to learn the fusion (we have a frozen cache; fusion must be head-side).
- tried_before: E39 / E41 (temporal feature fusion on the cache, +0.001 / +0.008 F1).
- verified: checked

### CARD SN19 - CenterLineDet: BEV heat maps from a separate stage, then a decision transformer
- papers: CenterLineDet: CenterLine Graph Detection for Road Lanes with Vehicle-mounted Sensors by Transformer for HD Map Generation (Xu et al., ICRA 2023; from memory) https://arxiv.org/abs/2209.07734
- depth: M (interface passages)
- stage: B, S1, S5
- idea: Stage 1 produces BEV centerline segmentation and an initial-vertex heat map (HDMapNet or its own FusionNet over 6 cameras + lidar sweeps), neighbouring-frame heat maps are warped and averaged; stage 2 crops an ROI around the current vertex and a transformer predicts the next vertices with validity.
- formulation: FusionNet F_bev = max_i { IPM(F_i) + phi_i(F_i) }; ROI = fused heat map concatenated with the ego historical map, cropped around the vertex (size not stated); N_hat = 0 end, 1 continue, > 1 push candidates; behaviour cloning over a BFS of the GT graph with noise; "two stages and cannot be trained end to end".
- input: BEV heat maps.
- output: graph via traversal.
- training: regime teacher_forcing (imitation).
- differentiable: no - separate stages, no gradient across.
- sensors_data: camera + lidar sweeps; nuScenes (698 / 148 sequences).
- results: Table I multi-frame P-F1 0.714, T-F1 0.518 (FusionNet), baselines 0.695 / 0.403 (FusionNet) and 0.685 / 0.400 (HDMapNet).
- compute: 0.41 s/frame.
- code: not checked.
- fit_to_us: baseline - effort M - same interface (BEV raster -> graph by local steps) as our trace network E94 but with DETR queries; confirms that stage-wise training on a separate BEV stage is accepted practice.
- compat_needs: BEV heat map + ego history map.
- compat_conflicts: separate stage, no gradient across.
- tried_before: E94 - E99 (learned trace network, partial).
- verified: checked

### CARD SN20 - BEVFusion (MIT): unified BEV with map-segmentation head
- papers: BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation (Liu et al., ICRA 2023; from memory) https://arxiv.org/abs/2205.13542 ; code https://github.com/mit-han-lab/bevfusion (URL from memory, not opened)
- depth: M
- stage: B
- idea: Camera features are lifted LSS-style (depth distribution scattered along rays), pooled to BEV with a precomputed, interval-reduction BEV pooling kernel (about 40 x faster), fused with VoxelNet lidar BEV by an elementwise operator + conv BEV encoder; one head per task.
- formulation: Swin-T + FPN at 1/8 of 256 x 704; depth [1, 60] m, 0.5 m step, D = 118; BEV pooling cell example 0.4 m; lidar voxel 0.075 m (detection) / 0.1 m (segmentation); map segmentation = one binary map per class (6: drivable, ped crossing, walkway, stop line, carpark, divider) with focal loss, evaluated on [-50, 50] m^2; grid-sample converts the fused BEV to the range / resolution of each head.
- input: 6 images + lidar.
- output: 3D boxes; 6 BEV maps.
- training: not stated in the page read.
- differentiable: yes.
- sensors_data: camera + lidar, nuScenes, Waymo.
- results: map segmentation val mean IoU: camera-only 56.6, lidar-only CenterPoint 48.6, fusion 62.7; camera-only beats lidar-only on maps while lidar beats camera by ~20 mAP on detection.
- compute: 119.2 ms latency (detection config).
- code: see papers line.
- fit_to_us: reuse_idea - effort S - map segmentation depends on the camera path; "grid-sample the fused BEV to each head's range / resolution" is exactly our bilinear `grid_sample` contract.
- compat_needs: lidar for the published fusion.
- compat_conflicts: lidar.
- tried_before: none.
- verified: checked

### CARD SN21 - BEVFusion (PKU): BEV-level fusion that survives lidar failure
- papers: BEVFusion: A Simple and Robust LiDAR-Camera Fusion Framework (Liang et al., NeurIPS 2022) https://arxiv.org/abs/2205.13790 ; code: ADLab-AutoDrive organisation named on the abstract page, https://github.com/ADLab-AutoDrive/BEVFusion (URL not opened)
- depth: M
- stage: B, T
- idea: An independent camera stream (Dual-Swin-T, LSS-style projector, spatial-to-channel BEV encoder) and any lidar-to-BEV network are concatenated and fused by a conv plus squeeze-and-excitation gate (dynamic fusion), so the camera stream still works if lidar fails.
- formulation: images 448 x 800; fusion: concat, 3x3 conv, SE-style sigmoid gate; two-stage training (streams separately, then fusion; lidar stream frozen for CenterPoint / TransFusion-L); robustness fine-tuning with limited-FOV and object-failure lidar augmentations (12 epochs, LR 1e-4).
- input: images + lidar.
- output: 3D boxes.
- training: staged; frozen lidar stream during fusion.
- differentiable: yes.
- sensors_data: camera + lidar, nuScenes.
- results: val 67.9 mAP / 71.0 NDS (TransFusion-L stream); test 69.2 / 71.8; lidar FOV failure (-pi/2, pi/2): TransFusion-L 27.8 -> 46.4 mAP.
- compute: camera projector > 957 ms, ~1.5 s per sample on a V100 (their implementation).
- code: see papers line.
- fit_to_us: reuse_idea - effort S - staged training with a frozen first stream is the same regime as ours; their robustness recipe (sensor-failure augmentation) matches our synthetic failures.
- compat_needs: lidar.
- compat_conflicts: lidar.
- tried_before: none.
- verified: checked

### CARD SN22 - BEVFormer: grid BEV queries, spatial cross-attention, recurrent temporal self-attention
- papers: BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers (Li et al., ECCV 2022; from memory) https://arxiv.org/abs/2203.17270 ; code https://github.com/fundamentalvision/BEVFormer (URL from memory, not opened)
- depth: M
- stage: B, S8
- idea: Learnable grid BEV queries (200 x 200, C = 256, +-51.2 m, 0.512 m cells) attend to image features through deformable spatial cross-attention on 4 z-reference points per pillar (only cameras the points project into, the "hit" views, are used) and to the previous BEV (ego-motion aligned) through temporal self-attention; recurrent: only the previous BEV is carried.
- formulation: 6 encoder layers; N_ref = 4 points at heights -5 to 3 m; 4 sampling points per reference point; ResNet101-DCN / VoVNet-99; 24 epochs; 4 frames in training (3 past without gradient); detection by Deformable DETR (900 queries); segmentation by a Panoptic-SegFormer-style mask decoder.
- input: 6 images + previous BEV.
- output: boxes, road / lane masks.
- training: regime hungarian_set (detection), joint segmentation.
- differentiable: partial - Hungarian at training.
- sensors_data: camera, nuScenes (200 x 200, +-51.2 m), Waymo.
- results: val NDS 51.7 / mAP 41.6 (R101); temporal ablation (BEVFormer-S val NDS by frames): 1 frame 0.448, 2 0.490, 3 0.510, 4 0.517; map segmentation val (seg only): road 80.1, lane 25.7 (LSS 72.9 / 20.0); velocity error 0.802 -> 0.394; recall on the 0-40 % visibility subset +6.0 over the static variant. Latency 130 ms (V100, R101-DCN, 1600 x 900).
- compute: 130 ms default.
- code: see papers line.
- fit_to_us: not_suitable - effort L - needs images and trainable queries; its hit-view test is reused in SN13 and its ego-aligned recurrent BEV is the encoder-side analogue of our time recursion (SN33).
- compat_needs: images; trainable queries.
- compat_conflicts: queries tied to the encoder.
- tried_before: none.
- verified: checked

### CARD SN23 - Lift-Splat-Shoot (and the frozen-encoder baseline inside it)
- papers: Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D (Philion, Fidler; ECCV 2020; from memory) https://arxiv.org/abs/2008.05711 ; code https://github.com/nv-tlabs/lift-splat-shoot (URL from memory, not opened)
- depth: M
- stage: B
- idea: Per-pixel categorical depth distribution times a context vector lifts features into a frustum, pillar pooling (cumulative-sum trick) splats into a BEV grid, a small BEV CNN decodes; planning is a cost-volume classification over 1000 template trajectories.
- formulation: images 128 x 352, depths 4 - 45 m at 1 m; BEV +-50 m, 0.5 m cells, 200 x 200; EfficientNet-B0 + ResNet-18 style BEV net, 14.3 M parameters; Adam 1e-3, 300k steps; BCE weights 1 vehicle, 5 lane, 1 road.
- input: 6 images.
- output: BEV vehicle / drivable / lane-boundary maps, cost map.
- training: regime matching_free.
- differentiable: yes.
- sensors_data: camera, nuScenes, Lyft.
- results: map IoU (Table 2): drivable 72.94, lane boundary 19.96; baselines CNN 68.96 / 16.51, OFT 71.69 / 18.07, Frozen Encoder 61.62 / 16.95; vehicles 32.07; transfer nuScenes -> Lyft vehicle: frozen encoder 15.82 vs LSS 22.59; performance falls roughly linearly with distance; night performance drops; single timestep only.
- compute: 35 Hz on a Titan V.
- code: see papers line.
- fit_to_us: reuse_idea - effort S - the "frozen encoder" row (frozen ImageNet encoder, trained lift and decoder) loses 11 IoU on drivable area against end-to-end; but it freezes a generic encoder, not a map-pretrained one like ours.
- compat_needs: images.
- compat_conflicts: none new.
- tried_before: none.
- verified: checked

### CARD SN24 - BEVDepth (and the BEVDet family): lidar-supervised, camera-aware depth for LSS
- papers: BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection (Li et al., AAAI 2023; from memory) https://arxiv.org/abs/2206.10092 ; BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View (Huang et al., 2021; title from memory) https://arxiv.org/abs/2112.11790 ; code https://github.com/Megvii-BaseDetection/BEVDepth , https://github.com/HuangJunJie2017/BEVDet (URLs from memory, not opened)
- depth: M (BEVDepth), M (BEVDet)
- stage: B
- idea: Explicit depth supervision from projected lidar points, a camera-parameter-aware depth module (intrinsics / extrinsics through an MLP into an SE gate), a depth refinement conv, a CUDA voxel pooling 80 x faster than the cumsum trick, multi-frame BEV concatenation. The camera-radar detectors here (CRN, RCBEVDet, HyDRa) start from this stream.
- formulation: BEVDepth: BCE depth loss (L1 changes little); ResNet-50 256 x 704, AdamW 2e-4, batch 64; depth bin range and BEV grid not stated in the page read. BEVDet: four modules (backbone + neck, LSS view transformer, ResNet-style BEV encoder with FPN-LSS, first-stage CenterPoint head); ROI 51.2 m at 0.8 m, depth [1, 60] m; image-view augmentation (IDA) and BEV-space augmentation (BDA: flip, rotation within +-22.5 deg, scale 0.95 - 1.05) are decoupled by the view transformer and IDA helps only together with BDA; Scale-NMS (class-specific box enlargement before NMS); AdamW 2e-4, batch 64, 20 epochs, CBGS, MMDetection3D.
- input: images (+ lidar for training).
- output: boxes.
- training: regime matching_free (heatmap detection head); depth supervised by lidar.
- differentiable: yes.
- sensors_data: camera (lidar at training), nuScenes.
- results: ablation val (mAP / NDS): baseline 0.282 / 0.327, + depth loss 0.304 / 0.344, + camera awareness 0.314 / 0.357, + refinement 0.322 / 0.367, + multi-frame 0.330 / 0.442; val R50 0.351 / 0.475, R101 0.412 / 0.535. Replacing learned depth by GT depth gives nearly +20 % mAP / NDS. BEVDet val (Table 2): Tiny 31.2 mAP / 39.2 NDS at 704 x 256, 15.6 FPS; Base 39.3 / 47.2 at 1600 x 640, 1.9 FPS; test 42.2 / 48.2; without augmentation Tiny peaks at 23.0 mAP and ends at 17.4 (overfits), with IDA + BDA 31.2; attribute error is worse than image-view methods (AAE 0.247 vs 0.170).
- compute: pooling 80 x faster; training 5 -> 1.5 days.
- code: see papers line.
- fit_to_us: reuse_idea - effort S - not usable as a block (image stream, lidar-supervised depth), but two lessons: depth quality limits camera BEV (the far-range loss), and BEV-space augmentation (BDA) is what regularises the BEV encoder and head (BEVDet without augmentation overfits within a few epochs); a head-side analogue on the cached raster (flip / small rotation / scale of raster and labels together) is cheap to test, but the sensor layout (blind spots, camera-specific artefacts) makes rotation only approximately valid (not tested).
- compat_needs: lidar depth at training.
- compat_conflicts: lidar.
- tried_before: none.
- verified: checked

### CARD SN25 - RCBEVDet: radar point + transformer backbone, RCS-aware BEV, cross-attention fusion
- papers: RCBEVDet: Radar-camera Fusion in Bird's Eye View for 3D Object Detection (Lin et al., CVPR 2024; from memory) https://arxiv.org/abs/2403.16440 ; code https://github.com/VDIGPKU/RCBEVDet (URL from memory, not opened)
- depth: M
- stage: B
- idea: RadarBEVNet = dual-stream radar backbone (PointNet-style branch + transformer branch with distance-modulated self-attention, linked by injection / extraction cross-attention) and an RCS-aware BEV encoder that scatters each point to nearby BEV pixels within an RCS-scaled radius; CAMF fuses with the camera BEV by deformable cross-attention in both directions plus conv blocks.
- formulation: radar features RCS + Doppler, multiple sweeps (number and ego compensation not stated); camera stream = BEVDepth with temporal fusion, BEVPoolv2, trained first then frozen; 12 epochs AdamW, CBGS.
- input: images + radar.
- output: boxes.
- training: staged (camera stream frozen in stage 2); regime matching_free (heatmap head).
- differentiable: yes.
- sensors_data: camera + radar, nuScenes, View-of-Delft.
- results: val R50 NDS 56.8 / mAP 45.3, 21.3 FPS; test V2-99 NDS 63.9 / mAP 55.0; ablation: + RadarBEVNet 55.7 NDS, + CAMF 56.4; radar dropout robustness better than CRN (Table 7).
- compute: 21.3 FPS (RTX 3090, FP16, batch 1).
- code: see papers line.
- fit_to_us: reuse_idea - effort M - RCS-radius splatting is a cheap radar densifier; no segmentation or map task in the paper (CTAB, SN36, adds one).
- compat_needs: images.
- compat_conflicts: detection only.
- tried_before: none.
- verified: checked

### CARD SN26 - CRAFT: proposal-based camera-radar fusion in polar coordinates
- papers: CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer (Kim et al., AAAI 2023; from memory) https://arxiv.org/abs/2209.06535
- depth: M
- stage: B
- idea: Image-based 3D proposals (CenterNet-style, DLA-34, 448 x 800) gather radar points through soft polar association (azimuth between the corners, adaptive radial margin from depth variance, up to 128 points); a spatio-contextual fusion transformer updates the proposals; no BEV raster exists in the method.
- formulation: gamma 5, delta 10 for the margin; radar PointNet++ on six sweeps (~0.5 s), 2048 points, RCS + ego-compensated radial velocity; 4 layers per encoder, 8 heads; camera detector 140 epochs then frozen, fusion 24 epochs.
- input: image proposals + radar points.
- output: boxes.
- training: staged; frozen camera detector; regime matching_free.
- differentiable: partial - proposal threshold 0.05 and hard association.
- sensors_data: camera + radar, nuScenes.
- results: test NDS 52.3 / mAP 41.1, 4.1 FPS; polar beats Cartesian association (AP 41.3 vs 27.7).
- compute: 4.1 FPS (RTX 3090).
- code: not checked.
- fit_to_us: not_suitable - effort L - needs per-object proposals; the idea of associating radar to a hypothesis in polar coordinates is the same geometry as our ray / tip.
- compat_needs: image proposals.
- compat_conflicts: object-centric, not for lanes.
- tried_before: none.
- verified: checked

### CARD SN27 - HyDRa (and RaCFormer): hybrid perspective / BEV camera-radar fusion
- papers: Unleashing HyDRa: Hybrid Fusion, Depth Consistency and Radar for Unified 3D Perception (Wolters et al., ICRA 2025 per arXiv v4) https://arxiv.org/abs/2403.07746 ; RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion (Chu et al., CVPR 2025) https://arxiv.org/abs/2412.12725 (abstract level only)
- depth: M (HyDRa), A (RaCFormer)
- stage: B
- idea: HyDRa fuses radar pillars with image columns in the perspective view (a height-association transformer) and refines in BEV with a radar-weighted depth-consistency attention; BEV features concatenated with radar-BEV and an SE block; an occupancy head reuses the BEV features. RaCFormer samples instance features from both BEV and the image view with polar queries and a Doppler-based implicit dynamic catcher.
- formulation: BEV 128 x 128 at 0.8 m (R50, 256 x 704), 256 x 256 for larger models; 118 depth bins; last eight history frames; trained 20 epochs per task.
- input: images + radar pillars (sweeps / ego compensation not specified).
- output: boxes, tracks, Occ3D occupancy.
- training: per task; regime matching_free (heatmap head).
- differentiable: yes.
- sensors_data: camera + radar, nuScenes, View-of-Delft, Occ3D.
- results: HyDRa val R50 NDS 58.5 / mAP 49.4, R101 61.7 / 53.6; test 64.2 NDS / 57.4 mAP; Occ3D mIoU 44.4 vs 40.7 (FB-OCC), night 30.1 vs 25.8; RaCFormer (abstract): 64.9 mAP / 70.2 NDS on nuScenes.
- compute: not stated.
- code: https://github.com/phi-wol/hydra and https://github.com/cxmomo/RaCFormer (both named on the pages read; not opened).
- fit_to_us: reuse_idea - effort M - the occupancy output on a camera-radar BEV shows radar helps dark / unobserved regions; both need images.
- compat_needs: images.
- compat_conflicts: perspective-view fusion inside the trained encoder.
- tried_before: none.
- verified: checked (HyDRa), partly (RaCFormer: abstract level)

### CARD SN28 - Fine-tuning versus linear probing under distribution shift (LP-FT)
- papers: Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution (Kumar, Raghunathan, Jones, Ma, Liang; ICLR 2022) https://arxiv.org/abs/2202.10054
- depth: A
- stage: T
- idea: Full fine-tuning gains in distribution but can lose out of distribution because the lower layers move while the head is still random; training the head first (linear probe) and then fine-tuning (LP-FT) keeps the pretrained features.
- formulation: not read.
- input: pretrained features.
- output: n/a.
- training: LP then FT; regime none (supervised).
- differentiable: yes.
- sensors_data: image classification benchmarks (10 shift datasets).
- results: abstract: fine-tuning about +2 % in-distribution and about -7 % out-of-distribution versus linear probing; LP-FT about +1 % ID and about +10 % OOD over full fine-tuning.
- compute: n/a.
- code: not checked.
- fit_to_us: reuse_idea - effort S - applies if we ever unfreeze the foundation's decoder: warm up our heads on the frozen features first, then unfreeze with a small LR; classification evidence only.
- compat_needs: a trained head before unfreezing.
- compat_conflicts: none.
- tried_before: none.
- verified: partly (abstract only)

### CARD SN29 - UDGA: adapters for multi-view BEV detectors under camera / dataset shift
- papers: Unified Domain Generalization and Adaptation for Multi-View 3D Object Detection (arXiv 2024) https://arxiv.org/abs/2410.22461
- depth: M (adapter section, Tables 2, 9 - 11)
- stage: T
- idea: Bottleneck adapters (down projection, activation, up projection, BatchNorm on input, parallel to the pretrained blocks, near-identity init) can be inserted into backbone, view transform, BEV encoder and head; the text reports that adapters only at the detection head decrease Lyft -> nuScenes performance.
- formulation: y = B(x) + A(x); best setup conv down, linear up, 8.8 M parameters (full fine-tuning 51.7 M; BEVFormer variant 12.2 M of 33.5 M).
- input: BEV detector features.
- output: boxes.
- training: parameter-efficient fine-tuning with target-domain labels at 1 - 100 % fractions; regime none (supervised).
- differentiable: yes.
- sensors_data: camera, nuScenes <-> Lyft, CARLA shifts.
- results: Lyft -> nuScenes NDS / mAP at 1 % target data: full FT 0.476 / 0.369, theirs 0.526 / 0.404, SSF 0.245 / 0.079; extrinsic-height shift in CARLA: NDS 0.666 -> 0.313 (Table 7).
- compute: 8.8 M trainable parameters.
- code: not checked.
- fit_to_us: reuse_idea - effort M - indicates that adapters inside the backbone matter more than head-only adaptation when the shift is geometric; ours is a scene / location shift on the same rig, so the effect may differ.
- compat_needs: access to inner layers of the foundation (not in the cache).
- compat_conflicts: the cache holds final features only.
- tried_before: none.
- verified: partly (reader could not map the head-only table row)

### CARD SN30 - OccFeat: pretraining a BEV network and evaluating map segmentation at 1 / 10 / 100 % labels
- papers: OccFeat: Self-supervised Occupancy Feature Prediction for Pretraining BEV Segmentation Networks (Sirko-Galouchenko et al., CVPRW 2024; from memory) https://arxiv.org/abs/2404.14027
- depth: M
- stage: T
- idea: Pretrain a camera BEV network to predict lidar-derived occupancy and DINOv2 features at occupied voxels, then fine-tune on vehicle and map segmentation; the gain is large with few labels and small with all labels.
- formulation: L = L_occ + 0.01 L_feat (negative cosine to DINOv2 ViT-S/14 features); BEV 200 x 200 over 50 m; Simple-BEV and BEVFormer; 50 epochs.
- input: images (+ lidar for the pretext).
- output: BEV maps.
- training: regime self_supervised pretraining, then full fine-tuning (no frozen / probe experiment).
- differentiable: yes.
- sensors_data: camera, nuScenes.
- results: aggregated map IoU (Simple-BEV EN-B0) none / OccFeat at 1 %: 20.1 / 25.4; 10 %: 28.1 / 33.7; 100 %: 45.3 / 47.3 (no per-layer numbers).
- compute: not stated.
- code: not checked.
- fit_to_us: reuse_idea - effort S - label-efficiency evidence for pretrained BEV features; ours is pretrained for the same layers, so the low-data regime of our 212 training scenes should favour frozen features.
- compat_needs: nothing for us.
- compat_conflicts: no frozen-encoder result.
- tried_before: none.
- verified: checked

### CARD SN31 - CRKD and BEVDistill: distilling a lidar BEV into a camera-radar or camera BEV
- papers: CRKD: Enhanced Camera-Radar Object Detection with Cross-modality Knowledge Distillation (Zhao, Song, Skinner; CVPR 2024) https://arxiv.org/abs/2403.19104 , project page https://song-jingyu.github.io/CRKD ; BEVDistill: Cross-Modal BEV Distillation for Multi-View 3D Object Detection (Chen et al., ICLR 2023) https://arxiv.org/abs/2211.09386 , code https://github.com/zehuichen123/BEVDistill (named in a search result)
- depth: M (CRKD), M (BEVDistill)
- stage: T
- idea: CRKD distils a frozen BEVFusion lidar-camera teacher into a camera-radar student with four losses: cross-stage radar distillation (radar features vs the teacher objectness heat map, L1), mask-scaling feature distillation (L2 on foreground-mask-weighted fused features, masks enlarged for far / fast objects), relation distillation (L1 between cosine affinity matrices), response distillation (quality focal + SmoothL1). BEVDistill distils a frozen lidar detector into BEVFormer-type camera students with a Gaussian-heat-map-weighted L2 on BEV features (dense) plus an InfoNCE-style contrastive loss on matched query features (sparse), because logit matching fails across modalities.
- formulation: CRKD weights 100 / 10 / 0.25 / 1 / 1 (detection); BEV 180 x 180 over [-54, 54] m; teacher keyframe + 9 sweeps, student keyframe + 6 radar sweeps; 20 epochs, LR 1e-4, 4 x A100. BEVDistill: Gaussian mask sigma 2 around GT centres, loss weights 1.0 / 0.25, BEV 128 x 128, 2 x 24 epochs, 8 x A100.
- input: teacher and student inputs.
- output: boxes.
- training: teacher frozen; regime none / self-distillation; labels only for the detection loss.
- differentiable: yes.
- sensors_data: CRKD: camera + radar student, camera + lidar teacher; BEVDistill: camera student, lidar teacher; nuScenes (BEVDistill also Waymo).
- results: CRKD val mAP / NDS: baseline CR 43.2 / 54.1, student 44.9 / 55.9, CRKD (SwinT) 46.7 / 57.3; test 48.7 / 58.7; gains by range 0-20 m +3.05 mAP, 30-50 m +1.03 mAP; sunny / rainy / night mAP +1.22 / +2.32 / +0.20. BEVDistill val BEVFormer-R50 NDS / mAP 42.3 / 35.2 -> 45.7 / 38.6; BEVFormer-T 48.8 / 38.3 -> 51.5 / 40.7; test BEVDepth 58.9 / 49.1 -> 59.4 / 49.8; ablation dense only 44.7 / 38.0, instance only 43.9 / 37.1.
- compute: not stated.
- code: see papers line.
- fit_to_us: not_suitable - effort L - needs a lidar teacher; reuse_idea: radar distillation does not rescue far range (CRKD 30-50 m +1.03 mAP only).
- compat_needs: a teacher BEV with the same grid.
- compat_conflicts: lidar teacher.
- tried_before: none.
- verified: checked

### CARD SN32 - BEV uncertainty maps: UFCE benchmark, online map uncertainty, direct BEV feature use
- papers: Predictive Uncertainty Quantification for Bird's Eye View Segmentation: A Benchmark and Novel Loss Function (Yu et al., ICLR 2025; authors from memory) https://arxiv.org/abs/2405.20986 ; Producing and Leveraging Online Map Uncertainty in Trajectory Prediction (Gu, Song, Gilitschenski, Pavone, Ivanovic; CVPR 2024) https://arxiv.org/abs/2403.16439 ; Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention (same authors; ECCV 2024) https://arxiv.org/abs/2407.06683 (both abstract level only)
- depth: M (UFCE, first ~100k characters), A (the two Gu et al. papers)
- stage: E, L
- idea: UFCE compares softmax entropy, ensembles, MC dropout and evidential (Dirichlet) heads on LSS, CVT and Simple-BEV and proposes an uncertainty-focal cross-entropy. Gu et al. add per-element uncertainty to online HD-map models and feed it to trajectory prediction (up to 50 % faster training convergence, up to 15 % better prediction on nuScenes), and expose the internal BEV features of the mapping model to the predictor (up to 73 % faster inference, up to 29 % more accurate predictions).
- formulation: not read in detail.
- input: BEV logits / features.
- output: per-cell uncertainty.
- training: evidential loss; regime matching_free.
- differentiable: yes.
- sensors_data: camera BEV (CARLA, nuScenes, Lyft, nuScenes-C).
- results: nuScenes vehicle, LSS: IoU 0.332 / ECE 0.00887 (entropy, CE) vs 0.343 / 0.00332 (ENN-UFCE); AUPR for misclassification detection stays < 40 %; the paper does not analyse uncertainty versus distance or occlusion.
- compute: not stated.
- code: not checked.
- fit_to_us: reuse_idea - effort M - an evidential head on the cached logits would give a learned "unreliable" signal to combine with the geometric mask (SN13); no paper shows that such uncertainty correlates with occlusion or range.
- compat_needs: logits and labels; extra training.
- compat_conflicts: none.
- tried_before: none.
- verified: partly

### CARD SN33 - Encoder-side temporal fusion versus our set-level recursion (summary of seen numbers)
- papers: BEVFormer SN22 ; StreamMapNet SN18 ; HDMapNet SN15 ; CRN SN05 ; BEVDepth SN24 ; Simple-BEV SN06 ; GaussianCaR SN01 ; CenterLineDet SN19
- depth: M (from the cards)
- stage: S8, B
- idea: Encoder-side temporal fusion aligns the previous BEV (or several) by ego motion and fuses by attention, GRU or concatenation; gains are largest for velocity and occluded objects, modest for static map geometry. GaussianCaR has no temporal camera state; its only temporal input is 7 radar sweeps.
- formulation: BEVFormer: recurrent temporal self-attention over ego-aligned B_(t-1); StreamMapNet: warp + GRU + LayerNorm, no gradient to previous frames; CRN: aligned fused BEV maps concatenated; CenterLineDet: warped neighbour heat maps averaged.
- input: previous BEV(s) + ego pose.
- output: fused BEV.
- training: end to end with the encoder.
- differentiable: yes.
- sensors_data: nuScenes.
- results: BEVFormer-S val NDS by frames 0.448 / 0.490 / 0.510 / 0.517 (1 - 4); HDMapNet map IoU 32.9 / 35.8 / 36.4 (1 / 2 / 4 frames); StreamMapNet AV2 ablation: BEV fusion +2.4 mAP; CRN mAP 42.9 -> 48.4 (1 -> 5 frames, orientation error unstable); BEVDepth NDS 0.367 -> 0.442 with multi-frame; GaussianCaR radar sweeps 54.9 / 56.2 / 57.3 vehicle IoU for 1 / 4 / 7.
- compute: n/a.
- code: n/a.
- fit_to_us: reuse_idea - effort S - our time recursion works on vertices with ego-warped validity; encoder-side fusion cannot be added to a frozen cache (needs the encoder), only head-side (input memory = ego-warped previous input; E39 / E41: +0.001 / +0.008 F1).
- compat_needs: ego poses.
- compat_conflicts: encoder-side fusion needs retraining the foundation.
- tried_before: E39 / E41 (temporal feature fusion on the cache).
- verified: checked

### CARD SN34 - Out-of-fold (held-out) features for second-stage training
- papers: A Robust Stacking Framework for Training Deep Graph Models with Multifaceted Node Features (arXiv 2022) https://arxiv.org/abs/2206.08473 ; classical stacked generalisation (Wolpert 1992; from memory)
- depth: A
- stage: T
- idea: If a first-stage model has memorised its training data, a second stage trained on its outputs for the same data sees features that are cleaner than at test time; the fix is to train the second stage only on predictions for data the first stage never saw (k-fold partition, k copies of the first model).
- formulation: not read.
- input: first-stage outputs.
- output: n/a.
- training: cross-fitting; regime none.
- differentiable: yes.
- sensors_data: graph benchmarks (not driving).
- results: none seen (search-summary statement of the principle).
- compute: k copies of the first stage.
- code: not checked.
- fit_to_us: reuse_idea - effort L for the foundation (retraining k copies is impossible with released weights) - but scene-level held-out features can be approximated by weighting the heads' training towards cached features of scenes outside the foundation's training split; this explains the 0.93 vs 0.82 gap qualitatively.
- compat_needs: knowing which scenes the foundation trained on (the nuScenes train list).
- compat_conflicts: we do not control the foundation's training.
- tried_before: none.
- verified: partly (search summary)

### CARD SN35 - FIN: camera-radar map segmentation with a distance analysis
- papers: FIN: Fast Inference Network for Map Segmentation (arXiv 2025) https://arxiv.org/abs/2510.00651
- depth: M
- stage: B
- idea: CRN-style radar-assisted view transformation and multi-modal deformable cross-attention with a ResNet-50 image backbone and a PAN radar backbone, read by a small two-level U-Net head; the paper evaluates map IoU in 10 m distance steps from 10 to 50 m.
- formulation: images 256 x 704; BEV 200 x 200 at 0.5 m (the text also says "200 m x 200 m grid", inconsistent with its -50..50 m figures); six classes (drivable, ped crossing, sidewalk, stop line, carpark, divider); six-term loss with a normalised boundary loss (Table 4: focal alone 46.3 mIoU, full set 52.4, + boundary 53.5; the text cites 49.6 for focal alone); head ablation: plain conv head 52.3 mIoU (1.2 M parameters) vs U-Net head 53.5 (1.7 M).
- input: 6 images + 5 radars.
- output: 6 map layers at 200 x 200.
- training: regime matching_free; nuScenes, 24 epochs in the ablation.
- differentiable: yes.
- sensors_data: camera + radar, nuScenes train / val, rainy and night subsets built from metadata.
- results: val mIoU 53.5; per class drivable 67.7, ped crossing 53.1, sidewalk 60.7, stop line 45.6, carpark 51.3, divider 48.2; 38 ms (A100), 58.4 M parameters. Per-distance numbers exist only in figures: degradation with distance is "proportional across classes", 0 - 25 m named the safe range, night worst; authors list occluded and distant structure as unsolved.
- compute: 58.4 M parameters, 26.3 FPS (A100).
- code: not checked.
- fit_to_us: reuse_idea - effort S - a U-Net head on a camera-radar BEV is the same read-out as our dense head; no per-range numbers are given for lanes anywhere in the camera-radar literature.
- compat_needs: images (RVT).
- compat_conflicts: none new.
- tried_before: E17 / E18 (U-Net head kept).
- verified: checked

### CARD SN36 - CTAB: a segmentation branch on RCBEVDet's coarse 0.8 m BEV upsampled to 0.5 m
- papers: Radar-Camera BEV Multi-Task Learning with Cross-Task Attention Bridge for Joint 3D Detection and Segmentation (arXiv 2026) https://arxiv.org/abs/2604.12918
- depth: M
- stage: B
- idea: Adds a BEV segmentation branch (instance-norm decoder, learnable upsampler) and a bidirectional detection-segmentation attention bridge (0.58 M parameters) to RCBEVDet; the shared BEV has 128 x 128 cells at 0.8 m ([-51.2, 51.2] m) and segmentation is upsampled to 200 x 200 at 0.5 m.
- formulation: seven classes (drivable, carpark, ped crossing, walkway, stop line, divider = road + lane dividers merged, vehicle); homoscedastic uncertainty loss weights; radar via RadarBEVNet (sweeps and feature dimensions not given).
- input: 6 images + radar.
- output: boxes + 7 BEV maps.
- training: regime matching_free; joint, nuScenes 28,130 / 6,019.
- differentiable: yes.
- sensors_data: camera + radar, nuScenes.
- results: val IoU baseline -> with CTAB: drivable 72.7 -> 73.3, divider 33.7 -> 34.4, vehicle 49.8 -> 49.8, mIoU-7 42.9 -> 43.5 (Table III); detection NDS 55.7 -> 55.6. Authors: features at 0.8 m cannot recover detail lost in the BEV projection; class sets differ across methods (cross-paper comparison is unsafe).
- compute: +0.58 M parameters.
- code: not checked.
- fit_to_us: reuse_idea - effort S - a coarse BEV upsampled to a finer grid caps thin structures (divider 34 here vs 45 - 50 for native 0.5 m camera-radar models in other papers; not protocol-matched); supports keeping the native 0.5 m cache rather than re-sampling.
- compat_needs: images.
- compat_conflicts: detection-oriented coarse BEV.
- tried_before: none.
- verified: checked

### CARD SN37 - TopoNet: which parts read the BEV and which need the perspective image
- papers: TopoNet, graph-based topology reasoning for driving scenes (Li et al., arXiv 2023; title from memory) https://arxiv.org/abs/2304.05277 ; code https://github.com/OpenDriveLab/TopoNet (URL from memory, not opened)
- depth: M (interface passages)
- stage: B, S6
- idea: A BEVFormer-style view transformer (3 layers, no temporal input) produces a BEV feature on a 200 x 100 grid with four height levels (cell size not stated; 0.5 m for +-50 x +-25 m is my inference); a lane-centerline decoder reads the BEV feature, a traffic-element decoder reads the front-view image feature (the perspective image is required there), and a scene-graph network over the queries predicts lane-lane and lane-element topology.
- formulation: two Deformable-DETR decoders (8 heads, 4 offset points, FFN 512), 200 lane queries, 100 element queries, dim 256; lane head 11 x 3 points; topology: two MLPs (128) per query type + sigmoid MLP on the pair; adjacency computed without gradients; one GNN layer is best; Hungarian matching with the loss as cost; loss weights lane cls 1.5, reg 0.025, topology 5.0; 24 epochs, batch 8, 8 x A100, AdamW 1e-4.
- input: multi-view images.
- output: 11-point 3D centerlines, boxes, adjacency.
- training: regime hungarian_set, end to end.
- differentiable: partial - Hungarian.
- sensors_data: camera, OpenLane-V2.
- results: subset A DET_l 28.5, TOP_ll 4.1, OLS 35.6; subset B 24.3, 2.5, 33.2 (older metric version; not comparable to later papers); 10.1 FPS (A100).
- compute: 10.1 FPS.
- code: see papers line.
- fit_to_us: reuse_idea - effort M - the lane branch needs only the BEV raster (compatible with a 39 -> 256 adapter), the traffic-element branch needs images (not available); Hungarian conflicts.
- compat_needs: BEV features 256-ch, 200 x 100.
- compat_conflicts: Hungarian; the element branch needs the front image.
- tried_before: E42 / E45 / E46 (TopoLogic-style and learned scoring within 0.01 of the rules).
- verified: checked

### CARD SN38 - MapTRv2: auxiliary dense losses (depth, BEV segmentation, PV segmentation) and one-to-many matching
- papers: MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction (Liao et al., arXiv 2023; title from memory) https://arxiv.org/abs/2308.05736
- depth: M (first ~100k of ~104k characters)
- stage: T, L
- idea: Keeps the MapTR decoder and adds three auxiliary dense losses on the encoder side (depth from lidar-rendered depth maps on perspective features, BEV segmentation on the BEV feature, perspective-view segmentation on image features), a one-to-many auxiliary query group, and decoupled attention; the auxiliary parts exist to train the encoder.
- formulation: one-to-many: T = 300 extra instance queries, each GT repeated K = 6, beta 1.0; decoupled self-attention O(N^2 + Nv^2) instead of O((N Nv)^2); default BEV encoder LSS with BEVPoolv2, cell 0.3 m; ranges x in [-15, 15] m, y in [-30, 30] m (a 100 x 50 m setting is not confirmed on the page read).
- input: multi-view images.
- output: classified point sets.
- training: regime hungarian_set + one-to-many; the page does not say whether auxiliary heads run at inference.
- differentiable: partial - Hungarian.
- sensors_data: camera (lidar for the depth target), nuScenes, Argoverse 2.
- results: nuScenes val mAP, 24 epochs R50: 61.5; ablation chain: MapTR 50.3, + LSS 50.1, + depth 55.2, + BEV seg 56.5, + PV seg 57.1, + decoupled attention 57.6, + one-to-many 61.5; 110 epochs R50 68.7.
- compute: 14.1 FPS (R50).
- code: not checked (same repository family as MapTR).
- fit_to_us: reuse_idea - effort S - tells which supervision belongs to the encoder: the BEV-segmentation auxiliary is already what GaussianCaR is trained on; perspective-view and depth auxiliaries cannot be recovered from a cached BEV raster.
- compat_needs: image features and lidar depth for the auxiliary losses.
- compat_conflicts: Hungarian one-to-one and one-to-many assignment.
- tried_before: E31 / E32 (matched sparse decoder dropped).
- verified: checked

### CARD SN39 - Radar BEV observability mask and unknown-region hallucination rate
- papers: On Learning Spatial Structure from Pre-Beamforming Per-Antenna Range-Doppler Radar Measurements (Sebastian, Berthold, Forkel, Pohl, Maehlisch; IEEE RA-L 2026) https://arxiv.org/abs/2604.01921
- depth: M (mask, loss and metric passages)
- stage: D, L, E
- idea: For a radar-to-BEV occupancy network, supervision is applied only where the radar field of view overlaps the lidar-observable area; lidar-unobservable cells inside the radar field are excluded from the loss but used to count how often the network "hallucinates" occupied cells there (unknown-region hallucination rate).
- formulation: lidar rays cast over azimuth bins of 0.05 deg after a height filter removes ground; the nearest non-ground return is the occluder, with no obstacle the farthest return sets the free extent; cells up to the endpoint are observable, beyond it unobserved; masked focal loss over supervised cells only; AP and occupied-class IoU computed inside the supervised region; BEV 0 - 60 m forward, +-38 m lateral, 0.5 m cells (120 x 152).
- input: radar range-Doppler data.
- output: BEV occupancy.
- training: masked supervised; regime matching_free.
- differentiable: yes (the mask is a fixed label-side tensor).
- sensors_data: radar + lidar (lidar for the mask only), the authors' own campus recordings (16,600 frames; 11,680 train / 4,920 validation), not nuScenes.
- results: no numbers seen.
- compute: not stated.
- code: not checked.
- fit_to_us: reuse_idea - effort S - gives the pair our censoring needs: a loss mask on unobserved samples plus a metric that counts positives hallucinated in unobserved cells; for us the lidar part would be replaced by the calibration mask (SN13) plus box shadows.
- compat_needs: a label-side observability map.
- compat_conflicts: lidar for the mask (project: no lidar input; label-side use is a project decision).
- tried_before: none.
- verified: partly (mask sections only)

### CARD SN40 - Other camera-radar detectors 2023 - 2025 (RCM-Fusion, SpaRC, CRT-Fusion, CR3DT): abstract level
- papers: RCM-Fusion: Radar-Camera Multi-Level Fusion for 3D Object Detection (Kim et al., ICRA 2024) https://arxiv.org/abs/2307.10249 ; SpaRC: Sparse Radar-Camera Fusion for 3D Object Detection https://arxiv.org/abs/2411.19860 ; CRT-Fusion: Camera, Radar, Temporal Fusion Using Motion Information for 3D Object Detection https://arxiv.org/abs/2411.03013 ; CR3DT: Camera-RADAR Fusion for 3D Detection and Tracking https://arxiv.org/abs/2403.15313
- depth: A (search-result summaries; not opened)
- stage: B
- idea: RCM-Fusion fuses at two levels (a radar-guided BEV encoder in which radar BEV features steer the camera-to-BEV transform, plus a radar grid-point refinement of instance proposals). SpaRC replaces dense BEV with a sparse fusion transformer (sparse frustum fusion, range-adaptive radar refinement) to avoid depth-estimation errors of dense BEV methods. CRT-Fusion models object-level dynamics to align moving objects across frames but still aggregates densely in BEV.
- formulation: not read.
- input: images + radar.
- output: boxes (CR3DT: tracks).
- training: not read.
- differentiable: yes (assumed from the architecture class; not read).
- sensors_data: camera + radar, nuScenes.
- results: SpaRC abstract: +2.9 NDS and +2.6 mAP on nuScenes over its baseline (reference method not seen); RCM-Fusion: state of the art among single-frame radar-camera methods at its time (no numbers seen).
- compute: not stated.
- code: not checked.
- fit_to_us: reuse_idea - effort M - none gives map or lane outputs; they show the two design poles (dense radar-guided BEV vs sparse query fusion) next to the GaussianCaR Gaussian-set pole.
- compat_needs: images.
- compat_conflicts: detection only; object-level temporal alignment does not apply to static lanes.
- tried_before: none.
- verified: partly (abstract-level search summaries)

### CARD SN41 - Horizon3D: range-binned radar-camera detection out to 150 m (TruckScenes)
- papers: Horizon3D: Sparse Radar-Camera Fusion for Long-Range 3D Perception in Autonomous Driving (arXiv 2026) https://arxiv.org/abs/2606.31096
- depth: M (first ~100k of ~105k characters)
- stage: B, E
- idea: Sparse radar BEV features (VoxelNeXt on 10 accumulated sweeps of six radars) and image-based Gaussian primitives are fused in a sparse BEV; radar keypoints (top 1024 BEV cells by a small MLP score) initialise the Gaussians; a dual-path temporal fusion is added. Evaluated on TruckScenes with range bins up to 150 m.
- formulation: three modules: keypoint-guided Gaussian initialisation, object-centric sparse fusion, dual-path temporal fusion; per-point radar features: position, RCS, radial velocity; camera depth range not stated.
- input: images + radar sweeps.
- output: boxes.
- training: not read in detail.
- differentiable: yes (top-K selection of keypoint cells is a hard operation).
- sensors_data: camera + radar, TruckScenes (747 scenes, evaluation range 150 m), not nuScenes.
- results: validation Table 9, mAP / NDS by range for BEVFusion (the compared camera-radar baseline): 0-25 m 28.8 / 35.7, 25-50 m 16.8 / 25.2, 50-100 m 11.7 / 20.1, 100-150 m 7.4 / 13.9; Horizon3D: 36.5 / 43.0, 23.3 / 37.8, 17.0 / 32.7, 8.9 / 18.4. Overall val (ResNet-50): Horizon3D 37.4 NDS / 23.6 mAP vs BEVFusion 30.4 / 18.2. Radar-only baseline 12.8 mAP / 27.1 NDS without range breakdown; no ablation of the radar contribution by range.
- compute: 8.5 FPS, 3.2 GB (BEVFusion 4.2 FPS, 6.52 GB).
- code: not checked.
- fit_to_us: reuse_idea - effort S - a range-binned reference showing that even the best radar-camera detector loses about 53 % of its mAP between 0-25 m and 50-100 m (36.5 -> 17.0); the task is vehicles, not lanes, and the dataset is trucks; use only as a shape of the range curve.
- compat_needs: sparse BEV; images.
- compat_conflicts: different dataset and task.
- tried_before: none.
- verified: partly (reader summary; supplement failure cases cut off)

## 3. Cluster compatibility notes

### (a) Interface contracts: what flows in and out, and how it maps to our cache
- Our cache: 200 x 200 at 0.5 m, +-50 m, heading-aligned ego frame, 32 PCA channels of the 256-ch decoder feature + 7 map logits (+ a second 32-ch fused feature), float16. Native GaussianCaR outputs per frame: 7 logits, a 256-ch decoder map, a 4-scale fused pyramid (128 ch at 200 / 100 / 50 / 25), 128-ch camera and radar BEV maps (SN01). The PCA to 32 channels is ours; no paper measures what it discards. The native fused pyramid already has the pyramid shape of our trace network (200 / 100 / 50), so re-caching it (instead of building a pyramid from the final decoder map) is a candidate; not tested.
- BEV conventions in mapping / segmentation papers (cells, extent, cell size; card):
  - 200 x 200, 100 x 100 m, 0.5 m: LSS, BEVFusion-MIT map evaluation, CRN segmentation, Simple-BEV (200 x 8 x 200), BEVCar, GaussianBeV, GaussianCaR, RESAR-BEV, FIN, CTAB (upsampled from 0.8 m), parametric-depth BEV, OccFeat -> the dense-segmentation convention; ours.
  - 200 x 200, +-51.2 m, 0.512 m: BEVFormer; 128 x 128 at 0.8 m (+-51.2 m): CRN / HyDRa / RCBEVDet detection; 180 x 180 over +-54 m: CRKD; Occ3D 200 x 200 over +-40 m at 0.4 m.
  - 200 x 100 for +-50 m x +-25 m at 0.5 m: LaneSegNet, TopoNet (cell size inferred), OpenLane-V2 evaluation; TopoBDA 200 x 104 (0.5 m). Our raster contains this: crop lateral +-25 m, no resampling.
  - 60 x 30 m: HDMapNet / MapTR / StreamMapNet (30 m); cell 0.15 m (400 x 200: MGMap, MV-Map) or 0.3 m / 0.75 m (MapTR tiny / nano; MapTRv2 0.3 m); 100 x 50 m for StreamMapNet (50 m) and MV-Map's long range at 0.25 m. A 60 x 30 m head on our cache needs a crop (+-30 x +-15 m) and a bilinear resolution adapter (0.5 m -> 0.15 m is a 3.3 x finer grid; the grid grows, the information does not).
  - Other: PON 50 m ahead, 25 m lateral at 0.25 m; RNTR 96 x 64 m; CenterLineDet not stated; radar BEV paper SN39 0 - 60 m forward, +-38 m at 0.5 m.
- Which heads can read a frozen cached BEV raster:

| head family | reads | needs images | encoder-tied queries | published training | fit to the cache |
|---|---|---|---|---|---|
| dense conv decoders (HDMapNet, CRN / BEVFusion / GaussianCaR / FIN heads, our U-Net) | BEV raster | no | no | matching-free, jointly with the encoder in the papers | direct |
| query heads on BEV (MapTR, MapTRv2, LaneSegNet, StreamMapNet, TopoBDA, TopoNet lane branch) | BEV features, about 256 ch, deformable attention at normalised reference points | no (training-time aux losses in MapTRv2) | no (own queries) | Hungarian, jointly with the encoder | adapter: 1 x 1 conv 39 -> 256 + normalised coordinates; Hungarian conflicts with matching-free losses (E31 / E32) |
| step / trace heads on BEV heat maps (CenterLineDet, our E94) | BEV heat map or raster crop | no | no | imitation, separate stage | direct |
| frozen encoder + sequence head (RNTR) | BEV features | no | no | encoder frozen, then optional joint fine-tune | direct for the regime; token head not differentiable |
| image-dependent (CRAFT, BEVCar lifting, HyDRa PV fusion, RaCFormer, CRN RVT, TopoNet element branch, BEVFormer) | perspective features or encoder-internal queries | yes | yes | end to end | not usable; ideas only |

- Our `grid_sample` at metric positions equals the deformable-attention sampling primitive and BEVFusion's "grid-sample the fused BEV to the head's range" step; no adapter beyond unit conversion is needed for sampling-based heads.

### (b) Known adapters and known conflicts
- Adapters: channel adapter (39 -> C), crop adapter (200 x 200 -> 200 x 100), resolution adapter (bilinear, cell size), coordinate normaliser (metres -> [0, 1]), foundation-side extra outputs (SN01: per-modality maps, fused pyramid, alpha coverage).
- Conflicts: Hungarian matching (MapTR, LaneSegNet, StreamMapNet, TopoBDA, TopoNet) vs matching-free losses; lidar-dependent training (BEVDepth, CRN depth supervision, CRKD / BEVDistill / RNTR-TIT teachers); detection-pretrained features (CRN) lose to map-pretrained ones for lanes (E27); non-commercial licence of the vendored rasteriser (SN01); FP32-only training of GaussianCaR; token-sequence heads need argmax; coarse detection BEVs (0.8 m) cap thin structures (SN36).
- Evaluation conflicts: cross-paper tables are not protocol-matched (RESAR-BEV re-evaluates BEVCar at 79.40 / 43.07 where the BEVCar paper has 83.3 / 45.3; CTAB notes that class sets differ); nuScenes scene-disjoint splits still share locations (SN03).

### (c) Combinations demonstrated by the papers
- LSS-family camera stream + radar: BEVDepth + radar -> CRN, RCBEVDet, HyDRa, FIN, CTAB; Simple-BEV bilinear lift + radar raster; BEVCar frozen DINOv2 + radar-guided queries; GaussianCaR = GaussianBeV-style Gaussian lift + PTv3 radar Gaussians + CMX fusion + DPT decoder.
- BEV encoder + map decoder: BEVFusion-MIT (6 binary maps, focal loss); CRN + CVT-style conv decoder; FIN + two-level U-Net; GaussianBeV + LSS / Simple-BEV backbone; MapTR with IPM, LSS, deformable attention, GKT; HDMapNet decoder on camera / lidar / fusion features; TopoBDA / BEVCar radar inside the encoder.
- Frozen BEV encoder + head: RNTR; staged: CenterLineDet, BEVFusion-PKU, RCBEVDet, CRAFT.
- Distillation: lidar-camera teacher -> camera-radar student (CRKD); lidar teacher -> camera student (BEVDistill); lidar BEV -> camera BEV, L1 (RNTR); DINOv2 feature prediction at occupied voxels (OccFeat).
- Temporal: BEVFormer TSA; StreamMapNet GRU; CRN concatenation; HDMapNet; CenterLineDet.
- Visibility: PON (mask + entropy), Occ3D (ray casting), radar BEV (lidar mask + hallucination metric), parametric depth (soft V).

### (d) Answers to the brief's questions, with pointers
1. Heads on a frozen cached raster: see the table in (a). Dense decoders and step / trace heads fit directly; query heads fit after a channel adapter but bring Hungarian matching; image-dependent designs do not fit (SN04, SN05, SN26, SN27, SN22, SN37).
2. Range / resolution mismatch: our 100 x 100 m at 0.5 m is the dense-segmentation convention; LaneSegNet / OpenLane-V2 heads use a 200 x 100 crop of the same grid; 60 x 30 m vector-map heads use 0.15 - 0.3 m cells (crop + upsample). StreamMapNet loses 10.9 mAP going from 30 m to 50 m range (33.9 -> 23.0, new split), the range cost is real.
3. Train-to-test shift (F1@1 0.93 on the foundation's training scenes vs 0.82 on unseen): (i) on official nuScenes splits 79 - 86 % of val / test samples lie within 5 m of a training sample, and removing the overlap lowers online-mapping scores by more than 35 mAP on average (SN03); RNTR lane-graph landmark F falls from 54.1 to 36.5 (SN02); StreamMapNet 62.9 -> 33.9 at 30 m (SN18); PON redistributed sequences already in 2020 (SN10); (ii) the cause named is memorisation of an implicit map, with dividers drawn in occluded areas (SN03) - the foundation's map layers are trained without visibility masking (SN01), so its logits in unobserved cells are priors; (iii) a second stage trained on the first stage's training-set outputs sees cleaner inputs than at test time and needs held-out outputs (SN34); (iv) frozen vs fine-tuned: a frozen generic encoder is much worse than end-to-end (LSS drivable 61.6 vs 72.9), but a frozen map-pretrained encoder with a trained head is indistinguishable from unfreezing on lane graphs (RNTR 48.6 vs 49.8 reachability F), fine-tuning can distort features under shift (LP-FT, classification evidence), and adapters inside the backbone beat head-only adaptation under geometric shift (UDGA); (v) pretraining helps most with few labels (OccFeat), which favours keeping the foundation frozen with 212 training scenes. No paper measures the feature gap of a frozen map-pretrained BEV encoder or tests mitigations; suggestions that follow (not tested): report seen / unseen scenes separately and a location-disjoint TEST; weight head training towards the foundation's unseen scenes; if unfreezing, use LP-FT order (head first).
4. Radar and far lanes: radar returns come from reflecting structure, not paint; lateral resolution is about 1.3 m at 24 m and 2.8 m at 50 m (near mode, derived from the datasheet); 3D radar has no height; one sweep is too sparse, 7 sweeps help (+2.4 vehicle IoU over 1); radar still lifts lane-divider IoU by +4.7 (BEVCar) and TopoBDA DET_l by +4.3; for vehicles the radar gain grows with range (BEVCar vs its camera-only twin: +5.3 at 0-20 m, +10.7 at 20-35 m, +11.9 at 35-50 m, Table III; my reading: vehicles are strong radar reflectors); for lanes no range-binned number exists, so the vehicle trend must not be transferred to lane paint. Camera: depth bins stop at 61 m in the foundation; ground sampling distance reaches about 1.3 m per pixel at 50 m (SN13); a third of the BEV is occluded under a depth-based definition and lane IoU there is about 12 against about 42 on visible cells (SN12).
5. Visibility from calibration alone: SN13 (FoV, depth cap, near-field ring, GSD) as a soft multiplier; vehicle occlusion needs boxes or lidar at label time (SN10, SN11, SN39) or a learned head; the foundation's own alpha coverage is a second source (SN01, derived).
6. Temporal: encoder-side gains are +1 - 3.5 points on map / detection metrics (SN33); our recursion is vertex-level and complementary; the frozen cache cannot be given encoder-side fusion.
7. Code and licence of the foundation: SN01 (Apache-2.0 for the repository, non-commercial licence for the vendored rasteriser).
8. Consolidated limits of the foundation for lane perception (all from the cards, SN01 unless noted): labels - the lane layer holds only lane polygons (no connectors), dividers are 1 m wide bands, map layers are trained without visibility masking; sensors - camera Gaussians stop at 61 m, ground sampling distance is about 1.3 m per pixel at 50 m (SN13), radar lateral resolution is about 1.3 m at 24 m and the radar sees reflectors, not paint (SN14), camera is single-frame; training - 700 scenes of two cities, 40 epochs, FP32, no depth supervision stated; evidence - only drivable area (82.9) and lane divider (50.1) are published, no per-range numbers (best published lane-divider IoU in its table, still only half overlap), no failure analysis; transfer - features on its own training scenes are cleaner than on unseen scenes (project: 0.93 vs 0.82) and the literature on location leakage (SN03, SN02) says that scene-disjoint is not location-disjoint.
9. Head-side regularisation from the BEV literature (suggestions, not tested): BEV-space augmentation of cached raster and labels together (BEVDet: augmentation decides whether a BEV head overfits, SN24), the PON-style entropy term on unobserved cells (SN10), feature-level noise to emulate unseen-scene feature quality (no paper found).

## 4. Reuse versus re-implement
- Reuse as is: the GaussianCaR checkpoint and code in env `gcar` (Apache-2.0 for the repo; the vendored rasteriser is non-commercial research only - fine for research use, a problem for products). Extra outputs and the alpha-coverage map: small code change, no retraining (S).
- Re-implement (S): the calibration-only visibility mask (SN13) with the foundation's exact crop; a location-distance-binned evaluation (SN03); a head-side BEV GRU / warp (SN18) if temporal fusion is revisited; a PON-style entropy term and a hallucination-rate metric for unobserved samples (SN10, SN39).
- Re-implement (M): an evidential uncertainty head on the logits (SN32); adapters in the foundation's decoder (SN29) only if the full decoder is exported; a 39 -> 256 channel adapter plus a query head if a LaneSegNet-style baseline is wanted (SN17), accepting Hungarian.
- Do not port: BEVFormer, BEVDepth, CRN, CRAFT, BEVCar, HyDRa, RaCFormer, RESAR-BEV (image path); MapTR / LaneSegNet / TopoNet / TopoBDA decoders (Hungarian, mmcv-era code; idea-level reuse only).

## 5. Open questions and gaps
- No paper trains lane heads on frozen cached camera-radar BEV features, and none measures the gap between features of the encoder's training scenes and unseen scenes. RNTR (frozen vs unfrozen) and the split papers (SN02, SN03, SN18) are the nearest evidence; the 0.93 vs 0.82 gap is therefore neither explained nor bounded by the literature.
- The foundation's training scene list and its overlap with our 255 scenes (and with location overlap) is in no paper; it decides how large the "seen" part is.
- GaussianCaR: per-layer map IoU for the "lane" polygon layer and the other five layers is unpublished; no limitation or failure-case analysis; the image resolution (448 x 800 in the paper vs 224 x 480 in the repo config) should be checked against the checkpoint before building the visibility mask; the devkit's mask excludes lane connectors, so connector areas are learned only indirectly (road_segment).
- Radar for lanes: only TopoBDA (+4.3 DET_l on OpenLane-V2 B) and BEVCar (+4.7 lane-divider IoU) show a gain; no study isolates radar-off effects on far (> 24 m) lanes; radar mounting and field of view per unit were not found.
- Occlusion by vehicles: no calibration-only method exists; the recipe in SN13 needs boxes or lidar at label time, or a learned visibility head.
- Range-binned map IoU (near / mid / far) for lane or divider layers is not published in text form (FIN: figures only); vehicle range tables exist (BEVCar, RESAR, CRN).
- Not read: SOLOFusion / BEVDet4D, StreamPETR, GaussianLSS, MapPrior; RCM-Fusion, SpaRC, CRT-Fusion, CR3DT only at abstract level (SN40).

## 6. Search log
1. GaussianCaR camera radar Gaussian BEV map segmentation nuScenes ICRA 2026
2. GaussianCaR github code license
3. camera radar fusion vectorized HD map construction nuScenes radar BEV online mapping
4. radar lane marking road boundary detection automotive radar mapping lane estimation neural network nuScenes radar
5. camera-radar BEV map segmentation drivable area lane divider IoU nuScenes 2025 radar fusion BEVGuide BEVCar comparison (extended)
6. BEVGuide BEV representation learning camera radar guide nuScenes vehicle segmentation arXiv
7. BEV semantic map prediction visibility mask lidar ray casting field of view mask evaluation only visible cells
8. Occ3D camera visibility mask ray casting voxel visible from camera
9. Localization Is All You Evaluate data leakage online mapping datasets nuScenes overlap
10. frozen BEV encoder features downstream head training linear probing
11. BEV pre-training self-supervised BEV encoder frozen evaluation only head trained (extended)
12. automotive radar road boundary estimation static targets guardrail
13. radar camera fusion lane detection BEV centerline road topology radar helps distant lanes
14. online map uncertainty per-vertex uncertainty trajectory prediction Gu CVPR 2024
15. BEV segmentation uncertainty unobserved occluded regions evidential
16. CRKD camera-radar cross-modality knowledge distillation
17. BEVDistill cross-modal BEV knowledge distillation
18. HyDRa camera-radar fusion hybrid depth radar
19. RaCFormer camera radar query-based SpaRC RCM-Fusion CRT-Fusion
20. parameter-efficient fine-tuning adapter BEV perception frozen BEVFormer
21. nuscenes CAM_FRONT calibrated_sensor camera_intrinsic (twice; values found on a third-party tutorial page)
22. GaussianLSS Gaussian splatting depth uncertainty BEV
23. 3D lane detection visibility vector lane anchor points
24. train second stage on frozen first-stage outputs out-of-fold stacking
25. HDMapNet BEV 60m x 30m range 0.15m resolution
26. BEV map segmentation IoU by distance range 0-20m 20-35m 35-50m lane divider
27. camera radar fusion BEV 2026 arXiv map segmentation lane divider outperform GaussianCaR BEVCar (extended)
28. Continental ARS 408-21 radar datasheet field of view range resolution
29. nuScenes sensor setup 5 x long range radar ARS 408-21 77GHz 13Hz
30. radar-camera fusion review sparse noisy ghost targets multipath elevation
31. pre-computed frozen BEV features offline cache train lightweight head
32. CRT-Fusion; RCM-Fusion; SpaRC camera radar fusion abstracts (extended)
- Page reads (arXiv HTML / ar5iv unless noted): 2602.08784, 2407.14108, 2403.11761, 2206.07959, 2304.00670, 2505.06515, 2510.00651, 2604.12918, 2403.16440, 2209.06535, 2403.07746, 2205.13542, 2205.13790, 2203.17270, 2008.05711, 2206.10092, 2412.18951 (two passes), 2304.05277, 2107.06307, 2208.14437, 2308.05736, 2312.16108, 2308.12570, 2402.08207 (two passes), 2209.07734, 2312.06420, 2003.13402, 2304.14365, 2307.04106, 2604.01921, 2403.19104, 2211.09386, 2112.11790, 2606.31096, 2404.14027, 2410.22461, 2405.20986, 1903.11027, 2203.11089, 2304.10410, 2202.10054 (abstract), 2407.06683 (abstract); GitHub raw files of santimontiel/gaussiancar (README, LICENSE, pyproject, configs, model, radar_encoder, fuser, dpt, render, loading, labels, nuscenes_map_api, vendored rasteriser LICENSE.md, issues page) and Pointcept/PointTransformerV3 (licence line); nuscenes-devkit data_classes.py and map_expansion/map_api.py; HF model page; nuScenes tutorial page (third-party) for CAM_FRONT values.
