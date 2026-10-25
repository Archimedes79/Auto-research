# Cluster B (loop 17): robustness of BEV perception and online mapping to sensor failure and corruption

Date of research: 2026-10-24. Prefix of cards: RB. Format: `SCHEMA.md` of lit16. Judged against `PROJECT_CONTEXT.md` (frozen GaussianCaR cached 39-channel BEV raster, 200 x 200 at 0.5 m, dense head -> differentiable sparse graph builder -> route head -> planner consumer).

## 1. Coverage statement

- Searched (WebSearch, about 45 queries) and opened (WebFetch, about 60 pages): arXiv abstract and HTML pages, ICLR / CVPR / ICML listing pages. Nothing was downloaded, no code was cloned, no model weights were touched.
- IMPORTANT limit of the reading method: WebFetch answers a question about a page through a small summarising model. Depth `M` here means "the method / experiment sections were queried and answered with equations, parameters, losses and table numbers"; I did not see raw LaTeX or figures, and when a summary contradicted itself (several do) I say so in the card. All numbers are as reported by the page; none are mine.
- Depth M (method and tables queried): RB01 PFS, RB02 ModalPatch, RB03 SafeMap, RB04 RoboMap, RB05 RAF, RB06 Grace-BEV, RB07 MetaBEV, RB08 UniBEV, RB11 RCBEVDet, RB13 BEVCar, RB14 GaussianCaR, RB16 ReliFusion, RB17 RoboBEV, RB18 MapBench, RB19 MultiCorrupt, RB20 Samsung-RAL, RB21 DINOv2-LoRA, RB22 LearnableBN, RB29 online-map uncertainty, RB30 BEV-UQ benchmark, RB31 camera health monitor, RB33 LIAR, RB34 adversarial camera-radar (first 100k characters of 117k).
- Depth A (abstract / listing page only): RB09 SB-BEVFusion (alphaxiv overview), RB10 RoCaRS, RB12 RobuRCDet, RB15 M-BEV, RB23 - RB25 TTA primitives, RB26 NCGR, RB27 CFT, RB28 MapLightning, RB32 LightDiff / DualCross, RB35 StreamMapNet / stability benchmark.
- Not found / not verifiable: RoCaRS has no arXiv or publisher page that I could open; I only saw its abstract text in search-result snippets of a Korean lab database (RB10 is marked "link not verified"). Retinex-BEVFormer / RetBEV (ICRA 2025 / journal) had no openable page. The stability benchmark PDF (2510.10660) was too large to fetch; only its abstract page was read.
- Honest gap: no paper found that does test-time adaptation on a BEV segmentation or lane output; no paper found that evaluates extrinsic-error robustness on map / lane output of a camera-radar network; no radar-camera map-segmentation robustness benchmark with radar corruptions.
- All card links below were opened by me unless the card says otherwise. arXiv ids 26xx exist (today is 2026-10-24) and were opened.

## 2. Cards (ordered by relevance to a frozen-foundation + small-head setting)

### CARD RB01 - Post Fusion Stabilizer (PFS): identity-initialised repair module between frozen fusion and head
- papers: Post Fusion Bird's Eye View Feature Stabilization for Robust Multimodal 3D Detection (Dong, Thakkar, Sargolzaei, Lin, arXiv 2026, v2 3 Sep 2026) https://arxiv.org/abs/2603.05623 (opened as HTML https://arxiv.org/html/2603.05623)
- depth: M
- stage: B, T
- idea: A small module sits between the fused BEV feature map and the unchanged detection head of a frozen host detector. It first normalises distribution shift, then estimates a per-cell reliability map, then runs expert correction / inpainting; every block starts as identity so clean performance is kept. Only the module is trained, in three curriculum stages, on clean and corrupted data.
- formulation: three blocks - shift normalisation (~99 K params), spatial reliability estimation (~664 K), expert correction and inpainting (~2.5 M); trained with detection and reliability losses (exact loss form not stated in what I read). Input and output tensors have the same shape.
- input: fused BEV feature map (shape not stated).
- output: corrected BEV feature map of the same shape (plus an internal reliability map).
- training: host frozen; module trained with a curriculum on clean + synthetic corruptions (camera rain lens, low light, occlusion, 6-camera dropout; LiDAR beam reduction, range / sector dropout, miscalibration; nuScenes-C fog, snow, sunlight).
- differentiable: yes - plain feed-forward on a raster.
- sensors_data: camera + LiDAR, nuScenes val; hosts BEVFusion and UniBEV; a 200-frame physical test with one camera and a 32-beam LiDAR, no fine-tuning.
- results: BEVFusion mAP clean 69.5 vs 69.2 baseline; Low Light 63.2 vs 58.8; Rain Lens 63.6 vs 60.0; 6-camera dropout 63.8 vs 62.6 (only +1.2); 8-beam 17.5 vs 12.1; nuScenes-C Fog 67.9 vs 54.1 (paper text and Table III disagree on snow / sunlight gains). Source: paper tables as summarised.
- compute: BEVFusion 40.8 M -> 44.1 M params; latency 70.9 -> 76.6 ms (+8.1 %, +5.7 ms); UniBEV 417 -> 436 ms (+4.5 %). GPU not stated in what I read.
- code: none found on the page.
- fit_to_us: reimplement - effort M - closest published match to our interface (frozen features in, same-shaped features out); we would train it on cached clean vs corrupted 39-channel rasters, but the gain for lost-camera (+1.2 mAP) is small and it was not tested on map outputs.
- compat_needs: a paired set of corrupted and clean cached features; a loss on the downstream head output or a feature-consistency loss.
- compat_conflicts: cannot restore information that the frozen foundation never encoded (lost camera views, wrong extrinsics inside the foundation).
- tried_before: close to "fine-tuning on failure frames" (helps night / blur, nothing for lost cameras / extrinsics) but PFS puts the capacity in front of the head and adds a reliability map; not tried as such.
- verified: partly

### CARD RB02 - ModalPatch: learned history-based feature prediction + uncertainty-gated fusion
- papers: ModalPatch (plug-and-play module for LiDAR / camera drops in multi-modal 3D detectors; full paper title not read; arXiv 2603.02481, 3 March 2026) https://arxiv.org/abs/2603.02481 (opened as HTML https://arxiv.org/html/2603.02481)
- depth: M
- stage: S8, B
- idea: When a modality drops for a frame, a memory of the last 6 frames' features is attended by a learnable BEV query (deformable attention) to predict the missing feature; an MLP then predicts per-location variance of the predicted (compensated) feature, and that uncertainty down-weights it in cross-modal enhancement. Detector weights stay frozen.
- formulation: memory bank tau = 6 frames; stage 1 (12 epochs) trains the history predictor with feature-prediction loss + detection loss; stage 2 (12 epochs) freezes it and trains the uncertainty branch with a Gaussian NLL + fusion + detection loss; AdamW lr 2e-4. Ego-motion alignment of the bank is not stated in what I read.
- input: BEV features of the present modality, memory of past BEV features.
- output: compensated BEV feature + per-cell variance.
- training: matching-free (feature regression + task loss); drop rates 10 / 30 / 50 % simulated.
- differentiable: yes.
- sensors_data: LiDAR + camera, nuScenes val; BEVFusion, CMT, UniBEV, MEFormer as frozen hosts.
- results: average gain over hosts +4.70 mAP / +2.29 NDS at 10 % drop, +11.14 / +5.28 at 30 %, +11.93 / +5.05 at 50 % (paper text); CMT 27.21 -> 44.21 mAP at 50 %. Detection only.
- compute: average throughput 5.33 -> 4.90 FPS at 50 % drop (about 8 % slower); ms not stated.
- code: announced at github.com/Castiel-Lee/MM3Det_MD, "will be available" (not checked live).
- fit_to_us: reimplement - effort M - this is the learned version of our training-free spatial hold: replace "copy the warped previous input" by "predict the current feature from 6 warped past features, with a variance".
- compat_needs: ego-warped feature memory (we already have the input memory); a variance / NLL head.
- compat_conflicts: needs a per-frame "this input is missing" signal or gating (our label-free heat monitor could provide it).
- tried_before: training-free spatial hold of the previous input (verdict in loop 17 prior: partial); the learned variant not tried.
- verified: partly

### CARD RB03 - SafeMap: robust online HD map with missing cameras (view reconstruction + BEV distillation)
- papers: SafeMap: Robust HD Map Construction from Incomplete Observations (Hao, Kong et al., ICML 2025) https://arxiv.org/abs/2507.00861 (opened HTML https://arxiv.org/html/2507.00861v1); builds on MapBench (RB18)
- depth: M
- stage: B, L, T
- idea: Two plug-ins for MapTR-type camera mappers. G-PVR rebuilds the features of a missing perspective view from the remaining views with a deformable-attention decoder whose reference points are Gaussian-distributed toward the neighbouring views. D-BEVC distils the BEV feature from the complete-view model into the BEV feature built from incomplete views with an MSE loss.
- formulation: L = L_map + 0.05 L_Rec + 5 L_Cor; sigma = 3 for the Gaussian reference sampling; MSE beat L1 (47.1) and KL (46.8) at 47.3 mAP; views randomly dropped in training; AdamW lr 4.2e-4, 8 epochs on nuScenes.
- input: multi-view image features (G-PVR) and BEV features (D-BEVC).
- output: reconstructed view features; corrected BEV features.
- training: Hungarian matching inherited from MapTR (not ours); the distillation term itself is matching-free.
- differentiable: yes (the plug-ins); the host loss uses matching.
- sensors_data: camera only; nuScenes (also Argoverse2); MapTR, HIMap.
- results: nuScenes mAP missing front: MapTR 31.3 -> 42.4, HIMap 38.0 -> 54.3; missing back: MapTR 28.5 -> 40.5, HIMap 33.1 -> 51.3; complete input 50.3 -> 52.5 (MapTR). 1 / 3 / 5 missing views average 39.9 -> 47.3 / 19.5 -> 22.8 / 4.0 -> 6.0. MapBench corruptions: HIMap mRR 56.6 -> 62.8, mCE 100 -> 83.2.
- compute: MapTR 39.1 M -> 39.5 M params, 21.5 -> 21.4 FPS, 2298 -> 2300 MB; HIMap 68.1 M -> 71.7 M, 9.7 -> 9.2 FPS (Table 8).
- code: none found on the page.
- fit_to_us: reuse_idea - effort S for D-BEVC - the clean-teacher MSE on the BEV feature can be applied to cached features (clean feature as target for a corrupted-input repair module); G-PVR needs image features, which we do not have.
- compat_needs: paired clean / incomplete features.
- compat_conflicts: view reconstruction is impossible on our cached fused raster.
- tried_before: none (failure fine-tuning uses the task loss only).
- verified: checked

### CARD RB04 - RoboMap: what matters for robust multi-sensor HD map construction (augmentation + fusion + modality dropout)
- papers: What Really Matters for Robust Multi-Sensor HD Map Construction? (arXiv 2507.01484, 2025) https://arxiv.org/abs/2507.01484 (opened HTML https://arxiv.org/html/2507.01484); project https://robomap-123.github.io/
- depth: M
- stage: B, T
- idea: A recipe study on MapTR / HIMap with camera + LiDAR: camera GridMask and LiDAR point dropout augmentation, a cross-modal interaction transformer (CIT, self-attention over concatenated flattened BEV tokens) with a dynamic fusion, and modality dropout of whole BEV feature maps.
- formulation: drop probability p_md; if dropped, LiDAR kept with p_L, camera with 1 - p_L (values not reported). ResNet50 + GKT camera, SECOND-style LiDAR, AdamW lr 4.2e-4.
- input: camera BEV and LiDAR BEV features.
- output: fused BEV feature for the map head.
- training: end-to-end; Hungarian loss of the host; no frozen foundation.
- differentiable: yes.
- sensors_data: camera + LiDAR; nuScenes val with the MapBench corruption suite.
- results: clean mAP MapTR (C+L) 62.5 -> 71.5; HIMap 74.3 -> 77.0 (Table I); mRS MapTR 69.51 -> 79.06, HIMap 67.69 -> 81.99 (Tables II / III). Ablation gains on MapTR (fusion / augmentation / dropout) +3.3 / +4.5 / +1.6 mAP; components help only in combination. Snow stays weak (19.55 -> 23.78); motion blur 70.00 -> 89.88.
- compute: not reported.
- code: none found (project page only).
- fit_to_us: reuse_idea - effort S - confirms that augmentation gives the largest single gain and that modality dropout is a small add-on; with a frozen camera-radar foundation, dropout can only be emulated by caching foundation features for each failure condition.
- compat_needs: per-condition feature caches.
- compat_conflicts: needs separable modality BEV maps; ours are already fused.
- tried_before: failure-frame training (corresponds to the augmentation part).
- verified: checked

### CARD RB05 - RAF: reliability map for the camera branch, supervised by pseudo labels from other sensors
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

### CARD RB06 - Grace-BEV: trust gate + failsafe fusion trained with modality dropout over a frozen vision backbone
- papers: Can BEV Perception Gracefully Degrade under Sensor Failures? (Zhang, Wang, Wang, Li, Zuo, arXiv 2605.30983, 29 May 2026) https://arxiv.org/abs/2605.30983 (opened HTML)
- depth: M
- stage: B, T
- idea: A plug-in for LSS-style BEV fusion. TrustGate pools the LiDAR BEV into a descriptor, a small MLP outputs a scalar trust alpha, and blends a LiDAR-guided expert with a vision-only expert; a FailSafe block applies a learned element-wise gate. Three-phase training: vision pre-training, frozen backbone with modality dropout (both 0.5, LiDAR-drop 0.25, camera-drop 0.25), then low-learning-rate end-to-end.
- formulation: bottleneck MLP 64 hidden units, ReLU, sigmoid; gate reduction 4 with dilated 3 x 3 conv, last bias init 0; loss not named.
- input: LiDAR BEV + image BEV features. output: fused BEV feature.
- training: staged, frozen backbone in stage 2.
- differentiable: yes.
- sensors_data: camera + LiDAR; nuScenes-R (LiDAR beam reduction / drop / limited FOV, camera view drop, object reflection noise), nuScenes-C weather.
- results: LiDAR drop BEVFusion-AD 0.0 -> 34.7 mAP, BEVFusion-MIT 0.0 -> 32.8; clean +1.1 / +1.4 mAP; ablation router alone 28.5, block alone 15.5, both 32.8 (LiDAR drop, MIT). Detection only; no map segmentation.
- compute: +1.2 M params; BEVFusion-MIT latency 40.76 -> 40.81 ms (A100, batch 1).
- code: none found.
- fit_to_us: reuse_idea - effort S - proves a scalar trust gate is almost free in ms (0.05 ms) and shows the three-phase frozen-backbone training recipe.
- compat_needs: a sensor-separable BEV pair.
- compat_conflicts: our fused features offer no pair; a gate would blend "repaired" and "raw" features instead.
- tried_before: none.
- verified: checked

### CARD RB07 - MetaBEV: modality-arbitrary decoder with switched-modality training
- papers: MetaBEV: Solving Sensor Failures for 3D Detection and Map Segmentation (Ge et al., ICCV 2023) https://arxiv.org/abs/2304.09801 (opened ar5iv https://ar5iv.labs.arxiv.org/html/2304.09801); CVF page https://openaccess.thecvf.com/content/ICCV2023/html/Ge_MetaBEV_Solving_Sensor_Failures_for_3D_Detection_and_Map_Segmentation_ICCV_2023_paper.html
- depth: M
- stage: B, T
- idea: Learnable dense BEV queries (180 x 180) cross-attend to camera and LiDAR BEV features through four deformable cross-modal layers plus two self-attention layers, with modality-specific MLPs producing offsets and weights, so any subset of sensors can be used. Mixture-of-expert FFNs (M2oE) serve detection and segmentation. Training randomly feeds camera-only, LiDAR-only or both at 1/3 each (switched-modality training) starting from a pretrained task model.
- formulation: detection and segmentation losses weighted 10 : 1 (as BEVFusion); AdamW wd 0.05, cyclic lr, 26 epochs (detection) / 20 (segmentation); Swin-T + VoxelNet encoders. Frozen backbones: not stated.
- input: camera BEV + LiDAR BEV. output: dense BEV queries -> heads (map segmentation head included).
- training: sensor-switching, no matching for the map head.
- differentiable: yes.
- sensors_data: camera + LiDAR, nuScenes val.
- results: map segmentation mIoU, missing camera: BEVFusion 4.1 vs MetaBEV 53.7; missing LiDAR: 36.7 vs 54.4; full 70.4 (Table 2; baselines run with the missing input zeroed). Detection NDS missing LiDAR 42.6.
- compute: latency, FPS, parameters not reported (limitations mention a slight increase).
- code: not checked.
- fit_to_us: reuse_idea - effort M - one of the few papers with BEV MAP results under sensor loss; shows zero-filling is a bad baseline and a learned decoder over "available subset" fixes it.
- compat_needs: separable modality BEV maps.
- compat_conflicts: conflicts with a pre-fused frozen raster.
- tried_before: none.
- verified: checked

### CARD RB08 - UniBEV: uniform BEV encoders, modality dropout and channel-normalised fusion
- papers: UniBEV: Multi-modal 3D Object Detection with Uniform BEV Encoders for Robustness against Missing Sensor Modalities (Wang, Caesar, Nan, Kooij, IEEE IV 2024) https://arxiv.org/abs/2309.14516 (opened HTML v3)
- depth: M
- stage: B, T
- idea: Both modalities are mapped to BEV with the same deformable-attention scheme; features are fused by channel-normalised weights (CNW) over the available modalities only (missing ones are skipped, not zero-filled). Training with modality dropout p_md = 0.5, LiDAR kept with p = 0.5.
- formulation: fusion options concat 51.9, average 52.3, CNW 52.5 mAP averaged over input combinations; roughly 50 % full, 25 % LiDAR-only, 25 % camera-only.
- input: camera + LiDAR BEV. output: fused BEV -> detection head.
- training: modality dropout.
- differentiable: yes.
- sensors_data: nuScenes val, R101-FPN + VoxelNet.
- results: camera-only NDS/mAP 42.4 / 35.0 vs BEVFusion 29.6 / 22.6 and MetaBEV 33.6 / 25.9; LiDAR-only 65.3 / 58.2; all combos average mAP 52.5 vs 43.5 (BEVFusion) and 48.7 (MetaBEV) (Table I, baselines retrained without augmentation).
- compute: about 1.6 FPS fused on a V100, batch 1 (BEVFusion 0.7, MetaBEV 1.4).
- code: github.com/tudelft-iv/UniBEV (from search listing, not opened).
- fit_to_us: reuse_idea - effort S - skip-missing averaging as the safe fusion operator; map segmentation explicitly left as future work.
- compat_needs/conflicts: as RB07.
- tried_before: none.
- verified: checked

### CARD RB09 - SB-BEVFusion: identity fusion when one sensor is absent, single-branch training
- papers: SB-BEVFusion: Enhancing the Robustness against Sensor Malfunction and Corruptions (arXiv 2605.11799, May 2026) https://arxiv.org/abs/2605.11799 (opened alphaxiv overview https://www.alphaxiv.org/abs/2605.11799)
- depth: A
- stage: B, T
- idea: Equalise channel counts of camera and LiDAR BEV features, then use an operator (average, max, cross-attention, progressive modality decay) that reduces to identity when only one sensor is active; each training scene is shown as L+C, L-only and C-only in shuffled mini-batches through one shared encoder and head.
- formulation: not read beyond the overview.
- input/output: BEV features -> fused BEV feature.
- training: mixed-availability batches.
- differentiable: yes.
- sensors_data: camera + LiDAR, MultiCorrupt (RB19), nuScenes.
- results: camera only: BEVFusion NDS about 0.70 -> 0.10; LiDAR only SB-BEVFusion 0.6959 vs UniBEV 0.653; average fusion best on mean resistance, attention variants better on spatial misalignment. No latency.
- code: none found.
- fit_to_us: reuse_idea - effort S - the "all availability subsets in every batch" trick, which we can do with cached feature sets per failure type.
- compat_needs/conflicts: as RB07.
- tried_before: none.
- verified: partly

### CARD RB10 - RoCaRS: camera-radar BEV segmentation for sensor failure (link not verified)
- papers: RoCaRS: Robust Camera-Radar BEV Segmentation for Sensor Failure Scenarios (Park, Kim, Cho, Hwang, 2025) - link not verified (no arXiv or publisher page opened; abstract text seen only in search snippets that came from a lab database listing)
- depth: A (snippet level)
- stage: B, T
- idea: A camera-radar BEV segmentation model with a radar-aware backbone and "feature spreading", a Dynamic Input Dropout Strategy (DIDS) for training with missing inputs, and a Bidirectional Feature Refinement (BFR) module that fills the missing branch from the present one.
- formulation: not read.
- input/output: camera + radar -> BEV map and vehicle segmentation.
- results (snippet, unverified): with all cameras removed +23.2 mIoU (map) and +30.0 IoU (vehicle) over baseline; keeps 99 % of radar-only and 103 % of camera-only performance when all cameras or all radars are disabled, without retraining. nuScenes.
- compute/code: not stated / none found.
- fit_to_us: reuse_idea - effort unknown - the only camera-radar map-segmentation failure paper found, but I could not read it; worth locating the real paper before relying on any detail.
- compat_needs/conflicts: separable camera / radar branches.
- tried_before: none.
- verified: from memory (snippets only)

### CARD RB11 - RCBEVDet / RCBEVDet++: radar-camera BEV fusion with explicit view / radar dropout tests
- papers: RCBEVDet: Radar-camera Fusion in Bird's Eye View for 3D Object Detection (Lin et al., CVPR 2024) https://arxiv.org/abs/2403.16440 (opened HTML); RCBEVDet++ (arXiv 2409.04979, abstract page opened: no robustness claim there) https://arxiv.org/abs/2409.04979
- depth: M
- stage: B
- idea: RadarBEVNet encodes radar with a point branch + a distance-modulated transformer branch and scatters points into BEV with an RCS-dependent footprint; a cross-attention multi-layer fusion (deformable, both directions) corrects azimuth misalignment before a conv fusion.
- formulation: deformable cross-attention cost O(HWC^2K); dropout test drops camera views (0 / 1 / 3 / all) or radar input.
- input/output: radar points + camera BEV -> fused BEV for detection.
- training: standard detection training.
- differentiable: yes.
- sensors_data: camera + radar, nuScenes, VoD.
- results (Table 7, car mAP): RCBEVDet drops camera views 72.5 / 66.9 / 53.5 / 16.5 (0 / 1 / 3 / all); CRN 68.8 / 62.4 / 48.9 / 12.8; BEVFusion 63.9 / 58.5 / 45.7 / 14.3. Radar-drop rows RCBEVDet 71.6 / 66.1 / 62.1 against CRN 64.3 / 57.0 / 43.8 (the summary's quoted drop percentages were inconsistent with these cells; use the cells). Camera loss costs far more than radar loss.
- compute: 21.3 FPS (R50, 256 x 704) vs CRN 20.4; RTX 3090, batch 1, FP16; no ms.
- code: github.com/VDIGPKU/RCBEVDet (announced on the abstract page).
- fit_to_us: baseline - effort S - the best available numbers on how much a camera-radar network keeps when radar or cameras go; supports "radar is the fallback, cameras the single point of failure".
- compat_needs/conflicts: detection only; no map output.
- tried_before: none.
- verified: checked

### CARD RB12 - RobuRCDet: radar noise taxonomy + weather-adaptive radar-camera fusion
- papers: RobuRCDet: Enhancing Robustness of Radar-Camera Fusion in Bird's Eye View for 3D Object Detection (Yue et al., ICLR 2025) https://arxiv.org/abs/2502.13071 (abstract page opened); ICLR listing https://proceedings.iclr.cc/paper_files/paper/2025/hash/21dabaacda3edba8bb281da45d7cbc17-Abstract-Conference.html (seen in search)
- depth: A
- stage: B
- idea: Systematic noise analysis (five noise kinds: radar position, RCS, velocity and others not named on the page), a 3D Gaussian Expansion module that spreads each radar return with a deformable kernel / variance controlled by RCS and velocity priors, and a weather-adaptive fusion that weights radar vs camera by an estimated camera confidence map.
- formulation, results, compute: not read; abstract says "competitive in regular and noisy conditions".
- sensors_data: camera + radar, nuScenes.
- code: announced, not confirmed.
- fit_to_us: reuse_idea - effort M - radar position / RCS noise is a failure type we did not test (we only switch radar off); the corruption recipe could extend our failure set.
- tried_before: radar-off only.
- verified: partly

### CARD RB13 - BEVCar: camera-radar fusion for BEV map and vehicle segmentation, day / rain / night split
- papers: BEVCar: Camera-Radar Fusion for BEV Map and Object Segmentation (Schramm, Vödisch et al., IROS 2024) https://arxiv.org/abs/2403.11761 (opened HTML); project https://bevcar.cs.uni-freiburg.de/
- depth: M
- stage: B
- idea: Frozen DINOv2 ViT-B/14 + adapter on six cameras; radar point encoding guides deformable-attention lifting of image features to BEV; radar BEV queries cross-attend to the lifted image features; one head outputs vehicle and map channels.
- formulation: 200 x 200 grid over 100 m x 100 m (0.5 m), 5 radar sweeps; vehicle BCE, map alpha-balanced focal loss (alpha 0.25, gamma 3).
- input: six images 448 x 896 + radar points. output: BEV segmentation (map + vehicle).
- training: end-to-end except the frozen DINOv2.
- differentiable: yes.
- sensors_data: camera + radar, nuScenes (28,130 train / 6,019 val; val split day 4,449 / rain 968 / night 602).
- results: mIoU 70.9 (camera + radar) vs 65.0 camera-only; night vehicle / map IoU 57.4 / 42.4 (BEVCar) vs 42.1 / 39.2 (camera-only BEVCar) vs 52.4 / 37.8 (Simple-BEV++); day map 57.3 vs night 42.4. Radar helps vehicles at night (+15 IoU) but map classes only a little (+3).
- compute: not reported here; GaussianCaR's paper lists 245.6 ms.
- code: github.com/robot-learning-freiburg/bevcar (from search listing).
- fit_to_us: baseline - effort S - the number to remember: radar does not repair night for planar map classes (lane-like layers); this is exactly our failure.
- tried_before: night / blur failure fine-tuning.
- verified: checked

### CARD RB14 - GaussianCaR (our foundation): no failure evaluation exists
- papers: GaussianCaR: Gaussian Splatting for Efficient Camera-Radar Fusion (Montiel-Marin et al., ICRA 2026) https://arxiv.org/abs/2602.08784 (opened HTML https://arxiv.org/html/2602.08784v1); code https://www.github.com/santimontiel/gaussiancar; project https://www.santimontiel.eu/projects/gaussiancar
- depth: M
- stage: B
- idea: Camera pixels (depth distribution + offset) and radar points (Point Transformer v3) become 3D Gaussians with 128-d features, alpha-blended into BEV; a four-stage multi-scale fusion (CMX-like modules) and a DPT decoder produce segmentation.
- results: vehicle 57.3, drivable area 82.9, lane divider 50.1 IoU; GaussianLSS camera baseline 46.1; radar sweeps 1 / 4 / 7 give 54.9 / 56.2 / 57.3.
- compute: 75.6 ms vehicle / 81.1 ms map on RTX 4090 at 448 x 800 (Table IV); BEVCar 245.6 ms; params not reported.
- robustness: the paper has NO sensor-failure, night, rain or modality-removal experiment (only qualitative figures); so the robustness of our frozen foundation is unmeasured in the literature and is purely our own finding.
- fit_to_us: baseline - effort S - any "radar saves the day" or "camera needed" claim must be measured on this model; the Gaussian representation might make a per-Gaussian opacity gate possible, but we cannot touch the frozen weights.
- tried_before: our own failure tests.
- verified: checked

### CARD RB15 - M-BEV: masked view reconstruction for missing cameras
- papers: M-BEV: Masked BEV Perception for Robust Autonomous Driving (Chen et al., AAAI 2024; arXiv 2312.12144) https://arxiv.org/abs/2312.12144 (abstract page opened); code https://github.com/Sranc3/M-BEV
- depth: A
- stage: B, T
- idea: During training random camera views are masked at the feature level; a masked-view reconstruction module (global or local, learned mask tokens, 3D position embedding, cross-view context) rebuilds them, with the original features as self-supervision.
- results: back camera missing: +10.3 % mAP on PETRv2 (abstract); detection only.
- compute: not stated.
- fit_to_us: reuse_idea - effort M - same family as SafeMap G-PVR; needs per-view features.
- tried_before: none.
- verified: partly

### CARD RB16 - ReliFusion: contrastive cross-modal reliability scores for confidence-weighted fusion
- papers: Reliability-Driven LiDAR-Camera Fusion for Robust 3D Object Detection (Sadeghian et al., arXiv 2502.01856, Feb 2025) https://arxiv.org/abs/2502.01856 (opened HTML v1)
- depth: M
- stage: B
- idea: Two MLPs map LiDAR and camera BEV features into a shared 128-d space; a contrastive loss (tau 0.07) separates matched from corrupted pairs; a sigmoid head gives one confidence per modality; confidence-weighted mutual cross-attention multiplies each modality's attended features by its confidence.
- training: contrastive + confidence regression pretraining (target not defined in what I read), confidence loss weight 0.05 in fine-tuning.
- results: clean 70.6 mAP / 73.2 NDS; LiDAR FOV +-pi/2: 52.4 / 59.6; zero FOV 24.6 / 28.7 (tables inconsistent between Table 2 and 4). No latency.
- code: none found.
- fit_to_us: reuse_idea - effort M - reliability from contrastive agreement between two streams; same limit as RB05.
- tried_before: none.
- verified: checked

### CARD RB17 - RoboBEV / nuScenes-C: corruption suite incl. Camera Crash, Frame Lost, Dark, Fog, Motion Blur
- papers: RoboBEV: Towards Robust Bird's Eye View Perception under Corruptions (Xie, Kong et al., 2023) https://arxiv.org/abs/2304.06719 (seen in search listing; page not opened); Benchmarking and Improving Bird's Eye View Perception Robustness in Autonomous Driving (expanded version, arXiv 2405.17426, May 2024) https://arxiv.org/abs/2405.17426 (opened HTML v2); code github.com/Daniel-xsy/RoBoBEV (from search listing)
- depth: M (for 2405.17426)
- stage: E, D
- idea: Eight corruptions at three severities (Bright, Dark, Fog, Snow, Motion Blur, Color Quant, Camera Crash, Frame Lost) applied to nuScenes val; 33 BEV models; metrics mCE / mRR; extends to multimodal sensor failure and offers a two-stage CLIP-based fix.
- results: map segmentation (CVT, IoU, Table IV): clean 0.348, Camera Crash 0.200, Frame Lost 0.170, Dark 0.200, Fog 0.247, Snow 0.177. Multimodal BEVFusion (Table VI): camera failure mAP 0.6852 -> 0.6247; LiDAR failure NDS 0.7138 -> 0.3340 (mAP drop about 89 %). Corruption-augmented CLIP model mRR 84.32 vs 58.54 for BEVDet baseline (Table X); frozen CLIP alone transfers poorly; end-to-end fine-tuning adds little.
- compute: not reported.
- fit_to_us: baseline - effort S - the corruption definitions (Dark, Fog, Motion Blur, Camera Crash, Frame Lost at 3 severities) can be reproduced in our failure synthesis; our failure suite should report RR = corrupted / clean per severity.
- tried_before: our own synthetic night / blur / radar-off.
- verified: partly (first paper by listing only)

### CARD RB18 - MapBench (and MSC-Bench): robustness benchmark for online HD map construction
- papers: Is Your HD Map Constructor Reliable under Sensor Corruptions? (Hao et al., NeurIPS 2024 D&B) https://arxiv.org/abs/2406.12214 (opened HTML v3); MSC-Bench: Benchmarking and Analyzing Multi-Sensor Corruption for Driving Perception (arXiv 2501.01037, Jan 2025) https://arxiv.org/abs/2501.01037 (abstract page opened)
- depth: M (MapBench) / A (MSC-Bench)
- stage: E, D
- idea: 29 corruptions (8 camera, 8 LiDAR, 13 multi-sensor combinations) at 3 severities on nuScenes val, 31 map constructors, metrics mAP, mCE (vs MapTR = 100), mRR (corrupted / clean).
- results: MapTR clean 50.3 mAP -> Camera Crash 15.0, Frame Lost 14.2; HIMap 65.5 -> 19.4 / 19.0. Snow worst (MapTR mRR 7.7 %). Fusion not clearly more robust (HIMap fused mRR 41.7 vs camera-only 56.6). What helps (all camera-only): Swin backbone (mCE -22.2 on PivotNet), temporal fusion (StreamMapNet mRR 47.1 -> 55.5, mCE 100 -> 85.9, Table 3), PhotoMetric augmentation (mRR 41.1 -> 49.3, mCE 100 -> 84.5), longer training; rotate / flip augmentation hurt. Combined camera + LiDAR failures worse than either alone. MSC-Bench: 16 combinations of camera / LiDAR corruption, 6 detectors and 4 map models, no radar (no numbers on the page).
- compute: not applicable.
- fit_to_us: baseline - effort S - the metric definitions (mRR, mCE) and the finding that temporal fusion and photometric augmentation are the two cheap robustness levers for map output.
- tried_before: temporal hold, failure fine-tuning (consistent with the two levers).
- verified: checked

### CARD RB19 - MultiCorrupt: ten corruptions incl. spatial misalignment (miscalibration) and missing camera
- papers: MultiCorrupt: A Multi-Modal Robustness Dataset and Benchmark of LiDAR-Camera Fusion for 3D Object Detection (Beemelmanns, Zhang, Eckstein, IEEE IV 2024) https://arxiv.org/abs/2402.11677 (opened HTML); code github.com/ika-rwth-aachen/MultiCorrupt (listing)
- depth: M
- stage: E, D
- idea: Darkness, brightness, points reducing, temporal misalignment, spatial misalignment (extrinsic rotation / translation offsets), motion blur, missing camera (random frame drops), beams reducing, fog, snow; resistance ability RA = corrupted / clean NDS, mRA over severities.
- results: mRA CMT 0.865, SparseFusion 0.834, BEVFusion 0.830, TransFusion 0.824, DeepInteraction 0.795; lowest single cells DeepInteraction fog 0.583, TransFusion spatial misalignment 0.685. Findings: masked-modal training (CMT) and independent modality handling help; LiDAR-only query initialisation and early deep coupling hurt.
- fit_to_us: baseline - effort S - gives a ready recipe for spatial misalignment and temporal misalignment synthesis; detection only, LiDAR-camera.
- tried_before: none (we lack an extrinsics test).
- verified: checked

### CARD RB20 - RoboDrive challenge and the Samsung-RAL map-segmentation entry: what practitioners did
- papers: The RoboDrive Challenge: Drive Anytime Anywhere in Any Condition (arXiv 2405.08816, 2024) https://arxiv.org/abs/2405.08816 (abstract page opened); Team Samsung-RAL: Technical Report for 2024 RoboDrive Challenge - Robust Map Segmentation Track (arXiv 2405.10567) https://arxiv.org/abs/2405.10567 (opened HTML v2)
- depth: M (technical report) / A (challenge)
- stage: T, E
- idea: Challenge: 140 teams, five tracks; top solutions used heavy augmentation, multi-sensor fusion, self-supervised error correction. Samsung-RAL: BEVerse with Swin backbones, temporal module, image augmentations (GridMask, hue, saturation, channel swap, value).
- results: Table 2 temporal: 21.68 -> 24.08 mIoU (+2.4); backbone Tiny / Small / Base 16.31 / 17.33 / 24.08; BEVerse-Tiny with augmentation 29.76 (phase 2, ranked third; tables mix phase 1 and 2).
- compute: 4 x A6000, 20 epochs.
- fit_to_us: baseline - effort S - ranking of levers: backbone strength (+7.8) > temporal (+2.4) > augmentation.
- tried_before: temporal hold, failure fine-tuning.
- verified: checked

### CARD RB21 - Robust BEV segmentation by adapting a frozen DINOv2 with LoRA
- papers: Robust BEV Segmentation via DINOv2 Adaptation (title as listed in search; arXiv 2409.10228, 2024) https://arxiv.org/abs/2409.10228 (opened HTML)
- depth: M
- stage: B, T
- idea: Keep a DINOv2 image encoder frozen and add LoRA matrices to the attention projections; a SimpleBEV decoder (about 5 M) lifts and decodes to BEV occupancy; evaluated on nuScenes-C.
- training: BCE on BEV occupancy; trainable parameters ViT-B LoRA 1 M, ViT-L LoRA 3 M, ResNet-101 37 M, full ViT-B 86 M.
- results: ViT-L beats SimpleBEV by at least 20 % relative on most corruptions; darkness: SimpleBEV falls below 30 % of its clean value, DINOv2 variants hold much better; fog loses under 20 %; camera crash and frame loss: no better than SimpleBEV. Exact numbers only in a figure.
- compute: latency not reported.
- fit_to_us: reuse_idea - effort L (needs trainable image front end) - evidence that a stronger frozen image foundation helps night and fog but not lost cameras, matching our own fine-tuning finding.
- tried_before: GaussianCaR vs CRN comparison (E27) is a different axis.
- verified: checked

### CARD RB22 - LearnableBN: test-time adaptation of BN for BEVFormer under corruptions
- papers: Improving Batch Normalization with Test-Time Adaptation for Robust Object Detection in Self-Driving (Liao, Qi, Liu, Ma, arXiv 2411.18860; AAAI 2026 listing) https://arxiv.org/abs/2411.18860 (opened HTML v1); AAAI page https://ojs.aaai.org/index.php/AAAI/article/view/37626
- depth: M
- stage: T
- idea: Each BN layer gets one learnable mixing coefficient between source and test statistics; only these coefficients are optimised online with entropy minimisation plus a "generalised-search" term; a two-stage update (lr 2e-8 then 2e-7) keeps only samples whose KL between stages is in the lowest 10 % of history.
- training: no labels; backbone frozen; batch size 1 for nuScenes-C.
- differentiable: yes (update step at test time); output not affected otherwise.
- sensors_data: camera, BEVFormer R101, nuScenes-C six corruption types.
- results (average NDS-type score, Table 1): Low Light baseline 0.3212 -> LearnableBN 0.3469; Motion Blur 0.3330 -> 0.3787; Fog 0.4796 -> 0.4808. In the same table TENT 0.2070 / 0.2823 / 0.3323, AdaBN 0.0772 / 0.1279 / 0.1333, ARM(BN) 0.0961 / 0.1484 / 0.1431 are far BELOW the baseline at batch size 1. Headline gain: about 8 % (arXiv v1) or 10 % (AAAI version).
- compute: no runtime reported; one extra forward / backward over BN coefficients.
- code: not released at the time of v1 ("soon").
- fit_to_us: reuse_idea - effort S - warns that plain AdaBN / TENT can collapse in a per-frame BEV stream; a coefficient-mixing BN (source vs batch statistics) is the safe variant; our +0.04 with adaptive BN on the lane head may be fragile.
- tried_before: adaptive batch-norm (+0.04).
- verified: checked

### CARD RB23 - TTA primitives: AdaBN, TENT, SAR
- papers: Revisiting Batch Normalization For Practical Domain Adaptation (AdaBN, 2016; authors from memory Li et al.) https://arxiv.org/abs/1603.04779; Tent: Fully Test-time Adaptation by Entropy Minimization (Wang et al., ICLR 2021, both from memory) https://arxiv.org/abs/2006.10726; Towards Stable Test-Time Adaptation in Dynamic Wild World (SAR, Niu et al., ICLR 2023) https://arxiv.org/abs/2302.12400 (all abstract pages opened; pre-2023 papers listed as background)
- depth: A
- stage: T
- idea: AdaBN replaces BN statistics by target statistics (parameter-free). TENT additionally minimises prediction entropy over BN scale / shift. SAR states TTA fails under mixed shifts, small batches and label imbalance, blames BN, and fixes it by dropping high-gradient samples and a sharpness-aware step.
- formulation: not read. results: none seen beyond RB22's batch-1 BEV table.
- compute: AdaBN no extra pass; TENT one backward over affine parameters per batch; SAR more.
- code: TENT / SAR links not seen on the pages.
- fit_to_us: reuse_idea - effort S - ordering by safety: AdaBN with mixing < SAR-style filtered entropy; entropy of a dense lane heat map is a possible loss but untested.
- tried_before: adaptive BN (+0.04).
- verified: checked (abstracts)

### CARD RB24 - Continual / memory-based TTA: CoTTA, RoTTA, distribution-aware tuning
- papers: Continual Test-Time Domain Adaptation (Wang et al., CVPR 2022) https://arxiv.org/abs/2203.13591; Robust Test-Time Adaptation in Dynamic Scenarios (RoTTA, Yuan et al., CVPR 2023) https://arxiv.org/abs/2303.13899; Distribution-Aware Continual Test-Time Adaptation for Semantic Segmentation (arXiv 2309.13604) https://arxiv.org/abs/2309.13604 (all abstract pages opened)
- depth: A
- stage: T
- idea: CoTTA: weight-averaged teacher + augmentation-averaged predictions + stochastic restore of a few weights to the source. RoTTA: robust BN, category-balanced memory bank with recency / uncertainty weighting, time-aware reweighting, for correlated streams such as driving. DAT: update only two small parameter groups (domain-specific for big shifts, task-relevant for small shifts) for driving segmentation.
- results: none seen (abstracts only; CoTTA's segmentation number appeared only in a third-party summary and is not used).
- compute: CoTTA multiplies forward passes (from memory); DAT is built for low cost.
- code: https://qin.ee/cotta ; https://github.com/BIT-DA/RoTTA.
- fit_to_us: reuse_idea - effort M - none was evaluated on BEV or lane outputs; the RoTTA memory bank is the closest to a streaming driving case.
- tried_before: adaptive BN only.
- verified: checked (abstracts)

### CARD RB25 - Analytic continual TTA for multi-modality corruption (MDAA)
- papers: Analytic Continual Test-Time Adaptation for Multi-Modality Corruption (arXiv 2410.22373, v2 July 2025) https://arxiv.org/abs/2410.22373 (abstract page opened)
- depth: A
- stage: T
- idea: Closed-form (analytic) classifiers avoid catastrophic forgetting; a dynamic late-fusion mechanism picks reliable modalities at test time, in continual multi-modal corruption including sensor malfunction.
- results: none seen; tasks not named on the page.
- fit_to_us: reuse_idea - effort M - ideas of closed-form updates and late fusion by reliability, but classification-style heads only.
- verified: partly

### CARD RB26 - NCGR: noise-conditional gated rectification for camera extrinsic perturbations
- papers: NCGR: Noise-Conditional Gated Rectification for Camera Extrinsic Perturbations in BEV 3D Object Detection (Pan et al., arXiv 2608.03895, 4 Aug 2026) https://arxiv.org/abs/2608.03895 (abstract page opened)
- depth: A
- stage: B, T
- idea: For spatial-cross-attention BEV detectors, predict a 2D offset per query-camera pair, scaled by a per-camera gate, to correct wrong projections instead of estimating a 6-DoF extrinsic; clean-teacher / perturbed-student training with a BEV-consistency loss; no perturbation metadata at inference.
- results: five-camera dynamic stress test 39.69 % NDS vs BEVFormer 28.00 and CAPE 33.23; clean performance comparable to BEVFormer.
- compute/code: not reported / none found.
- fit_to_us: reuse_idea - effort L - the repair sits inside the image-to-BEV transform, which we cannot change; the teacher-student BEV-consistency loss is portable to RB01-type modules.
- tried_before: none (failure fine-tuning did not help extrinsics).
- verified: checked (abstract)

### CARD RB27 - Calibration-free BEV (CFT) and extrinsic-sensitivity evidence
- papers: Multi-Camera Calibration Free BEV Representation for 3D Object Detection (arXiv 2210.17252, Oct 2022; outside the date window, included because it is the standard calibration-free reference) https://arxiv.org/abs/2210.17252 (abstract page opened); noisy-extrinsic numbers for BEVFormer / PETRv2 come from a search snippet that cited arXiv 2203.17270 / 2206.01256 (not opened)
- depth: A
- stage: B
- idea: Learn the image-to-BEV mapping with position-aware enhancement and view-aware attention without any intrinsics / extrinsics, so noisy calibration cannot hurt.
- results: 49.7 NDS on nuScenes leaderboard; no noise experiment on the page I opened (snippet only: BEVFormer loses 0.6 - 6.8 % and PETR 4.7 - 7.6 % NDS under small extrinsic noise while CFT stays stable - not verified).
- fit_to_us: not_suitable - effort L - the projection lives inside the frozen foundation.
- verified: partly

### CARD RB28 - MapLightning: 1D map tokens without camera projection
- papers: MapLightning: Online Vectorized HD Map Construction with 1D Map Tokens (Zheng, Ghosh, Ramanagopal, Narasimhan, arXiv 2610.01905, 1 Oct 2026) https://arxiv.org/abs/2610.01905 (abstract page opened)
- depth: A
- stage: B
- idea: Learnable 1D map tokens plus image tokens in one self-attention transformer; image tokens are discarded and map tokens decoded. Claim: because camera projection parameters are not used, it is robust to extrinsic perturbation (no experiment on the page).
- results: lightweight variant +10.1 mAP vs MapTRv2 on nuScenes, +16.2 on Argoverse 2, 1.73x faster (40+ FPS), 53 % less memory, 16.7x fewer tokens (abstract).
- code: announced, no link.
- fit_to_us: reuse_idea - effort L - same calibration-free argument as CFT; vectorized map decoders were already compared (E31, E32).
- tried_before: sparse DETR-style decoder dropped (E31, E32).
- verified: checked (abstract)

### CARD RB29 - Online map uncertainty: Laplace vertex output and class probabilities
- papers: Producing and Leveraging Online Map Uncertainty in Trajectory Prediction (Gu et al., 2024; venue CVPR 2024 from memory) https://arxiv.org/abs/2403.16439 (opened HTML); code github.com/alfredgu001324/MapUncertaintyPrediction
- depth: M
- stage: L, S10
- idea: Each vertex coordinate is predicted as a Laplace distribution (mean + scale); classification logits are the class uncertainty; training swaps L1 for the Laplace NLL; downstream trajectory predictors consume the uncertainty.
- formulation: loss sum[log(2b) + |v - mu| / b]; regression weight 0.03 (2 for StreamMapNet), lr 1.5e-4 (1.25e-4), grad-clip 3.
- results: HiVT miss rate MapTR 0.0981 -> 0.0834 (up to 15 % lower MR), minFDE up to 6 % better, DenseTNT converges in 44 vs 88 epochs; one configuration worse (MR +5 %); mapper mAP within about 2 %; no quantitative calibration metric; rain / night only qualitative.
- compute: no overhead reported (single RTX 4090 training).
- fit_to_us: reuse_idea - effort S - per-vertex scale on our vertex output feeds the planner consumer E117 and doubles as failure signal (scale grows under failure).
- tried_before: validity / keep probability only; no positional uncertainty.
- verified: checked

### CARD RB30 - BEV segmentation uncertainty benchmark and UFCE evidential loss
- papers: Predictive Uncertainty Quantification for Bird's Eye View Segmentation: A Benchmark and Novel Loss Function (Yu et al., ICLR 2025; arXiv 2405.20986) https://arxiv.org/abs/2405.20986 (opened HTML v2); code github.com/bluffish/ubev
- depth: M
- stage: L, E
- idea: Compares softmax entropy, energy score, deep ensembles, MC dropout and evidential networks (ENN) for calibration and misclassification / OOD detection in BEV segmentation, and proposes the Uncertainty-Focal-Cross-Entropy (focal loss expected under a Dirichlet, closed form) plus epistemic uncertainty scaling and evidence regularisation on pseudo-OOD samples.
- results: best OOD detection (average gain over runner-up about 2.7 AUROC, 6.5 AUPR, 4.3 FPR95, Table 2 caption); without pseudo-OOD all methods poor at epistemic uncertainty; includes nuScenes-C.
- compute (LSS, Table 3): inference energy 54 ms, ensemble 147 ms, dropout 457 ms, ENN and proposed 51 - 56 ms; training 10 h (30 h ensemble).
- fit_to_us: reuse_idea - effort S - the evidential head is free at inference and gives a per-cell failure signal for the heat map; ensembles and dropout are too slow.
- tried_before: label-free heat monitor (no calibrated uncertainty).
- verified: checked

### CARD RB31 - Camera health monitor: single-image degradation index with early warning
- papers: Safety-Critical Camera Reliability Monitoring for ADAS via Degradation-Aware Uncertainty Pattern Analysis (arXiv 2605.05439, May 2026) https://arxiv.org/abs/2605.05439 (opened HTML)
- depth: M
- stage: E, X
- idea: An EfficientNet-B2 with light heads predicts degradation presence, per-mode severity (12 modes), a spatial degradation map and a multiplicative health index (GSHI); regimes Healthy / Degraded / Critical at 0.9 / 0.6.
- results: health MAE 0.064; lead time 0.47 severity units before a YOLOv8n failure; GSHI correlates 0.95 with detector mAP on failure subset vs 0.90 for YOLO confidence and 0.60 for clean-feature OOD; DAWN zero-shot balanced accuracy 84.2 % (fog / rain / snow). Synthetic KITTI degradations only.
- compute: 7.91 M params, 0.72 GFLOPs, 2.27 ms at 224 x 224 on an RTX 5090.
- fit_to_us: reuse_idea - effort M - needs RGB, not BEV features; but the evaluation idea (correlation with downstream failure, lead time) is directly usable for our heat monitor.
- tried_before: label-free heat monitor.
- verified: checked

### CARD RB32 - Night images: LightDiff and DualCross (image-level fixes)
- papers: Light the Night: A Multi-Condition Diffusion Framework for Unpaired Low-Light Enhancement in Autonomous Driving (Li et al., CVPR 2024) https://arxiv.org/abs/2404.04804 (abstract page opened); DualCross: Cross-Modality Cross-Domain Adaptation for Monocular BEV Perception (authors not read, IROS 2023) https://arxiv.org/abs/2305.03724 (abstract page opened)
- depth: A
- stage: B, T
- idea: LightDiff enhances dark camera images with a diffusion model conditioned on a degraded image, LiDAR-derived depth and text, trained on synthetic day-night pairs and aligned to detectors with RL rewards. DualCross transfers LiDAR knowledge into a camera-only monocular BEV model for cross-domain use.
- results: abstract level, no numbers; LightDiff improves several 3D detectors at night on nuScenes.
- compute: not stated; a diffusion pass per image is expected to be expensive (my inference).
- fit_to_us: not_suitable - effort L - need raw images in the loop and add latency; the foundation is frozen and cached.
- verified: checked (abstracts)

### CARD RB33 - LIAR: illumination-guided sampling and projection for nighttime occupancy
- papers: See through the Dark: Learning Illumination-affined Representations for Nighttime Occupancy Prediction (arXiv 2505.20641) https://arxiv.org/abs/2505.20641 (opened HTML v3); code https://github.com/yanzq95/LIAR
- depth: M
- stage: B
- idea: A frozen Retinex-style illumination estimator (pretrained on night images, since joint training collapses on day-dominated data) gates enhancement, scales deformable offsets in 2D sampling and weights BEV residuals along vertical columns.
- results: Occ3D-nuScenes night mIoU 19.27 (1 frame) / 22.09 (2 frames) vs 20.01 second best; day+night training 23.67 / 27.33; nuScenes-C darkening easy / moderate / hard 30.66 / 25.67 / 17.31. Generic low-light preprocessors on other models gave only +0.08 to +0.37 mIoU.
- compute: 4.7 FPS, 64.39 M params, 5.05 GB (RTX 4090).
- fit_to_us: not_suitable - effort L - image-side; the lesson is that off-the-shelf enhancement barely helps and illumination priors must be integrated into sampling.
- verified: checked

### CARD RB34 - Adversarial-style degradation of camera-radar fusion (blur lens vs radar ghosts)
- papers: Adversarial Robustness of Camera-Radar Sensor Fusion for Autonomous Driving (Zhou et al., arXiv 2610.16012, 12 Oct 2026) https://arxiv.org/abs/2610.16012 (opened HTML, first 100 k of 117 k characters)
- depth: A (experiments table-level)
- stage: E
- idea: Eight camera perturbations (three blurs, four motion blurs, lens attack) and three radar perturbations (depth error, ghost targets, target disappearance) on RADIANT and CRN (detection) and radar-camera depth estimators.
- results: vertical motion blur kernel 20: CRN mAP / NDS 0.473 / 0.562 -> 0.4066 / 0.519; RADIANT 0.363 / 0.340 -> 0.290 / 0.301 (about 20 - 25 % drops); radar depth error 200 % barely changes RADIANT. Camera damage is much larger than radar damage; suggests reliability estimation, cross-modal consistency and failover but does not evaluate them.
- fit_to_us: baseline - effort S - supports prioritising blur / night over radar corruption in our failure mix.
- verified: partly

### CARD RB35 - Temporal redundancy: StreamMapNet and a stability benchmark
- papers: StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction (Yuan et al., WACV 2024) https://arxiv.org/abs/2308.12570 (abstract page opened); Stability Under Scrutiny: Benchmarking Representation Paradigms for Online HD Mapping (arXiv 2510.10660, Oct 2025) https://arxiv.org/abs/2510.10660 (abstract page opened; PDF too large to read)
- depth: A
- stage: S8, E
- idea: StreamMapNet carries a memory feature through the sequence, 14.2 FPS; the benchmark measures temporal stability (presence, localisation, shape stability, mean Average Stability) on 42 models and states that accuracy and stability are largely independent.
- results: temporal gain under corruption is quantified in MapBench (RB18): mRR 47.1 -> 55.5; MapBench and the Samsung entry (RB20) give +2.4 mIoU from a temporal module.
- fit_to_us: reuse_idea - effort S - evaluation: add a stability score to our temporal-consistency metric; supports learned recursion over the spatial hold (cf. RB02).
- tried_before: spatial hold; input memory in system B; temporal feature fusion (E41 +0.008 F1).
- verified: partly

## 3. Cluster compatibility notes

(a) Interface contracts. The literature repairs failures in four places: (1) image / view features before lifting (SafeMap G-PVR, M-BEV, NCGR, LIAR): not usable, we have only the cached fused raster; (2) the sensor-specific BEV maps before fusion (MetaBEV, UniBEV, Grace-BEV, SB-BEVFusion, ReliFusion, RAF): not usable as published, our camera and radar are already fused; (3) the fused BEV raster before the head (PFS, ModalPatch, SafeMap D-BEVC, M2-style repair): this is our interface; (4) head inputs / statistics (BN adaptation, TTA): usable. Our 39-channel 200 x 200 raster in, same-shaped raster out is exactly the PFS contract; ModalPatch adds a memory of ego-warped past rasters; both would be trained on cached clean / failure pairs.
(b) Known adapters / conflicts. Reliability or trust scalars blend two feature sets: in our case blend repaired and raw (identity-initialised) features. Modality dropout requires per-failure caches of the frozen foundation outputs. Entropy-based TTA conflicts with batch-size-1 BN (RB22). Hungarian-matching losses of the map papers are not ours; the feature-level terms (MSE distillation, Gaussian NLL, contrastive reliability) are matching-free.
(c) Combinations shown in papers: temporal memory + uncertainty variance + frozen host (ModalPatch); frozen backbone + dropout-trained gate (Grace-BEV); clean teacher + perturbed student (SafeMap, NCGR); augmentation + fusion + modality dropout, effects add only jointly (RoboMap).

## 4. Reuse versus re-implement
- Re-implement (small, method sections suffice): PFS-style identity-initialised repair block on the cached raster (RB01); ModalPatch history predictor with variance (RB02); clean-teacher MSE feature consistency (RB03/RB26); Laplace-NLL vertex scale (RB29); evidential Dirichlet head with UFCE (RB30); LearnableBN-style mixing coefficients (RB22).
- Reuse as protocol: corruption definitions and mRR / mCE / RA metrics (RB17-RB19).
- Cannot reuse: anything needing image features, sensor-separate BEV maps, or changes to the foundation (SafeMap G-PVR, M-BEV, NCGR, CFT, LIAR, LightDiff, RoCaRS).

## 5. Open questions and gaps
- No paper evaluates a frozen camera-radar foundation (GaussianCaR) under failure; RB14 shows none exists. Our measurement is new.
- No lane / centre-line output under failure was evaluated besides vectorized map benchmarks (MapBench, MSC-Bench, SafeMap, RoboMap); lane graph quality (junctions, connectivity) under failure is unstudied.
- Lost cameras: every strong fix (SafeMap +11 to +18 mAP, M-BEV, MetaBEV) works inside image / modality space; PFS gets only +1.2 mAP for 6-camera dropout at BEV level. Whether a BEV-level repair can do better with temporal memory is open.
- Extrinsic errors: only fixed inside the projection (NCGR) or by dropping projection (CFT, MapLightning); no downstream-only fix known.
- TTA on dense BEV outputs at batch size 1 not studied; RB22 suggests plain AdaBN / TENT can collapse, which conflicts with our +0.04.
- RoCaRS could not be read; its claim is the closest to our task and needs verification.
- Latency of most fixes is given as FPS only and on different GPUs; ms numbers: PFS +5.7 ms (BEVFusion), Grace-BEV +0.05 ms, ModalPatch about -8 % throughput, SafeMap about 0 %.

## 6. WHAT TO TAKE (concrete building blocks)
1. Identity-initialised repair block on the cached 39-channel raster (PFS, RB01), trained on clean / failure pairs with head loss + feature MSE to the clean feature (SafeMap D-BEVC, RB03) - medium - night, blur, partly radar-off; limited for lost cameras (+1.2 mAP in the paper).
2. Learned history predictor with per-cell variance (ModalPatch, RB02) replacing the training-free spatial hold; memory of 6 ego-warped rasters - medium - lost front cameras, radar-off, blur.
3. Reliability map gating repaired vs raw features, supervised by agreement between the heat monitor and the clean-teacher feature (RAF/ReliFusion idea, RB05, RB16) - medium - all types, also turns the heat monitor into a trained gate.
4. Train the heads with every failure combination in every batch (SB-BEVFusion/UniBEV/RoboMap recipe, probabilities 0.5 clean / 0.25 / 0.25, RB06-RB09) using per-failure feature caches - cheap - camera-off, radar-off, night.
5. Photometric / GridMask-type augmentation at the image level before caching foundation features (largest single lever in RB04, RB18, RB20) - cheap - night, blur, fog.
6. Safe TTA: BN mixing coefficients only (LearnableBN, RB22) or SAR-style sample filtering; do not use plain TENT / AdaBN at batch size 1 without a check (RB22) - cheap, near 0 ms - night, blur.
7. Per-vertex Laplace scale in the vertex output (RB29) feeding the planner consumer - cheap - all types, failure detection.
8. Evidential (Dirichlet) head with UFCE on the dense heat map for a per-cell uncertainty at no inference cost (RB30); avoid ensembles (147 ms) and MC dropout (457 ms) - cheap - failure detection, calibration.
9. Evaluate with RR = corrupted / clean over three severities and mCE-style normalisation, plus a stability score (RB17-RB19, RB35) and monitor lead-time against downstream F1 (RB31) - cheap - all.
10. Not available downstream: extrinsic errors need a change inside the foundation (NCGR, CFT, MapLightning); keep as an explicit limit - expensive - extrinsics.
11. Do not spend effort on image enhancement (LightDiff, LIAR): extra latency and generic preprocessors gave +0.08 to +0.37 mIoU (RB33) - expensive - night.
12. Locate and read the real RoCaRS paper before deciding on any bidirectional camera-radar refinement idea (RB10) - cheap - lost cameras, radar-off.

## 7. Search log
- RoboBEV benchmarking BEV perception robustness corruptions
- MapBench robustness online HD map corruption benchmark sensor failure
- radar-camera fusion robust to camera failure BEV missing modality
- RoCaRS camera-radar BEV segmentation sensor failure (two variants)
- BEVCar camera-radar fusion BEV map and object segmentation
- test-time adaptation BEV perception camera 3D detection sensor corruption (three variants)
- extrinsic calibration perturbation robustness BEV online HD map (two variants)
- SafeMap robust HD map incomplete observations
- Improving Batch Normalization with Test-Time Adaptation BEVFormer NuScenes-C
- continual test-time adaptation autonomous driving BEV memory
- online HD map uncertainty confidence calibration failure detection
- night-time BEV perception nuScenes domain adaptation
- robust BEV map segmentation missing camera temporal fusion
- RobuRCDet; RetBEV Retinex; test-time adaptation online HD map; reliability-aware gating radar camera
- GaussianCaR; UniBEV; MultiCorrupt; LightDiff; uncertainty BEV segmentation evidential
- lane topology OpenLane-V2 robustness; missing modality frozen foundation adapters
- robust HD map construction 2026; SAR; ReliFusion; RoboDrive challenge
- perception quality self-assessment monitor; CoTTA; dropped frame BEV feature compensation
- radar camera corruption benchmark; BEV feature restoration masked view; StreamMapNet
- Fetches: arXiv HTML / abs pages for every id cited above (see links).
