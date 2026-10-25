# Cluster A (loop 17): streaming / temporal memory for online vector-map, lane-graph and lane-topology perception, 2024 - Oct 2026

Prefix `A`. Date of research: 2026-10-24. Reading only (WebSearch / WebFetch); nothing downloaded or run by me.

## 1. Coverage statement

- **Searched:** streaming and memory models for online vectorised HD maps (query propagation, BEV recurrence, raster/global priors, previous-prediction priors), temporal lane-topology models on OpenLane-V2, state-space / Mamba memories, robustness benchmarks and sensor-failure fusion (camera, camera-radar, camera-LiDAR as transferable ideas), gating / reliability weighting, ego-motion handling, temporal-consistency losses, training-only tricks.
- **How "depth M" was reached (important caveat):** WebFetch does not return raw pages, it returns a small-model digest of the page for my question. So `M` below means: the method / experiment sections of the arXiv HTML page were queried for equations, hyper-parameters, losses, training and numbers, and the digest named them. I did not read the raw equations myself. Where the digest said "not stated" I wrote "not stated"; several digests also flagged inconsistencies inside the papers (noted in the cards). Numbers are as the digest reported them from the paper tables.
- **Depth A only:** TopoFG, UniTopo (PDF text not extractable, only title/authors/code link seen), LaneSegNet, ReSMap, ReliFusion, RoCaRS (no arXiv id found). **Not found / not opened:** UniTopo method section; any official code link for MapUnveiler (project page has none), MambaMap ("forthcoming"), SQD-MapNet ("available soon" at v1); I did not check code pages of most others (EV cluster does licences).
- **Key negative finding:** I found **no paper** that (a) evaluates a streaming memory for a **lane graph with explicit edges** (forks / merges) under **modality failure**, or (b) combines camera-radar BEV **rasters** with a memory under failure. Everything is camera-based query/feature memory evaluated on clean data; the only robustness evidence for temporal fusion is one ablation in MapBench (card A03). Junction / split / merge handling is not solved by any memory paper (MapTracker states this as a limitation).
- **2026 items seen (newest first):** UniTopo 2605.08911 (May), Lane Field 2610.25284 (Oct), SIGMA-Lane 2608.16338 (Aug, ECCV 2026), MapTCL 2608.05209 (Aug, IROS 2026), D2HDMap 2606.20725 (Jun), PredMapNet 2602.16669 (Feb), Post Fusion Stabilizer 2603.05623 (Mar), ReSMap (NeurIPS 2026 poster, no arXiv id found).

## 2. Cards (ordered by relevance to our design)

### CARD A01 - FASTopoWM: fast-slow pipelines with a latent world model for lane-segment topology
- papers: FASTopoWM: Fast-Slow Lane Segment Topology Reasoning with Latent World Models (authors not captured, arXiv 2025) https://arxiv.org/abs/2507.23325
- depth: M
- stage: S8, T
- idea: Two decoders share layers and heads: a "slow" branch that replaces the lowest-ranked initial queries by propagated stream queries and fuses the previous BEV by GRU, and a "fast" single-frame branch. Both are trained in parallel with the same losses; at inference the slow output is used when the pose is reliable, otherwise the fast output. Two small "world models" (query and BEV) predict the current frame's stream queries / BEV from the previous frame, conditioned on an action latent made from the relative ego pose, with an MSE / query loss against the actual current frame.
- formulation: slow loss = lane-segment loss + 0.3 x latent loss; fast loss = lane-segment loss. World-model depth 2 layers chosen (2/4/6 layers give 37.1 - 37.8 mAP). Action conditioning: none 36.5, predicted trajectory 31.2, relative ego pose 37.4 mAP. Architecture of the latent model: linear 35.2, MLP 36.2, transformer 37.4. Criterion for "pose reliable" is not stated.
- input: BEV features (from images), previous stream queries + BEV, relative ego pose.
- output: lane segments (centreline + boundaries) + topology (adjacency), Hungarian-matched DETR set.
- training: Hungarian matching; 24 epochs, first 12 single-frame; parallel fast-slow training 37.4 vs random alternation 35.3 temporal mAP.
- differentiable: partial - Hungarian matching in the loss; gradient flow through the propagated memory not stated.
- sensors_data: camera, OpenLane-V2 subset A / B.
- results: OpenLane-V2 subset A: temporal 37.4 mAP, TOP_lsls 29.6, 11.4 FPS; the same model run single-frame 34.1 mAP at 14.0 FPS; best prior temporal in its table (SQD-MapNet) 26.0, best prior non-temporal (Topo2Seq) 33.6. Subset B: 46.3 OLS temporal vs 41.8 single-frame.
- compute: 11.4 FPS (temporal) vs 14.0 FPS (single-frame); parameters not stated.
- code: not checked
- fit_to_us: reuse_idea - effort S/M - the principle (train with and without memory in parallel, switch by a reliability signal, condition the propagation on ego motion) fits our hold + recursion; the pose-reliability test is unspecified, we would use the foundation's own confidence.
- compat_needs: a signal that tells "memory trustworthy" (pose OK, previous frame OK); previous-frame features and relative pose.
- compat_conflicts: their queries / Hungarian set; world model is trained against the actual current features (needs a stable target, ok for cached features).
- tried_before: E39 / E41 temporal feature fusion (+0.008 F1, kept small); time recursion in builder (loop 16). The fast-slow switch and the latent predictor are new.
- verified: partly (digest of paper; authors and venue not captured)

### CARD A02 - SIGMA-Lane: gated Mamba2 temporal state that is protected against contaminated frames
- papers: SIGMA-Lane: Scale-pyramId Gated MAmba for Temporally Consistent Video Lane Detection (Zhang, Wang, Gao, Kong, Shen, Du, ECCV 2026) https://arxiv.org/abs/2608.16338
- depth: M
- stage: S8
- idea: Video lane detection with an SSM temporal core (Mamba2). The authors name the failure mode "state contamination": an occluded or corrupted observation written into the recurrent state damages later frames. They gate the write path (input gate: x' = x * (1 - m), m = token-wise occlusion probability) and the residual-fusion path (scalar retention ratio r from the mean mask via small MLP + sigmoid, initialised near 0.2). A lane-guided affine warp aligns the previous features before the SSM, and a cross-attention "structural spatial retrieval" recovers lane structure from the aligned past that the gates cannot restore.
- formulation: SSM update h_t = A_t h_{t-1} + B_t x_t, gate scales the write term; memory bank = past 8 frames' features + masks + previous lane mask; learnable start token; affine residual (2x3, tanh, zero-initialised final layer) from lane-masked pooled previous features + current features. Gate signal = obstacle mask from a frozen segmenter; gate placement matters more than mask precision (removing both gates drops F1 by 0.018 on OpenLane-V).
- input: aggregated frame features + obstacle masks (image space, not BEV).
- output: lane masks / coefficients (per-frame lane detection).
- training: base detector frozen, only the temporal aggregation modules trained (0.35 M params) with the base detector's losses. Hyper-parameters deferred to the supplement (not read).
- differentiable: yes in the module (gates, SSM, affine); gradient across frames not discussed.
- sensors_data: camera video, VIL-100, OpenLane-V; 384x640.
- results: VIL-100 mIoU 0.801, F1@0.5 0.940; ablation (VIL-100) baseline Mamba2 swap 0.775 mIoU / flicker R_F@0.5 0.052, both gates 0.787 / 0.024, full model 0.801 / 0.023. OpenLane-V mIoU 0.761, F1@0.5 0.838.
- compute: OpenLane-V, RTX 4090: 27.1 GFLOPs, 118.8 model FPS (96.7 end-to-end), 3.96 M main params, 0.35 M aggregation params.
- code: not checked
- fit_to_us: reuse_idea / reimplement - effort S/M - the gated write is a one-line change to our spatial hold (our hold already prefers the previous input where its lane heat is higher; this makes the write depend on an explicit per-pixel corruption probability). The only paper in the set that models contamination of the state explicitly.
- compat_needs: a per-pixel corruption / reliability map m (we know which cameras we switch off in training, so m can be supervised; at test time it must be predicted).
- compat_conflicts: image-space affine warp is a short-range approximation, not ego-motion; our BEV uses exact ego warp.
- tried_before: none (spatial hold is the training-free analogue).
- verified: partly

### CARD A03 - MapBench: temporal fusion under camera corruption (the one robustness measurement of temporal memory)
- papers: Is Your HD Map Constructor Reliable under Sensor Corruptions? (Hao, Zhang et al., NeurIPS 2024 D&B) https://arxiv.org/abs/2406.12214
- depth: M (only the temporal ablation tables were queried; 4.7 k characters of the page were not read)
- stage: E, S8
- idea: Benchmark of 29 camera / LiDAR corruptions on nuScenes val (camera crash, frame lost, dark, motion blur, snow ...), 31 constructors. One ablation compares StreamMapNet with and without temporal fusion under the corruptions.
- formulation: metrics mCE / mRR (corruption error / resilience ratio relative to a baseline). Temporal fusion is the StreamMapNet BEV + query memory.
- input / output: images; vector maps.
- training: models retrained on a new split for this ablation (absolute numbers differ from Table 1).
- differentiable: n/a (benchmark).
- sensors_data: camera (also LiDAR and fusion models); nuScenes val.
- results: StreamMapNet clean mAP 23.8 -> 28.0 with temporal; mRR 47.1 -> 55.5, mCE 100 -> 85.9. Per-corruption mAP (no temporal -> temporal): camera crash 5.0 -> 10.3, frame lost 5.7 -> 12.1, dark 11.2 -> 14.4, motion blur 12.1 -> 17.6, snow 3.2 -> 2.0 (only corruption where temporal is worse). The text separately reports large drops for missing cameras in other models (e.g. about 40 % mAP loss for one model, about 69 % for another, from a search snippet; not verified in the paper).
- compute: n/a
- code: project page mapbench.github.io (named by search result, not opened)
- fit_to_us: reuse_idea (evaluation protocol: crash, frame lost, dark, blur map directly onto our night / blur / cameras-off set) - effort S.
- compat_needs: corruption generators; our synthetic failures already cover dark, blur, crash.
- compat_conflicts: none.
- tried_before: our own failure set (night, blur, three front cameras off).
- verified: partly

### CARD A04 - StreamMapNet: BEV warp + GRU and query propagation
- papers: StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction (Yuan, Liu, Wang, Zhao, WACV 2024) https://arxiv.org/abs/2308.12570
- depth: M+C(framework only)
- stage: S8
- idea: Two memories. (1) BEV memory: previous BEV feature warped by the 4x4 ego transform T and fused with the current one by a GRU + LayerNorm. (2) Query memory: top k = 33 of N_q = 100 queries by confidence are kept; propagated by Q_t = MLP(concat(Q_{t-1}, flatten(T))) + Q_{t-1}; their polylines are transformed by T and used as reference points; concatenated with the remaining new queries.
- formulation: loss = 50 L_line (Smooth L1 over 20 points, min over permutations) + 5 L_focal + 5 L_trans (aux loss supervising the transformed queries). Gradients through memory do not flow back to previous frames; each training sequence is randomly split in 2 parts per epoch; first 4 epochs single-frame.
- input: BEV feature (from images), previous BEV and queries, ego pose T.
- output: vectorised map polylines (set, matching based).
- training: Hungarian / permutation matching, 8 x 3090, batch 32, AdamW 5e-4.
- differentiable: partial - memory detached across frames; Hungarian in loss.
- sensors_data: camera, nuScenes + Argoverse2; 60x30 and 100x50 m.
- results: new (geographically disjoint) nuScenes split mAP: 30 m 33.9 (vs MapTR 20.9), 50 m 23.0 (vs 14.8); 14.2 FPS (Argoverse2 setting); GPU memory / latency numbers not stated.
- compute: 13.2 FPS (nuScenes table), 14.2 FPS (abstract setting).
- code: https://github.com/yuantianyuan01/StreamMapNet (README: OpenMMLab stack mmcv-full 1.6.0, mmdet 2.28.2, mmdet3d 1.0.0rc6; GPL-3.0)
- fit_to_us: reuse_idea - effort S - GRU on warped BEV and a transformed-query MLP are small; "detach memory, single-frame warm-up" is the standard recipe our time recursion already follows.
- compat_needs: ego transform; confidence ranking.
- compat_conflicts: their code needs mmcv / mmdet3d (not in our env).
- tried_before: E41 temporal feature fusion (+0.008 F1); loop 16 one-step recursion.
- verified: checked

### CARD A05 - MapTracker: raster + vector memory with strided fusion and pose-noise training
- papers: MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping (Chen, Wu, Furukawa, ECCV 2024 oral) https://arxiv.org/abs/2403.15951
- depth: M+C(README)
- stage: S8
- idea: Mapping is tracking. BEV memory M_BEV (50x100x256, 15 m left/right, 30 m front/back, last 20 frames) is warped to the new frame by affine ego motion + bilinear sampling; pixels outside the previous map are filled with learnable per-pixel embeddings ("MaskBlend"). Vector memory = one 512-d latent per tracked element, propagated by a two-layer PropMLP fed with quaternion + translation (positionally encoded); new candidates (100) join the propagated set. Strided fusion: four past memories at about 1, 5, 10, 15 m travelled distance are aligned, concatenated, passed through a residual conv block (BEV) or cross-attention (vector).
- formulation: L = L_BEV (10 focal + 1 Dice) + L_track (5 focal + 50 line) + 0.1 L_trans. Stride ablation (nuScenes old split): {1,5,10,15} 76.1 mAP / 69.1 C-mAP, {5,10,15,20} C-mAP 51.5. Pose noise: Gaussian noise added to the transform during training (no standalone test-time evaluation). Clip training: 4 of the previous 10 frames. Thresholds 0.4 first frame, 0.5 propagated, 0.6 new.
- input: BEV features from images, ego pose.
- output: tracked polylines with identities (consistent map).
- training: tracking-style label inheritance + bipartite matching for new elements; 3 stages (18/6/48 epochs nuScenes); image backbone frozen on the first four frames; 8 x A5000, ~3 days.
- differentiable: partial - matching for new elements; track labels inherited; memory gradient not detailed.
- sensors_data: camera, nuScenes / Argoverse2.
- results: nuScenes consistent GT 76.1 mAP / 69.1 C-mAP vs StreamMapNet 70.4 / 56.4 and MapTRv2 69.5 / 50.5; Argoverse2 76.9 / 68.3 vs 70.3 / 57.5. No missing-frame experiment reported. Stated limitation: merges and splits of elements are not handled by the tracking formulation.
- compute: 11.5 FPS on RTX 6000 (StreamMapNet 14.2, MapTRv2 12.5); extra time from un-batched loops in vector fusion.
- code: https://github.com/woodfrog/maptracker (GPL-3.0, built on StreamMapNet code; framework not named in README)
- fit_to_us: reuse_idea - effort S/M - take MaskBlend (learned fill for cells without history), distance-based strides instead of time strides, pose-noise augmentation of the warp; the vector tracking itself needs identities (we have none).
- compat_needs: ego pose; distance-based buffer of past inputs.
- compat_conflicts: element identities / bipartite matching; our graphs have forks and merges, which they explicitly do not handle.
- tried_before: none for strides / MaskBlend; map-offset and pose-jump failures exist in our synthetic set.
- verified: checked

### CARD A06 - HRMapNet: a global rasterised map built from the network's own past predictions
- papers: Enhancing Vectorized Map Perception with Historical Rasterized Maps (Zhang et al., ECCV 2024) https://arxiv.org/abs/2409.00620
- depth: M+C
- stage: S8, S9
- idea: Local vector predictions are rasterised (3 channels, 0.3 m cells) and merged into a global grid by a per-pixel counter: +30 where the local map is 1, -1 where 0 (asymmetric evidence, 8-bit, about 1 MB/km). The current local patch is retrieved with the ego pose (round(R p + t)), thresholded, and used twice: extra BEV queries at non-empty cells inside the BEV encoder (F_BEV = Conv(Concat(F_I + F_M, M_l))), and prior embeddings that initialise the decoder queries via cross-attention.
- formulation: S_th not stated; no decay or log-odds clamp stated; map starts empty each epoch in training; whether gradients pass through the map is not stated. Pose noise: sigma_t 0.2 m, sigma_r 0.02 rad gives 63.8 mAP (baseline MapTRv2 61.5); training-time pose augmentation helps robustness at some accuracy cost (sigma_r 0.1 rad: 63.1 vs 65.8 clean).
- input: image BEV features + retrieved raster; output: vector map (set).
- training: standard map loss of the host; empty map first.
- differentiable: no for the map itself (rasterise / threshold / counter update); yes in the fusion convolutions.
- sensors_data: camera, nuScenes / AV2.
- results: nuScenes mAP StreamMapNet 60.4 -> 66.3, MapTRv2 61.5 -> 67.2 (24 epochs); night 39.6 -> 42.6, rainy 50.8 -> 62.6, with an empty initial map; a map pre-built from training data gives 83.7 (overlap with val, not fair); location-disjoint split StreamMapNet 33.9 -> 37.1.
- compute: FPS 22.5 -> 21.1 (StreamMapNet), 19.6 -> 17.0 (MapTRv2); +9 GB training memory; no ms figures.
- code: https://github.com/HXMap/HRMapNet (MIT; MMDetection3D-style folder, mmcv not confirmed)
- fit_to_us: reuse_idea - effort S/M - the asymmetric counter update is a cheap non-learned long-horizon memory for lane evidence and is the one place where night / rain gains were quantified; it is a non-differentiable side memory, so it can only be an input channel.
- compat_needs: global pose; vector -> raster step.
- compat_conflicts: requires consistent global pose over a drive; our pose-jump failure would corrupt it.
- tried_before: E40 / E48 map prior at the grid with learned registration (+0.075 F1 exact pose, -0.057 unaligned) - same family, same lesson.
- verified: checked

### CARD A07 - PrevPredMap / Uni-PrevPredMap: previous predictions as queries and map priors, trained with perturbed priors and mode dropout
- papers: PrevPredMap: Exploring Temporal Modeling with Previous Predictions for Online Vectorized HD Map Construction (2024) https://arxiv.org/abs/2407.17378 ; Uni-PrevPredMap: Extending PrevPredMap to a Unified Framework of Prior-Informed Modeling (2025) https://arxiv.org/abs/2504.06647
- depth: M (both)
- stage: S8, T
- idea: The previous frame's top-k predictions (k = 10 on nuScenes) are ego-transformed and used directly as decoder queries (category and confidence -> content query, location -> position query), no BEV memory. Training randomly picks single-frame mode with p = 0.5. The unified version stores predictions and map priors in a UTM-tile global vector map (write threshold 0.4 train / 0.9 test) and trains a tri-mode model (no prior, temporal prior, temporal + map prior) with perturbed priors (about half the vectors displaced by up to +-6 m).
- formulation: performance saturates at about 12 previous queries; the paper notes a very large k adds uncertain predictions that hurt. Robustness table (nuScenes, perfect map 85.6): instance displacement +-3/6/9 m 81.9/80.9/80.3; frame displacement +-3/6/9 m 77.0/75.7/74.6 (empty prior 74.0); rotation +-15 deg and scale 0.8-1.2 both 77.0.
- input: previous predictions (+ optional map vectors); output: vector map.
- training: dual-mode; Hungarian host. Gradient across frames not stated.
- differentiable: partial (prediction dictionary updated during training, matching).
- sensors_data: camera, nuScenes / AV2.
- results: PrevPredMap nuScenes 66.3 mAP temporal vs 65.7 single mode (24 epochs), 71.3 at 110 epochs; FPS 15.7 vs 16.1. Uni-PrevPredMap: no prior 64.9, temporal prior 74.0 (24 ep) / 77.0 (72 ep), map prior 71.3, both 80.9 / 81.8; FPS 14.2 / 12.2 / 13.1 / 11.5 (A6000).
- compute: see above; the previous-prediction prior costs about 6.5 % FPS.
- code: not checked
- fit_to_us: reuse_idea - effort S - this is the closest published analogue of our "prior from the previous frame's graph" recursion; the transferable tricks are (i) train with p = 0.5 no-history, (ii) perturb the prior by metres, (iii) confidence-ranked top-k, (iv) test unseen perturbation types.
- compat_needs: previous graph in ego frame with confidences; our vertices (position + token + validity) are directly usable as "queries".
- compat_conflicts: they decode with a DETR set + Hungarian; ours is a tracer; the global tile store needs pose.
- tried_before: loop 16 one-step recursion (prior from previous graph) - this paper's perturbation / mode-dropout recipe was not reported as used there (check).
- verified: partly (PrevPredMap pose handling: only the phrase "ego transformation" was found)

### CARD A08 - SQD-MapNet: stream query denoising (training-only)
- papers: Stream Query Denoising for Vectorized HD-Map Construction (Wang et al., ECCV 2024) https://arxiv.org/abs/2401.09112
- depth: M
- stage: T, S8
- idea: In training, the previous frame's ground truth is ego-warped into the current frame, noised, and fed as extra "denoising" queries that must reconstruct the current frame's ground truth. Adaptive Temporal Matching pairs previous and current instances by bidirectional Chamfer distance below a threshold relative to instance size; the noise magnitude decays with the matched distance. Removed at inference: zero cost.
- formulation: noise = box shift (lambda1, lambda2 x width / height) and box scale ((1 +- lambda3), (1 +- lambda4)); values of lambda not stated; L = L_map + lambda4 L_focal^DN + lambda5 L_line^DN. Best matching scale alpha = 0.1, decay gamma = 0.2.
- input: previous GT (training); output: n/a at inference.
- training: same losses as the map head on the denoising queries; single-frame training uses ordinary denoising.
- differentiable: partial (matching).
- sensors_data: camera, nuScenes + AV2.
- results: nuScenes 60x30 m 63.9 mAP vs StreamMapNet 60.4; ablation 59.9 (single frame) -> 59.2 (streaming alone) -> 62.9 (dynamic noising) -> 63.9 (+ATM). No quantitative consistency metric in the paper.
- compute: no inference cost (training only).
- code: "available soon" at v1; not checked since
- fit_to_us: reimplement - effort S - for our recursion: train the tracer on the previous-frame ground-truth graph warped and perturbed in position, scale, and with dropped / spurious pieces, so the prior can be wrong; note the ablation shows streaming alone slightly hurt (59.2) until denoising was added, i.e. a memory that is trusted blindly can lose.
- compat_needs: GT graph of the previous frame (we have it), ego transform.
- compat_conflicts: none (loss-level).
- tried_before: none seen.
- verified: partly

### CARD A09 - MapTCL: bidirectional temporal consistency loss (training-only)
- papers: MapTCL: Temporal Consistency Learning via Bidirectional Alignment for Vectorized HD Map Construction (2026, IROS 2026) https://arxiv.org/abs/2608.05209
- depth: M
- stage: L, T
- idea: Two auxiliary losses that vanish at inference. BVCL: past-frame predictions with confidence above 0.3 are warped into the current frame and Hungarian-matched to current predictions (instance + point level), and the reverse; SmoothL1 on points + KL on class distribution, weights decay with frame distance. RCL: a BEV raster head with focal loss; past raster maps warped to the current frame and compared with a focal consistency term with decaying weights.
- formulation: total = map loss + BVCL + RCL (beta = 1.0 for RCL); memory length N_t = 5 best (AV2 56.7 mAP, 51.6 at N_t = 7); tau = 0.3 best (0.1 -> 54.0 mAP / 23.8 C-mAP). Applied only in the temporal training stage.
- input: model outputs of past and current frames; output: loss.
- training: 4 x A6000, batch 16, 24 epochs nuScenes / 6 AV2.
- differentiable: partial (Hungarian inside BVCL; RCL fully differentiable).
- sensors_data: camera, nuScenes / AV2 new splits.
- results: new split 60x30 m mAP / C-mAP: StreamMapNet nuScenes 35.2 / 25.7 -> 38.9 / 28.5; SQD-MapNet 36.6 / 25.4 -> 37.9 / 26.8; MapTracker 39.4 / 30.8 -> 41.0 / 32.4; heavy occlusion StreamMapNet 29.5 -> 32.4.
- compute: none at inference.
- code: no link on the page
- fit_to_us: reimplement - effort S - the RCL variant (consistency between warped previous dense output and current dense output) needs no matching and is exactly our heat-map output; bidirectional warp avoids favouring either frame. Directly attacks flicker, relevant to our "temporal consistency" metric.
- compat_needs: dense output per frame, ego warp; for BVCL a matching which we avoid -> use RCL only.
- compat_conflicts: BVCL uses Hungarian (our rule: not in our losses).
- tried_before: none.
- verified: partly

### CARD A10 - MemFusionMap: working memory + temporal overlap heatmap
- papers: MemFusionMap: Working Memory Fusion for Online Vectorized HD Map Construction (Song et al., WACV 2025) https://arxiv.org/abs/2409.18737
- depth: M
- stage: S8
- idea: The last T_WM = 4 BEV features (warped by ego motion) are concatenated with the current BEV and a "temporal overlap heatmap" feature, then passed through three dilated (2,2) 3x3 conv layers + LayerNorm; a GRU still carries the long state. The heatmap starts as ones, is warped by ego motion with zero padding and incremented by 1 per step, so it counts how often a cell has been seen; a small conv stack with sigmoid turns it into a 32-channel feature that tells the network how much to trust the present vs the past.
- formulation: T_WM best 4 (mAP 24.3 / 25.9 / 27.6 / 26.2 / 27.0 for 1 / 2 / 4 / 6 / 8, 50 m nuScenes new split); ablation 100x50 m: no GRU 19.5, + working memory 25.5, + dilation 25.9, + heatmap 27.6; sigmoid vs ReLU on heatmap 27.6 vs 26.5. No gating; memory tensor detached. The heatmap is unbounded over long sequences and not occlusion-aware (paper's own limitation).
- input: BEV features, ego motion; output: BEV feature for a MapTR-style decoder.
- training: two-stage (single frame, then temporal), 8 x V100.
- differentiable: yes in the fusion; detached across frames.
- sensors_data: camera, nuScenes / AV2 new splits.
- results: nuScenes 30 m 38.0 vs StreamMapNet 34.1; 100x50 m 27.6 vs 22.2; AV2 60.6 vs 58.3; max gain 5.4 mAP. About 15.1 FPS (StreamMapNet 15.3, MapTracker 11.6, A6000). No pose-error or missing-frame test.
- compute: 15.1 FPS.
- code: not checked
- fit_to_us: reimplement - effort S - an "age / coverage" channel next to the held input costs one channel and tells the tracer how stale each cell is; it is the simplest published trust cue. Can be extended with a "this frame failed" channel.
- compat_needs: warped history stack (we have the hold).
- compat_conflicts: none.
- tried_before: E41 temporal feature fusion (small gain); hold with decay (loop 16). Age channel not tried.
- verified: partly

### CARD A11 - InteractionMap: key-frame stacking next to streaming
- papers: InteractionMap: Improving Online Vectorized HDMap Construction with Interaction (CVPR 2025) https://arxiv.org/abs/2503.21659
- depth: M (temporal part only)
- stage: S8
- idea: Local fusion = warped previous BEV + GRU + LayerNorm + residual block (streaming). Global fusion = key-frame selection by distance stride (d_stride = 5 m): either KFS-streaming (previous global feature warped, GRU) or KFS-stacking (up to N_pre = 4 warped key-frame submaps concatenated, residual conv). Argument: pure streaming forgets long-range history.
- formulation: training samples previous submaps randomly within N_pre x d_stride; first half of training single-frame. Ablation (nuScenes mAP): baseline 50.2, plain streaming 54.0, KFS-stacking 57.6, KFS-streaming 57.4.
- input: BEV features, 4x4 relative pose; output: BEV feature for the decoder.
- training: first N_total/2 epochs single-frame.
- differentiable: yes in the fusion.
- sensors_data: camera, nuScenes / AV2.
- results: nuScenes val 71.6 (R, 24 ep) / 75.5 (110 ep); AV2 73.7.
- compute: 12.1 FPS, 49.9 M params (R); 11.0 FPS, 116.7 M (C) (units as written in the paper: "MB").
- code: not checked
- fit_to_us: reuse_idea - effort M - a stack of 2 - 4 distance-spaced warped inputs gives long memory without a recurrent state that can be poisoned; cost is linear in the stack depth on our 200x200x42 raster (conv only).
- compat_needs: distance-based buffer.
- compat_conflicts: more channels into the trace-network input (42 x K).
- tried_before: E41 temporal fusion (+0.008 F1).
- verified: partly

### CARD A12 - MambaMap: BEV and instance-query fusion with a gated (diagonal) state-space block
- papers: MambaMap: Online Vectorized HD Map Construction using State Space Model (Yang et al., IROS 2025) https://arxiv.org/abs/2507.20224
- depth: M
- stage: S8
- idea: Memory bank of the last N = 4 refined BEV features and instance queries. BEV Mamba Fusion: warp history by 4x4 affine, concat, conv + LayerNorm, scan in four directions through a gated SSM block (GELU gate, expansion 4, DSS branch with reduction 0.5), average the four scans. Instance Mamba Fusion: historical queries aligned to current ones by L2-distance Hungarian matching, two sequences (instance-first, temporal-first) through gated SSM blocks. The authors note the SSM is DSS, "inspired by" Mamba.
- formulation: L_map = 5 L_pts + 50 L_cls(as stated; weights differ in orientation across papers). Memory features detached; two-stage training (single frame, then sequences). Ablation 60x30 nuScenes: baseline 61.5, + BMF 65.2, + IMF 63.7, both 67.3; SSM variant: S4 64.6, Mamba 65.3, Mamba-2 65.2, DSS 67.3; N = 4 best (67.3), N = 6 66.5.
- input: BEV features + queries; output: vector map.
- training: AdamW 2.5e-4, batch 8, 2 x A30.
- differentiable: partial (Hungarian in query alignment and loss; memory detached).
- sensors_data: camera, nuScenes / AV2.
- results: nuScenes 60x30 67.3 mAP (StreamMapNet 63.4), 100x50 66.5 (63.5); AV2 64.9 (61.1); geo-disjoint nuScenes 40.1 (MemFusionMap 38.0, StreamMapNet 34.1). Paper text claims a 3.1 mAP minimum gain on the disjoint split; table gives 2.1 over the strongest baseline.
- compute: 12.7 FPS vs StreamMapNet 14.9 (nuScenes), 13.6 vs 15.9 (AV2): slower; no FLOPs / params stated.
- code: "forthcoming" per search summary; not checked
- fit_to_us: baseline / reuse_idea - effort M - evidence that an SSM memory is not faster than GRU at 4 frames; the linear-complexity argument pays only for long sequences. Not recommended as first move.
- compat_needs: grid-shaped features (fine for 200x200).
- compat_conflicts: query alignment uses Hungarian.
- tried_before: none.
- verified: partly

### CARD A13 - MapUnveiler: clip tokens and a Token-Turing-Machine-style memory
- papers: Unveiling the Hidden: Online Vectorized HD Map Construction with Clip-Level Token Interaction and Propagation (Kim, Seong, Ji, Jang, NeurIPS 2024) https://arxiv.org/abs/2411.11002
- depth: M
- stage: S8
- idea: Per clip (T = 3 frames, stride S = 2): 50 compact clip tokens are made from the frame-level map queries by cross-attention; the BEV is updated by cross-attending to the clip tokens (BEV queries, token keys) instead of dense spatio-temporal attention; the map generator decodes with deformable attention. An inter-clip memory of M = 96 tokens is read and written in Token-Turing-Machine style; the first clip uses learned embeddings.
- formulation: memory 96 or more similar, smaller degrades; clip tokens 50 best (25: 69.2, 100+ worse); read: memory + map queries 69.8 vs memory only 68.9; write: clip tokens 69.1, + map tokens 69.8, adding BEV 68.9; frozen frame-level MapNet 66.7 vs end-to-end 69.8; T = 5, S = 3 gives 70.1 but over 40 GB. Ego-pose usage and cross-clip gradient flow: not found.
- input: BEV features from MapTRv2 backbone; output: vector map.
- training: built on MapTRv2, frame-level pre-train then main; 8 x A100, batch 16; the model runs clip inference three times per step and applies the loss after each.
- differentiable: partial (Hungarian host; gradient across clips not stated).
- sensors_data: camera, nuScenes / AV2.
- results: nuScenes val 68.0 (60x30) / 68.7 (100x50) vs MapTRv2 61.5 / 58.6; AV2 70.5 / 65.4; heavy-occlusion split 63.8 vs StreamMapNet 53.1 and MapTRv2 47.1; geo-disjoint 39.4 vs 36.6. (Slide listing quotes different occlusion gains; the digest used the proceedings figure.)
- compute: 12.7 FPS vs MapTRv2 15.6 (A100); peak GPU memory about 2x and parameters about 3x of MapTRv2.
- code: no repository on the project page (https://mapunveiler.github.io)
- fit_to_us: reuse_idea - effort M/L - a fixed-size learned memory (96 tokens, read / write) is the best published answer to long-horizon memory with forgetting; heavy for our latency budget.
- compat_needs: token interface; our vertex tokens (128-d) could be the clip tokens.
- compat_conflicts: cost (2x memory, 3x params).
- tried_before: none.
- verified: partly

### CARD A14 - PredMapNet: per-instance raster history memory + short-term forecast on top of MapTracker
- papers: PredMapNet: Future and Historical Reasoning for Consistent Online HD Vectorized Map Construction (2026) https://arxiv.org/abs/2602.16669
- depth: M
- stage: S8
- idea: Each tracked instance keeps a stored BEV mask, blended with the new prediction (confidence-weighted, beta = 0.9), warped by ego motion each step, pruned when the track is lost. History-Map Guidance lets track queries cross-attend to BEV features sampled where the stored mask is above 0.5. Short-Term Future Guidance: an MLP predicts per-point offsets to the next frame from the last n decoded polylines.
- formulation: loss adds mask (Dice 2, BCE 1), Chamfer trajectory loss and auxiliary depth loss. Ablation on MapTracker (74.7 mmAP baseline): +SAQG 0.3, +HMG 0.4, +STFG 0.9, +depth 0.6 -> 76.9.
- input / output: as MapTracker.
- training: 72 epochs nuScenes in three stages, 4 x A100; the paper says masks come from the query generator so "end-to-end differentiability" is preserved; detachment of the memory not stated.
- differentiable: partial (matching) / memory claim not verified.
- sensors_data: camera, nuScenes + AV2.
- results: nuScenes old split 76.9 mmAP / 69.7 C-mmAP (MapTracker 76.1 / 69.1); AV2 77.3 / 69.1; non-overlapping splits +1.8 and +0.9 mmAP.
- compute: 10.1 FPS vs MapTracker 10.9.
- code: not checked
- fit_to_us: reuse_idea - effort M - confidence-weighted blending of a stored dense mask with decay is close to our hold; the gain over MapTracker is small (+0.8 mmAP) for 7 % slower.
- compat_needs: tracked instances.
- compat_conflicts: identity tracking.
- tried_before: none.
- verified: partly

### CARD A15 - TopoStreamer: streaming for lane-segment topology (with explicit adjacency)
- papers: TopoStreamer: Temporal Lane Segment Topology Reasoning in Autonomous Driving (2025) https://arxiv.org/abs/2507.00709
- depth: M
- stage: S8, S6
- idea: LaneSegNet-style lane segments with a streaming layer: BEV warped + GRU, top 30 % of queries (66 of 200) propagated through an MLP of previous queries + ego transform with residual. Streaming attribute constraints supervise coordinates, class, boundary type and a BEV mask on stream queries using instance IDs of matched positives; dynamic lane-boundary positional encoding recomputed per decoder layer from refined reference points; lane-segment denoising incl. topology (adjacency focal loss).
- formulation: topology = MLP on concatenated query projections + sigmoid, focal loss, threshold 0.5; ablation: static PE 33.8 vs dynamic 36.6 mAP; no ID tracking 34.4.
- input: BEV features, ego transform; output: lane segments + adjacency matrix.
- training: AdamW 2e-4, batch 8, 24 epochs, first 12 single-frame; Hungarian matching. Gradient through memory not stated.
- differentiable: partial (Hungarian).
- sensors_data: camera, OpenLane-V2 subset A.
- results: lane-segment mAP 36.6 vs LaneSegNet 32.6; TOP_lsls 28.5 vs 25.4; centerline OLS 44.4 vs 40.7; 13.6 FPS vs 14.7.
- compute: 13.6 FPS; parameters not stated.
- code: not checked
- fit_to_us: reuse_idea - effort M - confirms that memory helps topology (+3.1 TOP_lsls) and that the cost is about 1 FPS; their topology head is a matched adjacency MLP, which we do not use.
- compat_needs: Hungarian-matched instance ids.
- compat_conflicts: our edges come from the tracer; their adjacency is a pairwise head (dropped in E42 / E45).
- tried_before: E42, E45, E46 learned link heads: dropped (rules decide).
- verified: partly

### CARD A16 - Score (CORTE): SD-map-seeded lane queries with ConvGRU temporal fusion
- papers: Coherent Online Road Topology Estimation and Reasoning with Standard-Definition Maps (Pham, Witte, Behley, Betz, Stachniss, 2025) https://arxiv.org/abs/2507.01397
- depth: A (page via aggregator; paper body not opened)
- stage: S8, S9
- idea: Single network over multi-view images + SD map: 50 lane queries whose reference points are sampled along SD edges plus learned offset, DN-style denoising queries, lane attention, ConvGRU recurrent BEV fusion with ego-motion compensation, topology post-processing blending learned similarity and a distance term (alpha = 2.5, beta = 0.8, tuned on validation).
- formulation: not read.
- input: images, SD map; output: lane segments, boundaries, traffic elements, topology.
- training: not read.
- differentiable: not read.
- sensors_data: camera + SD map, OpenLane-V2 subset A.
- results: aggregator page: OLUS 54.9 val / 55.8 test; temporal fusion raises TOP_ll from 36.2 to 40.0 (per the aggregator, not verified in the paper).
- compute: not stated.
- code: not checked
- fit_to_us: reuse_idea - effort M - one of the few places where SD-map seeding and temporal memory are combined for lane topology; the +3.8 TOP_ll from ConvGRU is the size of gain to expect.
- compat_needs: SD map aligned in ego frame (assumed aligned; misplacement not corrected).
- compat_conflicts: as A15.
- tried_before: our route head E104b handles SD map separately.
- verified: partly (aggregator page; numbers not cross-checked)

### CARD A17 - TopoLogic: geometric-distance topology prior (family reference, no memory)
- papers: TopoLogic: An Interpretable Pipeline for Lane Topology Reasoning on Driving Scenes (Fu et al., NeurIPS 2024) https://arxiv.org/abs/2405.14747
- depth: M
- stage: S6
- idea: Connectivity = combination of (i) a learnable monotone function of the distance between the end point of one lane and the start point of the next, f = exp(-x^alpha / (lambda sigma)) (sigma = std of the distance matrix; alpha init 0.2, lambda init 2) and (ii) a semantic similarity from two 3-layer MLPs on the lane queries; G = lambda1 G_dis + lambda2 G_sim. The geometric part can be added to a trained model without retraining.
- formulation: loss on G_sim only; G_dis trained indirectly via the detection loss through the GNN.
- input: lane queries + end-point geometry; output: lane-lane topology matrix.
- training: standard OpenLane-V2.
- differentiable: yes.
- sensors_data: camera, OpenLane-V2.
- results: subset A TOP_ll 23.9 vs TopoNet 10.9; OLS 44.1 vs 39.8; post-hoc use: TopoNet 10.9 -> 22.3, LaneSegNet TOP_lsls 25.4 -> 29.6. Temporal use: none stated in the digest.
- compute: not stated.
- code: https://github.com/Franpin/TopoLogic (as named by search result; not opened)
- fit_to_us: baseline - effort S - relevant to memory only as a topology scorer that is robust to end-point shifts; our rule system already does this.
- compat_needs: lane end points.
- compat_conflicts: none.
- tried_before: E42, E45, E46, E51, E56 TopoLogic-style scoring: within 0.01 of rules, dropped.
- verified: checked

### CARD A18 - TopoFG, UniTopo, LaneSegNet: the non-temporal topology models that the streaming ones build on
- papers: Fine-Grained Representation for Lane Topology Reasoning (TopoFG, 2025) https://arxiv.org/abs/2511.12590 ; Unified Modeling of Lane and Lane Topology for Driving Scene Reasoning (UniTopo, Li, Gao, Liu, Wang, Liu, Mu, 2026) https://arxiv.org/abs/2605.08911 ; LaneSegNet: Map Learning with Lane Segment Perception (ICLR 2024) https://arxiv.org/abs/2312.16108
- depth: A (TopoFG abstract page; UniTopo PDF text not extractable, only title / authors / code link https://github.com/homothetic/UniTopo seen; LaneSegNet from search listing + README)
- stage: S6, S4
- idea: TopoFG: hierarchical prior extractor (BEV mask -> global spatial prior, in-lane keypoint sequences -> local sequential prior), region-focused decoder (fine-grained queries from priors), boundary-point topology reasoning with topological denoising. UniTopo: models lane and lane topology jointly (predecessor / successor), built on TopoNet per the search summary. LaneSegNet: lane segment = centreline + boundaries in one query; base for TopoStreamer / FASTopoWM / Score.
- formulation: not read.
- input / output: BEV features -> lanes + topology.
- training: not read; Hungarian-matched DETR family.
- differentiable: partial (Hungarian).
- sensors_data: camera, OpenLane-V2.
- results: TopoFG OLS 48.0 (subset A), 45.4 (subset B) per the abstract page. UniTopo: search summary says +6.0 % TOP_lsls over LaneSegNet (not verified). LaneSegNet: +5.6 mAP lane segment per search listing (not verified). No temporal component was found for TopoFG; temporal use in UniTopo not determined.
- compute: LaneSegNet README: OpenMMLab stack mmcv-full 1.5.2, mmdet 2.26.0, mmdet3d 1.0.0rc6, Apache-2.0.
- code: https://github.com/OpenDriveLab/LaneSegNet ; UniTopo link above (not opened)
- fit_to_us: baseline - effort - - noted so the lane-topology family is covered; none of the three has a memory, so the "TopoLogic family temporal fusion" in the brief is carried by TopoStreamer, FASTopoWM and Score (A15, A01, A16).
- compat_needs: Hungarian set.
- compat_conflicts: matching (our losses forbid it).
- tried_before: E31 / E32 sparse matched decoder dropped; E42 ... E56 link heads dropped.
- verified: partly

### CARD A19 - GlobalMapNet: online global vector map with merging and a rendered-map input
- papers: GlobalMapNet: An Online Framework for Vectorized Global HD Map Construction (Shi et al., 2024) https://arxiv.org/abs/2409.10063
- depth: M
- stage: S8, S7
- idea: GlobalMapBuilder transforms local elements to world coordinates, Hungarian-matches them (Chamfer) to a clipped part of the global map and merges by replacing the matched sub-sequence; Map NMS removes duplicates by buffered IoU. GlobalMapFusion renders a local clip of the global map as soft Gaussian masks (max over elements of exp(-d / tau)) plus a "traced region" mask and concatenates them with the BEV feature (zeros when no map).
- formulation: training with an empty map for the first 4 epochs, only 1/4 of scenes update the map. Pose-error / drift effect not analysed.
- input: BEV + rendered global map; output: global vector map.
- training: host losses; geometric builder not differentiable.
- differentiable: no for builder (Hungarian, NMS, replace); yes for rendering and fusion.
- sensors_data: camera, nuScenes / AV2.
- results: nuScenes 60x30 m mGAP 13.9 -> 17.6, mAP 32.9 -> 34.8; AV2 mGAP 31.7 -> 35.6; cross-scene nuScenes 10.5 -> 15.2. Latency not reported.
- compute: not stated.
- code: not checked
- fit_to_us: reuse_idea (the "traced region" mask: a channel that says where the system has already looked) - effort S; the global builder not suitable (matching, pose drift).
- compat_needs: global pose.
- compat_conflicts: Hungarian merging.
- tried_before: E40 / E48 map prior at the grid.
- verified: partly

### CARD A20 - Neural Map Prior: learned global BEV feature memory with gated ConvGRU update
- papers: Neural Map Prior for Autonomous Driving (Xiong et al., CVPR 2023) https://arxiv.org/abs/2304.08481
- depth: M
- stage: S8
- idea: Sparse tile-based global neural map (0.3 m default). Current BEV queries cross-attend to the prior (current-to-prior attention, 10x10 patches); a 2D ConvGRU (update + reset gate) writes fused features back to the tiles. Pose transforms each BEV cell to the global frame.
- formulation: pose noise not discussed in the digest.
- input: BEV features + global neural map; output: refined BEV (segmentation or vector heads).
- training: end-to-end with gated update.
- differentiable: yes (also through the memory update).
- sensors_data: camera + GPS / IMU, nuScenes.
- results: mIoU +4.32 (HDMapNet), +5.02 (LSS), +5.50 (BEVFormer); VectorMapNet +3.90 mAP; larger gains in rain (+5.92) and night (+7.73) than normal (+5.20); night-rain +2.90.
- compute: inference "comparable to previous methods"; the HRMapNet paper says NMP's BEV-feature memory needs over 11 GB for Boston vs about 120 MB for a raster map (from HRMapNet digest).
- code: not checked
- fit_to_us: reuse_idea - effort - - the gated ConvGRU write is the template for a learned gate; the large night gains show a prior helps most when the sensor is weak. Global memory itself needs repeat drives and consistent pose (not our setting: 255 scenes, scene-disjoint).
- compat_needs: global localisation.
- compat_conflicts: scene-disjoint split makes the global prior useless in our protocol.
- tried_before: E40 / E48.
- verified: partly (2023 paper, included as base of the 2024+ line)

### CARD A21 - Online map uncertainty: per-vertex Laplace scale as a confidence output
- papers: Producing and Leveraging Online Map Uncertainty in Trajectory Prediction (Gu, Song, Gilitschenski, Pavone, Ivanovic, CVPR 2024) https://arxiv.org/abs/2403.16439
- depth: M
- stage: S10, L
- idea: Each vertex is modelled by two univariate Laplace distributions; the regression head outputs (mu, b) per coordinate; training swaps L1 for the Laplace NLL sum[log(2b) + |v - mu| / b]. Vertex features for the predictor = [mu; b; class probs]. Uncertainty correlates with range, occlusion (StreamMapNet's memory lowers it in occluded regions), night for crossings, rain.
- formulation: Gaussian training was unstable; Laplace matches the existing L1 loss.
- input: online map head; output: map + per-vertex scale.
- training: NLL replaces L1; no explicit confidence weighting is applied downstream.
- differentiable: yes.
- sensors_data: camera, nuScenes.
- results: MR -15 % for HiVT + MapTR and HiVT + MapTRv2-Centerline; DenseTNT + StreamMapNet minADE 0.9492 -> 0.9036, MR 0.2569 -> 0.2359; mixed for DenseTNT + MapTRv2 (minADE 1.214 -> 1.262). Table inconsistencies noted in the digest.
- compute: not stated.
- code: not checked
- fit_to_us: reimplement - effort S - a log-scale output per vertex (cheap) gives the memory and the planner consumer a calibrated confidence; this is the planner interface of our E117 (edges + route probability).
- compat_needs: regression head with scale output.
- compat_conflicts: none.
- tried_before: keep gate 0.872 (E120) is a validity probability, not a position scale.
- verified: checked

### CARD A22 - Stability under scrutiny: a stability metric for temporal map models
- papers: Stability Under Scrutiny: Benchmarking Representation Paradigms for Online HD Mapping (2025) https://arxiv.org/abs/2510.10660
- depth: M (first 100 k characters)
- stage: E
- idea: mAS = mean average stability, per element presence x weighted mix of localisation jitter and shape (curvature) difference across frames; 42 models evaluated.
- formulation: details of the weights not captured.
- results: StreamMapNet temporal vs not: mAS 83.8 -> 91.9, mAP 51.7 -> 63.3; accuracy and stability are largely independent (HRMapNet with training-map priors mAP 83.6 but mAS 76.7; MapTracker 75.95 / 90.4); camera-only MapTR mAS 71.6 -> 61.9 when the sampling interval grows to 10 frames.
- compute: n/a. code: not checked.
- sensors_data: camera, nuScenes.
- differentiable: n/a.
- fit_to_us: reuse_idea - effort S - our "temporal consistency" metric is undefined in the context; presence x jitter x shape on matched pieces is a ready recipe; also says global raster priors improve accuracy but not stability.
- compat_needs: matched elements across frames (we have a piece graph).
- compat_conflicts: none.
- tried_before: temporal consistency metric exists (definition not given in context).
- verified: partly

### CARD A23 - Post Fusion Stabilizer: a 3.3 M adapter between frozen fused BEV and frozen head
- papers: Post Fusion Bird's Eye View Feature Stabilization for Robust Multimodal 3D Detection (Dong, Thakkar, Sargolzaei, Lin, 2026) https://arxiv.org/abs/2603.05623
- depth: M
- stage: B, T
- idea: A near-identity module between the fused BEV map and the frozen detection head, trained on corrupted data only (host frozen). Block 1: global-context shift normalisation, gate sigmoid(alpha), alpha init -5 (about 0.0067). Block 2: per-pixel reliability map R in [0,1] from shifted features + raw LiDAR BEV (bias init +4), F_clean = R * F_shift; BCE to a point-density ratio target on corrupted samples plus an anchor pushing R to 1 on clean samples (weight 0.2). Block 3: semantic and geometric experts give correction dF, spatial gate G (bias -4), output F_shift + G * dF.
- formulation: three-stage curriculum (6 / 4 / 4 epochs; AdamW 2e-4 / 1e-4 / 2e-4); corruptions per modality with probability 0.30 each: 49 % clean, 21 % camera-only, 21 % LiDAR-only, 9 % both.
- input: fused BEV feature (+ raw LiDAR BEV); output: corrected BEV feature.
- training: host detection losses + reliability BCE; host frozen.
- differentiable: yes.
- sensors_data: camera-LiDAR, nuScenes val.
- results: BEVFusion mAP clean 69.2 -> 69.5; camera dropout (6 views) 62.6 -> 63.8; low light 58.8 -> 63.2; rain lens 60.0 -> 63.6; 8-beam LiDAR 12.1 -> 17.5. No temporal element.
- compute: BEVFusion 40.8 M -> 44.1 M params, 70.9 -> 76.6 ms (+8.1 %), 14.1 -> 13.1 FPS on RTX 4080 Super.
- code: not checked
- fit_to_us: reimplement - effort S/M - this is the cleanest match to our situation (frozen foundation, cached BEV, small head trained for failures): a near-identity residual adapter with a supervised reliability map, trained by a curriculum that leaves clean performance untouched. We know the corruption mask exactly in training (cameras off, night, blur).
- compat_needs: a reliability target (density of valid input per cell); our radar and camera are already fused, so the target must come from known camera coverage.
- compat_conflicts: our features are PCA channels + logits (not a raw fused map); the adapter would act on the 42-ch trace input.
- tried_before: failure-frame fine-tuning (project memory); a supervised reliability map and near-identity gating not tried.
- verified: partly (digest notes text vs table inconsistencies in the real-world test)

### CARD A24 - MetaBEV: modality-arbitrary BEV queries with switched-modality training
- papers: MetaBEV: Solving Sensor Failures for BEV Detection and Map Segmentation (Ge et al., ICCV 2023) https://arxiv.org/abs/2304.09801
- depth: M
- stage: B, T
- idea: Dense learnable BEV queries are refined by a decoder with cross-modal deformable attention over camera BEV, LiDAR BEV, or both (modality-specific MLPs give offsets / weights). A missing modality's features are simply dropped (baselines zero-fill). Switched-modality training samples camera-only, LiDAR-only, both with probability 1/3 each, one set of weights.
- input: camera and LiDAR BEV; output: detection + map segmentation.
- training: starting from task-specific pretrained models; mixture-of-experts layers (M2oE) to reduce multi-task conflict.
- differentiable: yes.
- sensors_data: camera, LiDAR, nuScenes.
- results: map segmentation mIoU 70.4 full, 53.7 camera missing, 54.4 LiDAR missing; no temporal component; params / latency not stated.
- code: not checked
- fit_to_us: reuse_idea - effort S - modality dropout with equal-probability modes; for us "radar off" and "cameras off" modes already exist in the failure set.
- compat_needs: per-modality BEV (we only have fused rasters).
- compat_conflicts: foundation is fused; modality-wise dropping needs the second (fused) feature or retraining.
- tried_before: failure-frame fine-tuning.
- verified: partly (2023, base of the line)

### CARD A25 - SafeMap: robust mapping from incomplete camera sets
- papers: SafeMap: Robust HD Map Construction from Incomplete Observations (2025) https://arxiv.org/abs/2507.00861
- depth: M
- stage: B, T
- idea: G-PVR reconstructs the feature of a missing view with a learnable query attending to available views (Gaussian-distributed reference points on a panorama tiled by spatial distance, MAE-like decoder, reconstruction loss). D-BEVC pulls the BEV features of the incomplete input toward those of the complete input by MSE (distillation).
- formulation: loss = map + 0.05 L_rec + 5 L_corr; Gaussian sigma = 3; views masked / dropped randomly in training. Temporal use not stated in the paper digest.
- input: multi-view images; output: vector map (MapTR / HIMap host).
- training: fine-tuning 8 epochs nuScenes, 4 x 3090.
- differentiable: yes.
- sensors_data: camera, nuScenes + AV2.
- results: MapTR mAP all views 50.3 -> 52.5; front missing 31.3 -> 42.4; back missing 28.5 -> 40.5; 3 missing views 19.5 -> 22.8; 5 missing 4.0 -> 6.0; HIMap front missing 38.0 -> 54.3.
- compute: MapTR 39.1 M params, 21.5 FPS -> 39.5 M, 21.4 FPS; HIMap 68.1 M, 9.7 FPS -> 71.7 M, 9.2 FPS.
- code: not checked
- fit_to_us: reuse_idea - effort S - the distillation loss (incomplete-input BEV pulled toward complete-input BEV) is a feature-level teacher that can be applied to our 42-channel input: teacher = clean input, student = failed input; no inference cost.
- compat_needs: paired clean / failed samples (we synthesise them).
- compat_conflicts: view reconstruction needs image features (we have none).
- tried_before: failure-frame fine-tuning (label loss only); feature-level distillation not tried.
- verified: partly

### CARD A26 - ReSMap and other reliability-weighted fusions (ReSMap, ReliFusion, RoCaRS)
- papers: ReSMap: Recasting Satellite Priors for Robust and Accurate Online HD Map Construction (Kim, Lee, Jeong, Lim, Park, Hwang, NeurIPS 2026 poster) https://neurips.cc/virtual/2026/poster/149564 ; ReliFusion: Reliability-driven LiDAR-camera fusion (2025) https://arxiv.org/abs/2502.01856 (listing seen in search, abstract not opened) ; RoCaRS: Robust Camera-Radar BEV Segmentation for Sensor Failure Scenarios (Park, Kim, Cho, Hwang, 2025) - no arXiv id found, link not verified (only a database page seen)
- depth: A (ReSMap page abstract; others from search snippets)
- stage: B, S8
- idea: ReSMap: Predict-then-Fuse weights camera and satellite BEV features by each branch's own prediction confidence as a per-pixel reliability weight (no router); decoder with per-modality attention and a learned per-token gate; satellite as redundancy when cameras fail. ReliFusion: per-modality confidence scores feed cross-attention, plus a spatio-temporal aggregation module (not read). RoCaRS: dynamic input dropout + bidirectional feature refinement for camera-radar BEV segmentation.
- formulation: not read.
- results: ReSMap "state of the art under camera failure", no numbers on the page. RoCaRS (abstract summary, unverified): keeps 99 % of the radar-only model when all cameras are off, +23.2 mIoU for map without camera over the baseline.
- differentiable: yes (weights are products of predictions).
- compute: not stated.
- code: ReSMap project page dehsurc.github.io/resmap-page (named, not opened)
- fit_to_us: reuse_idea - effort S - "use the model's own confidence heat map as per-pixel weight between present and held input" is what our hold does without learning; ReSMap gives a published template for making it smooth rather than a hard max.
- compat_needs: two branches that both produce confidence (current frame, held frame).
- compat_conflicts: none.
- tried_before: hold = hard selection where previous heat is higher.
- verified: partly (A only)

### CARD A27 - OnlineBEV: recurrent BEV memory with motion-guided deformable alignment and a consistency loss
- papers: OnlineBEV: Recurrent Temporal Fusion in Bird's Eye View Representations for Multi-Camera 3D Perception (2025) https://arxiv.org/abs/2507.08644
- depth: M
- stage: S8, L
- idea: One recurrent BEV state H_{t-1}, updated per layer (L = 3) by a motion-guided fusion: a motion feature extractor takes the difference between target and history queries (FC + channel attention) and predicts deformable-attention offsets and weights that align the ego-compensated history to the current frame. HTC-loss: shared heads produce heatmaps from H_t and from the aligned history; L2 between them with stop-gradient on the target branch, weight 2.
- formulation: loss = focal (1) + L1 (0.25) + 2 HTC. Whether BPTT or detach: not stated; memory resets at scene start. Two phases: 6 epochs single-frame then 90 epochs recurrent.
- input: BEV queries; output: detection heatmaps.
- training: 4 x 3090, batch 32.
- differentiable: yes in module.
- sensors_data: camera, nuScenes.
- results: val R50 44.4 mAP / 54.5 NDS; ablation (NDS): single frame 41.9, + recurrent fusion +8.5, + motion-guided +1.0, + HTC +0.5.
- compute: 79.3 ms (SOLOFusion 72.8), 205.7 GFLOPs, 65.0 M params, 3.4 GB, RTX 3090.
- code: not checked
- fit_to_us: reuse_idea - effort M - learned residual alignment after the ego warp (offsets from a motion feature) is the only concrete answer to "ego motion error" at feature level; plus a training-only heat-map consistency loss with stop-gradient on the target branch (cheap, A09-like).
- compat_needs: dense BEV (ours); deformable attention over 200x200 is expensive; a plain predicted flow field + grid_sample would be the cheap version.
- compat_conflicts: designed for moving objects (flow), our lanes are static, so the offsets would mainly absorb pose error.
- tried_before: learned registration for map priors (E48).
- verified: partly

### CARD A28 - MambaBEV: Mamba2 temporal fusion with four-direction scan
- papers: MambaBEV: An efficient 3D detection model with Mamba2 (2024) https://arxiv.org/abs/2410.12673
- depth: M
- stage: S8
- idea: TemporalMamba block: BEV serialised in four scan directions through Mamba2 and averaged; past BEV features aligned by ego rotation angle only (as described); 3 history frames (tiny) / 4 (base).
- formulation: conv fusion beats concatenation by about 1 point; kernel 3 beats 5; four-direction scan beats one direction by 1.02 % NDS / 1.28 % mAP.
- results: nuScenes val base 51.7 NDS / 42.7 mAP; claims +3.51 mAP and +5.97 NDS over BEVFormer-S and a 37 % drop in velocity error; tiny-model numbers inconsistent across tables.
- compute: no latency; limitations section says still a bit more expensive than deformable attention.
- code: not checked
- training: AdamW 2e-4, batch 1 per GPU, no augmentation.
- differentiable: yes.
- sensors_data: camera, nuScenes.
- fit_to_us: baseline - effort M - like A12: SSM is not shown to be faster than a GRU at 3 - 4 frames. Not an early candidate.
- compat_needs: grid features.
- compat_conflicts: none.
- tried_before: none.
- verified: partly

### CARD A29 - StreamPETR: object-centric memory queue with motion-aware layer norm
- papers: Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection (Wang et al., ICCV 2023) https://arxiv.org/abs/2303.11926
- depth: M
- stage: S8
- idea: Memory queue N x K = 4 x 256 top-scoring queries (FIFO). Propagation transformer with motion-aware layer norm: affine parameters (gamma, beta) are produced from the ego transform, velocity and time interval; hybrid attention concatenates stored and current queries. Gradients detached for the first 6 of 8 training frames, loss on the last 2.
- formulation: E = E_t^-1 E_{t-1}; Q~ = E Q (Eq. 8-9); gamma, beta = linear(ego transform, v, dt) (Eq. 10). Explicit motion compensation alone did not help (0.380 vs 0.378 mAP), which they attribute to error propagation early in training. Memory length: mAP 39.4 / 40.1 / 40.2 for 1 / 2 / 4 frames (saturates at about 2 frames, about 1 s).
- results: nuScenes val R50: 43.2 mAP / 54.0 NDS at 27.1 FPS; light version 45.0 / 55.0 at 31.7 FPS; MLN ablation: pose encoding +2.0 mAP, +time and velocity to 40.2 mAP.
- compute: 27.1 - 31.7 FPS.
- code: not checked
- training: AdamW 4e-4, batch 16; random frame skip as augmentation.
- differentiable: partial (detach).
- sensors_data: camera, nuScenes.
- fit_to_us: reuse_idea - effort S - (a) condition a normalisation on ego motion and dt instead of (or beside) a hard warp; (b) truncated-gradient training: detach most of the history, backprop through the last frames only; (c) random frame skip.
- compat_needs: token memory.
- compat_conflicts: detection-specific (velocity).
- tried_before: none.
- verified: partly

### CARD A30 - CRT-Fusion: camera-radar recurrent temporal fusion guided by predicted motion
- papers: CRT-Fusion: Camera, Radar, Temporal Fusion Using Motion Information for 3D Object Detection (Kim, Seong, Choi, NeurIPS 2024) https://arxiv.org/abs/2411.03013
- depth: M
- stage: S8, B
- idea: Recurrent over N timestamps with a memory bank; a motion feature estimator predicts per-pixel BEV velocity and occupancy; features are shifted by velocity x dt where speed exceeds 1.0 m/s, concatenated with the next BEV and multiplied by the occupancy map, 1x1 conv. Ego motion compensation not described in the digest.
- training: 24 epochs, first 6 without the temporal module; loss weights depth 3, segmentation 25, velocity 1, occupancy 30; 4 x 3090.
- differentiable: yes.
- sensors_data: camera + radar, nuScenes.
- results: R50 val 57.2 NDS / 50.0 mAP vs CRN 56.0 / 49.0; night mAP 33.0 vs CRN 30.4 and BEVDepth 16.8; sunny 54.7 vs 54.8.
- compute: 14.5 FPS (R50), light 20.5 FPS, total 67.1 ms (light 48.8 ms).
- code: not checked
- fit_to_us: reuse_idea - effort - - only camera-radar temporal paper found; shows radar-camera temporal gives night gains in detection; the velocity-shift part is for moving objects, not relevant to static lanes.
- compat_needs: radar + camera BEV.
- compat_conflicts: moving-object motion model.
- tried_before: none.
- verified: partly

### CARD A31 - TempCoBEV: a pluggable temporal module that is trained on cached embeddings and survives lost inputs
- papers: Unlocking Past Information: Temporal Embeddings in Cooperative Bird's Eye View Prediction (TempCoBEV, 2024) https://arxiv.org/abs/2401.14325
- depth: M
- stage: S8, T
- idea: Module between a fused BEV embedding and a frozen decoder, trained on pre-computed stored embeddings (backbone removed): an importance map per embedding (channel-max + sigmoid, softmax across embeddings), importance-weighted sum as query for deformable cross-attention (4 keypoints, 3 stacked blocks), final importance-weighted blend of current and temporal output so the current frame keeps priority. No explicit ego warp: alignment only by learned deformable offsets.
- formulation: more than one historical frame reduced IoU in the ablation; removing importance components lowered performance.
- training: IoU loss through frozen decoder; per-epoch training time about 50 min -> 3 min.
- differentiable: yes inside; not end-to-end with backbone.
- sensors_data: camera cooperative BEV, simulated OPV2V.
- results: when communication is lost, IoU gain over base models for t+1..t+4 of +19, +13.1, +7.9, +3 (CoBEVT), up to 2 % for the current frame.
- compute: training-time saving; inference latency not stated.
- code: link in paper (not opened)
- fit_to_us: reuse_idea - effort S - same recipe as ours (cached embeddings, frozen decoder, small trained module) and the only paper that evaluates memory as a replacement for lost inputs, but on simulated data; shows that the importance weighting keeps the present dominant.
- compat_needs: stored embedding sequences.
- compat_conflicts: simulated cooperative setting.
- tried_before: E41 temporal feature fusion on cached features (+0.008 F1).
- verified: partly

### CARD A32 - Online lane graph extraction from video: early temporal aggregation in BEV
- papers: Online Lane Graph Extraction from Onboard Video (Can, Liniger, Paudel, Van Gool, 2023) https://arxiv.org/abs/2304.00930
- depth: M
- stage: S8, S6
- idea: Compared three ways to use several frames for a lane graph: merging per-frame graphs (post-merge), a transformer over spatio-temporal embeddings, and early aggregation (proposed): warp each frame's features to BEV by homography, mask, align by odometry, aggregate with a max operator after a residual block per frame. Detection and connectivity head: DETR-style, Bezier control points + existence + association features.
- formulation: max beats mean; training with 3 frames (1 past, 1 future), test spacing 2 s; the model accepts any number of frames including one.
- results: nuScenes M-F / detection / C-F: 1 frame 58.3 / 60.0 / 61.0, 3 frames 58.8 / 60.3 / 61.3; post-merge 57.3 / 59.9 / 56.0; Argoverse 3 frames 58.2 / 62.2 / 58.1. Connectivity benefits from future frames only if at least one past frame is present.
- compute: about 10 FPS (1 frame) vs 3 FPS (3 frames).
- code: "will be made public" (no link found)
- training: matched DETR set.
- differentiable: partial (matching).
- sensors_data: camera, nuScenes + Argoverse.
- fit_to_us: reuse_idea - effort S - evidence that early aggregation of aligned dense features beats merging graphs (our E41 +0.008 F1 is consistent with the small gains); max aggregation across frames is the simplest operator and matches our hold's "higher heat wins".
- compat_needs: aligned BEV.
- compat_conflicts: future frames are not available online.
- tried_before: E39 / E41.
- verified: partly

### CARD A33 - D2HDMap: driveline prior with noise-trained robustness and removable prior
- papers: D2HDMap: Non-visible Driveline Map Prior for Online Vectorized HD Map Prediction (2026) https://arxiv.org/abs/2606.20725
- depth: M
- stage: S9, T
- idea: A static rasterised driveline (centreline sequence) prior and Uni-PrevPredMap's temporal prior are retrieved with the ego pose, rasterised, concatenated and used to initialise queries. Trained with translation noise 0 - 2 m and rotation noise 0 - 5 deg on the prior.
- formulation: at 1.0 m / 2.5 deg noise mAP 33.3 vs 44.8 clean; robust training gives +3.86 mAP on average and shrinks the worst-case drop from -19.4 to -13.1.
- results: nuScenes geosplit 60x30 m 44.8 mAP vs MapDiffusion 35.6; without prior at test time 37.8 (-7.0), versus -42.3 for an HD-map-prior model. AV2 numbers inconsistent between table and text.
- compute: latency not reported.
- code: not checked
- training: as host.
- differentiable: partial (host).
- sensors_data: camera, nuScenes / AV2.
- fit_to_us: reuse_idea - effort S - recipe for a prior that can be missing or wrong: train with noise and with prior removed (modes), report the no-prior drop; no lane-graph metric given.
- compat_needs: prior raster.
- compat_conflicts: none.
- tried_before: E40 / E48 (map prior), E104b (route head with SD raster).
- verified: partly

### CARD A34 - Lane Field: dense lane-structure field as conditioning, with uncertainty gates
- papers: Learning Lane-Structured Conditioning Fields for Online HD Mapping (Bertogalli et al., 2026) https://arxiv.org/abs/2610.25284
- depth: M
- stage: S2, S9
- idea: A 5-channel BEV field (3 for unoriented lane direction, proximity, direction confidence) built analytically from centreline polylines (closed-form Gaussian line integral), predicted by an auxiliary head from BEV features or built from SD-map polylines; used for SPADE-like feature modulation and to stretch deformable-attention sampling along the local lane direction. Guidance is gated by direction confidence, a predicted-uncertainty gate (Laplacian log-scale) and a learnable per-head gate.
- formulation: gradient-flow ablation: fully attached gradients 36.1 mAP vs stop-gradient variants 34.1 - 34.8. No temporal component, no sensor-failure test, no code (project page https://andberto.github.io/LaneField/).
- results: LaneSegNet OpenLane-V2 subset A mAP 32.1 -> 36.1, TOP_lsls 25.5 -> 28.4; SD-map variant 36.8.
- compute: LaneSegNet 14.6 -> 13.9 FPS (RTX 4000 Ada), params 45.4 M -> 52.6 M.
- code: none found
- training: end-to-end, closed-form target.
- differentiable: yes.
- sensors_data: camera + optional SD map.
- fit_to_us: reuse_idea - effort S/M - our heat + 2 direction channels are the same kind of field; the sampling-direction idea matches our fan sampling; the uncertainty gate shows how to switch the guidance off when unsure.
- compat_needs: dense field head (we have).
- compat_conflicts: none.
- tried_before: dense head heat + direction (ours).
- verified: partly

## 3. Cluster compatibility notes

**(a) Interface contracts.** In the literature the memory sits at one of four interfaces: (1) dense BEV feature (StreamMapNet, MemFusionMap, InteractionMap, MambaMap, OnlineBEV; warp by 4x4 ego transform, bilinear, fuse by GRU / conv / SSM); (2) query / token (StreamMapNet queries, MapTracker, StreamPETR, MapUnveiler); (3) prediction as prior (PrevPredMap, PredMapNet, HRMapNet, GlobalMapNet, Uni-PrevPredMap; vectors or rasters from earlier output re-enter as queries or as an extra raster channel); (4) output-level loss only (SQD, MapTCL, HTC-loss). Our cached 200x200x42 input is interface (1) with the memory in **input space** (the hold) and interface (3) in the builder (prior from the previous graph). Our vertex (position, 128-d token, validity, edges) maps one-to-one to interface (2)/(3): position = reference point, token = content query, validity = confidence for top-k selection. All "set" methods need Hungarian matching for their own losses; the memory part itself does not, except query-to-query alignment (A12) and BVCL (A09).

**(b) Adapters and conflicts.** Adapter: vector prior -> raster (Gaussian rendering, A06 / A19) and raster -> grid_sample queries. Conflicts: identity tracking (MapTracker, PredMapNet) vs forks / merges; global maps (NMP, HRMapNet, GlobalMapNet) vs scene-disjoint data and our pose-jump failure; mmcv / mmdet3d code (A04, A05, A06, A18) vs our env. A memory that is trusted blindly can lose: SQD's streaming-alone row is below the single-frame baseline (59.2 vs 59.9) until noised-prior training is added.

**(c) Combinations demonstrated.** HRMapNet + StreamMapNet or MapTRv2 (A06); SQD-MapNet as a base for MapTCL (A09); MapTracker as base for PredMapNet (A14) and MapTCL; Uni-PrevPredMap + D2HDMap driveline prior (A33, A07); LaneSegNet as base for TopoStreamer, FASTopoWM, Score, Lane Field (A15, A01, A16, A34); fast-slow parallel training with a pose-reliability switch (A01); gated SSM write + affine alignment + structural retrieval (A02); frozen backbone + small trained robustness module (A23, A31, A02).

## 4. Reuse versus re-implement

- **Re-implement (small, from the method section):** gated write with a supervised reliability map (A02 + A23); age / coverage channel (A10); previous-graph denoising training (A08, A07); training-only consistency losses on the dense output (A09 RCL, A27 HTC); pose-noise augmentation of the warp (A05, A06, A33); MaskBlend learned fill (A05); distance-based strided buffer (A05, A11); Laplace scale output (A21).
- **Reuse only as idea:** fast-slow dual path (A01), MapUnveiler read / write memory (A13), StreamPETR motion-aware norm (A29), learned deformable alignment (A27), SafeMap distillation (A25).
- **Do not port code:** all repos in this cluster that were opened need the OpenMMLab stack (A04, A06 partial, A18 LaneSegNet) or are GPL-3.0 (A04, A05); we have no mmcv / mmdet.
- **Not suitable:** global maps with merging (A19, A20), identity tracking (A05, A14), Hungarian-based BVCL (A09), query-to-query Hungarian alignment (A12).

## 5. Open questions and gaps

- No paper tests a memory under **modality** failure for a map / lane-graph model; MapBench (A03) measures only StreamMapNet with and without temporal fusion under camera corruptions (gains +3 to +6 mAP except snow). Whether the memory gain survives **three cameras off for several consecutive frames** is unknown; only FASTopoWM (A01) and SIGMA-Lane (A02) address a bad current frame, and the former gives no quantitative test of its pose-failure fallback.
- Nobody reports **ms-level** cost of the memory in the map setting: published numbers are FPS of whole models (cost of memory 1 - 3 FPS out of 11 - 15, i.e. roughly 5 - 20 %).
- No paper handles **splits / merges** in a memory; MapTracker states it as a limitation. Junction evidence accumulation is open.
- Error in ego motion: only noise augmentation (A05, A06, A33) and a fallback switch (A01); no learned pose correction for online mapping found (OnlineBEV A27 aligns objects, not pose).
- Gradient through the memory: most papers detach (A04, A10, A12, A29); NMP (A20) trains through the update; for others it is not stated.
- Method depth is limited to digests (see section 1); UniTopo, ReliFusion, RoCaRS, Score not read at method depth.

## 6. WHAT TO TAKE (max 12 lines)

1. Gated write into the held input: replace the hard "previous heat higher" selection by a learned, supervised per-pixel reliability R (BCE against the known failure mask in training; anchor R = 1 on clean frames) with input gate x*(1 - m) as in A02/A23 - **cheap**; failure robustness.
2. Near-identity residual adapter on the 42-ch trace input (gate bias -4, three-stage curriculum, failures at p = 0.3 per mode, 49 % clean) as in A23, +8 % latency in their case - **medium**; failure robustness without hurting clean.
3. Fast-slow training: one shared tracer trained with memory and without memory in parallel, switch by confidence of the current frame (A01); also p = 0.5 no-history mode dropout (A07) - **cheap**; failure robustness.
4. Previous-graph denoising for the time recursion: perturb the warped previous GT graph (shift, scale, drop, spurious pieces; metres of error) as in A08/A07/A33, plus test on unseen perturbation types - **cheap**; failure robustness, shallow forks (wrong prior forks).
5. Training-only consistency loss between warped previous dense output and current dense output, both directions, decaying weight, stop-gradient on the target branch (A09 RCL, A27 HTC) - **cheap**, zero inference cost; speed neutral, flicker.
6. Age / coverage channel and "this frame failed" channel next to the held input (A10) so the tracer can see staleness - **cheap**; failure robustness.
7. Pose-noise and pose-jump augmentation of the warp during training (A05, A06) plus a learned fill for cells without history (MaskBlend, A05) - **cheap**; failure robustness (pose jumps).
8. Distance-spaced buffer of 2 - 4 warped inputs (A05 strides 1/5/10/15 m; A11 5 m key frames) instead of a longer recurrent chain; memory beyond 4 frames saturates or hurts (A10, A12, A09, A29) - **medium** (extra channels in f0); failure robustness for multi-frame outages.
9. Feature-level teacher: pull the failed-input representation toward the clean-input representation (A25 D-BEVC; MSE) - **cheap**; failure robustness, no inference cost.
10. Laplace log-scale per vertex and keep a calibrated confidence for the planner and for ranking what is propagated (A21, top-k by confidence A04/A07) - **cheap**; failure robustness, planner interface.
11. Asymmetric evidence accumulation (+30 / -1 counter, A06) or a "traced region" mask (A19) as a non-differentiable side channel for junction evidence - **medium**; shallow forks (unproven: no paper tests it for forks).
12. Speed: none of the memory papers is faster than a GRU / conv memory at 3 - 4 frames (A12 is slower than StreamMapNet); keep the memory as conv / gate on cached rasters; use SSM only if horizons beyond about 8 frames matter - **expensive/unproven**; speed.

## 7. Search log

1. streaming temporal online vectorized HD map construction memory 2025 arXiv
2. lane topology reasoning temporal streaming TopoStreamer OR TopoLogic OR TopoFG OR UniTopo arXiv
3. Mamba state space model BEV temporal fusion online HD map construction
4. online HD map construction sensor failure robustness missing camera temporal memory
5. SQD-MapNet stream query denoising ... ; HRMapNet ... ; MapUnveiler ... ; Neural Map Prior arXiv 2304.08481
6. Is Your HD Map Constructor Reliable under Sensor Corruptions MapBench arXiv
7. ReSMap satellite priors robust online HD map camera failure confidence-weighted fusion
8. 2026 online HD map temporal consistency memory lane graph streaming arXiv (extended)
9. lane segment topology temporal fusion OpenLane-V2 2026 ... (extended); TopoLogic ...; UniTopo OR LaneSegNet ...
10. state space model temporal BEV perception streaming Mamba 3D object detection OR occupancy
11. camera radar fusion BEV robust sensor failure temporal reliability gating arXiv 2025
12. BEV temporal fusion ego pose noise robust alignment learned offset deformable flow history BEV
13. MetaBEV solving sensor failures ...; Producing and Leveraging Online Map Uncertainty ...; GlobalMapNet ...; RoCaRS ...; Post Fusion Stabilizer ...
14. online lane graph centerline perception temporal aggregation previous frame graph prior (extended)
15. StreamPETR object-centric temporal modeling ...; linear attention OR RWKV OR xLSTM OR ConvLSTM BEV long horizon forgetting (no BEV hit)
16. CRT-Fusion ...; MapMamba OR Mamba lane detection temporal (found SIGMA-Lane); temporal BEV fusion robust to ego pose error ... (extended)
17. online map construction history reliability gating unreliable memory ... 2026 (extended); arXiv 2026 vectorized HD map streaming query propagation ... (extended)
18. temporal BEV map segmentation camera radar sequence history frames ... (found TempCoBEV, PointBeV; PointBeV https://arxiv.org/abs/2312.00703 seen but dropped from cards: sparse BEV with Submanifold Attention for temporal fusion, abstract only)
19. github pages opened: StreamMapNet, maptracker, HRMapNet, LaneSegNet, mapunveiler.github.io; search for other repos returned nothing relevant.
20. WebFetch of arXiv HTML pages: 2403.15951, 2308.12570, 2409.00620, 2401.09112, 2411.11002, 2507.20224, 2409.18737, 2507.00709, 2507.23325, 2507.00861, 2406.12214, 2608.05209, 2602.16669, 2407.17378, 2503.21659, 2610.25284, 2507.08644, 2410.12673, 2603.05623, 2409.10063, 2403.16439, 2303.11926 (ar5iv), 2304.09801 (ar5iv), 2401.14325, 2304.08481, 2304.00930, 2411.03013, 2405.14747, 2504.06647, 2606.20725, 2510.10660, 2608.16338, 2605.20301 (Co-Fusion4D, read, dropped: no failure experiments, 266 ms), abs pages 2511.12590, 2312.00703. PDF-only fetches that failed (binary): 2605.08911, 2407.17378 (re-read via html). Note: the WebFetch tool saved two fetched PDFs into the Claude tool-results folder on its own; I wrote nothing else to disk.
