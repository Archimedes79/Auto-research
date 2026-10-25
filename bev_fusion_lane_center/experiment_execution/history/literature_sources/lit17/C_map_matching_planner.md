# Cluster C (loop 17): SD-map / route conditioning, map matching inside networks, lane-level assignment, robustness and abstention

Date of research: 2026-10-24. Reader: loop-17 synthesis. Card prefix `C`. Format: `report_final/text/lit16/SCHEMA.md`. Project facts: `lit16/PROJECT_CONTEXT.md`.

## 1. Coverage statement

- Searched (see section 7): SD-map priors for online mapping, route / command conditioned planners, BEV-to-map localisation, learned matching with optimal transport, robustness to wrong / shifted / missing maps, uncertainty hand-over to planners, classical HMM map matching with lane markings. Window 2024 - Oct 2026, a few older anchor papers (OrienterNet 2023, GC-PGP 2023, PDM 2023, Hidden Biases 2023, SuperGlue 2019, DGR 2020) where they are the origin of a block we may reuse; they are marked "older anchor".
- **How I read.** Every page was read through the WebFetch tool, which returns a model-written extraction of the page for a prompt I gave it (not the raw page). Where I asked for equations / hyper-parameters, the card carries them as extracted; where the extraction said "not stated" I wrote "not stated". Depth code `M` therefore means "method section content was extracted", not that I read every line myself. No PDF was downloaded, no code was cloned; code pages were only opened for ONR (MAT), NavMapFusion / UMPE links were taken from the paper text.
- Depth M: ONR/MAT, UMPE, NavMapFusion, SD-RouteFusion, D2HDMap, Uni-PrevPredMap, Bateman, RTMap, SMERF, P-MapNet, SDTagNet, NTT target points, GoalFlow, PLUTO, Diffusion Planner, GC-PGP, PDM-Closed, Hidden Biases, CaRL (first 100k chars of 153k only), DiffusionDrive, VADv2, SparseDrive, Hydra-MDP, RAVE (first 100k of 117k), U-ViLAR, HOLO, MapLocNet, OrienterNet (+OSMLoc, 2603.01613, DiffVL), EgoVM, SuperGlue, LightGlue, HMM map matching, Gu et al. Depth A only: DGR (abstract page; weighting details not seen), NavigScene (only the fusion path was extracted; Geo-VLA similar).
- **Not found / not opened:** no paper that does Sinkhorn / optimal transport between a perceived lane graph and a map lane graph (searched twice; the nearest are ONR's softmax-over-roads and SuperGlue itself, from image matching). No paper that tests a route-conditioned end-to-end planner against a deliberately wrong route with a controlled protocol (SD-RouteFusion only has natural corruption; GoalFlow has a shadow trajectory but no wrong-goal ablation). No SE(2) weighted-Procrustes paper in driving. Bateman PDF on the CVF server returned HTTP 403, I read the arXiv HTML instead. ICRA / ITSC / IV papers behind IEEE paywalls were not opened (e.g. P-MapNet RA-L version).
- Date caution: several IDs are 2026 (2605.*, 2606.*, 2607.*, 2608.*); their dates were read from the arXiv listing / HTML (e.g. SD-RouteFusion submitted 1 July 2026, UMPE 4 May 2026, Geo-VLA 18 Aug 2026). I have no independent check beyond the arXiv page itself.
- One search-engine summary was wrong on a detail (it said UMPE is evaluated on NAVSIM v2); the paper HTML says nuScenes with a VAD backbone. Cards follow the paper text.

## 2. Cards (ordered by relevance to the route head / joint fine-tune)

### CARD C01 - Online Navigation Refinement, MAT (SD road route -> perceived lane assignment)
- papers: Online Navigation Refinement: Achieving Lane-Level Guidance by Associating SD and Online Perception Maps (Wan et al., ICLR 2026) https://arxiv.org/abs/2507.07487 ; code https://github.com/WallelWan/OMA-MAT
- depth: M+C (README only)
- stage: S9 (also S6, E)
- idea: Turn a road-level route on an SD map into lane-level guidance by assigning every perceived centre-line to exactly one SD road (many lanes per road, one road per lane). The assignment is a plain softmax over roads of a dot product between a centre-line token and the mean-pooled road token; a topology-aware post-processing then picks a start token and runs a bidirectional beam search along road connectivity.
- formulation: vector = [p1x,p1y,p2x,p2y,theta], 2-layer MLP embedding for road / centre-line / boundary vectors. MAT block = Path-Aware attention (tokens re-ordered by path index, grouped attention, group size k, cost O(N) instead of O(N^2)) + Spatial attention (tokens quantised to (x,y,orientation), Hilbert-curve serialisation, grouped attention) + FFN. Prob_ij = softmax_j(F_cl_i . mean(F_road_j) / sqrt(d)). Loss = 1.0 CE + 0.01 CTC (order / sequence consistency). 50 epochs AdamW, lr 1e-4, wd 0.05, batch 128, cosine, 2 warm-up epochs. Variants MAT-T (tiny) / MAT-L. Parameter counts: not stated.
- input: SD map +-75 m (road vectors), perceived centre-line graph and road boundaries +-15 x +-30 m (training on GT, test on MapTR / MapTRv2 / SeqGrowGraph outputs).
- output: per centre-line vector a road id (scores); sequence of SD links per path.
- training: dense CE on manually annotated association labels (OMA dataset, ~30 k scenarios, 480 k road paths, 2.6 M lane vectors, nuScenes + OSM), trained from scratch; no Hungarian matching.
- differentiable: partial - the network and softmax are differentiable; the beam-search post-processing is a rule. It is trained stand-alone, not inside a perception network.
- sensors_data: any perceived map (vector); nuScenes Boston / Singapore, OSM roads; non-overlapping split.
- results: NR-F1(50:95) val: MAT-T 78.2 at 34 ms, MAT-L 78.7 at 70 ms; best classical map-matching baseline 72.9 at 345 ms (EAM3), FastMAC 70.3 at 62 ms (A6000). Test on perceived graphs: MAT-L 41.9 / 45.0 / 54.9 for MapTR / MapTRv2 / SeqGrowGraph (latency 74 ms). Robustness (val / test): global shift 10 / 20 / 50 % 77.5 / 76.0 / 74.6 (test 43.3 / 42.9 / 41.4); missing roads 10 / 12 / 30 % 77.5 / 76.4 / 75.5; element jitter 5 / 10 / 20 % 78.3 / 77.5 / 77.5. nuScenes-C zero-shot rain / fog / dark hard 40.6 / 42.1 / 41.6 (test-style numbers). 5 % of the training data still gives 77.1 val. Ablation: PTv3 baseline 61.8, +PA 74.1, +PA+SA 77.8, +boundary 78.5, +post-processing 78.7.
- compute: 34 ms (MAT-T) on an A6000 including all tokens of a scene; params not stated. This alone is as long as our whole 31 ms builder, so only the idea (not the module) fits.
- code: https://github.com/WallelWan/OMA-MAT (MIT, Pointcept-based per README; checkpoint and dataset released on HuggingFace per README news - not downloaded)
- fit_to_us: reuse_idea - effort S/M - our route head already does cross-attention graph -> SD tokens; the new parts are (a) a softmax over ROAD pools instead of per-vertex route flags, (b) path-ordered attention along edges, (c) a sequence / order loss.
- compat_needs: a centre-line graph with a path index (we have edges + successor chain); road tokens pooled per road segment.
- compat_conflicts: their ground truth is a manual association label; we derive route labels from the driven lane sequence (no road-id labels per lane yet). No wrong-road test, no abstain mechanism (the paper states no "wrong road" category; unmatched not handled).
- tried_before: route head by cross-attention (E104b, 0.853 - 0.820) kept; correlation volume (E103) dropped. Road-pool softmax and path-aware attention: none.
- verified: partly (extraction via summarising fetch; params and appendix tables 22 not seen)

### CARD C02 - UMPE: unified map-prior encoder with SE(2) micro-alignment, confidence and gating
- papers: Unified Map Prior Encoder for Mapping and Planning (Zhang et al., ICRA 2026 accepted, arXiv May 2026) https://arxiv.org/abs/2605.02762 ; code https://github.com/Ethan-Zheng136/UMPE (link from paper)
- depth: M
- stage: S9, B (prior fusion), S10
- idea: Fuse any subset of HD vector, SD vector, SD raster and satellite priors into BEV features. Vector priors get a per-frame rigid correction, polyline tokens get a sigmoid confidence that enters cross-attention as a log bias, raster priors get a regressed pose that is applied with grid_sample, and a per-channel softmax gate with a presence mask mixes the sources; missing sources are trained by source dropout.
- formulation: vector branch: frame-wise (dx,dy,dtheta) applied to all polyline points, regulariser lambda_t * ||T||^2 + lambda_r * dtheta^2 (lambdas not stated; the predictor of (dx,dy,dtheta) not stated). Cross-attention logit bias log(clamp(U, eps, 1)). Gate: softmax over sources of (L + log(pi + eps)), pi = presence mask. Raster branch: ResNet-18 + FiLM, pose regressed from raster + BEV reference, converted to a normalised affine (metres per pixel) and applied with grid_sample; zero-initialised residual fusion. SourceDropout p = 0.3 per encoder during training. 60 epochs, 8 x A800.
- input: BEV features of the host mapper (MapTRv2 / MapQR / VAD) + prior polylines / rasters.
- output: prior-augmented BEV features (+ polyline tokens with confidence).
- training: end-to-end with the host loss (mapping or VAD planning loss); no extra labels.
- differentiable: yes in form (grid_sample, softmax gate); the paper does not claim a gradient path for the vector correction.
- sensors_data: camera BEV; nuScenes, Argoverse2.
- results: nuScenes mapping MapTRv2 61.5 -> 67.4 mAP, MapQR 66.4 -> 71.7; Argoverse2 +4.1 mAP. Planning with VAD backbone (nuScenes val, average): L2 0.72 -> 0.42 m, collision 0.22 -> 0.12 %; vector encoder only 0.48 / 0.17, raster only 0.55 / 0.19; other priors in the same table: P-MapNet 0.68 / 0.66, PriorDrive 0.73 / 0.61. Test-time subsets (no retraining): SD raster only 64.5 mAP vs 63.5 for a single-prior model. No injected-noise experiment, no wrong-prior experiment (stated in the paper content I read).
- compute: MapTRv2 40.3 M params / 11.57 FPS -> 54.8 M / 10.98 FPS with UMPE on an RTX 3090; planning model latency not stated.
- code: https://github.com/Ethan-Zheng136/UMPE (not opened)
- fit_to_us: reuse_idea - effort S/M - take (1) presence-mask gate + source dropout for "missing map", (2) zero-init residual fusion, (3) a small SE(2) micro-alignment with grid_sample on the SD raster; the host BEV encoder is not ours.
- compat_needs: map tokens or raster with metric size, a place to add a residual (our SD raster tokens before the cross-attention).
- compat_conflicts: SE(2) correction is global per frame - cannot fix per-road lateral bias.
- tried_before: learned registration of the SD raster (E40 / E48) kept; "no map" training mode: not run as an explicit trained mode.
- verified: partly (paper vague about the correction network; gradient path unconfirmed)

### CARD C03 - NavMapFusion: SD-map cross-attention with noise / wrong-map training, discrepancy = diffusion noise
- papers: NavMapFusion: Diffusion-based Fusion of Navigation Maps for Online Vectorized HD Map Construction (Monninger et al., WACV 2026) https://arxiv.org/abs/2512.03317 ; code https://github.com/tmonnin/navmapfusion
- depth: M
- stage: S9, T
- idea: A diffusion decoder denoises queries into map elements; the SD map enters via a Map Cross-Attention before the sensor cross-attention, and the paper views the mismatch between map and sensor as the noise the diffusion is supposed to remove. The practical value is its systematic robustness study: translation noise, mis-aligned maps, element dropout, wrong location, wrong frame.
- formulation: 6 decoder layers; BEV grid 100 x 50; BEV dropout d = 0.30 (zeroed BEV grid, to avoid over-reliance on sensors); AdamW 2e-4, cosine, 24 epochs; k denoising steps (k = 1 / 3 / 5), final k = 5, eta 0.5, tau 0.5.
- input: BEV features + SD map (OSM) features; random queries.
- output: vectorised HD map elements.
- training: standard detection loss with perturbed SD map; training perturbations with probability p (e.g. 10 %).
- differentiable: yes (no matching on the map side).
- sensors_data: camera; nuScenes (A6000, 100 x 50 BEV).
- results: mAP 26.6 aligned; Gaussian translation noise hurts beyond sigma = 1 m, training with sigma = 1 m keeps performance up to test sigma = 3 m (values only in a figure). Aligned- vs unaligned-trained: 26.6 / 26.0 vs 26.8 / 26.5. Wrong map: random location (p = 1) 26.6 -> 20.1; other frame's map (p = 1) -> 17.8; training with p = 0.1 limited this to 24.4 / 24.0 (still above the 22.4 MapDiffusion baseline, no-map reference); element dropout above 0.7 falls below the no-SD-map reference, 10 % dropout in training most robust. FPS 14.7 (k = 1), 10.3 (k = 3), 8.1 (k = 5).
- compute: A6000 FPS above; params not stated.
- code: https://github.com/tmonnin/navmapfusion (not opened)
- fit_to_us: reuse_idea - effort S - take the training recipe: sigma = 1 m shift noise, 10 % element dropout, p = 0.1 "wrong map" (other frame / random place) samples; and report the same sweep for route success.
- compat_needs: SD map available as random-access tokens, ability to swap in another frame's map in the data loader.
- compat_conflicts: diffusion decoder itself is not needed.
- tried_before: map offsets 0 - 3 m in route-head training and evaluation (yes); explicit wrong-map and element-dropout training samples: not recorded as run.
- verified: partly

### CARD C04 - SD-RouteFusion: route-led vs perception-led trajectory with a learned gate
- papers: SD-RouteFusion: Ego-Trajectory Prediction with SD-Map Route Conditioning (Voloshyn et al., FUSION 2026) https://arxiv.org/abs/2607.01139 ; code https://github.com/zenseact/SD-RouteFusion
- depth: M
- stage: S10, S9
- idea: Encode the SD route (OSM route between start and end, sequence of 2D points, re-centred to the first point) with a small MLP to a 64-d vector; two mirrored cross-attention branches predict a route-led and an image-led trajectory; an MLP gate reads both embeddings and the difference of the two predictions and selects ONE branch (hard choice).
- formulation: gate logit trained with BCE against a soft target derived from the per-sample L2 errors of the two branches, scaled by a temperature (loss weights not stated). Route built by matching OSM links between GNSS start and end with a 20 m search radius growing in 50 m steps to 1 km; straight-line fallback for "no route".
- input: front camera, kinematics, route points.
- output: 8 s trajectory.
- training: supervised on logged trajectories; end-to-end; gate supervised by relative branch errors.
- differentiable: partial - the final selection is a hard branch choice (gate itself is trained with a soft target).
- sensors_data: front camera; 480 k private scenarios in 10 European countries + US (not public); public ZOD for reproduction.
- results: 8 s horizon ADE: image+kinematics 2.19, +route MLP fusion 1.96, SD-RouteFusion 1.82 (FDE 5.58 / 4.51 / 4.32). Image branch alone 2.59, route branch alone 1.91. Gate chose the route branch in 66 % of test scenarios. No controlled corruption experiment (natural corruption only: drift, outdated maps, connectivity errors), qualitative examples plus one failure case.
- compute: not stated.
- code: https://github.com/zenseact/SD-RouteFusion (released per paper; not opened)
- fit_to_us: reuse_idea - effort S - a gate that decides "trust route vs trust perception" is the closest published thing to abstention; supervise it with the error difference of two planner passes (with and without route).
- compat_needs: two planner passes (route masked / route given) - our planner has 93 k parameters so the cost is small.
- compat_conflicts: route is a global point list, not a lane attribute; hard branch choice blocks gradient through the selection.
- tried_before: none (planner loss into route head E119 no gain is a different thing).
- verified: partly (gate internals only as extracted)

### CARD C05 - D2HDMap: noise-aware training of a map-prior model with translation and rotation errors
- papers: D2HDMap: Non-visible Driveline Map Prior for Online Vectorized HD Map Prediction (2026) https://arxiv.org/abs/2606.20725 (authors not extracted)
- depth: M
- stage: T, S9
- idea: Lane drivelines (centre-lines) as a non-visible prior rasterised and used to initialise map queries; the useful part is the clean table of pose-noise training vs test.
- formulation: noise = zero-mean Gaussian translation (sigma 0 / 0.5 / 1 / 2 m) and rotation (0 / 2.5 / 5 deg), the same perturbed pose used for prior retrieval and for accumulating the temporal map. Whether the training noise set equals the test set is unclear in the text.
- input: BEV + rasterised driveline prior; output: vector map. Training: detection loss with noisy prior; regime matching_free is not the case (host uses DETR matching).
- differentiable: yes.
- sensors_data: nuScenes geosplit, Argoverse 2; clean mAP 44.8.
- results: robust model test mAP: 0.5 m 38.8; 1.0 m 36.0; 2.0 m 31.8; 2.5 deg 37.3; 5 deg 34.7; 2 m + 5 deg 32.3. Robust vs non-robust: +3.86 mAP average, +6.9 at 2 m / 5 deg (32.3 vs 25.4); mean loss 14.2 -> 10.3, worst case -19.4 -> -13.1. No misalignment estimation.
- compute: not stated. code: none found (no link in the content).
- fit_to_us: baseline/reuse_idea - effort S - note that mAP falls roughly linearly with shift even with noise training: shifts are absorbed only partly; our route success loses 0.03 per metre which is a much flatter curve.
- compat_needs: none. compat_conflicts: none.
- tried_before: map offsets 0 - 3 m: yes.
- verified: partly

### CARD C06 - Uni-PrevPredMap: tri-mode training (no prior / temporal prior / both) and perturbation table
- papers: Uni-PrevPredMap: Extending PrevPredMap to a Unified Framework of Prior-Informed Modeling (2025) https://arxiv.org/abs/2504.06647 ; code https://github.com/pnnnnnnn/Uni-PrevPredMap
- depth: M
- stage: T, S9, S8
- idea: One model that runs with no prior, a temporal prior, or a map prior plus a temporal prior; the training batch samples the three modes with fixed ratios so "missing prior" is a normal operating mode. Priors are rasterised heatmaps concatenated to BEV features and used to initialise queries.
- formulation: ratio non-prior : temporal : both = 0.5 : 0.3 : 0.2. Training perturbation: instance displacement +-6 m. Test table (nuScenes mAP): perfect map 85.6; instance displacement +-3 / 6 / 9 m 81.9 / 80.9 / 80.3; instance addition 85.2; deletion 82.5; frame displacement +-3 / 6 / 9 m 77.0 / 75.7 / 74.6; frame rotation +-15 deg 77.0; frame scale 0.8 - 1.2 77.0; empty map 74.0.
- training: standard detection loss; the "outdated prior" test mirrors a map about the trajectory start (qualitative only: predictions fall back toward the empty-prior behaviour).
- differentiable: yes. sensors_data: nuScenes (+ Argoverse2), camera.
- results: see formulation; FPS (A6000): no prior 14.2, temporal 12.2, map 13.1, both 11.5.
- code: https://github.com/pnnnnnnn/Uni-PrevPredMap (not opened)
- fit_to_us: reuse_idea - effort S - add a mode-sampling training scheme (no map / SD map / SD map shifted) for the route head and report "empty map" as the floor; their frame displacement is nearly the empty-map result, i.e. a frame-level shift of 3 - 9 m is almost as bad as no map.
- compat_conflicts: none. tried_before: none (explicit no-map mode).
- verified: partly

### CARD C07 - Bateman et al.: synthetic vs real map change gap
- papers: Exploring Real World Map Change Generalization of Prior-Informed HD Map Prediction Models (Bateman et al., CVPRW 2024) https://arxiv.org/abs/2406.01961
- depth: M
- stage: T, E
- idea: Prior tokens replace queries in a MapTR-style model; trained with synthetic prior perturbations, then tested against a real 3-year-old map version (2020 prior, 2023 truth). Reports which synthetic perturbation best predicts real robustness and shows the gap remains.
- formulation: discrete perturbations (feature dropout, duplication, wrong class), continuous (control point perturbation, location perturbation, global rotation / shift, Perlin warp). Training 75 k steps on 32 A100.
- results: mAP no-noise baseline 0.8980 synthetic / 0.8239 real; "Low All Noise" 0.9934 / 0.8571 (gap ~0.136). Best single settings on real data: localisation noise 0.5 m / 0.5 deg 0.8648, control-point 0.5 m 0.8640, Perlin warp 1.0 m 0.8623, duplicate p = 0.5 0.8604. Heavily corrupted priors can be worse than no prior; for complex real changes (new medians, construction) the model tends to copy the prior.
- data: Houston + Mountain View, 13.7 k training scenes, 1,240 real-change holdout scenes; private, so not reproducible by us.
- compute / code: not stated / none found.
- fit_to_us: reuse_idea - effort S - a warning for the metric: synthetic map offsets overstate robustness; keep a "copy-the-map" check (does the route head follow the map when the lanes disagree?).
- compat_conflicts: none. tried_before: map-agreement monitor (label-free) is related.
- verified: partly

### CARD C08 - RTMap: localisation against a prior map + change detection inside one network
- papers: RTMap: Real-Time Recursive Mapping with Change Detection and Localization (Du et al., ICCV 2025) https://arxiv.org/abs/2507.00980 ; code announced https://github.com/CN-ADLab/RTMap (availability not verified)
- depth: M
- stage: S9, S8
- idea: Prior-map elements are queries that are either "map" (still valid) or "fake" (outdated) at training time using ground-truth change labels; at inference outdated elements show up as low category confidence. Pose against the prior map is obtained either by an end-to-end delta-pose MLP (smooth L1) or by an LM least-squares step on matched vertices with a Mahalanobis residual.
- formulation: per-vertex Laplace density with NLL + Manhattan loss, weights 0.03 and 5.0 (lambda1, lambda2) over 20 vertices per element; Hungarian matching for new elements.
- differentiable: partial - MLP pose yes; LM pose reported as better but gradient through the solver not stated; Hungarian in the detection loss.
- sensors_data: camera, nuScenes + TbV-style data.
- results: TbV lateral error mean / 90th 0.125 / 0.256 m (with map queries) vs 0.163 / 0.318; nuScenes (LM pose) lateral 0.121 m, longitudinal 0.586 m, yaw 0.368 deg; change detection mAcc 57.4 vs 54.1 (changed class 40.0 -> 48.9). FPS: not reported in the content.
- fit_to_us: reuse_idea - effort M - the "map query confidence = is this map element still true" idea fits as a per-road validity output of the route head; the LM / MAP pose step is a template for per-road lateral bias with covariance.
- compat_conflicts: needs Hungarian for new elements (we avoid it).
- tried_before: label-free map-agreement monitor (project) is the same goal without labels.
- verified: partly

### CARD C09 - SMERF: SD-map transformer encoder fused into BEV by cross-attention
- papers: Augmenting Lane Perception and Topology Understanding with SD Navigation Maps (Luo et al., ICRA 2024) https://arxiv.org/abs/2311.04079 ; code https://github.com/NVlabs/SMERF (mmdet3d 1.0.0rc6, Python 3.8, PyTorch 1.9.1 per README)
- depth: M+C (README)
- stage: S9
- idea: Polylines of the OSM roads around the GPS pose are sampled at N = 11 points, encoded with sinusoidal embeddings (d = 32, T = 1000, coordinates normalised to (0, 2 pi)) plus one-hot road type, passed through a 6-layer transformer encoder, and cross-attended after each spatial cross-attention step of a BEV lane-topology model; trained end-to-end without extra labels.
- results (OpenLane-V2 subset A, OLS): baseline 30.2 -> 34.8; TopoNet 34.5 -> 39.4 (DET_l 28.2 -> 33.4); gains largest far away and at intersections; on a geographically disjoint split smaller (TopoNet 21.7 -> 23.4). No pose-noise, no wrong-map experiment, latency not reported.
- differentiable: yes. sensors_data: camera, OpenLane-V2.
- fit_to_us: reuse_idea - effort S - confirms the cross-attention graph/BEV <-> SD tokens pattern (same as E104b); nothing new for matching.
- tried_before: cross-attention to SD raster tokens (E104b) kept.
- verified: partly

### CARD C10 - P-MapNet: SD raster tokens with BEV-query cross-attention against weak alignment
- papers: P-MapNet: Far-seeing Map Generator Enhanced by both SDMap and HDMap Priors (Jiang et al., RA-L 2024) https://arxiv.org/abs/2403.10521
- depth: M
- stage: S9, B
- idea: Rasterised OSM is encoded by a CNN with sine position embedding into tokens; BEV features are down-sampled by d = 4 (50 x 25 grid) to queries; two cross-attention layers let each BEV query find the best-aligned road token, which is the paper's answer to weak GPS alignment. An optional MAE-pretrained HD-map refiner (mask ratio 50 %) is attached.
- results: nuScenes 120 x 60 m C+L mIoU: HDMapNet 49.50, P-MapNet SD 60.20, SD+HD 61.77; cross-attention 60.20 vs concatenation 54.73 / CNN concat 56.27; excluding OSM "service" roads +about 2 mIoU; at 240 x 60 m (camera) 26.77 -> 45.50. FPS 21.2 (HDMapNet), 18.7 (SD), 9.6 (SD+HD) on RTX-class GPU. No pose-noise experiment; paper only discusses misalignment qualitatively.
- differentiable: yes. code: project page https://jike5.github.io/P-MapNet/ (from abstract, not opened).
- fit_to_us: reuse_idea - effort S - same pattern as ours; useful numbers: two cross-attention layers best, four overfit; drop low-value road classes from the SD raster.
- tried_before: E104b cross-attention kept.
- verified: partly

### CARD C11 - SDTagNet: point-level SD-element tokens with text tags and element identifiers
- papers: SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction (Immel et al., NeurIPS 2025) https://arxiv.org/abs/2506.08997 ; code https://github.com/immel-f/SDTagNet
- depth: M
- stage: S9
- idea: Every SD point is a query = sin/cos position + OSM-tag embedding (compact BERT, dim 144, contrastively pretrained) + orthogonal random-feature element identifiers that tie points to their polyline; resample polylines to 10 points; fuse with an extra cross-attention in the MapTRv2 decoder.
- results (mAP): Argoverse 2 far range (120 x 60 m) 13.0 -> 18.9 (+5.9); P-MapNet best SD baseline 15.7; near range 46.5 -> 48.1; nuScenes far 3.9 -> 8.0, near 22.9 -> 23.3. 42.12 M -> 50.54 M params, 16.0 -> 14.5 FPS (H100), < 7 ms extra.
- robustness: authors say position-noise and element-drop augmentation WORSENED performance in their setting; they note SD/HD discrepancies (missing tunnels / service roads) and wrong one-way directions (attributed to missing point order).
- differentiable: yes. fit_to_us: reuse_idea - effort S - element identifiers (all points of one road share a code) are a cheap way to let attention know which vertices belong to the same SD road; that is exactly the pooling unit that C01 uses. Caution: a counter-result to the "always augment with noise" rule.
- tried_before: none. verified: partly

### CARD C12 - Navigation-path target points for end-to-end planning (VAD-based)
- papers: Utilizing Navigation Paths to Generate Target Points for Enhanced End-to-End Autonomous Driving Planning (NTT, 2024) https://arxiv.org/abs/2406.08349
- depth: M
- stage: S10, S9
- idea: A rough Google Maps route is interpolated to 5 m spacing, the 11 points nearest the ego (about 50 m) are encoded as [d_i, cos h_i, sin h_i] by a 2-layer VectorNet-style encoder; a target-generation module scores dense candidates on reconstructed lane centre-lines and in front of the ego by cross-attention over scene tokens; the best candidate is added to the route feature as driving-intent query for a 6-waypoint decoder (every 0.5 s, 3 s).
- design for meter-level error: models mainly the DIRECTION of the route (difference vectors + heading), because the route may deviate by metres.
- results (nuScenes val): average L2 0.61 m / collision 0.20 % vs VAD-Base 0.72 / 0.22; turning scenes L2 1.40 vs 1.88 m (collision 0.47 vs 1.09 %); navigation command variant (target + command) 1.86 / 1.28 % worse than target + learned embedding; target module ablation 0.63 -> 0.61 L2, 0.71 -> 0.20 % collision. No noise test of the path.
- differentiable: partial - the target is the argmax of candidate scores (selection). Code: none found.
- fit_to_us: reuse_idea - effort S - confirms "candidate scoring on the lane centre-lines" as the consumer-side use of route information; our consumer reads the route probability per edge which is the lane-graph version of this.
- tried_before: planner consumer E117 (0.34 m at 20 m). verified: partly

### CARD C13 - GoalFlow: goal point scoring and the shadow trajectory as a reliability check
- papers: GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation (Xing et al., CVPR 2025) https://arxiv.org/abs/2503.05689 (HTML v6 read) ; code https://github.com/YvanYin/GoalFlow
- depth: M
- stage: S10, S4
- idea: A vocabulary of goal points (K-means of training end points x, y, heading; 4096 or 8192 centres) is scored by distance and a drivable-area (DAC) score; the best goal conditions a rectified-flow trajectory generator. A "shadow trajectory" is generated with the goal masked; if it deviates strongly from the goal-conditioned one, the goal is treated as unreliable and the shadow trajectory is output (no threshold given, no quantitative ablation).
- formulation: goal CE losses weights 1.0 (distance) and 0.005 (DAC); noise sigma 0.1 best; conditioning randomly masked in training (classifier-free style).
- results: NAVSIM PDMS 90.3; with ground-truth end point as goal 92.1; human 94.8; TransFuser 84.0. Inference per sample: 20 steps 177.8 ms (PDMS 89.9), 10 steps 92.4 ms, 5 steps 49.0 ms (90.3), 1 step 10.4 ms (88.9). No wrong-goal ablation.
- differentiable: partial - goal is an argmax from a scored vocabulary at inference.
- fit_to_us: reuse_idea - effort S - the shadow pass (condition dropped at training time, compare outputs at test time) is a label-free monitor of "does the route change the plan" for our planner; the GT-goal gap (90.3 -> 92.1) shows the value an oracle route could give.
- tried_before: oracle route 0.316 m path error (E119 note) - same kind of upper bound.
- verified: partly (no command usage in GoalFlow; goal chosen from scene features)

### CARD C14 - PLUTO: reference lines from the route lane graph as lateral queries
- papers: PLUTO: Pushing the Limit of Imitation Learning-based Planning (Cheng et al., 2024) https://arxiv.org/abs/2404.14327
- depth: M
- stage: S10, L
- idea: Lanes within a radius are chained by depth-first search along lane connectivity into reference lines (truncated to a uniform length, resampled); embedded reference lines become lateral queries, combined with learnable longitudinal queries into a grid, factorised self-attention along each axis, cross-attention to scene. A separate head decodes one trajectory without a reference line. The imitation loss projects the GT end point to the nearest reference line and selects the matching longitudinal query; contrastive learning uses positive augmentations (state perturbation, non-interactive agent dropout) and negative ones (leading agent dropout / insertion, traffic-light inversion).
- results: nuPlan Val14 score 93.21 (R-score 92.06) vs PDM-Closed 93.08, PlanTF-H 89.96, expert 93.68; without post-processing 89.04. Runtime not reported in the paper content (a secondary table elsewhere lists 250 ms - not verified). Wrong route not addressed.
- differentiable: yes for the network; post-processing is a rule.
- fit_to_us: reuse_idea - effort M - route as chains of lane centre-lines used as queries is the lane-attribute interface; our graph already gives chains; a "no reference line" fallback head is the abstention path.
- tried_before: none. verified: partly

### CARD C15 - Diffusion Planner: route lanes as an AdaLN condition
- papers: Diffusion-based Planning for Autonomous Driving with Flexible Guidance (Zheng et al., ICLR 2025) https://arxiv.org/abs/2501.15564 ; project page https://zhengyinan-air.github.io/Diffusion-Planner/ (no repo link in the content)
- depth: M
- stage: S10
- idea: Route = K lanes with P points each (coordinates only), encoded by an MLP-Mixer to Q_n, added to the diffusion timestep embedding and applied through adaptive layer norm to all tokens; lanes are 70 polylines x 20 points, 12 features per point.
- results: nuPlan Val14 89.87 / 82.80 (non-reactive / reactive, learning-based, no refinement), Test14-hard 75.99 / 69.22; with refinement Test14 94.80 / 91.75. About 20 Hz on an A6000 (0.04 s in one table, 0.05 s in the limitations section - the paper is inconsistent). 3 blocks, hidden 192, 6 heads, 10 denoising steps; params not stated. Wrong route not addressed.
- fit_to_us: reuse_idea - effort S - global route embedding via AdaLN is an alternative to per-edge route attributes; they differ in whether the planner can tell WHICH lane is the route.
- differentiable: yes. tried_before: none. verified: partly

### CARD C16 - GC-PGP: goal / route as soft or hard mask on lane-graph edges (older anchor, 2023)
- papers: From Prediction to Planning With Goal Conditioned Lane Graph Traversals (Janjos et al., ITSC 2023) https://arxiv.org/abs/2302.07753 ; project page https://mh0797.github.io/gc-pgp/
- depth: M
- stage: S9, S10
- idea: PGP samples traversals on a lane graph; the route (nodes on a path from the vehicle to the goal) modifies edge probabilities: soft mask = learnable additive bonus beta on on-route edges, hard mask = off-route edges get probability zero (inference only, or also in training).
- results (nuPlan only, no nuScenes): open-loop intersection split GC-PGP 0.74 vs PGP 0.56 vs Urban Driver 0.83; closed-loop 0.46 vs 0.37 vs Urban Driver 0.58 vs IDM/MOBIL 0.75; plan instability 2.80 m vs 3.65. Some nuPlan route labels were wrong; the hard mask applied during TRAINING was much less robust to such labelling errors.
- differentiable: soft mask yes; hard mask is a rule. code: project page claims code (not opened).
- fit_to_us: reuse_idea - effort S - route as an additive bias on edge probabilities (soft), hard only at inference; directly applicable to our consumer (E117 reads route probability per edge).
- tried_before: none. verified: partly

### CARD C17 - PDM-Closed: centre-line of the route lane sequence + simulation scoring (older anchor, 2023)
- papers: Parting with Misconceptions about Learning-based Vehicle Motion Planning (Dauner et al., CoRL 2023) https://arxiv.org/abs/2306.07962
- depth: M
- stage: S10, E
- idea: Graph search finds the lane sequence along the route, its centre-line feeds IDM for the longitudinal profile; proposals at 5 speeds x 3 lateral offsets (-1, 0, +1 m), 4 s at 10 Hz, LQR + bicycle simulation, best proposal extended to 8 s, emergency brake. PDM-Hybrid adds a learned offset MLP beyond 2 s.
- results (Val14, CLS-R / CLS-NR / OLS / ms): PDM-Closed 92 / 93 / 44 / 91 ms; PDM-Hybrid 92 / 93 / 84 / 96 ms; without lateral offsets CLS-R 89 at 55 ms. Route / centre-line errors not analysed; no lane changes (collision risk) and needs HD maps.
- differentiable: no (rule + simulation). 
- fit_to_us: baseline - effort S - the strong "route centre-line is nearly the plan" prior: a baseline planner for our route-labelled graph would be: take the lane chain with highest route probability, offset it by learned residual; our 0.34 m path error should be compared with it.
- tried_before: oracle route planner reference 0.316 m (project note). verified: partly

### CARD C18 - Hidden Biases: the target-point shortcut
- papers: Hidden Biases of End-to-End Driving Models (Jaeger, Chitta, Geiger, ICCV 2023) https://arxiv.org/abs/2306.07957 (older anchor)
- depth: M
- stage: S10, T
- idea: Planners that get target points (GNSS points about 30 m apart on lane centres) steer towards the nearest point and so recover from errors, but can cut turns when the point is far; the decoder puts about 25 % of its attention on the target-point token (16.5 x uniform).
- results (CARLA): TransFuser with target points 39 DS / 84 RC with no route deviations, with discrete commands 32 DS / 56 RC; AIM 86 RC (target points) vs 50 RC (commands). Mitigations: transformer-decoder pooling instead of global average pooling (RC 84 -> 93, static collisions halved), shift / rotation augmentation (about 5 x fewer static collisions), predicting path + classified target speed.
- fit_to_us: reuse_idea - effort S - risk for our consumer: it may rely on the route probability instead of the geometry; use route-attribute dropout and shifted-map augmentation.
- differentiable: yes. tried_before: none. verified: partly

### CARD C19 - Route as lane-union BEV channel: CaRL and direct flow-matching policy
- papers: CaRL: Learning Scalable Planning Policies with Simple Rewards (Jaeger et al., 2025) https://arxiv.org/abs/2504.17838 ; Learning Direct Control Policies with Flow Matching for Autonomous Driving (Ceresini et al., ITSC 2026) https://arxiv.org/abs/2605.14832
- depth: M (CaRL: first 100k of 153k characters)
- stage: S10, S9
- idea: Both render the ROUTE as a lane-level mask (A* route lanes in CaRL; union of the lanes of the planned route in the second paper, pixel value encodes ego speed) in a BEV raster next to obstacles / drivable area. CaRL reports that the policy largely ignores lane markings and road channels and "abuses" the route mask for steering (lane changes at fixed places); drawing the route only inside intersections cut these crashes but sometimes caused wrong turns there.
- results: CaRL longest6 v2 DS 64 / RC 82 (PDM-Lite 73 DS, PlanT 62); input ablation: wider FoV and route only outside junctions removed DS 41 -> 31, RC 94 -> 63; about 8 ms per step (CARLA), about 14 ms (nuPlan), nuPlan Val14 CLS-NR 91.3. Flow-matching policy: closed loop in-distribution collision 3.75 %, drivable area 93.75 %, route progress 98.12 % (NFE = 10, 25 ms on an A6000; NFE = 1 gives 9.25 % collisions); no route perturbation in either paper.
- differentiable: policy yes (RL / flow matching); route mask is an input.
- fit_to_us: reuse_idea - effort S - this is "route as lane attribute" in raster form; same shortcut risk as C18. Code CaRL: https://github.com/autonomousvision/CaRL (not opened).
- tried_before: none. verified: partly

### CARD C20 - DiffusionDrive: anchored truncated diffusion planner (command only)
- papers: DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving (Liao et al., CVPR 2025) https://arxiv.org/abs/2411.15139 ; code https://github.com/hustvl/DiffusionDrive
- depth: M
- stage: S10, S4
- idea: K-means trajectory anchors (20 for NAVSIM, 18 for nuScenes) are diffused only 50/1000 steps; 2 denoising steps, cascade decoder with deformable BEV cross-attention and agent cross-attention, parameters shared across steps. The extracted page did not describe the use of the navigation command.
- results: NAVSIM 88.1 PDMS, 45 FPS on a 4090 (3.8 ms per step, 7.6 ms planning), about 60 M params; nuScenes L2 0.57 m, collision 0.08 %, 8.2 FPS.
- differentiable: yes except top-1 selection. fit_to_us: baseline - effort L - our consumer is 93 k parameters; take the idea of few-step anchored decoding only if we need multimodal paths.
- tried_before: none. verified: partly (command handling not seen)

### CARD C21 - VADv2 and SparseDrive: discrete command / navigation embedding in the planner
- papers: VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning (Chen et al.) https://arxiv.org/abs/2402.13243 ; SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation (Sun et al.) https://arxiv.org/abs/2405.19620 ; code https://github.com/hustvl/VAD , https://github.com/swc-17/SparseDrive
- depth: M
- stage: S10
- idea: VADv2: vocabulary of 4096 trajectories, navigation + ego state embedded by MLPs and added to the decoder output before a sigmoid per-action probability; CE against the closest demonstration + conflict loss. SparseDrive: three commands (left / right / straight), planning queries sized per command (N_cmd x K_p x C) from K-means intention points, 6 modes x 6 steps, winner-takes-all, selection filtered by the command and by a collision-aware rescore (collision 0.12 -> 0.08 %).
- results: VADv2 CARLA Town05 Long DS 85.1 / RC 98.4; 3DGS benchmark 125 ms, 0.40 B params (RTX 4090). SparseDrive-S: nuScenes L2 0.61 m, collision 0.08 %, 9.0 FPS, 85.9 M params; SparseDrive-B 0.58 / 0.06 %; VAD 0.72 / 0.21, UniAD 0.73 / 0.61.
- route: global command only; no route geometry, no SD map, no wrong-command test in either paper.
- fit_to_us: baseline - effort L - the contrast case: command-only interface; our lane-attribute interface carries strictly more information.
- differentiable: yes (except top-1). tried_before: none. verified: partly

### CARD C22 - Hydra-MDP: distilling non-differentiable rule scores into per-trajectory heads
- papers: Hydra-MDP: End-to-end Multimodal Planning with Multi-target Hydra-Distillation (Li et al., 2024) https://arxiv.org/abs/2406.06978 ; code (announced) https://github.com/woxihuanjiangguo/Hydra-MDP
- depth: M
- stage: L, S10
- idea: Every vocabulary trajectory (K-means of 700 k nuPlan trajectories, 4096 or 8192) is scored offline by the simulator metrics (no-at-fault collision NC, drivable area DAC, TTC, comfort, ego progress EP); one prediction head per metric learns these scores with BCE, so no post-processing is needed at inference.
- results: NAVSIM navtest PDMS best single model 86.5 (PDM-Closed with GT perception 89.1), scaled variant 91.0; 8192 better than 4096. The page did not mention navigation commands; latency not reported.
- differentiable: yes (scores are learned heads).
- fit_to_us: reuse_idea - effort M - add a "route compliance" score (fraction of the path inside the route lanes, computable from labels) as a distilled head or auxiliary loss on the consumer.
- tried_before: none. verified: partly

### CARD C23 - NavigScene (and Geo-VLA): navigation text / map knowledge fused to a driving model
- papers: NavigScene: Bridging Local Perception and Global Navigation for Beyond-Visual-Range Autonomous Driving (Peng et al., 2025) https://arxiv.org/abs/2507.05227 ; Geo-VLA (arXiv Aug 2026) https://arxiv.org/abs/2608.21440
- depth: A/M (fusion path only)
- stage: S9, S10
- idea: NavigScene generates textual navigation guidance (Google Maps static images -> GPT-4o descriptions) and fuses the frozen VLM output (compressed by an MLP, concatenated with BEV features) before the task heads. Geo-VLA does the opposite: map semantics are internalised into the VLM backbone by training on Geo-QA (3,000 QA pairs) and NO map is used at inference; inputs there are front image, ego history, state and a high-level command (no route).
- results: NavigScene nuScenes open loop, SparseDrive average L2 1.01 -> 0.76 m, collision 32.48 -> 20.71 (units as in the paper); VAD 1.22 -> 0.99 m. Navigation errors not analysed; unclear how the guidance is obtained at inference. Geo-VLA NAVSIM PDMS ReCogDrive 90.8 -> 91.2, DynVLA 91.0 -> 92.1; no latency.
- differentiable: partial (VLM frozen). code: none found.
- fit_to_us: not_suitable - text / VLM interface and heavy compute; only the observation that planners gain a little from route context is reusable.
- tried_before: none. verified: partly

### CARD C24 - RAVE: correlation-volume flow alignment + decoupled pose solver + transformer refinement
- papers: RAVE: End-to-end Hierarchical Visual Localization with Rasterized and Vectorized HD map (2025) https://arxiv.org/abs/2503.00862
- depth: M (first 100k of 117k characters)
- stage: S9 (localisation), S5
- idea: FLORA estimates a BEV flow between misaligned map features and BEV features with a correlation volume and iterative refinement, projects the flow on a low-rank basis (rank 16) to suppress noise and warps the map features before fusion. DEMA solves yaw (polar / FFT features), then longitudinal and lateral offsets separately (GeM-pooled axis features, ZNCC), turning a product of hypotheses into a sum (96 % less pose-solver memory); POET (6 + 6 layer transformer) refines using vector-map queries and runs only when the DEMA peak probability is below 0.4.
- formulation: search +-2 m, +-1 m, +-2 deg with steps 0.4 m, 0.2 m, 0.4 deg (11 hypotheses per DoF); BEV +-15 x +-30 m at 0.15 m; NLL loss weight 0.1 per DoF + smooth L1 on POET; AdamW lr 1e-4, batch 8, 10 + 15 epochs.
- results: nuScenes MAE 0.132 m longitudinal, 0.128 m lateral, 0.392 deg; about 150 ms per frame on an RTX 3090. Code link: not found.
- differentiable: yes end to end (stage 2). sensors: six cameras, HD map.
- fit_to_us: reuse_idea - effort M - decoupled (yaw, then along, then across) 1-D scoring is a cheap alternative to a full 3-DoF volume; confidence-gated second stage is a template for "refine only if uncertain".
- compat_conflicts: needs a rasterised map and a BEV semantic mask of comparable resolution.
- tried_before: correlation-volume localisation (E103) dropped in favour of cross-attention; registration front end E74 kept.
- verified: partly

### CARD C25 - U-ViLAR: differentiable perception-to-map association with uncertainty
- papers: U-ViLAR: Uncertainty-Aware Visual Localization for Autonomous Driving via Differentiable Association and Registration (Li et al., ICCV 2025) https://arxiv.org/abs/2507.04503
- depth: M
- stage: S9
- idea: Similarity matrix between visual BEV and rasterised-map BEV features -> softmax association distribution (Gaussian soft target); a pixel-wise perceptual uncertainty map is concatenated so uncertain regions (occlusion borders) are relaxed; a decoder outputs independent x, y, heading bin distributions (Shannon entropy per DoF as uncertainty), a cost volume around the coarse pose is fused with the joint prior by a learnable weight, refined pose = softmax-weighted average.
- training: uncertainty-weighted reconstruction, CE to the Gaussian target, symmetric contrastive loss in local windows (weights not stated).
- results: nuScenes HD map multi-view MAE lateral 0.040 m, longitudinal 0.140 m, orientation 0.075 deg (BEV-Locator 0.076 / 0.178 / 0.510); OSM, nuScenes multi-view position recall@1m 34.53 % vs MapLocNet 20.10 %; 28 FPS on a V100, 15 FPS on Orin with TensorRT INT8. GPS noise simulated as +-2 m / +-2 deg (fine) or +-30 m / +-30 deg (relocalisation); no dedicated map-error study.
- differentiable: yes (softmax-weighted pose). code: none found in content.
- fit_to_us: reuse_idea - effort M - soft-argmax pose (weighted average over a cost volume) with an entropy-based uncertainty is a gradient-friendly pose block and gives a confidence output for abstention.
- tried_before: correlation volume localisation (E103) dropped. verified: partly

### CARD C26 - HOLO: homography-guided pose on SD (OSM) maps with local correlation lookups
- papers: HOLO: Homography-Guided Pose Estimator Network for Fine-Grained Visual Localization on SD Maps (2026) https://arxiv.org/abs/2601.02730
- depth: M
- stage: S9
- idea: BEV and OSM raster (roads, buildings) features go to a dense correlation volume; each of 6 refinement iterations crops local patches around the current projection (no re-computation of attention); the decoder predicts corner displacements, converted to a homography by DLT; x, y from projecting the BEV centre, heading from a projected auxiliary point.
- formulation: loss BCE semantic (weight 1000) + L2 translation (1) + L1 yaw (10); AdamW + OneCycle, batch 16, 180 k iterations. Training perturbation +-30 m and +-30 deg of the map patch.
- results: nuScenes recall@1 m 36.41 % (with drivable-area supervision), 61.21 % @2 m, 73.74 % @1 deg; road-only SD variant 30.40 % @1 m; MapLocNet 20.10 / 45.54; stress test +-40 m still 18.24 % @1 m, 76.87 % @5 m. 25.6 FPS with 2 iterations, 19.9 FPS with 6 (RTX A6000). Differentiability of the DLT step not stated explicitly (supplement mentions a differentiable homography estimator). Code announced, not available.
- fit_to_us: reuse_idea - effort M - RAFT-style local correlation lookup around the current estimate is cheaper than a full volume.
- tried_before: E103 (correlation volume) was not better than cross-attention. verified: partly

### CARD C27 - MapLocNet: coarse-to-fine transformer registration to a navigation map (regression head)
- papers: MapLocNet: Coarse-to-Fine Feature Registration for Visual Re-Localization in Navigation Maps (Wu et al., 2024) https://arxiv.org/abs/2407.08561
- depth: M
- stage: S9
- idea: Rasterised OSM patch (128 x 128 m, 0.5 m) -> U-Net / VGG-16 map features; BEV (LSS) and map features are fused (7 x 7 conv, 4 x down-sampled), 3-layer transformer encoder + 3-layer MLP regress a 3-DoF offset (coarse), then the same on fine features; final pose = composition T2 T1. Smooth L1 on both stages + BCE segmentation, weights 1; training noise +-30 m and +-30 deg.
- results: nuScenes six cameras position recall @1 / 2 / 5 / 10 m 20.10 / 45.54 / 77.70 / 91.89; orientation @1 deg 58.61; single camera 8.96 / 27.05 / 64.57 / 85.86 vs OrienterNet 5.83 / 18.92 / 52.83; 22.9 M params, 56.15 GFLOPs, 18.2 FPS (V100) for six cameras, 24.4 FPS one-stage; OrienterNet 54.9 M and 8.1 FPS single camera. 200 epochs, about 48 h on 8 V100.
- differentiable: yes (regression head, no differentiable optimiser).
- fit_to_us: reuse_idea - effort S/M - plain regression of (dx, dy, dtheta) from fused tokens is the simplest differentiable registration head; accuracy (20 % within 1 m) shows regression on weak OSM is crude.
- tried_before: registration front end E74 / E48 kept. verified: partly

### CARD C28 - OrienterNet family (OrienterNet, OSMLoc, hierarchical re-localisation 2603.01613, DiffVL): exhaustive probability volume over pose
- papers: OrienterNet (Sarlin et al., CVPR 2023, older anchor) https://arxiv.org/abs/2304.02009 ; OSMLoc (2024) https://arxiv.org/abs/2411.08665 ; Uncertainty-Aware Hierarchical Re-Localization in OpenStreetMap (arXiv 2603.01613, v2 June 2026) https://arxiv.org/abs/2603.01613 ; DiffVL (2025) https://arxiv.org/abs/2509.14565
- depth: M
- stage: S9
- idea: Neural map F (U-Net on rasterised OSM, 50 cm/px, N = 8 channels) is correlated with a confidence-masked neural BEV T (32 x 32 m at 0.5 m) for every (x, y, theta); posterior P = softmax(M + Omega) with a learned per-cell location prior; rotations by batched multiplication in the Fourier domain (K = 64 train, 512 test); loss = NLL of the GT pose; multi-frame fusion by warping and multiplying per-frame volumes.
- results: OrienterNet MGL position recall @1 / 3 / 5 m 15.78 / 47.75 / 58.98; KITTI 20 s sequences lateral @1 m 79.71 %; about 94 ms per localisation (37 ms BEV + 51 ms matching, RTX 2080); multi-frame fusion roughly doubles recall; fails in repetitive scenes and where the map lacks distinctive elements. OSMLoc: MGL 15.30 / 46.05 / 59.40 vs 10.69 / 42.12 / 54.01, 9.27 FPS (4090). 2603.01613: coarse 1 m / 6 deg grid by FFT correlation, fine 0.5 m / 2 deg; uncertainty U = 1 - (peak mass in a neighbourhood) only sets the fine search window, NOT an abstain; 25.80 FPS vs OrienterNet 7.51. DiffVL: refinement ablation 0.1554 vs 0.0978 recall@1 m (KITTI); runtime not reported.
- differentiable: yes (volume + softmax; argmax only at readout).
- sensors_data: gravity-aligned camera, OSM; training prior offsets up to +-20 m.
- fit_to_us: reuse_idea - effort M - the volume is too heavy for 31 ms with our 200 x 200 grid, but the loss (NLL of the true pose in a small volume, e.g. across-road offsets only, 1-D) and temporal product of volumes are exactly the missing tools for per-road lateral bias.
- tried_before: correlation-volume localisation E103 dropped (accuracy near cross-attention, more cost).
- verified: partly

### CARD C29 - EgoVM: histogram pose solver with covariance on vectorised lane maps
- papers: EgoVM: Achieving Precise Ego-Localization using Lightweight Vectorized Maps (2023, older anchor) https://arxiv.org/abs/2307.08991
- depth: M
- stage: S9
- idea: Grid-search offsets (+-3 m in x, y; +-3 deg yaw): map elements are projected into the BEV, BEV features are bilinearly sampled and scored against map embeddings by a shared MLP; softmax -> posterior; pose = posterior mean, covariance = posterior spread.
- training: covariance-weighted RMSE on the offset + KL loss on the grid + KL on random poses (mixture of t / von Mises / uniform) + focal segmentation loss; weights not stated.
- results: nuScenes full mode longitudinal / lateral MAE 10.9 / 3.4 cm, yaw 0.089 deg; 120 ms average (RTX 3080). Initial-pose noise range not stated.
- differentiable: yes. code: none found.
- fit_to_us: reuse_idea - effort M - "bilinear sampling of BEV features at projected map points, shared MLP score" is the cheapest volume (cost scales with candidates x map points); it is the same sampling primitive as our trace network (grid_sample on the pyramid).
- tried_before: E103-like volume dropped. verified: partly

### CARD C30 - SuperGlue: Sinkhorn optimal transport with a dustbin
- papers: SuperGlue: Learning Feature Matching with Graph Neural Networks (Sarlin et al., CVPR 2020, older anchor) https://arxiv.org/abs/1911.11763
- depth: M
- stage: S6, S9 (assignment)
- idea: Score matrix S_ij = <f_i^A, f_j^B>, extended by one learnable scalar z in an extra row and column (dustbin); marginals [1_M, N] and [1_N, M]; Sinkhorn with T = 100 iterations; dustbin dropped afterwards; unmatched points get a probability mass in the dustbin, which is the abstention mechanism.
- loss: NLL of ground-truth matches plus NLL of the dustbin entries of unmatched points; keypoints with ambiguous status are left unlabelled. Attentional GNN of 9 self / cross layers (4 heads, d = 256), about 12 M parameters (appendix); about 69 ms per pair on a GTX 1080; test threshold 0.2.
- differentiable: yes (Sinkhorn is differentiable; T = 100 is heavy).
- fit_to_us: reuse_idea - effort S - add ONE dustbin column ("no SD road / off-map") to the vertex -> road softmax with an NLL for lanes that are not on any map road, and a dustbin row for map roads with no perceived lane; two or three Sinkhorn iterations are enough for a (vertices x roads) matrix of a few hundred entries (my judgement, not tested by the paper).
- compat_needs: unmatched labels (lanes without a road, roads without a lane) - we have them from the map expansion.
- compat_conflicts: our losses avoid Hungarian matching; Sinkhorn is not Hungarian but needs one-to-one marginals, whereas road-to-lane is many-to-one (C01) - use dustbin + softmax, not full OT.
- tried_before: none. verified: partly

### CARD C31 - LightGlue: matchability scores instead of Sinkhorn, early exit and pruning
- papers: LightGlue: Local Feature Matching at Light Speed (Lindenberger et al., ICCV 2023, older anchor) https://arxiv.org/abs/2306.13643 ; code https://github.com/cvg/LightGlue
- depth: M
- stage: S6, S9
- idea: Per-point matchability sigma_i = sigmoid(linear(x_i)); assignment = sigma_i sigma_j * softmax_row * softmax_col ("double softmax", no Sinkhorn); a match needs score above tau = 0.1 and mutual best; per-layer confidence head (not back-propagated into the states) allows early exit once 95 % of the points are confident; confident low-matchability points are pruned (beta = 0.01); 9 layers, 4 heads, d = 256.
- results: relative pose MegaDepth-1500 44.2 - 44.5 ms vs SuperGlue 70.0 ms, adaptive 31.4 ms; adaptive depth / width about 33 % faster overall.
- differentiable: yes (thresholding only at inference).
- fit_to_us: reuse_idea - effort S - a sigmoid "has a road on the map" head per vertex (matchability) plus a double softmax is the cheapest abstain-capable assignment; the per-layer exit confidence is not worth it at our size.
- tried_before: none (validity / keep gate E120 exists for vertices, not for the map assignment). verified: partly

### CARD C32 - Differentiable weighted Procrustes (Deep Global Registration)
- papers: Deep Global Registration (Choy, Dong, Koltun, CVPR 2020, older anchor) https://arxiv.org/abs/2004.11540
- depth: A (abstract page; how the weights are produced and the gradient proof were not on the page)
- stage: S9 (registration step)
- idea: A network predicts a confidence per correspondence; a closed-form weighted Procrustes solves the rigid transform; gradients flow through the weights (not through coordinates), linear in the number of correspondences; a robust gradient-based refinement follows. A search-engine summary of the paper states the weights-not-coordinates property and the linear cost (from the paper's text as quoted there); not verified on the page I opened.
- differentiable: yes in form. SE(2): not covered - no SE(2) variant seen anywhere; the 2-D closed form (2 x 2 SVD, or the atan2 closed form) would need to be re-derived by us (not claimed by any paper I read).
- fit_to_us: reuse_idea - effort S - vertex <-> map-road-point correspondences with soft weights from the assignment softmax give a per-frame or per-road (dx, dy, dtheta) in closed form; cost negligible.
- compat_conflicts: assumes point-to-point correspondences; lanes are curves (point-to-polyline residual, as in our E120, is the better residual).
- tried_before: none. verified: partly (A)

### CARD C33 - HMM lane-level map matching with lane markings (non-differentiable reference)
- papers: Driving with Context: Online Map Matching for Complex Roads Using Lane Markings and Scenario Recognition (Bi et al., 2025) https://arxiv.org/abs/2505.05007 ; code https://github.com/TRV-Lab/LMSR-OMM
- depth: M
- stage: S9 (classical baseline)
- idea: Hidden state = road segment; transitions follow road connectivity (direct successors vs via intermediate roads); observation factors: GNSS + lane markings associated with OSM roads (association probability max over sampled points, Gaussian distance and heading terms) + a scenario factor (elevated road / tunnel via detections and fuzzy inference); Viterbi online decoding; ICP of lane-marking points with a penalty for solid vs dashed mismatch.
- results: Zenseact Open Dataset F1 98.04 % vs 96.13 %; Shanghai multilevel roads 94.60 % vs 88.75 %; ablation baseline 66.32 -> scenario only 92.74, lane markings only 79.22. Runtime not reported.
- differentiable: no.
- fit_to_us: baseline - effort S - the temporal HMM prior over "which road am I on" (transition by road connectivity) is the classical form of our temporal accumulation; its log-odds form could be fused with the learned route score.
- tried_before: temporal consistency in route head: partial. verified: partly

### CARD C34 - Online map uncertainty handed to a downstream consumer (Laplace per vertex)
- papers: Producing and Leveraging Online Map Uncertainty in Trajectory Prediction (Gu et al., CVPR 2024) https://arxiv.org/abs/2403.16439 ; code https://github.com/alfredgu001324/MapUncertaintyPrediction ; related: Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention https://arxiv.org/abs/2407.06683
- depth: M
- stage: S10, L
- idea: The map head outputs, per vertex, a Laplace location mu and scale b per coordinate (NLL: log 2b + |v - mu| / b) plus class probabilities; mu, b and class probabilities are concatenated into the predictor's polyline encoder (sigma = sqrt(2) b when rotating). Related paper: skip the vector map and let agents attend to BEV patches (patch 20 x 20, about 6 m).
- results: HiVT + MapTR minADE 0.4015 -> 0.3854, minFDE 0.8418 -> 0.7909, MR 0.0981 -> 0.0834; DenseTNT + MapTR convergence 88 -> 44 epochs; not every pair improves (DenseTNT + MapTRv2-centerline MR 0.1520 -> 0.1593); mapping mAP within about 2 %. Direct BEV attention: up to 29 % better minFDE and up to 73 % faster inference (HiVT + MapTR 22.4 -> 9.1 ms, RTX 4090), trained not jointly.
- differentiable: yes (the uncertainty is a plain regression output).
- fit_to_us: reuse_idea - effort S - add per-vertex log-scale and the route probability / entropy to the consumer input; E117 already receives edges + route probability, so only the scale is new.
- tried_before: none for scale. verified: partly

## 3. Cluster compatibility notes

(a) Interface contracts in the literature.
- SD map in: (1) rasterised OSM -> CNN tokens -> cross-attention from BEV queries (P-MapNet, SMERF-like); (2) polyline tokens (sin/cos + type) -> transformer -> cross-attention (SMERF, SDTagNet, UMPE vector); (3) map-element queries (RTMap, Uni-PrevPredMap, D2HDMap). Ours: vertex tokens attend to SD raster tokens (E104b). All published SD-map integrations feed the map INTO the perception model; only ONR (C01) and the OMA benchmark treat association as an output between two graphs, and none of them is trained inside the perception net.
- Route to planner: (1) discrete command (VAD / VADv2 / SparseDrive / DiffusionDrive); (2) route points as a sequence (SD-RouteFusion, NTT); (3) route lanes as polylines / tokens (PLUTO reference lines, Diffusion Planner route lanes, PDM centre-line); (4) route as a BEV mask of lanes (CaRL, flow-matching policy); (5) route as a mask / bias on lane-graph edges (GC-PGP). Only (3) - (5) are lane-attribute interfaces like ours; (5) is the closest (soft additive bias on route edges).
- Localisation: all learned localisers consume a raster map and a BEV raster and output a pose or a pose distribution (RAVE, U-ViLAR, MapLocNet, HOLO, OrienterNet, EgoVM). They estimate ONE global SE(2) offset per frame. No published block estimates per-road lateral bias.

(b) Adapters and conflicts.
- Graph tokens -> map tokens: our cross-attention already is the adapter; the missing piece for ONR-style assignment is a road-level pooling (mean of vertex tokens per SD road, or per matched road) and a softmax over roads (+ dustbin).
- Pose block <-> route head: SE(2) via grid_sample (UMPE, C02) or 1-D across-road volume (OrienterNet-style) can be placed before the cross-attention; both are differentiable, both are global.
- Conflicts: Hungarian-based pieces (RTMap new elements, DETR hosts) conflict with our matching-free rule; hard route masks in training conflict with robustness (GC-PGP); hard branch choice (SD-RouteFusion gate) blocks gradients through the selection; the SDTagNet result that noise augmentation hurt is the only counter-example to noise training.

(c) Combinations demonstrated by the papers themselves.
- UMPE + VAD planning: prior-fused BEV features improve planning L2 0.72 -> 0.42 m, so SD-map benefit does reach the planner end to end (but through BEV features, not through a lane graph).
- SMERF / P-MapNet / SDTagNet + MapTR(v2) or TopoNet: cross-attention fusion.
- Map uncertainty + predictor (Gu et al.): yes, trained separately, then concatenated.
- Localiser + HD-map encoder trained end to end (RAVE, U-ViLAR): yes, pose loss through the BEV network.
- ONR + perception: association network trained separately from the mappers; evaluated on three perceived map sources, so it transfers across mappers (41 - 55 NR-F1) but at a clear loss against GT graphs (78 on GT).

## 4. Reuse versus re-implement

| item | reuse / re-implement | reason |
|---|---|---|
| MAT code (C01) | adapt only the idea | Pointcept-based, 34 ms alone; our route head is 0.69 M params and must fit a 31 ms frame |
| vertex -> road softmax with dustbin (C01 + C30 + C31) | reimplement, S | a few lines on top of E104b |
| presence-mask gate + source dropout (C02, C06) | reimplement, S | no framework dependency |
| perturbation recipe (C03, C05, C07) | reimplement, S | data-loader level |
| SE(2) micro-alignment via grid_sample (C02) | reimplement, S/M | we already use grid_sample everywhere |
| 1-D across-road NLL volume (C28, C29) | reimplement, M | tiny volume |
| weighted Procrustes (C32) | reimplement, S | closed-form, derive SE(2) ourselves |
| soft route bias on edges (C16) | reimplement, S | consumer change only |
| two-pass route-masked consumer (C04, C13) | reimplement, S | consumer is 93 k parameters |
| RAVE / U-ViLAR / HOLO / MapLocNet / OrienterNet | not code, ideas only | need image features or heavy volumes; no released code found for most |

## 5. Open questions and gaps

- No published block does per-road lateral bias; all localisers fit one global offset. Whether a per-road offset is identifiable from a few vertices is open (my reading of the literature, not a result).
- No paper trains a route-conditioned planner against deliberate wrong routes; the only abstain-like mechanisms are the SD-RouteFusion gate (natural corruption) and the GoalFlow shadow trajectory (no ablation). So there is no external number for "planner behaviour under a wrong route" to compare to.
- ONR evaluates association quality, not the planner; it does not test wrong roads or missing lanes quantitatively (appendix says only qualitatively); its NR-F1 drops from 78 on GT graphs to 41 - 55 on perceived graphs, i.e. perception errors dominate.
- Sinkhorn / OT between perceived and map lane graphs: not found in driving. Everything is softmax.
- Latency of the matching step is almost never reported in the same unit as ours; ONR (34 ms) and RAVE (150 ms) are the only explicit numbers for the matching stage.
- Real map change vs synthetic perturbation: gap remains (C07); our synthetic offsets share this limit.
- Depth limits: values come from summarised page extractions; appendix-only tables were not seen for ONR (params), UMPE (λ values), SD-RouteFusion (loss weights).

## 6. WHAT TO TAKE (concrete, tagged)

1. cheap - Add a dustbin column ("no SD road") to the vertex-to-road softmax of the route head and an NLL for lanes without a map road; optional dustbin row for roads without a perceived lane (C30, C31, C01). Gives an abstain output.
2. cheap - Pool vertex tokens per SD road and softmax over roads (ONR Eq. 4 style) next to the per-vertex route flag; add road-level element identifiers to the SD tokens (C01, C11).
3. cheap - Training modes: no map / SD map / shifted map (sigma = 1 m) / wrong map (other frame, p = 0.1) / element dropout 10 %, with presence-mask gate and source dropout 0.3 (C03, C02, C06); keep part of the batch clean (C07); report route success on "wrong map" and "empty map" as floor.
4. cheap - Soft additive route bias on edges in the consumer, hard masks only at inference; route-attribute dropout during consumer training against the target-point shortcut (C16, C18, C19).
5. cheap - Two-pass consumer (route given vs route masked), difference of the two paths as a label-free reliability signal; optionally a small gate trained with BCE on the error difference (C04, C13).
6. cheap - Hand per-vertex Laplace log-scale and route entropy to the consumer (C34).
7. medium - SE(2) micro-alignment of the SD raster via regressed pose + grid_sample with zero-init residual and L2 regulariser, in front of the cross-attention (C02, C27).
8. medium - Weighted Procrustes in SE(2) on soft vertex-to-road correspondences (point-to-polyline residual) to get a per-road (dx, dy, dtheta) and a covariance; use the output as extra token features (C32, C08, C29).
9. medium - 1-D across-road NLL volume (11 - 21 hypotheses, +-3 m) per matched road with temporal product of the volumes (log-odds fusion) to attack per-road lateral bias (C28, C29, C33).
10. medium - Distil a route-compliance score (share of the path inside route lanes) as an extra consumer head, Hydra-style (C22); add path-order loss (CTC-like, weight 0.01 in ONR) along route chains (C01).
11. medium - Joint fine-tune only with the consumer's route-masked / route-given pair and wrong-map samples; E119 (no gain) suggests the route loss alone is not the lever - the lever the literature offers is robustness training and abstention, not accuracy.
12. expensive - Full 3-DoF correlation / transformer pose stack (RAVE, U-ViLAR, HOLO) or MAT-sized association network: cost and accuracy do not beat our cross-attention at 31 ms (E103).

## 7. Search log (queries, WebSearch unless noted)

1. SMERF SD map encoder online HD map construction arXiv
2. P-MapNet far-seeing map generator SD map prior arXiv
3. navigation route conditioned end-to-end planner lane-level route SD map 2025 arXiv
4. BEV localization against OpenStreetMap map differentiable pose refinement camera 2025 arXiv
5. SMERF arXiv 2311.04079 title query
6. end-to-end driving navigation map OpenStreetMap route planning nuScenes 2025 NavigScene OR SD map planning transformer
7. MapLocNet coarse-to-fine feature registration BEV localization
8. differentiable Procrustes OR Sinkhorn lane graph matching perception map association
9. SuperGlue Sinkhorn dustbin learned feature matching
10. lane graph matching optimal transport OR Sinkhorn map element association HD map OpenLane-V2
11. detecting outdated or erroneous HD map vs online perception mismatch change detection
12. robustness of online HD map construction to SD map prior noise misalignment evaluation benchmark
13. NavigScene bridging local perception and global navigation
14. GoalFlow goal-driven flow matching
15. planner conditioned on route polyline lane centerline graph end-to-end NAVSIM OR nuPlan
16. end-to-end driving lane-level navigation route as input SD map route token planner
17. vectorized map localization online-constructed vector map to prior HD map matching
18. PLUTO OR PlanTF route lane encoding planner nuPlan
19. OpenStreetMap routing end-to-end driving policy conditioned on route map BEV wrong route
20. Hydra-MDP multimodal planning
21. VLA driving SD map navigation route input
22. differentiable Levenberg-Marquardt OR Gauss-Newton layer BEV localization
23. weighted Procrustes differentiable SE(2) registration learned correspondences
24. Producing and Leveraging Online Map Uncertainty in Trajectory Prediction
25. lane-level map matching navigation route with perceived lane detection
26. SatMap satellite maps prior misalignment (only listing read; SATMapTR / AerialFusionMapNet noted by the search summary but not opened - they report misalignment tests with Gaussian ego-pose noise and small mAP drops, unverified)
27. Bateman Exploring Real World Map Change Generalization
- WebFetch (page extractions): arXiv HTML / abs of 2507.07487 (two chunks), 2607.01139 (abs + html), 2605.02762 (abs + html), 2406.08349, 2311.04079, GitHub NVlabs/SMERF, 2403.10521, 2506.08997, 2503.00862, 2507.04503, 2304.02009 (abs; ar5iv html), 2601.02730, 2603.01613, 2407.08561, 2512.03317, 2606.20725, 2504.06647, 2507.05227, 2503.05689, 2306.07962, 2411.15139, 2402.13243, 1911.11763 (ar5iv), 2307.08991, 2408.08802 (PriorMapNet: opened, no SD / noisy prior, therefore no card), 2501.15564, 2404.14327, 2605.14832, 2306.07957, 2507.00980, 2504.17838, 2302.07753, 2306.13643, 2608.21440, 2406.06978, 2411.08665, 2509.14565, 2405.19620, 2403.16439, 2505.05007, 2407.06683, 2406.01961, 2004.11540 (abs), GitHub WallelWan/OMA-MAT. CVF PDF of Bateman: HTTP 403. The OrienterNet arXiv PDF exceeded the fetch size limit.
