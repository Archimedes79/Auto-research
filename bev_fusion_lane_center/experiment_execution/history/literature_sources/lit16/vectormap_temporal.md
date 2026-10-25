# Cluster VM - online vectorised HD-map construction and temporal propagation (loop 16 literature)

Date: 2026-10-23. Prefix VM. Reading only (WebSearch / WebFetch); nothing downloaded or cloned by me. The earlier abstract-level review `report_final/text/lit_p16_tree.md` was skimmed and its links reused (Q6 temporal rows, Sparse4D, StreamMapNet, SQD-MapNet, MapTracker, TopoStreamer, FASTopoWM, mAS).

## 1. Coverage statement

- Searched: nuScenes / Argoverse 2 online vector-map papers (single frame and streaming), temporal propagation in sparse 3D detectors, stability / consistency metrics, geographic splits, log-odds / Bayesian map fusion, 2025 - 2026 papers (found up to Oct 2026).
- Read at method level (depth M) through fetched web text (arXiv HTML, ar5iv). A small model extracted the answers from the page, so single numbers can be mis-transcribed; ambiguous ones are flagged in the cards. Papers: MapTR, MapTRv2, StreamMapNet, SQD-MapNet, MapTracker, TopoStreamer, FASTopoWM, PivotNet, BeMapNet, VectorMapNet, HDMapNet, Sparse4D v2 / v3, StreamPETR, SparseDrive, MapQR, GeMap, MGMap, PredMapNet, PrevPredMap / Uni-PrevPredMap, HRMapNet, HisTrackMap, MemFusionMap, InteractionMap, MapExpert, MapUnveiler, MambaMap, Neural Map Prior, DTCLMapper, MapTCL, AMap, SparseMeXt, MapVR, MapGR, AugMapNet, Mask2Map, DAMap, ADMap, MapLightning, GaussianMap, MapDiffusion, Mapping like a Skeptic, L2G-Map, Stability Under Scrutiny (mAS), Lilja et al. (geographic splits).
- Code pages read (depth C, no cloning): MapTR `decoder.py` and `maptr_assigner.py`, HDMapNet `postprocess/vectorize.py` and `data/dataset.py` (GitHub web view), repository front pages of MapTracker, HDMapNet, PivotNet. Other code URLs come from papers or search results and were not opened.
- Seen only as search hits, not read (depth A or less): MapExpert follow-ups, DiffSemanticFusion, OnlineBEV, FastMap, MachMap, PriorMapNet, UMPE, the HRDX dataset, Sparse4D v1; LaneGAP and TopoLogic belong to the topology cluster.
- Not found / not obtainable: an SQD-MapNet repository (paper: "soon"; its supplement PDF is not readable as text), numeric noise magnitudes of SQD-MapNet and SparseMeXt, a code link inside the PivotNet paper (the repository exists), PDFs above 10 MB (BeMapNet, VectorMapNet, HDMapNet were read through ar5iv instead). Five PDF fetches failed (the three large PDFs by size limit; the Neural Map Prior PDF and the SQD-MapNet supplement came back as binary and were not used; Neural Map Prior was then read through ar5iv). The fetch tool itself cached the two binary responses in its own tool-results folder under the Claude project directory; I did not open or use them.
- Not covered: re-runs, perspective-view-only variants, aerial road-graph methods (other clusters).

## 2. Cards

Reading guide (theme -> cards): temporal recursion on element sets and tokens VM01 - VM11; raster / feature memories and validity fusion VM12 - VM22; temporal consistency losses and distillation VM23 - VM25; matching, ambiguity, representation and loss VM26 - VM36; decoders that sit on a BEV raster VM37 - VM43; evaluation (stability, consistency, geographic splits) VM44 - VM46. Author lists and venues marked "from memory" were not on the pages read.

### CARD VM01 - StreamMapNet (top-k query propagation + warped BEV memory)
- papers: Streaming Mapping Network for Vectorized Online HD Map Construction (Yuan et al., WACV 2024) https://arxiv.org/abs/2308.12570
- depth: M
- stage: S8, S4
- idea: Carries the k most confident decoder queries of frame t-1 to frame t. Each carried query is updated by a residual MLP over [query, flattened 4x4 ego transform T]; its previous predicted polyline is mapped by T and becomes the initial reference points. In parallel a dense BEV memory: the previous BEV feature is warped by T and fused with the current BEV by a GRU (+ LayerNorm). "Multi-point attention" uses all N_p predicted points of the previous layer as reference points (not a single centre).
- formulation: k = 33 of Nq = 100 (rank-based, no threshold); the first decoder layer runs on the learned queries, after it the top Nq - k = 67 of them are kept and the 33 carried queries are concatenated. Reference points p' = (T [x y 0 1])_xy per point. Transformation loss: smooth-L1 between Reg(carried query) and the previous PREDICTED polyline mapped by T, summed over 20 points, no permutation minimum (identity-based, needs no GT); weight 0.1 (value from the follow-up papers MemFusionMap and MapTCL; the StreamMapNet text is ambiguous). BEV: B = LN(GRU(B_t, warp(B_{t-1}, T))). Main loss: focal (class, weight 5) + smooth-L1 line loss (weight 50) minimised over MapTR's permutation group. N_p = 20. Training: sequences randomly cut in 2 parts per epoch, no gradient into previous-frame memory, first 4 epochs single-frame, inference over full sequences; AV2 subsampled to 2 Hz; 24 epochs nuScenes (30 AV2), AdamW 5e-4, batch 32, 8 GPUs.
- input: surround images -> ResNet-50 + 1-layer BEVFormer; previous queries (Nq x C), previous polylines (20 x 2 each), previous BEV feature, pose T.
- output: per query a class score and 20 ordered points (absolute coordinates in the ego frame); fixed query set.
- training: bipartite matching (algorithm not named on the page) + focal + smooth-L1 + transformation loss; labels = vector map GT; regime hungarian_set plus an identity-based auxiliary loss.
- differentiable: partial - rank-based top-k (gradient into the selected queries, none into the previous frame because memory is detached), bipartite matching in the main loss.
- sensors_data: camera only; nuScenes and Argoverse 2, original and NEW location-disjoint splits (VM46); ranges 60 x 30 m and 100 x 50 m; BEV grid size not stated.
- results: nuScenes NEW split mAP, 60 x 30 m: StreamMapNet 33.9, MapTR 20.9, VectorMapNet 18.0; 100 x 50 m: 23.0 / 14.8 / 8.8 (paper Table 2). AV2 new split 60 x 30 m: 58.1 vs MapTR 51.1 (Table 1). nuScenes original split, 30 m range: 62.9 vs MapTR 48.7, BeMapNet 59.8 (Table 4). 13.2 - 14.2 FPS.
- compute: FPS only.
- code: https://github.com/yuantianyuan01/StreamMapNet
- fit_to_us: reuse_idea - S/M - the ego-warped previous set plus a pose-conditioned token update plus an identity-based transformation loss is exactly the primitive of our time recursion; the BEV-GRU is our already tested feature fusion.
- compat_needs: ego transform between consecutive frames; a persistent per-element token (our 128-d step token); a confidence for the rank rule; fixed or padded set size. Cached-BEV: the memory part works on any BEV raster; the query part needs our own token set.
- compat_conflicts: the main loss assumes a fixed query set + bipartite matching (we have none); carried queries are concatenated into a DETR decoder (we have a tracer / refiner); frame 0 has no prior (needs a single-frame mode).
- tried_before: E39 / E41 temporal feature fusion (+0.008 F1, small, kept) = BEV-GRU analogue; E31 / E32 matched sparse decoder dropped; System B input memory (ego-warped previous input, PROJECT_CONTEXT) = raster-side idea; set / token propagation with a transformation loss: none.
- verified: partly (fetched text; k and the loss weight needed a second read).

### CARD VM02 - SQD-MapNet (stream query denoising)
- papers: Stream Query Denoising for Vectorized HD Map Construction (Wang et al., ECCV 2024) https://arxiv.org/abs/2401.09112
- depth: M
- stage: T, L, S8
- idea: Training-only add-on to StreamMapNet. The previous frame's GROUND TRUTH is mapped to the current frame by the ego transform, each element is wrapped in its bounding rectangle which is randomly shifted / rescaled, and the result is fed as extra "stream" queries whose targets are the current-frame GT instances found by "adaptive temporal matching" (ATM: bidirectional Chamfer between ego-warped previous GT and current GT with a per-instance threshold). The decoder practises turning a stale, shifted, partly out-of-range set into the current set, which its own early stream queries cannot teach. Removed at inference.
- formulation: box shift |dx| < l1 w/2, |dy| < l2 h/2; scale w in [(1-l4) w, (1+l4) w], h in [(1-l3) h, (1+l3) h]; points move with the box, so no point jitter and no topology change. Noise decays with the matched Chamfer distance D: R = 1 - D / (gamma (w+h)/2); noised box = {x,y,w,h} + {dx,dy,dw,dh} * R; gamma = 0.2 best (0.1 / 0.3 / 0.5 / 0.7 worse). ATM threshold delta = alpha (w+h)/2, alpha = 0.1 best (0.05 / 0.2 / 0.3 worse; fixed thresholds 2.5 / 4 / 5 gave 62.0 - 62.8). L_DN = focal + line loss on the ATM-matched current GT; L = L_map + L_DN. Magnitudes l1 - l4, number of groups / queries and the attention mask are NOT stated on the page; no repository to check.
- input: GT polylines of frame t-1 + pose (training only).
- output: nothing extra at inference (model = StreamMapNet).
- training: denoising branch assigned by ATM (a threshold rule), main branch keeps StreamMapNet's matching; regime denoising + hungarian_set.
- differentiable: yes for the DN branch; ATM is a label rule, so no gradient is needed there.
- sensors_data: camera; nuScenes, AV2; 100 x 50 and 200 x 100 ranges (page writes "mm", read as metres); R50 + 1-layer BEVFormer, AdamW 5e-4, batch 32, 24 / 30 epochs.
- results: nuScenes (old split, R50, 480 x 800, 24 ep, Table 1): StreamMapNet 60.4 -> 63.9 mAP (60 x 30 m), 60.6 -> 64.0 (100 x 50); with extra tricks 62.9 -> 65.0; V2-99, 110 ep: 74.0 / 75.2. AV2 (30 ep): 61.5 -> 63.3 (30 m), 57.7 -> 59.3 (50 m). Ablation (Table 3, cumulative, nuScenes 60 x 30 m): single-frame baseline 59.9; + StreamMapNet's temporal stream 59.2 (-0.7); + dynamic query noising 62.9; + ATM 63.9 (full model). Table 6: no denoising 59.2 (= the temporal model), normal denoising on current-frame GT 62.7, stream denoising 63.9. The paper explains the -0.7 by the network finding it hard to learn the constant change of curves between frames. New-split numbers only in the supplement (not seen).
- compute: no inference cost; training cost not stated.
- code: none found (paper: "will be available").
- fit_to_us: reuse_idea - S/M - give the time-recursion block (ego-warped previous set -> current set) a training source that does not depend on its own early outputs: warp the previous GT lane graph, perturb it, regress the current GT with our matching-free point-to-polyline loss. The noise decay rule and the ATM rule (which prior elements still have a counterpart) are cheap to copy; ATM gives the validity target of the prior.
- compat_needs: GT of frames t-1 and t plus ego transform in the data loader; a recursion block that accepts a prior set as input; per-element extent for the noise scale. Cached-BEV: yes (only the prior path is involved).
- compat_conflicts: noise acts on whole boxes: no per-point jitter, no dropped / duplicated / merged elements, no topology errors; noised GT is cleaner than real predictions (train-test mismatch only partly covered); built to run as extra queries beside matched queries with attention masking.
- tried_before: none.
- verified: checked (parameters missing on the page are listed above).

### CARD VM03 - PrevPredMap / Uni-PrevPredMap (previous vector predictions as priors)
- papers: PrevPredMap: Exploring Temporal Modeling with Previous Predictions for Online Vectorized HD Map Construction (arXiv 2407.17378; authors not read) https://arxiv.org/abs/2407.17378 ; Uni-PrevPredMap: Extending PrevPredMap to a Unified Framework of Prior-Informed Modeling (Peng et al., IEEE RA-L vol. 11 no. 10, Oct 2026, arXiv 2504.06647) https://arxiv.org/abs/2504.06647
- depth: M
- stage: S8, T
- idea: The previous frame's top-k VECTOR predictions (class, confidence, location), ego-transformed, become content + position queries of the current frame. Training uses no GT-derived prior: each sample's prior is a model prediction stored in a dictionary from earlier iterations, so it has the real error distribution. Dual-mode training: p = 0.5 of steps run single-frame; inference is single-frame on the first frame, temporal afterwards. Uni- version: predictions (and an optional corrupted HD-map prior) are stored in a tile-indexed global vector map retrieved by UTM position, rasterised into heatmaps and concatenated with the BEV feature (as in HRMapNet); tri-mode training none / temporal / temporal + map (ratio 0.5 : 0.3 : 0.2).
- formulation: k = 10 (nuScenes), 8 (AV2) of 100 queries, saturating near 12. Category + confidence -> linear -> content query; location -> initial location prediction -> sinusoidal + linear -> position query; position queries re-encoded per decoder layer; group-wise one-to-many branch with 6 groups. 100 queries x 20 points, 6 layers, R50 + LSS BEVPoolv2, AdamW 6e-4, wd 0.01, batch 16, 4 A100, 24 / 110 epochs. Uni-: the only training perturbation of the map prior is instance-level displacement with magnitude uniform in [-6, 6] m on subsets (about half of the vectors, mean about 3 m); additions, deletions, rotation, scaling appear only as inference robustness tests. Transform formula and loss weights are not on the pages.
- input: previous predicted polylines (top-k) + pose (Uni-: global pose); current BEV.
- output: polylines + class (fixed set).
- training: MapTRv2-style hierarchical matching + one-to-many group branch; regime hungarian_set with stored predictions as prior.
- differentiable: partial - top-k, stored predictions carry no gradient, Hungarian.
- sensors_data: camera; nuScenes (old and non-overlapping), AV2 2 Hz / 10 Hz; 60 x 30 m (and 100 x 50 m for Uni-), 0.3 m grid.
- results: nuScenes 24 ep: PrevPredMap 66.3 mAP vs SQD-MapNet 65.0, StreamMapNet 63.4 (30 ep); 110 ep: 71.3 vs MapTRv2* 69.2; AV2 2D 2 Hz: 68.2 vs 63.3 vs 61.5. Uni-PrevPredMap, nuScenes standard split, no prior: 74.0 (24 ep) / 77.0 (72 ep); with map prior 80.9 / 81.8; non-overlapping split 39.7 -> 52.5 with map prior.
- compute: not stated.
- code: https://github.com/pnnnnnnn/PrevPredMap ; https://github.com/pnnnnnnn/Uni-PrevPredMap (not opened)
- fit_to_us: reuse_idea - S - vector-level prior independent of the BEV encoder; the stored-predictions training and the single-frame / temporal mode switch are what a recursion trained on its own outputs needs, with a single-frame fallback for frame 0 and for failures.
- compat_needs: a per-frame store of model predictions refreshed across epochs; ego transform; a prior-input path in the recursion block. Cached-BEV: yes for the prior path.
- compat_conflicts: stored predictions are stale by up to one epoch and need a warm-up epoch; the decoder is query / Hungarian based; Uni- needs a global pose.
- tried_before: none.
- verified: partly (fetched text; transform and loss details missing).

### CARD VM04 - MapTracker (mapping as tracking, strided memory, consistent GT, C-mAP)
- papers: MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping (Chen et al., ECCV 2024 oral) https://arxiv.org/abs/2403.15951
- depth: M+C
- stage: S8, E, L
- idea: Every map element is a track: its 512-d vector latent is carried frame to frame; a raster latent (BEV, 50 x 100 x 256) accumulates BEV evidence. Buffers keep the last 20 frames but fusion uses only 4 memories chosen by travelled distance (nearest to 1, 5, 10, 15 m ago); BEV memories are warped by the ego affine transform (bilinear), vector latents by a pose MLP (quaternion + translation) and fused by cross-attention with relative-frame encoding. Assignments of tracked elements are inherited from t-1 (identity persists); only new elements pass through Hungarian matching. Proposes a "consistent GT" (elements merged over time) and the consistency-aware mAP (VM45).
- formulation: BEV loss focal + Dice (10, 1); vector loss 5 x focal + 50 x line; transformation loss from StreamMapNet 0.1; Gaussian noise added to the inter-frame transform P^t_{t-1} that feeds the vector PropMLP during training (sigmas not on the page; whether the BEV warp is perturbed is not stated; no noisy-pose test); consistent GT = per-frame cleaning, then bipartite matching of IoU between motion-aligned rasterised masks of adjacent frames, chained into tracks (look-back 1); thresholds 0.4 (first frame), 0.5 (propagated) / 0.6 (new); 100 new-candidate embeddings, 20-point polylines, 6 decoder layers (instance self-attention -> BEV multi-point cross-attention -> memory cross-attention). Stages: BEV pretraining 18 ep, vector warm-up 6, joint 48 (nuScenes; AV2 12 / 3 / 20); clips: 4 of the previous 10 frames, length 5; AdamW 5e-4; 8 RTX A5000, about 3 days.
- input: images -> BEV (StreamMapNet code base); 20-frame BEV and vector buffers; pose per frame.
- output: segmentation raster + 20-point polylines with class and persistent IDs.
- training: hierarchical Hungarian for new elements, inherited assignment for tracked ones; focal + line + BEV segmentation.
- differentiable: partial - Hungarian for new elements, thresholded birth / death, distance-rule memory selection.
- sensors_data: camera; nuScenes, AV2; 60 x 30 m (BEV 50 x 100, derived 0.6 m cells), also 100 x 50 m; original and non-overlapping splits.
- results: nuScenes consistent GT mAP / C-mAP 76.1 / 69.1 vs StreamMapNet 70.4 / 56.4; original GT 74.7 vs 69.1; non-overlapping split (60 x 30 m) nuScenes 40.3 / 32.5 vs 33.5 / 22.2, AV2 71.3 / 63.2 vs 64.4 / 54.4. Ablation: single memory 70.8 / 62.4; fusion without stride 74.9 / 68.1; with stride 76.1 / 69.1; stride sets {1,5,10,15} 76.1, {1,2,3,4} 75.8, {5,10,15,20} 73.8, {0,0,0,0} 75.0. 11.5 FPS (RTX 6000, batch 1) vs StreamMapNet 14.2.
- compute: FPS; parameters not stated.
- code: https://github.com/woodfrog/maptracker (README: official, built on StreamMapNet, GPL-3.0 stated, framework not named)
- fit_to_us: reuse_idea - M - the distance-strided memory selection, the consistent-GT idea and C-mAP are portable; the tracker as a whole is not (query identity).
- compat_needs: ego pose per frame; element IDs or tokens; frame buffers; consistent GT (our piece graph has ID continuity). Cached-BEV: the memory modules yes; stage 1 trains the BEV encoder.
- compat_conflicts: identity inheritance has no meaning at forks / merges of a graph unless defined; the three-stage 72-epoch training is far beyond our budget.
- tried_before: E41 (single recent memory, +0.008 F1); distance-strided multi-memory: untested.
- verified: partly (fetched text + front page).

### CARD VM05 - SparseMeXt (physical-prior denoising of map polylines, query memory bank)
- papers: SparseMeXt (arXiv 2505.08808; full title and authors not on the pages read, title from memory: "Unlocking the Potential of Sparse Representations for HD Map Construction") https://arxiv.org/abs/2505.08808
- depth: M
- stage: T, S8, B
- idea: A sparse map decoder (no dense BEV at inference) with a query memory bank and a "physical-prior denoising" (PPDN) adapted to polylines instead of boxes: four noise types on GT polylines - rotation about the mean point, one common x/y shift of all points, scale of the x/y coordinates, and a per-point curvature-based displacement (curvature from second-order tangent differences). No categorical noise (rationale: static geometry, strong dependence among road elements).
- formulation: noise magnitudes, number of groups and queries, assignment rule, matching and loss weights: not stated. One non-temporal decoder stage + four temporal stages (ablation). 100 epochs, lr 0.006 half-cosine, AdamW wd 0.01, grad clip 35, 4 x A100, input 704 x 256, nuImages-pretrained R50 init for Base / Tiny.
- input: images (SiMo neck on ResNet C5); a BEV segmentation head exists only as a training auxiliary (+0.9 mAP).
- output: sparse queries -> polylines + class.
- training: DN-DETR-style reconstruction of the source GT (assumed, not confirmed); matching not stated.
- differentiable: unknown (matching not stated).
- sensors_data: camera; nuScenes only (official 700 / 150 split); 60 x 30 m base range, 90 x 60 m long range.
- results: nuScenes val mAP: Tiny R18 55.5 (32.9 FPS), Base R50 65.2 (25.4 FPS), Large R101 68.9 (20.2 FPS); MapTRv2-Base 68.7; centerline average mAP 58.8 vs MapTRv2 54.0; long range 47.6.
- compute: FPS above (RTX 3090).
- code: none given.
- fit_to_us: reuse_idea - S - the noise TAXONOMY (rotate, shift, scale, curvature jitter on a polyline) is the closest published model for noising a previous lane set; magnitudes must be chosen by us.
- compat_needs: GT polylines; a recursion block that accepts noised priors. Cached-BEV: yes for the noise part, no for the image-sampling decoder.
- compat_conflicts: matching behind the denoising targets is unknown.
- tried_before: none.
- verified: partly.

### CARD VM06 - Sparse4D v3 (temporal denoising, quality estimation, instance IDs)
- papers: Sparse4D v3: Advancing End-to-End 3D Detection and Tracking (Lin et al., arXiv 2311.11722) https://arxiv.org/abs/2311.11722
- depth: M
- stage: T, S8, S3
- idea: (1) Temporal instance denoising: noisy anchors built from GT are propagated into the next frame together with the normal instances (ego + velocity compensation; their features initialise the next frame's), so the temporal path sees realistic stale instances. (2) Decoupled attention: concatenate (not add) anchor embedding and feature for q / k. (3) Quality estimation: centerness C = exp(-||xyz_pred - xyz_gt||) and yawness Y = dot of (sin, cos) of predicted and GT yaw, supervised by focal / CE; centerness x confidence ranks detections. (4) End-to-end tracking by propagating IDs of instances above a threshold.
- formulation: noise per anchor component uniform: positive set in (-x, x), second set in (-2x, -x) U (x, 2x) (x NOT stated); M = 5 noise groups, M' = 3 random groups propagated; groups independent (attention mask); each group matched to GT by bipartite matching (at most one positive per GT per group; avoids DINO's fixed assignment). The N_t = 600 highest-confidence instances of the previous frame are carried (900 instances in the decoder); carried confidence c' = max(c', c * 0.6); ID given when confidence >= 0.25 and none yet; IDs persist; no tracking-specific training. 6 layers, dim 256, 7 fixed + 6 learnable keypoints; 100 epochs, AdamW.
- input: multi-view image features + instance memory.
- output: 3D boxes + class + velocity + quality + IDs.
- training: Hungarian per group + focal / CE quality losses; regime denoising + hungarian_set.
- differentiable: partial - top-k, matching.
- sensors_data: camera; nuScenes.
- results: nuScenes val R50 256 x 704: mAP 0.469, NDS 0.561, AMOTA 0.490, 19.8 FPS (v2: 0.439 / 0.539); R101 512 x 1408: 0.537 / 0.623 / AMOTA 0.567; test V2-99: 0.570 / 0.656.
- compute: 19.8 FPS (R50).
- code: https://github.com/linxuewu/Sparse4D
- fit_to_us: reuse_idea - S - the confidence decay of carried instances (x 0.6) is a hand-made validity decay; a calibrated quality head for ranking and propagation of noised GT through the same operator as real instances are portable.
- compat_needs: instance geometry + features; bipartite matching for DN groups (replaceable by a nearest-GT rule). Cached-BEV: only the memory logic.
- compat_conflicts: 3D boxes, not polylines.
- tried_before: none.
- verified: partly.

### CARD VM07 - Sparse4D v2 (recurrent instance propagation)
- papers: Sparse4D v2: Recurrent Temporal Fusion with Sparse Model (Lin et al., arXiv 2305.14018) https://arxiv.org/abs/2305.14018
- depth: M
- stage: S8, S4
- idea: 3D detection instances (anchor box + velocity, feature vector) are propagated recurrently: the most confident instances of frame t-1 are re-projected (anchor moved by velocity x dt, then by the ego rotation / translation; feature copied unchanged; anchor re-encoded) and concatenated with fresh instances. The current frame samples image features only at the current time.
- formulation: 900 instances per frame in training: 600 temporal + 300 single-frame; anchor A = {x,y,z,w,l,h,sin yaw,cos yaw,vx,vy,vz}; Eq. 2: position + v dt, then R, T; yaw vector and velocity rotated by R. Single-frame layer scores and selects the new instances; multi-frame layers have temporal cross-attention + self-attention. Depth reweighting of v1 dropped once dense depth supervision (L1) exists. 100 epochs, AdamW, no CBGS; lr, batch, history length, loss / matching and gradient treatment of carried instances NOT stated on the page.
- input: multi-view features (R50 / R101 / V2-99 + FPN) + instance memory.
- output: 3D boxes + class + velocity.
- training: not stated on the page (from memory: Hungarian one-to-one).
- differentiable: partial (top-k).
- sensors_data: camera; nuScenes val / test.
- results: nuScenes val R50 256 x 704: mAP 0.439, NDS 0.539, 20.3 FPS; R101 512 x 1408: 0.485 / 0.580 at 8.4 FPS; test V2-99: 0.557 / 0.638.
- compute: 20.3 FPS (R50).
- code: https://github.com/linxuewu/Sparse4D
- fit_to_us: reuse_idea - S - the plain template of "ego-warped previous set": store (geometric anchor, feature) per instance, transform only the geometry, keep the feature, concatenate with fresh instances.
- compat_needs: geometry separable from a feature vector; ego transform; confidence.
- compat_conflicts: 3D boxes, not polylines.
- tried_before: none.
- verified: partly.

### CARD VM08 - StreamPETR (object-centric memory queue, motion-aware layer norm)
- papers: Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection (Wang et al., ICCV 2023; authors / venue from memory) https://arxiv.org/abs/2303.11926
- depth: M
- stage: S8
- idea: A FIFO memory queue keeps the top-K foreground queries of the last N frames; current queries read them by "hybrid attention" (self-attention over current + memory queries) and PETR-style cross-attention to image tokens. Ego-motion / velocity / time gap enter via Motion-aware Layer Normalisation (MLN): affine parameters (gamma, beta) computed from (ego pose E, velocity v, dt) modulate the position encoding of the ego-compensated centre and the context embedding. No BEV warping.
- formulation: memory N x K = 4 frames x 256 objects; entry = (dt, context embedding Q_c, 3D centre Q_p, velocity v, ego pose E). E_{t-1}^t = E_t^-1 E_{t-1}; Q_p~ = E_{t-1}^t Q_p; gamma = xi_1(M), beta = xi_2(M); Q_pe~ = gamma LN(psi(Q_p~)) + beta, Q_c~ = gamma LN(Q_c) + beta; current frame v = 0, dt = 0. Explicit compensation alone did not help (attributed to early error propagation). 644 fresh + 256 propagated queries = 900 (lightweight 300 + 128). Training: local sliding windows of 8 key frames, one random frame skipped, gradients of the first 6 frames detached (losses on the last 2); AdamW 4e-4, batch 16, 24 / 60 epochs. Detection loss / matching not on the page (denoising from PETRv2).
- input: image tokens with 3D position encoding + memory queue.
- output: 3D boxes (queries).
- training: hungarian_set (from memory).
- differentiable: partial - TopK + gather, earlier-frame gradients detached.
- sensors_data: camera; nuScenes.
- results: nuScenes val R101 512 x 1408: mAP 0.504, NDS 0.592, 6.4 FPS; R50 256 x 704: 0.432 / 0.540, 27.1 FPS; test V2-99: 0.550 / 0.636, AMOTA 0.653.
- compute: 27.1 FPS (R50); flash attention cuts memory 61 -> 27 GB at 1600 x 640 (A100).
- code: https://github.com/exiawsh/StreamPETR
- fit_to_us: reuse_idea - S - MLN (pose-conditioned affine modulation of carried tokens) is a cleaner alternative to concatenating a flattened 4x4 pose; the FIFO of top-K tokens with (dt, pose) per entry is the template for a multi-frame recursion memory.
- compat_needs: tokens with an explicit centre; per-entry timestamp and pose. Cached-BEV: only the memory logic (the model reads image tokens).
- compat_conflicts: detection queries (boxes), not polylines.
- tried_before: none.
- verified: partly.

### CARD VM09 - TopoStreamer (streaming lane-segment topology, OpenLane-V2)
- papers: TopoStreamer (arXiv 2507.00709; title "Temporal Lane Segment Topology Reasoning in Autonomous Driving" and authors from memory / repository owner) https://arxiv.org/abs/2507.00709
- depth: M
- stage: S8, S6
- idea: Extends a LaneSegNet-style lane-segment decoder with (1) streaming attribute constraints (stream queries must predict coordinates, class, boundary type and BEV mask of the NEXT frame, supervised by previous-frame GT mapped by the pose), (2) lossless streaming supervision (follow the IDs of stream queries that were positively matched instead of re-matching pose-transformed GT; dropping it costs 36.6 -> 34.4 mAP), (3) a dynamic lane-boundary positional encoding (8 boundary reference points per query, re-encoded every layer), plus lane-segment denoising incl. topology denoising.
- formulation: carried queries top-66 of 200 (30 %: 10 % 34.8, 30 % 36.6, 50 % 34.3, 75 % 32.9 mAP); query transform = MLP over [query, pose] + residual; BEV warp + GRU as StreamMapNet. DN: box-shift factor 0.2, class-label flip probability 50 %; 240 DN queries (120 -> 35.0, 240 -> 36.6, 360 -> 35.1); topology denoised by two MLP projections (predecessor / successor) -> pairwise sigmoid logits, topology itself not noised. Lane segment = centerline + left / right boundary, 10 ordered points each, adjacency thresholded at 0.5. Weights: vector L1 0.025, seg 3.0, class 1.5, boundary type 0.01, topology focal 5.0. 6 layers, 200 queries, 24 epochs (first 12 single-frame), AdamW 2e-4, batch 8, V100.
- input: 7 cameras -> R50 + FPN + BEVFormer-style, BEV 200 x 100 over +-50 x +-25 m.
- output: lane segments (3 polylines) + adjacency matrix + class + boundary type.
- training: Hungarian + DN + streaming supervision; regime hungarian_set + denoising.
- differentiable: partial (top-k, Hungarian).
- sensors_data: camera; OpenLane-V2 subsets A (Argoverse 2) and B (nuScenes).
- results: subset A lane segments mAP 36.6 (AP_ls 35.0, AP_ped 38.1), TOP_lsls 28.5, 13.6 FPS vs LaneSegNet 32.6 / 25.4, SQD-MapNet 26.0 / 16.6; centerline subset A OLS 44.4, DET_l 35.2, TOP_ll 28.8; subset B (pseudo lane-segment labels from a 4 m width) 42.6 / 30.9 / 29.4 vs LaneSegNet 38.7 / 27.5 / 24.9. No temporal-consistency metric.
- compute: 13.6 FPS.
- code: https://github.com/YimingYang23/TopoStreamer
- fit_to_us: reuse_idea - M - the only streaming method that propagates and denoises TOPOLOGY (adjacency from predecessor / successor projections) and uses ID-following supervision; centerline direction is fixed by data (no permutation set).
- compat_needs: queries with identity; adjacency head; segment IDs in GT. Cached-BEV: the temporal parts yes.
- compat_conflicts: Hungarian; segment-level not vertex-level; OpenLane-V2 labels differ from nuScenes map-expansion pieces.
- tried_before: topology heads E42 / E45 / E46 / E51 / E56 (within 0.01 of the rules, dropped); streamed topology: none.
- verified: partly.

### CARD VM10 - FASTopoWM (fast-slow latent world model)
- papers: FASTopoWM (arXiv 2507.23325; title "Fast-Slow Lane Segment Topology Reasoning with Latent World Models" from memory) https://arxiv.org/abs/2507.23325
- depth: M
- stage: S8, T
- idea: Two streams share decoder weights: a fast single-frame stream and a slow stream with history through "world models" - a query world model and a BEV world model (2 transformer layers each, conditioned on the flattened relative pose) predict the NEXT frame's stream queries and BEV features, then fused with the current BEV by a GRU. At test time the slow output is used when the pose is reliable, the fast one otherwise.
- formulation: L = L_slow + L_fast; L_slow = L_ls + 0.3 L_latent; L_latent = MSE(predicted stream BEV, current BEV) (self-supervised) + transformation loss on predicted queries (L1 coordinates, focal / CE class and type, CE + Dice mask) against previous-frame GT mapped to the current frame. About 30 % of the 200 queries are stream slots (inferred, K not stated). No denoising. The BEV world model runs on a 100 x 50 average-pooled BEV. Weights 0.025 / 3.0 / 1.5 / 0.01 / 5.0; 24 epochs (first 12 single-frame); AdamW 2e-4; batch 8.
- input: as VM09 (200 x 100 BEV, +-50 x +-25 m, 7 cameras).
- output: lane segments + adjacency.
- training: Hungarian; regime hungarian_set + self_supervised (next-BEV MSE).
- differentiable: partial.
- sensors_data: camera; OpenLane-V2 subsets A and B.
- results: subset A lane segments mAP 37.4 (TOP_lsls 29.6, 11.4 FPS) vs its single-frame variant 34.1 and Topo2Seq 33.6; subset B centerline OLS 46.3, DET_l 35.1, TOP_ll 33.0 (single-frame 41.8 / 31.6 / 27.1; LaneSegNet 38.7 / 27.5 / 24.9; SQD-MapNet 29.1 / 21.9 / 13.3). Pose robustness not tested (future work).
- compute: 11.4 FPS (fast only 14.0).
- code: https://github.com/YimingYang23/FASTopoWM
- fit_to_us: reuse_idea - M - (a) predict the next frame's BEV / elements from the previous ones given the pose, trained self-supervised against what the next frame shows (an MSE on cached features is possible for us), (b) a fast fallback when pose or history is unreliable - fits our synthetic pose-jump and failure tests.
- compat_needs: ego pose as a conditioning vector; a single-frame path of the same network. Cached-BEV: yes for the world-model part.
- compat_conflicts: two forward paths; Hungarian.
- tried_before: E41 (fusion, small); predictive next-frame memory: none.
- verified: partly.

### CARD VM11 - SparseDrive map branch (temporal map instances next to a planner)
- papers: SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation (Sun et al., ICRA 2025) https://arxiv.org/abs/2405.19620
- depth: M
- stage: S8, S10
- idea: Map polylines are sparse instance anchors (N_m = 100, 20 points each) handled by the same instance-bank mechanism as detected agents (temporal decoders cross-attend last-frame instances to the current ones; the page reports 33 temporal map instances); a parallel motion planner reads agents and map with agent-temporal / agent-agent / agent-map attention and an ego-instance memory queue of 3 frames.
- formulation: L_map = 1 x focal + 10 x L1 with Hungarian matching; permutation / direction handling for polylines not described; top-k selection and ego transform of map anchors not described on the page. Stage 1 perception from scratch (S 100 epochs, B 80), stage 2 joint with the planner (10 epochs).
- input: images -> sparse map and agent instances.
- output: map polylines, agents, ego trajectory.
- training: hungarian_set.
- differentiable: partial.
- sensors_data: camera; nuScenes, map range 60 x 30 m, AP thresholds not stated.
- results: nuScenes val map mAP: B 56.2 (ped 53.2, divider 56.3, boundary 59.1), S 55.1 (VAD 47.6); planning: B L2 0.58 m, collision 0.06 %; S 0.61 m, 0.08 %.
- compute: not stated here.
- code: https://github.com/swc-17/SparseDrive
- fit_to_us: reuse_idea - S - confirms that published stacks give map instances the same temporal instance memory as agents and hand them to the planner by attention; the planner-side interface (S10) is what E117 already does with edges, route probability and ego motion.
- compat_needs: instance map output with confidences.
- compat_conflicts: Hungarian; planner trained jointly end to end (ours: planner loss through positions gave no gain).
- tried_before: E117 planner consumer (0.34 m at 20 m), E119 / E121 planner loss into route head / positions (no gain).
- verified: partly.

### CARD VM12 - L2G-Map (hand-specified log-odds fusion of warped local maps, nuScenes)
- papers: L2G-Map: Local-to-Global Mapping via Hierarchical Diffusion Refinement and Elliptical Bayesian Fusion (arXiv 2503.02578; authors not read) https://arxiv.org/abs/2503.02578
- depth: M
- stage: S8, E
- idea: Builds a GLOBAL BEV semantic map (divider, crossing, boundary) from per-frame local maps by a closed-form recursive Bayesian log-odds update, weighted by the distance of each cell from the ego (observations near the perception edge count less); each local map is first refined by a conditional diffusion U-Net conditioned on the previous-frame map and BEV features warped into the current frame and on an OSM-centerline prior.
- formulation: L_t = L_{t-1} + dL_t; dL_t = logit(p_obs) - logit(p_prior); class priors 0.1 (divider), 0.02 (crossing), 0.09 (boundary); weighted increment dL_w = dL x c x f(P) with c = 1 / (1 + d_n^2), d_n = sqrt((l_x/r_x)^2 + (l_y/r_y)^2) the elliptical distance in units of the perception radii (c = 1 at the vehicle, 0.5 at the boundary) and f(P) bilinear weights that spread a sub-cell-shifted observation over 4 neighbours; final map sigma(L_0 + sum dL_w). The page describes no decay, forgetting or clamping. Diffusion U-Net: DDIM, two-stage training (20 epochs with 50 % masking, 10 fine-tune), focal loss, Adam + StepLR, RTX A6000. The fusion itself is NOT learned (fixed formulas and empirical priors).
- input: local BEV semantic maps from HDMapNet / P-MapNet / GenMapping + vehicle pose.
- output: global semantic raster (and a closed loop with StreamMapNet).
- training: diffusion network only; regime denoising; fusion none.
- differentiable: the log-odds sum is a closed form (differentiable), but the page does not train through it.
- sensors_data: camera; nuScenes, Argoverse 2; local extents 60 x 30, 120 x 60, 240 x 60 (units as printed); global resolution not stated.
- results: nuScenes 60 x 30 semantic mIoU 39.03 vs 37.13 (P-MapNet); Argoverse 2 48.93 vs 48.23 (GenMapping); nuScenes 240 x 60 48.37 vs 45.50; global fusion ablation (Table IV, Global IoU): Bayesian only 35.06, + bilinear 35.13, + confidence weight 35.57; the paper states 5.51 % Global IoU and 7.11 % mIoU gain over a probabilistic-mean fusion.
- compute: about 0.13 s per frame for the panoramic input (as printed; ambiguous).
- code: announced https://github.com/lynn-yu/L2G-Map (not released when the paper was written)
- fit_to_us: reuse_idea - S - the only log-odds recursion with a reliability weight on nuScenes map layers that I found. For vertex validity: p_obs = the per-frame keep gate, prior = its base rate, weight = distance-to-ego (the BEV border is less reliable) times bilinear spread; add a decay (absent here) to bound memory.
- compat_needs: pose alignment; cells instead of moving vertices (association is implicit on a grid). Cached-BEV: yes (works on any raster).
- compat_conflicts: for a graph the validity lives on vertices that move, so a soft association (nearest vertex or splat to a grid) is needed; the published version assumes a global pose.
- tried_before: none.
- verified: partly (extracted from the HTML; closed-loop numbers not used).

### CARD VM13 - HRMapNet (global historical raster of past vector outputs)
- papers: Enhancing Vectorized Map Perception with Historical Rasterized Maps (Zhang et al., ECCV 2024) https://arxiv.org/abs/2409.00620
- depth: M
- stage: S8, B
- idea: A global BEV raster (3 class channels, 0.3 m per pixel, uint8) in world coordinates is updated additively from past vector outputs with the GLOBAL ego pose; the local crop is (i) concatenated into the BEV feature and used to place extra BEV queries (feature aggregation) and (ii) turned into prior embeddings that initialise the map queries.
- formulation: pixel update +S+ = 30 if the local rasterised prediction is 1, -S- = 1 if it is 0 (occupancy-grid-style additive score, NOT log-odds); read-out 1 where score > S_th (value not stated). Local map downsampled to 0.6 m for query initialisation. F_BEV = Conv(Concat(F_img + F_M, M_local)). Training: the global map is reset to empty at the start of each epoch and filled from the model's own predictions; test maps start empty; no GT-derived maps in the main setting. Pose-noise test at inference: translation sigma 0 - 0.2 m, rotation 0 - 0.02 rad: 67.2 -> 63.8 mAP at worst.
- input: BEV feature + retrieved local raster (GT pose by default).
- output: polylines of the host (MapTRv2 or StreamMapNet).
- training: host losses (hierarchical matching).
- differentiable: partial - rasterisation of vector outputs and the integer update are non-differentiable; the memory is an input only.
- sensors_data: camera; nuScenes, AV2; 60 x 30 m, 0.3 m, R50.
- results: nuScenes original split 24 ep: StreamMapNet 60.4 -> 66.3, MapTRv2 61.5 -> 67.2 (110 ep 68.7 -> 73.6); AV2 StreamMapNet 61.5 -> 64.3; NEW split: nuScenes StreamMapNet 33.9 -> 37.1, AV2 58.1 -> 61.5. Ablation (MapTRv2): feature aggregation +3.1, query initialisation +2.6. A pre-built training-area map gives 83.7 mAP but is leakage on the old split.
- compute: map storage about 1 MB per km (120 MB Boston); training cost not stated.
- code: https://github.com/HXMap/HRMapNet
- fit_to_us: reuse_idea - M - a raster memory of past graph OUTPUT in a rolling ego frame is a cheap second temporal channel; its additive score (uint8, so bounded; no log-odds) is a near relative of log-odds fusion (ours could use true log-odds with decay).
- compat_needs: rasterisation of our graph (0.5 m, optional direction channels), an ego-warp of the raster each frame, clamp / decay. Cached-BEV: the raster memory and query initialisation yes; the aggregator uses image projection (no).
- compat_conflicts: global-pose retrieval implies localisation in a world map (conflicts with the registration-free premise unless the memory is a rolling local window); rasterisation breaks gradients.
- tried_before: E39 / E41 (feature level), System B input memory (input level); raster memory of the OUTPUT graph: none.
- verified: partly.

### CARD VM14 - PredMapNet (decayed instance-mask history + short-term future polylines)
- papers: PredMapNet: Future and Historical Reasoning for Consistent Online HD Vectorized Map Construction (arXiv 2602.16669, Feb 2026) https://arxiv.org/abs/2602.16669
- depth: M
- stage: S8, E
- idea: Built on MapTracker. (i) A History Rasterized Map Memory keeps per-instance BEV masks, decayed (beta = 0.9) with confidence weighting and warped to the next frame by a grid-based ego warp; a mask-guided module (threshold 0.5) lets each track query cross-attend to BEV features under its own warped mask. (ii) A short-term future module predicts per-point offsets from the stacked past polylines (P_{t+1} = P_t + dP) and mean-pools the predicted polyline into the track query; supervised by Chamfer to the GT future polyline. (iii) A Mask2Former-style semantic-aware query generator.
- formulation: tracks pruned below tau_t = 0.5, births above tau_d = 0.4; tau_L = 0.5; Nq = 100, Np = 20, C = 512; loss = BEV + track (MapTracker) + seg (2 Dice + 1 BCE) + future Chamfer + depth, unweighted sum; 72 epochs nuScenes (18 + 6 + 48), 35 AV2; 4 x A100; BEV range, lr, batch not stated.
- input: as MapTracker; memory = instance masks + last n polylines per track.
- output: polylines with class and track IDs.
- training: as MapTracker plus future Chamfer; whether history is GT or predicted during training is NOT described.
- differentiable: partial (thresholds, Hungarian for births).
- sensors_data: camera; nuScenes, AV2; old and non-overlapping splits.
- results: nuScenes old split val, 24 ep: 73.2 mAP / 64.3 C-mAP (Mask2Map 71.6 / 55.8, MapTRv2 61.5 / 41.7, HRMapNet 67.3 / 49.2, StreamMapNet 63.4 / 38.4); 72 ep: 76.9 / 69.7 at 10.1 FPS vs MapTracker 76.1 / 69.1 at 10.9; AV2 77.3 / 69.1; non-overlapping nuScenes 42.1 / 33.7 vs MapTracker 40.3 / 32.5.
- compute: 10.1 FPS.
- code: project page https://astronirav.github.io/predmapnet (no repository link on the page)
- fit_to_us: reuse_idea - S/M - the decayed, confidence-weighted, ego-warped per-instance mask memory is the closest relative of a recursively fused validity; the "future" offsets are a motion prior (static lanes need none).
- compat_needs: per-instance masks (our pieces / edges can be rasterised); ego warp; track identity. Cached-BEV: memory yes.
- compat_conflicts: relies on MapTracker identities and Hungarian births; the decay constant is hand-set, not a log-odds model.
- tried_before: none.
- verified: partly.

### CARD VM15 - HisTrackMap (per-track history maps, global-map metric)
- papers: HisTrackMap: Global Vectorized High-Definition Map Construction via History Map Tracking (arXiv 2503.07168; authors not read) https://arxiv.org/abs/2503.07168
- depth: M
- stage: S8, E
- idea: Each track query owns a rasterised history map (H x W = BEV size). New instances initialise it from their rasterised vector scaled by score; tracked instances decay it by lambda = 0.95 and add the new rasterisation; it is warped to the new ego position each step. "Map-Trajectory Prior Fusion" samples PV and BEV features where the history map exceeds tau_map = 0.5 and cross-attends the track queries to them. Also defines a global-map metric (G-mAP, VM45).
- formulation: tau_det = 0.4, tau_track = 0.5, lambda = 0.95, Nq = 100, Np = 20, C = 512; the Motion-MLP transformation loss was removed, other losses as MapTracker (weights not stated); 72 epochs nuScenes (18 / 6 / 48), 35 AV2, 8 A100; BEV range, lr not stated.
- input: as MapTracker.
- output: polylines with IDs.
- training: as MapTracker; regime hungarian_set with identity.
- differentiable: partial.
- sensors_data: camera; nuScenes, AV2; old and non-overlapping splits.
- results: nuScenes 72 ep: mAP 76.6, C-mAP 68.7, G-mAP 50.2, 10.3 FPS vs MapTracker 76.1 / 69.1 / 49.4, 10.9 FPS; AV2 35 ep: 77.7 / 69.5 / 48.6 vs 76.8 / 68.3 / 48.0; non-overlapping splits: nuScenes 41.5 / 32.8 / 28.5, AV2 71.4 / 62.7 / 49.2 (mAP / C-mAP / G-mAP).
- compute: 10.3 FPS.
- code: none; project page https://yj772881654.github.io/HisTrackMap/
- fit_to_us: reuse_idea - S - decayed per-instance raster memory (same family as VM14); G-mAP defines how to evaluate a MERGED multi-frame map.
- compat_needs: as VM14.
- compat_conflicts: as VM14; the prior fusion samples perspective-view features (not available on cached BEV).
- tried_before: none.
- verified: partly.

### CARD VM16 - MemFusionMap (working memory + temporal overlap heatmap)
- papers: MemFusionMap: Working Memory Fusion for Online Vectorized HD Map Construction (Song et al., WACV 2025) https://arxiv.org/abs/2409.18737
- depth: M
- stage: S8, B
- idea: A fixed-lag working memory of the last 4 unified BEV features (warped by ego transform, concatenated, 3 conv layers) plus a "temporal overlap heatmap": a one-channel BEV map counting how long each cell has been inside the field of view, propagated recursively and fused as a 32-channel feature.
- formulation: H_t0 = 1; H_t = Warp(H_{t-1}) + 1 with zero padding, so a cell that leaves view resets to 1 (a count of continuous visibility; unbounded, no normalisation; text and formula slightly disagree); Conv_H 1 -> 16 -> 16 -> 32 channels + sigmoid; fusion ConvMem: 3 x 3 conv x 3, dilation (2,2), 256 ch, LayerNorm; first BEV repeated 4 times at sequence start. Built on StreamMapNet: loss 5 focal + 50 line + 0.1 transformation; stage 1 single-frame (44 ep nuScenes), stage 2 temporal with detached memory. BEV 50 x 100 x 256, ranges 60 x 30 and 100 x 50 m, AdamW 5e-4, batch 32, 8 V100.
- input: as StreamMapNet.
- output: polylines.
- training: as StreamMapNet; hungarian_set.
- differentiable: partial (Hungarian, detached memory); the heatmap recursion itself is differentiable.
- sensors_data: camera; nuScenes and AV2 new splits.
- results: new split mAP vs StreamMapNet: nuScenes 60 x 30 38.0 vs 34.1, 100 x 50 27.6 vs 22.2; AV2 60.6 vs 58.3, 54.3 vs 51.5; original split nuScenes 65.2 vs 63.4; 15.1 vs 15.3 FPS (A6000).
- compute: FPS as above.
- code: none; project page https://song-jingyu.github.io/MemFusionMap
- fit_to_us: reuse_idea - S - an explicit "how often observed" channel next to the fused evidence is the natural companion of a log-odds validity memory (tells the head how much to trust it).
- compat_needs: ego warp of a raster; conv fusion. Cached-BEV: yes.
- compat_conflicts: none technical; gain measured only on a StreamMapNet host.
- tried_before: E41 (feature fusion without a count channel).
- verified: partly.

### CARD VM17 - InteractionMap (key-frame hierarchical fusion, geometry-aware classification)
- papers: InteractionMap: Improving Online Vectorized HDMap Construction with Interaction (Wu et al., CVPR 2025) https://arxiv.org/abs/2503.21659
- depth: M
- stage: S8, L
- idea: (1) Key frames chosen by DISTANCE (stride 5 m, N_pre = 4 submaps), local fusion (warp previous BEV + GRU + residual block) into a "submap", global fusion by streaming (GRU over a global BEV) or stacking (warp and concatenate the N_pre submaps). (2) Explicit point-to-instance relation prior added to the attention logits (point level: log-scaled coordinate differences + edge-direction cosine differences; instance level: sign of score difference x Chamfer distance). (3) Geometry-aware classification: the positive target of the focal-style BCE is a soft score s_geo (p2p: 1 - mean Manhattan distance of normalised points; dir: 0.5 + mean edge cosine / 2; GIoU variant) and the same score enters the matching cost.
- formulation: s_p2p = 1 - (1/(2 N_p)) sum Manhattan(pred, GT) on normalised points; best = s_p2p + s_dir; matching cost C = s_geo BCE(s_geo, p) - alpha p^gamma BCE(1-p, 1). First 50 % of epochs single-frame. Range +-15 x +-30 m, BEV 200 x 100 (0.3 m), R50, AdamW 6e-4 cosine, batch 32, 8 A800, 24 / 110 epochs. One-to-many assignment not described.
- input: images -> BEV + submap memory.
- output: polylines + class (MapTRv2 family).
- training: Hungarian with geometry-aware cost; regime hungarian_set.
- differentiable: partial (Hungarian).
- sensors_data: camera; nuScenes, AV2 (2 Hz, 10 Hz); standard val sets (no new-split row on the page).
- results: nuScenes val: MapTRv2 61.5 -> InteractionMap-R 71.6 / -C 71.8 (24 ep); 68.7 -> 75.5 / 76.0 (110 ep). AV2 2D 2 Hz 30 ep: 64.3 -> 73.7. 11.0 - 12.1 FPS (A800).
- compute: 11.0 - 12.1 FPS; model size 49.9 - 116.7 MB.
- code: none on the page.
- fit_to_us: reuse_idea - S - (a) key-frame stride by distance (agrees with MapTracker's strided memory), (b) soft quality target for the validity head (same family as VM34 and VM06).
- compat_needs: ego warp; GRU on BEV; validity head with a soft target. Cached-BEV: the fusion part yes.
- compat_conflicts: the relation prior and matching cost live inside a DETR decoder.
- tried_before: E41 (single-memory GRU).
- verified: partly.

### CARD VM18 - MapExpert (recurrent single-state BEV fusion, sparse expert block)
- papers: MapExpert: Online HD Map Construction with Simple and Efficient Sparse Map Element Expert (arXiv 2412.12704; authors not read) https://arxiv.org/abs/2412.12704
- depth: M
- stage: S8, B
- idea: Argues that stacking S frames gives the current frame weight 1/S so history dominates as S grows, and that map elements fill under 5 % of a raster so extra history mostly adds noise. Replaces stacking by a recurrent "learnable weighted moving descent" with ONE previous fused BEV feature, and adds a sparse mixture-of-experts block (8 experts, top-2 routing, intended one expert per element type) in the tracking decoder.
- formulation: Eq. 7: C_t = beta F_t + f_t(F_t, C_{t-1}), beta a learnable parameter (shape not stated), f_t an unspecified network; router R(x) = Softmax(TopK(x W_r)), output = weighted sum of top-K SwiGLU experts; balance loss alpha N sum f_i P_i (alpha not given); tracking query TQ(t) = [TQ_element(t-1), TQ_init]; ego-pose alignment not specified; ResNet-50 + FPN, BEVFormer-style BEV as in MapTracker; 100 epochs, AdamW, cosine, 500 warm-up steps, 6 x A800.
- input: BEV features + previous fused BEV + tracking queries.
- output: polylines with class.
- training: as MapTracker; hungarian_set.
- differentiable: partial.
- sensors_data: camera; nuScenes, AV2; ranges 30 m and 50 m as printed, BEV grid not stated.
- results: nuScenes original split 76.5 mAP (easy set) vs MapTracker 74.7 (+1.8; hard set 60.4 vs 58.13); new split 42.0 vs 40.3; AV2 original 30 m 72.8 vs 71.4, new split 71.8 vs 71.3; ablation 73.9 -> 76.5; FPS not reported.
- compute: not reported.
- code: none on the page.
- fit_to_us: reuse_idea - S - a single-state recurrence with a learned weight on the current observation is the minimal feature memory; the 1/S argument is a reason to keep the memory weight below the current-frame weight.
- compat_needs: one stored fused state. Cached-BEV: yes.
- compat_conflicts: the MoE block and tracking queries are host-specific.
- tried_before: E41 (GRU fusion of one previous memory).
- verified: partly.

### CARD VM19 - Neural Map Prior (global neural map memory with GRU update)
- papers: Neural Map Prior for Autonomous Driving (Xiong et al., CVPR 2023) https://arxiv.org/abs/2304.08481
- depth: M (read through ar5iv)
- stage: S8, B
- idea: A global feature grid (sparse tiles, 0.3 m, indexed by world coordinates) is read by pose, fused with the current BEV feature by cross-attention (patches 10 x 10 cells = 3 m, dim 256) and updated by a 2D-conv GRU; the updated local prior replaces the stored content.
- formulation: z = sigma(Conv[p, o]), r = sigma(Conv[p, o]), p~ = tanh(Conv[r*p, o]), p_new = (1 - z) p + z p~; the paper's moving-average update is only an ablation baseline. Modules before the BEV features are frozen; test samples processed chronologically; inter-trip fusion +3.81 All mIoU vs intra-trip +2.30; 70 % less memory than dense tiles; resolution 0.3 / 0.6 / 1.2 m gives +3.86 / +3.25 / +1.78.
- input: BEV feature, global pose.
- output: refined BEV for a map segmentation / detection head (HDMapNet, LSS, BEVFormer, VectorMapNet hosts).
- training: host losses; epoch counts not given.
- differentiable: yes end to end through the GRU; tiles written back by replacement.
- sensors_data: camera (+ lidar variants); nuScenes, 60 x 30 m, also 100 x 100 and 160 x 100 m; assumes precise GPS / IMU pose.
- results: HDMapNet 32.73 -> 37.05 All IoU; LSS 39.78 -> 44.80; BEVFormer mIoU 49.51 -> 55.01; VectorMapNet mAP 40.9 -> 44.8; range 160 x 100 m +4.65; night +7.73, rain +5.92 (All IoU).
- compute: not stated.
- code: none on the page.
- fit_to_us: reuse_idea - M - the gated conv-GRU as a learned fusion cell on a rolling local raster; not suitable as published (global pose, multi-traversal revisits of the same place).
- compat_needs: global pose; revisits of places. Cached-BEV: yes.
- compat_conflicts: conflicts with the registration-free premise; it is a place memory, so it is leakage-prone on random splits (VM46).
- tried_before: E40 / E48 map prior at the grid (+0.075 F1 exact pose, -0.057 unaligned) is the HD-map analogue.
- verified: partly.

### CARD VM20 - MapUnveiler (clip tokens and memory tokens)
- papers: Unveiling the Hidden: Online Vectorized HD Map Construction with Clip-Level Token Interaction and Propagation (Kim et al., NeurIPS 2024) https://arxiv.org/abs/2411.11002
- depth: M
- stage: S8
- idea: Processes clips (T = 3 frames, stride 2, overlap 1 frame): 50 clip tokens generated from the frame-level map queries interact with the clips' BEV features (BEV updater: BEV as queries, clip tokens as keys / values) and a map generator produces map tokens; 96 memory tokens, read and written by token summarisers, carry information between clips. No ego-pose alignment is mentioned.
- formulation: Nc = 50 (25 -> 69.2, 100 -> 68.5 mAP), M = 96; MapTRv2-style generator (one-to-one loss 2 / 5 / 0.005, PV seg 2; frame-level network also depth 3, BEV seg 1); frame-level MapNet pre-trained, then clip training; ResNet-50 + LSS, BEV 100 x 200; 24 ep nuScenes, 6 AV2; 8 A100, batch 16, AdamW 6e-4. First clip: memory written from learnable embeddings.
- input: frame BEV + queries for T frames.
- output: map tokens -> polylines.
- training: Hungarian (MapTR one-to-one).
- differentiable: partial.
- sensors_data: camera; nuScenes, AV2; 60 x 30 and 100 x 50 m; a geo-disjoint nuScenes split is reported.
- results: nuScenes mAP 60 x 30: 68.0 (SQD 63.9, StreamMapNet 62.9, MapTRv2 61.5); 100 x 50: 68.7; occlusion subset 63.8 vs MapTRv2 47.1; geo-disjoint split 39.4 vs MapTRv2 36.6, StreamMapNet 33.9; 12.7 FPS (A100).
- compute: 12.7 FPS.
- code: none; project page https://mapunveiler.github.io
- fit_to_us: reuse_idea - M - compact learned memory tokens instead of full BEV memories; low priority because gains over tracker-style memory are unclear and pose is ignored.
- compat_needs: clip dataloader; token summariser. Cached-BEV: yes (works on per-frame BEV + queries).
- compat_conflicts: map tokens are DETR queries.
- tried_before: none.
- verified: partly.

### CARD VM21 - MambaMap (state-space memory)
- papers: MambaMap: Online Vectorized HD Map Construction using State Space Model (arXiv 2507.20224; authors not read) https://arxiv.org/abs/2507.20224
- depth: M
- stage: S8
- idea: A sliding memory bank of the last N = 4 frames holds refined BEV features and refined instance queries (detached); history BEV is warped by a 4x4 affine, history queries are reordered to the current ones by Hungarian matching on L2 distance between query vectors; a gated diagonal-state-space block scans the BEV (50 x 100 = 5000 tokens, four directions) and the instance sequences (length (N+1) Nq).
- formulation: best variant DSS (Mamba and Mamba-2 scored lower); N = 6 gave 66.5 vs 67.3 at N = 4; loss 5 x smooth-L1 points + 50 x focal; Deformable-DETR-style decoder, Hungarian as StreamMapNet; two-stage training (single frame, then random sequences); R50 + BEVFormer, lr 2.5e-4, batch 8, 2 x A30, 30 / 24 epochs.
- input: BEV + queries of N + 1 frames.
- output: polylines.
- training: hungarian_set.
- differentiable: partial.
- sensors_data: camera; nuScenes, AV2; 60 x 30 and 100 x 50 m.
- results: nuScenes 60 x 30: 67.3 (12.7 FPS) vs StreamMapNet 63.4, SQD-MapNet 65.8; 100 x 50: 66.5; new split 30 m: 40.1 vs StreamMapNet 34.1, SQD 37.3, MemFusionMap 38.0; AV2 new split 61.0.
- compute: parameters not reported.
- code: announced https://github.com/ZiziAmy/MambaMap (may not be live)
- fit_to_us: reuse_idea - M - low priority; an SSM memory is an alternative to the GRU, with evidence only from this paper.
- compat_needs: BEV sequence. Cached-BEV: yes.
- compat_conflicts: query reorder by Hungarian.
- tried_before: E41 (GRU-type fusion).
- verified: partly.

### CARD VM22 - Mapping like a Skeptic (probabilistic BEV projection, confidence-weighted temporal merge)
- papers: Mapping like a Skeptic: Probabilistic BEV Projection for Online HD Mapping (arXiv 2508.21689; authors not read) https://arxiv.org/abs/2508.21689
- depth: M
- stage: B, S8
- idea: Two small CNNs on the camera features predict per-pixel offsets of a static geometric mapping and a covariance + confidence map; each BEV cell samples K image locations from the Gaussian and weights features by confidence x likelihood; the cell confidence is the weight sum. Historical BEV features and their confidences are warped to the current time and merged by confidence-weighted averaging (memory: last 20 BEV maps, T = 4 chosen by vehicle position), then a learned temporal module fuses them. The decoder follows MapTracker.
- formulation: K, losses, epochs and BEV grid are deferred to the supplement; 63.8 M parameters.
- input: camera features + historical BEV + confidences.
- output: polylines (MapTracker head).
- training: as MapTracker.
- differentiable: yes in the projection and merge; partial in the head.
- sensors_data: camera; nuScenes, AV2; 60 x 30 and 100 x 50 m, StreamMapNet-style new splits.
- results: mAP on the new splits: nuScenes 60 x 30 45.4 vs MapTracker 40.3 and StreamMapNet 33.5; 100 x 50 42.8 vs 36.2 / 23.0; AV2 60 x 30 72.4 vs 71.3 / 64.4; ablation nuScenes 41.0 -> 43.5 (probabilistic projection) -> 45.4 (full); 15.9 FPS (A100).
- compute: 63.8 M parameters.
- code: https://github.com/Fatih-Erdogan/mapping-like-skeptic
- fit_to_us: reuse_idea - S - confidence-weighted merging of warped past BEV evidence (weights = confidence) is a non-learned relative of log-odds fusion at feature level.
- compat_needs: per-cell confidence maps.
- compat_conflicts: the projection needs camera lifting (not available on cached BEV); the merge itself is portable.
- tried_before: E41 (unweighted GRU).
- verified: partly.

### CARD VM23 - MapTCL (bidirectional temporal consistency loss, training only)
- papers: MapTCL: Temporal Consistency Learning via Bidirectional Alignment for Vectorized HD Map Construction (arXiv 2608.05209, Aug 2026; authors not read) https://arxiv.org/abs/2608.05209
- depth: M (HTML read)
- stage: L, S8, E
- idea: An auxiliary training loss that penalises geometric and semantic differences between matched past and current instances (vector level, both directions after ego warp) and between warped past raster maps and the current one (raster level, mask focal). No inference cost.
- formulation: BVCL: past instances with confidence > tau = 0.3 are warped into the current frame, current ones into each past frame by inverse motion; matched by MapTR's hierarchical scheme (instance Hungarian with L1 class + smooth-L1 position cost, then best point order, N = 100, N_p = 20), forward and backward; terms: smooth-L1 between matched points (no Chamfer) and KL divergence between class distributions; temporal weights w = [9, 7, 5, 3, 1] (position), u = [1.0, 0.7, 0.5, 0.3, 0.1] (semantic), N_t = 5 past frames. RCL: a BEV segmentation head (mask focal, beta = 1.0) and a weighted mask focal loss between warped past raster maps and the current one, weights [1.0, 0.7, 0.5, 0.3, 0.1]. Detachment of past predictions not stated. tau ablation (AV2 StreamMapNet): 0.1 -> 54.0 mAP / 23.8 C-mAP, 0.5 -> 56.0 / 36.5.
- input: memory of N_t past predictions and rasters.
- output: none at inference.
- training: added to StreamMapNet (6 + 18 ep, loss only in the temporal stage), SQD-MapNet, MapTracker (6 / 2 / 16); nuScenes 24 ep, AV2 6 ep, 4 x A6000, batch 16.
- differentiable: partial (matching inside BVCL; RCL is fully differentiable).
- sensors_data: camera; nuScenes, AV2, new and original splits, 60 x 30 m (and 100 x 50 m).
- results: nuScenes new split mAP / C-mAP: StreamMapNet 35.2 / 25.7 -> 38.9 / 28.5; SQD-MapNet 36.6 / 25.4 -> 37.9 / 26.8; MapTracker 39.4 / 30.8 -> 41.0 / 32.4; AV2 new split StreamMapNet 53.6 / 35.4 -> 56.7 / 37.9; original nuScenes MapTracker 67.9 / 53.6 -> 70.2 / 59.8. Only C-mAP is reported as consistency metric (no mAS).
- compute: no inference cost; FPS unchanged.
- code: none on the page.
- fit_to_us: reuse_idea - S - consistency between the ego-warped previous prediction and the current one, applied both ways. We can drop the instance matching by using our point-to-polyline loss in both directions between the two predicted graphs (matching-free), and the raster part as a mask loss between warped rasterised graphs.
- compat_needs: memory of past predictions (detach policy open); ego warp. Cached-BEV: yes.
- compat_conflicts: the vector part as published needs Hungarian + permutation min.
- tried_before: none (temporal consistency is only a metric in our project).
- verified: checked (HTML); detachment unknown.

### CARD VM24 - DTCLMapper (dual temporal consistent learning: contrastive instances + occupancy consistency)
- papers: DTCLMapper: Dual Temporal Consistent Learning for Vectorized HD Map Construction (arXiv 2405.05518; authors not read) https://arxiv.org/abs/2405.05518
- depth: M
- stage: L, S8
- idea: Two training-time consistency components on a MapTR-style host: instance consistency (a vector point pre-selection module selects reference points from an instance map; contrastive learning pulls the current instance embedding to historical embeddings of the same class that are close in position) and map consistency (rasterise the predicted vector instances to a grid map and compare ego-warped historical grid maps with the current one by an L1 "map occupancy" loss; self-supervised, no GT).
- formulation: Loss_cst = log[1 + sum_c sum_{k+} sum_{k-} exp(v_c . k- - v_c . k+)]; Loss_ol = sum_{i=1..m} L1(V2GMap'_{t-i}, V2GMap_t) with ego rotation / translation warp (Eqs. 15 - 16); only instances with normal predicted class confidence are rasterised (threshold 0.4); total loss weights 2 / 5 / 0.005 (base) + 0.1 / 1 / 1 / 0.1 for the consistency terms (assignment to terms not read). Grid 0.15 m, map 400 x 200, range 60 x 30 m; batch 16, lr 1.5e-3, 24 ep nuScenes, 6 ep AV2, RTX A6000. Core temporal parts are training-time only; the Stream and Tracker variants still use memory at inference.
- input: images + history grid maps (training).
- output: as MapTR.
- training: hungarian_set + consistency losses.
- differentiable: yes for the occupancy loss through the rasteriser (soft rasteriser details not read).
- sensors_data: camera; nuScenes, AV2.
- results: nuScenes mAP DTCLMapper-l (LSS) 61.9, -b 53.6 (MapTR 50.3, PivotNet 57.6, MapTRv2* 59.9); +StreamMapNet 60.3 -> 61.4, +MapTracker 65.4 -> 66.9; AV2 65.1. No temporal-consistency metric reported.
- compute: not stated.
- code: https://github.com/lynn-yu/DTCLMapper
- fit_to_us: reuse_idea - S - a GT-free, matching-free temporal consistency loss between rasterised outputs of consecutive frames after ego warp.
- compat_needs: a differentiable rasteriser of our graph (VM32); ego warp. Cached-BEV: yes.
- compat_conflicts: none for the raster loss; the contrastive part needs instance embeddings with positions.
- tried_before: none.
- verified: partly.

### CARD VM25 - AMap (future-prior distillation into a single-frame student)
- papers: AMap: Distilling Future Priors for Ahead-Aware Online HD Map Construction (arXiv 2512.19150; authors not read) https://arxiv.org/abs/2512.19150
- depth: M
- stage: T, S8
- idea: A MapTracker-based teacher gets FUTURE context (4 cached future BEV features, future-frame queries aligned by an MLP); a student without any temporal path (R18, learnable queries only) is trained to match the teacher's BEV features (masked L2 with a GT-derived spatial mask) and query logits (Hungarian + KL). Targets better prediction of the road ahead.
- formulation: total = basic (BEV + vector + transformation) + distill (feature + logits), all weights 1.0; teacher R50 frozen, 4 frames sampled from 10; AdamW 5e-4, wd 0.01, cosine to 1.5e-6, batch 16, 8 H20. BEV range / epochs not stated.
- input: teacher: past + future frames; student: current frame.
- output: polylines.
- training: teacher-student; Hungarian for logits.
- differentiable: partial.
- sensors_data: camera; nuScenes, AV2 (MapTracker partitioning).
- results: nuScenes R18 student mAP 64.49 (A-mAP 66.28, R-mAP 65.11, 31.5 FPS) vs non-temporal baseline 62.81; R18 temporal MapTracker 69.34; AMap R50 non-temporal 69.26 (20.1 FPS); MapTR-based R18 baseline 32.35 -> 47.84.
- compute: 31.5 FPS (R18 student).
- code: none on the page.
- fit_to_us: reuse_idea - M - privileged-information distillation: a recursion that sees past AND future frames can serve as teacher for a causal student; gains were measured on the "ahead" region.
- compat_needs: future frames in training (we have the sequences). Cached-BEV: yes.
- compat_conflicts: logit distillation uses Hungarian.
- tried_before: none.
- verified: partly.

### CARD VM26 - MapTR (permutation-equivalent point sets, hierarchical matching)
- papers: MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction (Liao et al., ICLR 2023; authors / venue from memory) https://arxiv.org/abs/2208.14437 ; code read: decoder.py and maptr_assigner.py on GitHub
- depth: M+C
- stage: L, S4, S5
- idea: Every map element is an ordered set of N_v = 20 points with an EQUIVALENT-PERMUTATION group: polylines have 2 permutations (forward, reversed), polygons 2 N_v (cyclic shifts x 2 orientations). The loss uses, per matched pair, the permutation with the lowest point-to-point cost, so start point and direction are not forced. Hierarchical queries q_ij = q_i^ins + q_j^pt (N x N_v) go through a deformable decoder.
- formulation: instance matching cost = focal(class) + point2point Manhattan cost (Chamfer cost 47.5 vs 50.3 mAP), Hungarian; point level: gamma* = argmin over Gamma of sum_j Manhattan(v^_j, v_gamma(j)). Loss L = 2 L_cls(focal) + 5 L_p2p + 0.005 L_dir with L_dir = - sum cos(e^_j, e_gamma*(j)) (edge vectors, cyclic). Fixed order 44.4 vs permutation-equivalent 50.3 mAP; N_v 10 / 20 / 40 -> 48.0 / 50.3 / 50.0; queries N 25 / 50 / 75 -> 41.6 / 50.3 / 52.2; layers 1 / 2 / 3 / 6 / 8 -> 29.1 / 42.4 / 44.0 / 50.3 / 47.6. Code: reference points are updated per layer as sigmoid(offset + inverse_sigmoid(ref)) and DETACHED before the next layer; the assigner builds the cost per (query, GT, order) and takes the minimum over orders, then scipy `linear_sum_assignment` on CPU (the assigner's default point cost is a Chamfer class, the paper states Manhattan; the config was not seen).
- input: surround images (1600 x 900 scaled 0.5), ResNet-50, GKT BEV (alternatives IPM / LSS / deformable attention), 6 decoder layers.
- output: N = 50 instances x 20 points (normalised BEV coordinates) + class (3 classes).
- training: Hungarian + focal + L1 + direction; regime hungarian_set.
- differentiable: partial (Hungarian; the minimum over permutations passes gradients to the chosen order).
- sensors_data: camera (+ lidar variants); nuScenes; range x [-15, 15] m, y [-30, 30] m (30 x 60 m), BEV grid 0.3 m (tiny) / 0.75 m (nano); AP at Chamfer thresholds {0.5, 1.0, 1.5} m.
- results: nuScenes val: MapTR-tiny R50 110 ep 58.7 mAP (ped 56.2, div 59.8, bound 60.1), 24 ep 50.3, 11.2 FPS; nano R18 45.9 at 25.1 FPS; VectorMapNet 40.9, HDMapNet 23.0 (camera); extrinsic noise sigma_t 0.5 m -> 34.0 mAP (50.3 clean), sigma_rot 0.02 rad -> 42.0.
- compute: 35.9 M parameters (tiny), 89.3 ms per frame (backbone 62 %).
- code: https://github.com/hustvl/MapTR
- fit_to_us: reuse_idea - S - only the point-level "minimum over the permutation group" and the edge-direction cosine loss; the full model is the matched sparse decoder that E31 / E32 dropped.
- compat_needs: polyline element with a fixed point count; Hungarian for the instance level. Cached-BEV: the decoder yes, the GKT lifting no.
- compat_conflicts: needs Hungarian; our vertex sets have variable lengths and branching, which a fixed 20-point set cannot hold.
- tried_before: E31 / E32 (matched sparse DETR-style decoder dropped: grid +0.058 F1, +0.18 route F).
- verified: checked.

### CARD VM27 - MapTRv2 (decoupled attention, one-to-many, dense auxiliaries, centerlines)
- papers: MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction (Liao et al., IJCV 2024; authors / venue from memory) https://arxiv.org/abs/2308.05736
- depth: M
- stage: L, S4, B
- idea: Keeps MapTR's hierarchical query + permutation-equivalent matching and adds (1) decoupled self-attention (inter-instance and intra-instance separately, O(N^2 + N_v^2)), (2) an auxiliary one-to-many branch (T = 300 extra queries, each GT repeated K = 6 times) with the same hierarchical matching, (3) dense supervision (depth from lidar projected to perspective views, BEV segmentation, perspective-view segmentation), (4) a fourth class "centerline": direction is given, so the permutation group has one element; centerlines are treated as paths via a graph traversal (following LaneGAP).
- formulation: L = 1 L_one2one + 1 L_one2many + 1 L_dense, one2one weights 2 / 5 / 0.005 (class / p2p / direction); dense = 3 depth + 1 BEV seg + 2 PV seg; default BEVPoolv2 (LSS), range x [-15, 15], y [-30, 30] m, 0.3 m grid, 50 instances x 20 points, 6 layers, AdamW wd 0.01, lr 6e-4, batch 32, 24 / 110 epochs.
- input: surround images -> BEV (LSS / GKT / BEVFormer variants).
- output: as MapTR (+ centerline class).
- training: hungarian_set (+ one-to-many).
- differentiable: partial.
- sensors_data: camera, lidar fusion; nuScenes, AV2.
- results: nuScenes val mAP: R18 110 ep 52.3 (33.7 FPS), R50 24 ep 61.5, R50 110 ep 68.7 (14.1 FPS), V2-99 110 ep 73.4; camera + lidar R50 24 ep 69.0; with centerline class, 24 ep: 54.0 average mAP (as read); AV2 R50 6 ep 67.4 (2D), 64.7 (3D), centerline 62.6.
- compute: 14.1 FPS (R50).
- code: https://github.com/hustvl/MapTR
- fit_to_us: reuse_idea - S - (a) a centerline class with fixed direction and path-wise graph traversal is the published way to carry lane graphs in this family, (b) the one-to-many auxiliary is the published counter to sparse positive supervision (our dense heat map already supervises all cells).
- compat_needs: Hungarian; perspective-view features for the depth / PV auxiliary losses (not available).
- compat_conflicts: PV-specific heads unusable on cached BEV.
- tried_before: E31 / E32 (matched sparse decoder dropped).
- verified: partly (last 4 k characters of the page not read).

### CARD VM28 - PivotNet (dynamic number of pivot points, sequence matching by dynamic programming)
- papers: PivotNet: Vectorized Pivot Learning for End-to-end HD Map Construction (Ding et al., ICCV 2023) https://arxiv.org/abs/2308.16477
- depth: M+C (repository front page)
- stage: L, S1
- idea: Each element is represented by pivot points (the vertices of a Visvalingam-Whyatt simplification of the polyline; collinear points are removable) and a per-slot pivot classifier makes the vertex COUNT dynamic. Matching of predicted slots to GT pivots uses a dynamic-programming sequence alignment (O(N T)) with first and last point fixed, instead of Hungarian over permutations; a "point-to-line mask" module lets the point queries of one instance form an instance line feature that masks cross-attention over the BEV feature.
- formulation: L_DVS = 5 L_pivot-point(L1) + 2 L_collinear-point(linearity) + 2 L_pivot-class(BCE); line-aware mask loss (BCE + Dice) weight 5; BEV segmentation (BCE + Dice) weight 3; max points per instance N = 10 (divider), 2 (crossing), 30 (boundary), max instances 20 / 25 / 15, 200 / 50 / 450 point queries; VW threshold not stated; BEV 64 x 32 features for 30 m front / back, 15 m left / right; backbones EfficientNet-B0, ResNet-50, Swin-T; AdamW 2e-4, wd 1e-4, 24 / 110 epochs, 8 x 2080Ti, batch 1 per GPU. Direction / start point: not discussed, fixed by the endpoints in matching.
- input: surround images -> BEV features (4 enc / 4 dec).
- output: variable-length pivot sequences per instance, set-based (no autoregression).
- training: DP matching inside instances + instance assignment (not detailed).
- differentiable: partial (matching, pivot selection).
- sensors_data: camera; nuScenes (AV2 45.0 at 10 epochs).
- results: nuScenes mAP at thresholds {0.2, 0.5, 1.0}: SwinT 24 ep 40.8, 110 ep 49.2 (44.8 M parameters, 5.1 FPS) vs MapTR R50 27.3 (re-evaluated), VectorMapNet 21.3, HDMapNet 21.3; at {0.5, 1.0, 1.5}: 66.8 at 110 ep.
- compute: 44.8 M parameters, 5.1 FPS.
- code: https://github.com/wenjie710/PivotNet (front page: official, MMCV 1.7.1 + Detectron2 + custom CUDA ops, MIT licence stated)
- fit_to_us: reuse_idea - S/M - (a) a learned dynamic vertex count per element via a "keep this slot" classifier is the nearest published relative of our validity gate, (b) a simplified GT (VW here, Douglas-Peucker in our oracle: 14.3 vertices per 100 m against 22 learned in E94, PROJECT_CONTEXT) defines the vertex budget the keep gate should learn, (c) a linearity loss for non-pivot points keeps dropped vertices on the chord.
- compat_needs: ordered slots per instance; GT simplification. Cached-BEV: the matching/loss part yes.
- compat_conflicts: the sequence DP assumes one ordered chain per instance (no branching).
- tried_before: none directly (vertex budget measured against a Douglas-Peucker oracle in loop 14+).
- verified: partly.

### CARD VM29 - BeMapNet (piecewise Bezier representation, Point-Curve-Region loss)
- papers: End-to-End Vectorized HD-map Construction with Piecewise Bezier Curve (Qiao et al., CVPR 2023) https://arxiv.org/abs/2306.09700
- depth: M
- stage: L, S5
- idea: Each element is a piecewise Bezier curve <k, n> (k segments of degree n, consecutive segments sharing an endpoint): divider <3,2>, crossing <1,1>, boundary <7,3>. The head predicts, per instance, control points and an end index (softmax over the maximum number of pieces) that sets the piece count. The loss supervises control points, restored curve points and a rasterised region.
- formulation: GT fitting: sample m = 100 points, least squares C = B+ P with the Bernstein matrix, extend a segment while the Chamfer distance stays below epsilon; k+1 explicit points (on the curve) and nk - k implicit points as offsets from the segment centre. L = 5 L_point(L1 on control points) + 10 L_curve(L1 on the restored m points) + 1 L_region (Dice on dilated rasterisations, width 5); auxiliary BEV / instance mask CE + Dice (1, 5). IPM-PE Align: sin-cos encodings of image-grid points (flat-ground assumption) and BEV grid points aligned by one shared FC. Semantic BEV decoder 2 enc + 4 dec layers with 64 x 32 BEV queries; instance Bezier decoder 6 masked cross-attention layers with 60 instance queries (20 / 25 / 15). Range [30, 30, 15, 15] m, 0.15 m per pixel, input 896 x 512, AdamW 2e-4, 30 / 110 epochs, 8 GPUs. Start-point / direction ambiguity: not addressed; matching bipartite, cost not given.
- input: images -> BEV.
- output: control points + end index per instance (post-processing free).
- training: set prediction with bipartite matching.
- differentiable: partial (matching).
- sensors_data: camera; nuScenes 28,130 train / 6,019 val; thresholds {0.2, 0.5, 1.0} m.
- results: nuScenes mAP: Swin-T 30 ep 43.7 (div 49.1, cross 42.2, bound 39.9); Swin-B 110 ep 50.5; at {0.5, 1.0, 1.5}: 67.7. Abstract: at least +18.0 mAP over prior best in the same setting.
- compute: not stated here.
- code: https://github.com/er-muyue/BeMapNet
- fit_to_us: reuse_idea - S - Bezier pieces with a predicted piece count and a curve-restoration loss (control points -> dense points -> L1) is a per-element loss that needs no point correspondence between prediction and GT other than index order; relevant to our Bezier connectors.
- compat_needs: piece-count head; curve sampler. Cached-BEV: the Bezier head and PCR loss yes.
- compat_conflicts: IPM-PE is camera-specific.
- tried_before: E53 Bezier connectors (split / merge F1 0.55 / 0.61 -> 0.71 / 0.64, kept).
- verified: partly.

### CARD VM30 - HDMapNet (dense heads + clustering + direction-guided tracing)
- papers: HDMapNet: An Online HD Map Construction and Evaluation Framework (Li et al., ICRA 2022) https://arxiv.org/abs/2107.06307
- depth: M+C
- stage: S1, S6, S7
- idea: Per-pixel BEV prediction of semantic class, instance embedding and lane DIRECTION (classification over angle bins, forward and backward both positive), followed by classic post-processing: DBSCAN clustering of the embedding, max-pool NMS, and greedy tracing of vertices along the predicted direction with a fixed step. The vectors are never trained; splits and merges are not evaluated.
- formulation: losses: CE (semantic), discriminative loss (alpha = beta = 1, delta_v = 0.5, delta_d = 3.0), softmax CE on lane pixels (direction, weight not stated). Code: angle_class = 36 (10 degree bins, 37 channels with background), label line thickness 5 px, canvas 200 x 400 at 0.15 m (60 x 30 m), DBSCAN eps = 1.5 and min_samples = 50, NMS by 1 x 5 / 5 x 1 max-pooling, tracing step 7 pixels (1.05 m), top-2 direction bins with a 5-bin suppression window, instance confidence = mean probability over the mask, no resampling interval.
- input: images (EfficientNet-B0) + MLP/geometric view transform, optional lidar pillars; BEV decoder ResNet.
- output: raster heads, then polylines by rules.
- training: dense losses, no matching; regime none (supervised dense).
- differentiable: no after the heads (clustering, NMS, greedy tracing are rules).
- sensors_data: camera + lidar; nuScenes; 60 x 30 m.
- results: nuScenes semantic IoU: fusion 44.5, surround camera 32.9; instance AP at Chamfer {0.2, 0.5, 1.0} m: fusion 30.6, camera 22.7; bidirectional Chamfer distance of fusion 0.639 m. In MapTR's comparison (thresholds 0.5 - 1.5 m): camera 23.0, fusion 31.0 mAP.
- compute: 0.8 FPS camera (MapTR's table).
- code: https://github.com/Tsinghua-MARS-Lab/HDMapNet (PyTorch + torch-scatter, GPL-3.0 stated, no mmcv)
- fit_to_us: baseline - S - the dense-heads-plus-rule-tracing design is the ancestor of our dense head + rule system A (heat map, direction channels, greedy trace); direction as a CLASSIFICATION over bins with both directions positive is the published way to avoid a direction ambiguity without matching.
- compat_needs: any BEV raster. Cached-BEV: yes (heads and post-processing are independent of the encoder).
- compat_conflicts: the rule post-processing is not differentiable; no fork / merge handling.
- tried_before: dense head + tracer (E31 / E32 grid kept), rule system A.
- verified: checked (paper text + code).

### CARD VM31 - VectorMapNet (two-stage detector + autoregressive polyline generator)
- papers: VectorMapNet: End-to-end Vectorized HD Map Learning (Liu et al., ICML 2023; authors from memory) https://arxiv.org/abs/2206.08920
- depth: M
- stage: L, S5, X
- idea: A DETR-style detector predicts per element a class and keypoints (bounding-box corners, k = 2, best), then an autoregressive transformer decodes the vertices of each polyline as discrete tokens (coordinates quantised on a 200 x 100 grid of 0.3 m cells, EOS token ends the polyline). Order encodes direction.
- formulation: detector Hungarian loss (NLL class + smooth-L1 + IoU on keypoints; coefficients garbled on the page); generator NLL over quantised tokens, teacher forcing on GT keypoints by default, a second stage fine-tunes on predicted keypoints (exposure bias); augmentation: Gaussian noise added to input vertices with probability 0.3; vertices sampled by curvature (RDP), fixed 1 m spacing worse; no canonical-direction rule or start-point rule described; decoding rule (greedy / sampling) not stated. IPM camera BEV at heights -1, 0, 1, 2 m, 200 x 100 x 128 features, ResNet-50, N_max = 100 elements, 6 + 6 layers, dropout 0.2, 110 epochs, batch 32, 8 x 3090.
- input: BEV (camera IPM / lidar PointPillars / fusion).
- output: variable-length vertex sequences.
- training: Hungarian + NLL, teacher forcing then fine-tuning.
- differentiable: no end to end (tokenisation, argmax decoding).
- sensors_data: camera, lidar; nuScenes, 60 x 30 m; thresholds {0.5, 1.0, 1.5} m.
- results: nuScenes mAP: camera 40.9 (ped 36.1, div 47.3, bound 39.3); camera + fine-tune 46.0; lidar 34.0; fusion 45.2, fusion + fine-tune 53.7.
- compute: not stated.
- code: https://github.com/Mrmoore98/VectorMapNet_code (linked as "Code" on the project page https://tsinghua-mars-lab.github.io/vectormapnet/; repository not opened)
- fit_to_us: not_suitable - L - autoregressive tokens are not differentiable through the sequence; reuse only the exposure-bias remedy (second-stage fine-tuning on own predictions) and the vertex-noise augmentation.
- compat_needs: a sequence order of the vertices.
- compat_conflicts: argmax / tokenisation break gradients; no branching.
- tried_before: E94 teacher-forced tracer (partial: F1 0.851 vs rules 0.867).
- verified: partly.

### CARD VM32 - MapVR (differentiable rasterisation as supervision)
- papers: Online Map Vectorization for Autonomous Driving: A Rasterization Perspective (Zhang et al., NeurIPS 2023) https://arxiv.org/abs/2306.10502
- depth: M
- stage: L, E
- idea: Vector outputs are rendered by a differentiable rasteriser and supervised with a Dice loss against the rasterised GT; the branch is removed at inference. Also proposes a rasterisation-based AP (AP_raster) that is more sensitive to small deviations.
- formulation: polylines I = exp(-D / tau) with D = distance of a pixel to the nearest segment, tau = 2.0 best (0.5 - 6.0 tested); polygons sigmoid(C D / tau) with inside / outside sign; L = l1 L_render(Dice) + l2 L_cls(focal) + l3 L_dir + l4 L_reg(L1), weights not stated; Hungarian matching with cost l1 L_render + l2 L_cls + l3 L_reg, so supervision is per matched instance; training raster 256 x 128 (inferred default), metric raster 480 x 240 at 0.125 m, dilation 2 px, IoU thresholds 0.25:0.05:0.50 (lines), 0.50:0.05:0.75 (polygons). Direction / start point: not addressed; the distance-based rendering is order-independent (reader's inference).
- input: MapTR host (R50, GKT).
- output: polylines (raster branch training-only).
- training: hungarian_set + raster loss.
- differentiable: yes through the rasteriser, Hungarian for pairing.
- sensors_data: camera, camera + lidar; nuScenes, AV2; +-15 x +-30 m.
- results: MapTR -> MapTR + MapVR on nuScenes camera 24 ep: Chamfer AP 50.3 -> 51.2, AP_raster 24.3 -> 31.2; 110 ep AP_raster 35.0 -> 38.5; no inference change (18.4 FPS).
- compute: no inference cost.
- code: https://github.com/ZhangGongjie/MapVR ; metric toolkit https://github.com/jiahaoLjh/MapVectorizationEvalToolkit
- fit_to_us: reuse_idea - S/M - the soft line rasteriser exp(-D / tau) is a differentiable bridge from our vertex output to the dense BEV (also needed by VM13 / VM24 / VM33-type losses and memories); related to but not the same as our point-to-polyline loss.
- compat_needs: rasteriser (a pure-PyTorch version is feasible at 200 x 200). Cached-BEV: yes.
- compat_conflicts: published use is per matched instance.
- tried_before: E120 point-to-polyline loss both ways (the geometric analogue).
- verified: partly.

### CARD VM33 - MapGR (global raster loss over all queries)
- papers: Learning Global Representation from Queries for Vectorized HD Map Construction (arXiv 2510.06969; authors not read) https://arxiv.org/abs/2510.06969
- depth: M
- stage: L
- idea: Adds a whole-map objective: every query feature is mapped (MLP -> small 2D map), all are concatenated, passed through a light conv net and upsampled to a BEV raster supervised by the GT raster (BCE), so unmatched queries also get gradients; a guidance branch encodes the predicted global map and fuses it back into each query (first two of six decoder layers only, because the global constraint conflicts with final convergence).
- formulation: GRL loss = BCE(pred raster, GT raster), weight ratio global : other = 1.0 : 0.1 in one place and 0.2 in a table (inconsistent); gradient weakening 0.8; global embedding dim 2048 best (512 -> 64.6, 2048 -> 65.0, 8192 -> 64.2 mAP2); 50 (MapTR) / 100 (MapQR) queries; range x [-15, 15], y [-30, 30] m; 24 / 110 epochs; lr 6e-4; R50.
- input: queries of a MapTR-family decoder.
- output: raster + polylines.
- training: baseline matching kept for instance losses; the global loss needs no matching.
- differentiable: yes.
- sensors_data: camera; nuScenes, AV2; standard splits (no new split).
- results: nuScenes mAP1 / mAP2: MapTR 24 ep 27.3 / 50.3 -> 31.2 / 54.5; MapTRv2 24 ep 37.2 / 61.5 -> 41.1 / 65.0; MapQR (reproduced) 42.1 / 65.3 -> 45.3 / 68.1; ablation MapTRv2 mAP2: GRL only 63.7, GRL + GRG 65.0.
- compute: +11 M parameters, FPS unchanged (MapTRv2 19.6 -> 19.4).
- code: none given (supplement).
- fit_to_us: reuse_idea - S - a MATCHING-FREE loss on the aggregate of ALL output elements (rasterise the whole output graph with validity weights and compare with the GT raster) that supervises duplicates and false positives without any assignment.
- compat_needs: a rasteriser (VM32) or a learned small decoder as in the paper.
- compat_conflicts: none.
- tried_before: none (we supervise the dense head, not the raster of the output graph).
- verified: partly.

### CARD VM34 - DAMap (distance-aware soft classification targets, task-modulated attention)
- papers: DAMap: Distance-aware MapNet for High Quality HD Map Construction (Dong et al., ICCV 2025) https://arxiv.org/abs/2510.22675
- depth: M
- stage: L, S3
- idea: In one-to-many matching all candidates of a GT get label 1 although their localisation differs. DAMap (i) replaces the positive classification target by a localisation confidence P_dist = exp(-lambda * L1 distance between predicted and assigned GT points), lambda = 1, in a distance-aware focal loss DAFL(p, y) = -(y - p)^gamma (y log p + (1 - y) log(1 - p)); (ii) a hybrid scheme: ordinary focal loss in the first decoder layer(s), DAFL in the later ones (one FL layer best: 61.6 vs 60.9 all-DAFL); (iii) task-modulated deformable attention: query split into Q_cls / Q_loc with task-specific attention weights, shared sampling offsets.
- formulation: see idea; query count, K, loss weights not stated; lr 3e-4 cosine, batch 16, 4 x 3090, 24 / 6 / 110 epochs, R50.
- input: MapTRv2 / MapQR / Mask2Map hosts.
- output: polylines + class (quality-aware scores).
- training: one-to-many matching.
- differentiable: partial.
- sensors_data: camera; nuScenes, AV2; range x +-15, y +-30 m.
- results: nuScenes R50 24 ep, hard set {0.2, 0.5, 1.0} m / easy set {0.5, 1.0, 1.5} m: MapTRv2 36.6 / 60.4 -> 39.0 / 62.8; MapQR + DAMap 46.0 / 68.8; Mask2Map + DAMap 51.2 / 72.6; 110 ep MapTRv2 + DAMap 47.4 / 70.4.
- compute: 12.1 FPS, 52 M parameters (vs 40 M).
- code: announced https://github.com/jpdong-xjtu/DAMap (not verified live)
- fit_to_us: reuse_idea - S - a calibrated VALIDITY target: train the keep probability toward the localisation quality (exp of the distance to the GT polyline) instead of a hard 0/1; with our point-to-polyline distance this needs no matching; same family as InteractionMap (VM17) and Sparse4D v3 (VM06).
- compat_needs: a distance of each output element to GT. Cached-BEV: the loss yes.
- compat_conflicts: the published form assumes an assigned GT.
- tried_before: keep gate of E120 (0.872) - soft versus hard targets not stated in PROJECT_CONTEXT.
- verified: partly.

### CARD VM35 - ADMap (anti-jitter direction-difference loss)
- papers: ADMap: Anti-disturbance framework for reconstructing online vectorized HD map (Hu et al., arXiv 2401.13172; ECCV 2024 version retitled "Anti-Disturbance Framework for Vectorized HD Map Construction" per search) https://arxiv.org/abs/2401.13172
- depth: M
- stage: L, E
- idea: Targets "point-order jitter" - jagged predicted point sequences from prediction bias - with three parts: a multi-scale perception neck (BEV features downsampled, upsampled and merged; per-level maps feed the decoder during training only), instance interactive attention (instance self-attention first, then point self-attention inside each instance) and a vector direction difference loss comparing the direction of consecutive predicted segments with the corresponding GT segments, weighting sharper GT turns more.
- formulation: VDDL: cosine between predicted segment j and GT segment j (normalised dot product); weight W = exp((1 - cos_gt) / 2) where cos_gt compares each GT segment with its predecessor (endpoints weight 1.0); loss = sum (1 - cos_pred) * W over segments, range [0, 2]; training weight 1.0 (0.0 - 3.0 ablated, 1.0 best, 53.3 mAP camera-only); L1 weight 5.0; v2 adds dense and one-to-many losses (replication 3). Jitter measure (Table 8): among predicted instances whose internal chamfer-distance sum is below 1.5, the mean chamfer distance to GT: MapTR 0.597, ADMap 0.518, MapTRv2 0.560, ADMapv2 0.422. Point cloud range [-15, -30, 15, 30], voxel 0.15 m; R50 / SECOND; 8 GPUs, batch 8, lr 6e-4, AdamW cosine, 24 / 110 epochs.
- input: MapTR-family BEV features (camera or lidar).
- output: MapTR-family polylines.
- training: MapTR-style matching (details not stated).
- differentiable: partial.
- sensors_data: camera, lidar; nuScenes, AV2.
- results: nuScenes camera mAP ADMap 54.5 vs MapTR 50.3; ADMapv2 62.9 vs MapTRv2 61.5; C&L ADMap 68.0 vs 62.5, ADMapv2 70.8 vs 69.0 (110 ep 72.7); AV2 camera 64.7 / 68.7.
- compute: no inference change for the multi-scale supervision; FPS not stated.
- code: https://github.com/hht1996ok/ADMap
- fit_to_us: reuse_idea - S - a turn-aware direction-consistency regulariser on consecutive vertex segments for E120-type position refinement; it needs GT segments aligned with predicted ones (for us: the projection of each predicted segment onto the GT polyline).
- compat_needs: index-aligned or projected GT segments.
- compat_conflicts: index alignment comes from matching in the published form.
- tried_before: E120 refinement (point-to-polyline both ways, +0.012 F1; vertex error 0.54 -> 0.43 m); a direction-difference regulariser: none.
- verified: partly.

### CARD VM36 - GeMap (Euclidean-invariant geometric clues, geometry-decoupled attention)
- papers: GeMap: Online Vectorized HD Map Construction using Geometry (authors not on the page, ECCV 2024 from memory) https://arxiv.org/abs/2312.03341
- depth: M
- stage: L, S5
- idea: Describes each polyline by translation- and rotation-invariant quantities - edge lengths d_u, turning angles alpha_u, and inter-instance angles and point distances - and adds an Euclidean loss on them to the MapTR loss; geometry-decoupled attention applies a shape attention (same instance) then a relation attention (different instances).
- formulation: L = 0.005 L_Euc + 2 L_cls + 5 L_pts + 0.005 L_dir + 1 L_seg + 3 L_dep (+ BEV 1, PV 2); angle term |cos a^ - cos a| + |sin a^ - sin a|; GDA beats two plain self-attention layers by +1.2 mAP (51.4 -> 52.6); N = 50 instances x 20 points (10 better for dividers), 6 blocks, GKT 1 layer, lr 6e-4 batch 4 (R50), 110 epochs nuScenes / 24 AV2.
- input: MapTR-family.
- output: polylines.
- training: Hungarian (MapTR matching).
- differentiable: partial.
- sensors_data: camera, lidar; nuScenes, AV2; no geographic split, no temporal part.
- results: nuScenes camera R50 110 ep with dense losses 69.4 mAP (div 69.8, ped 67.1, bnd 71.4), 15.8 FPS; without dense 62.7; V2-99 76.0; Swin-T 72.0; AV2 71.8; weather split sunny / cloudy / rainy 73.1 / 71.0 / 59.3.
- compute: 15.8 FPS.
- code: https://github.com/cnzzx/GeMap
- fit_to_us: reuse_idea - S - pose-invariant shape descriptors (lengths and turning angles) as a loss are invariant to start point and global shift (not to direction reversal of the angle sign); a vertex-shape regulariser for our graph.
- compat_needs: ordered polyline.
- compat_conflicts: ordered chains only.
- tried_before: none.
- verified: partly.

### CARD VM37 - AugMapNet (predicted raster fed back into the latent BEV grid)
- papers: AugMapNet: Improving Spatial Latent Structure via BEV Grid Augmentation for Enhanced Vectorized Online HD Map Construction (Monninger et al., WACV 2026) https://arxiv.org/abs/2503.13430
- depth: M
- stage: B, S1
- idea: A parallel raster decoder predicts a 3-class segmentation (100 x 50 x 3) from the latent BEV grid; a 4-layer CNN encodes it ({32, 64, 128, 256} channels) and ADDS it back to the BEV grid before the vector decoder. The raster prediction is effectively detached ("immutable prior").
- formulation: B_aug = B + e_raster(M^_raster); Dice loss for the raster branch (weight not stated); host StreamMapNet (also SQD-MapNet); lr 2.5e-4, batch 1, 8 V100, 24 epochs; 52.70 M parameters (+6.3 %). Argument: vector decoders supervise a sparse subset of BEV cells, raster decoders all cells.
- input: BEV grid.
- output: polylines (host) + raster.
- training: hungarian_set (host) + Dice.
- differentiable: partial (raster path detached by design).
- sensors_data: camera; nuScenes (location-disjoint split), AV2; 60 and 100 m ranges.
- results: nuScenes location-disjoint mAP: StreamMapNet 33.8 -> 38.3 (+13.3 %) at 60 m, 23.2 -> 28.9 at 100 m; SQD-MapNet 36.9 -> AugSQD 40.5; oracle GT raster 91.4; AV2 60 m 57.1 -> 58.8.
- compute: +6.3 % parameters.
- code: https://github.com/tmonnin/augmapnet
- fit_to_us: reuse_idea - S - this is what our design already does (dense head heat map + direction channels concatenated to the 39 feature channels = 42-channel trace-network input); outside evidence that it helps vector decoding and that an oracle raster is a very high upper bound (91.4).
- compat_needs: a raster head on the BEV.
- compat_conflicts: none.
- tried_before: dense head + tracer (E31 / E32, grid kept).
- verified: partly.

### CARD VM38 - Mask2Map (BEV masks -> instance queries -> key points; inter-network denoising)
- papers: Mask2Map: Vectorized HD Map Construction Using Bird's Eye View Segmentation Masks (Choi et al., ECCV 2024) https://arxiv.org/abs/2407.13517
- depth: M
- stage: S1, S2, T
- idea: Stage 1 (IMPNet) produces Mask2Former-style instance queries and BEV masks; stage 2 (MMPNet) refines point queries with (i) a positional query from the mask-weighted sinusoidal embedding (pixels > 0.3) and (ii) a geometric feature extractor: pixels > 0.8, strongest pixel per 4 x 4 window ("max survival"), farthest-point sampling of N_S = 20 key points, BEV features pooled at those points; then a deformable decoder. Inter-network denoising (INDT): noisy GT queries (class flip probability 0.2; GT masks built from noisy ordered points, magnitude not stated) are matched to their own GT without Hungarian, to fix inconsistent assignments between the two networks (query utilisation 24.7 % -> 74.7 %).
- formulation: masks tau_M = 0.5 for masked attention; BEV encoder 6 layers, mask transformer 3 layers, decoder 6 layers; loss = L_seg + L_map (L1, focal, cosine direction) + L_aux (depth, 2D seg) + L_dn, apparently unweighted; range x +-15, y +-30 m; R50; AdamW 6e-4, 24 / 110 epochs, 4 x 3090.
- input: BEV features + BEV masks.
- output: polylines.
- training: Hungarian for learnable queries, identity assignment for noisy ones.
- differentiable: partial.
- sensors_data: camera, lidar fusion; nuScenes, AV2.
- results: nuScenes camera R50 24 ep 71.6 mAP (MapTRv2 61.5), 10.1 FPS; 110 ep 74.6 (68.7); C + L 78.4; rasterised metric 54.7 vs 36.7; AV2 71.5 vs 67.4.
- compute: 10.1 FPS.
- code: https://github.com/SehwanChoi0307/Mask2Map
- fit_to_us: reuse_idea - S - key-point selection from a thresholded mask (4 x 4 max survival + FPS) is a published analogue of our heat-map-peak seeds; INDT shows identity assignment for noised queries.
- compat_needs: a BEV mask (our dense head's heat map could play that role).
- compat_conflicts: stage-1 mask queries are trained jointly with the BEV encoder; the paper does not test external BEV masks.
- tried_before: seeds = heat-map peaks (192 peaks limited E94).
- verified: partly.

### CARD VM39 - MGMap (mask-guided instance queries and point-patch refinement)
- papers: MGMap: Mask-Guided Learning for Online Vectorized HD Map Construction (Liu et al., CVPR 2024) https://arxiv.org/abs/2404.00876
- depth: M
- stage: S2, S5
- idea: An enhanced multi-level neck (3 residual blocks with channel / spatial attention, upsample and fuse) feeds (i) a mask-activated instance decoder - 50 instance masks M_ins = sigmoid(Conv(F)), lane query = mask x features, then a deformable decoder with iterative point offsets - and (ii) position-guided mask patch refinement: for each point a patch of normalised size 0.1 around it is ROI-aligned to a 5 x 5 feature on a densified binary mask feature + position grid, refreshed by attention over 2 stages and regressed.
- formulation: L_ins weight 2, L_b weight 15 (mask CE + Dice); other weights not stated; BEV 200 x 100 over 60 x 30 m (also 60 x 60, 30 x 90); 50 instances x 20 points; AdamW 6e-4, batch 6 per GPU, 8 V100, 30 / 110 epochs; no separate dense head (masks from the same feature, training auxiliary).
- input: BEV multi-level features.
- output: polylines.
- training: bipartite matching (cost not stated).
- differentiable: partial.
- sensors_data: camera, lidar; nuScenes, AV2 subset.
- results: nuScenes camera R50 30 ep Chamfer avg 61.4 (ped 57.4, div 63.5, bou 63.3), raster avg 37.2, 11.6 FPS; 110 ep 66.5; lidar (SECOND) 67.9; fusion 71.7. No temporal part.
- compute: 11.6 FPS.
- code: https://github.com/xiaolul2/MGMap
- fit_to_us: reuse_idea - S - local patch pooling around predicted points on a mask feature is the point-level refinement idea of our E120 (bilinear samples at predicted positions).
- compat_needs: a mask feature. Cached-BEV: yes.
- compat_conflicts: matching.
- tried_before: E120 refinement (+0.012 F1).
- verified: partly.

### CARD VM40 - MapQR (scatter-and-gather query)
- papers: Leveraging Enhanced Queries of Point Sets for Vectorized Map Construction (MapQR; authors not on the page, ECCV 2024 from memory) https://arxiv.org/abs/2402.17430
- depth: M
- stage: S4, S5
- idea: Self-attention runs among the N = 100 INSTANCE queries only (cheaper than MapTR's N x N_v point queries); each instance query is then scattered into n = 20 copies with a shared content vector and individual positional embeddings from the reference points, cross-attended to the BEV, and gathered back by concatenation + MLP. Pluggable into MapTR, MapTRv2 and StreamMapNet.
- formulation: layer = gather(CA(scatter(SA(q)) + {P_ij}, F_bev)); P_ij = LP(PE(A_ij)); matching and loss as MapTRv2 (one-to-one + one-to-many + dense); GKT-h BEV 200 x 100 over +-15 x +-30 m; lr 6e-4; 24 / 110 epochs.
- input: BEV features.
- output: polylines.
- training: hungarian_set + one-to-many.
- differentiable: partial.
- sensors_data: camera; nuScenes, AV2; no temporal part (host-only).
- results: nuScenes mAP1 (0.2 / 0.5 / 1.0 m) / mAP2 (0.5 / 1.0 / 1.5 m): R50 24 ep 43.3 / 66.4 at 17.9 FPS; 110 ep 50.5 / 72.6; SwinB 24 ep 45.6 / 70.1; plug-in gains: MapTR 27.3 / 50.3 -> 34.4 / 58.0, MapTRv2 37.2 / 61.5 -> 40.1 / 63.4, StreamMapNet 36.7 / 61.1 -> 38.4 / 62.1; AV2 2D mAP2 68.2 vs MapTRv2 67.4.
- compute: 17.9 FPS.
- code: https://github.com/HXMap/MapQR
- fit_to_us: reuse_idea - S/M - an instance token that scatters into points and gathers back is our "step token -> vertex positions" pattern.
- compat_needs: Hungarian (as published). Cached-BEV: the decoder yes.
- compat_conflicts: Hungarian.
- tried_before: none.
- verified: partly.

### CARD VM41 - MapLightning (1D map tokens instead of a BEV grid; near-extrapolation split)
- papers: MapLightning: Online Vectorized HD Map Construction with 1D Map Tokens (arXiv 2610.01905, Oct 2026; authors not read) https://arxiv.org/abs/2610.01905
- depth: M (first 100 k characters)
- stage: B, S8
- idea: K learnable map tokens (K = 50 x cameras x timesteps = 1200 on nuScenes) are concatenated with flattened image tokens of the current and 3 past frames and processed by a 4-layer full self-attention Transformer; only the map tokens are kept and decoded by a MapTRv2-style 6-layer decoder with full cross-attention. No intrinsics, extrinsics, depth or ego-pose alignment. Past-frame features are cached.
- formulation: learned row / column, camera and time embeddings; matching, loss weights, epochs, lr not in the visible text; 16.7 x fewer intermediate tokens than a dense BEV (20 k vs 1.2 k).
- input: image tokens of T = 4 frames.
- output: polylines.
- training: not stated (MapTRv2-style decoder).
- differentiable: partial.
- sensors_data: camera; nuScenes, AV2; default range 60 x 30 m on the Near-Extrapolation split, also StreamMapNet split and original split.
- results: (mAP / FPS) Near-Extrapolation split, ResNet-50: MapTR 19.0, StreamMapNet 26.1, MapTracker 27.5, MapTRv2 26.7, AugMapNet 27.5, ours (self-attention) 35.3 at 18.9 FPS (nuScenes); Argo2 63.3. StreamMapNet split: 45.8 (nuScenes 60 x 30), 42.1 (100 x 50); original split 77.7 (paper notes memorisation). Real-time variants 30.3 - 30.6 mAP at 40 - 43 FPS.
- compute: 18.9 FPS (R50).
- code: none ("will be released").
- fit_to_us: not_suitable - S - removes the BEV grid altogether; only the split protocol and the fact that the best published numbers on location-disjoint splits are about 45 mAP are useful.
- compat_needs: raw image tokens.
- compat_conflicts: no use of cached BEV.
- tried_before: none.
- verified: partly.

### CARD VM42 - GaussianMap (Gaussian primitives splatted to BEV)
- papers: GaussianMap: Learning Gaussian Representation for Multi-Sensor Online HD Map Construction (arXiv 2606.31177, Jun 2026; authors not read) https://arxiv.org/abs/2606.31177
- depth: M
- stage: B
- idea: 20,000 Gaussian primitives (mean, scale, rotation, opacity, feature) replace the dense BEV grid; they sample LSS camera (and lidar) BEV features, interact by self- and cross-attention and are splatted (mixture responsibilities at cell centres) into a BEV feature that feeds a MapQR + DAMap decoder.
- formulation: M = 20,000 matched to 200 x 100 cells; init uniform means, fixed scales, identity rotation, opacity 0.5; 3-layer encoder, 16 reference points per Gaussian; loss = map (class, p2p, direction) + dense (depth, BEV seg, PV seg), lambdas not given; 6 layers, 100 queries x 20 points; lr 6e-4, batch 32, 24 / 6 epochs.
- input: sensor features.
- output: polylines.
- training: not stated.
- differentiable: yes in the splatting.
- sensors_data: camera, lidar; nuScenes, AV2 standard val; no radar; no temporal part.
- results: nuScenes val camera 70.5 mAP (MapQR + DAMap 68.8, MapTRv2 61.5); camera + lidar 78.3 (75.2, 69.0); AV2 camera 6 ep 70.1, C + L 79.0.
- compute: FPS and parameters not reported.
- code: none ("will be made publicly available").
- fit_to_us: not_suitable - M - an encoder-side representation; the splatting idea could turn sparse graph vertices into a dense feature but nothing else transfers.
- compat_needs: sensor encoders.
- compat_conflicts: replaces the BEV encoder.
- tried_before: none.
- verified: partly.

### CARD VM43 - MapDiffusion (diffusion over polyline vertices, sample-based uncertainty)
- papers: MapDiffusion: Generative Diffusion for Vectorized Online HD Map Construction and Uncertainty Estimation in Autonomous Driving (arXiv 2507.21423; authors not read) https://arxiv.org/abs/2507.21423
- depth: M
- stage: S5, X
- idea: Polyline vertex coordinates are diffused (forward noise on GT, reverse by a DETR-style decoder with deformable cross-attention to a BEV grid and a time-step embedding); class scores are not diffused. Several samples give a per-location variance map as uncertainty.
- formulation: T = 1000, cosine schedule; inference DDIM eta = 0.5, 5 steps; outputs with score below 0.5 are re-noised; n = 10 samples rasterised and Gaussian-smoothed (3 m kernel, sigma 0.6 m), U = sum over classes of the sample variance (unnormalised); fixed l queries, padding GT with Gaussian noise (mean 0.5, sd 0.25) best; permutation handling and matching not described; BEV 100 x 50 over 60 x 30 m, StreamMapNet configuration (24 ep, lr 2e-4).
- input: BEV grid (trained encoder).
- output: sampled polylines.
- training: denoising.
- differentiable: partial.
- sensors_data: camera; nuScenes (StreamMapNet split); temporal only via StreamMapNet BEV aggregation.
- results: nuScenes (location-disjoint) mAP 35.6 vs StreamMapNet 33.8, SQD-MapNet 33.1, MapTRv2 29.9, MapVR 22.8, MapTR 22.1, VectorMapNet 18.0; 8.0 FPS with 5 steps (12.8 with 1); uncertainty is 31 % higher in invisible than visible areas (Occ3D visibility); a frozen pretrained encoder reached only 31.7 mAP.
- compute: 8.0 FPS.
- code: none found.
- fit_to_us: not_suitable - M - iterative generative sampling is costly; the sample-variance uncertainty map and the -3.9 mAP for a FROZEN encoder are the only transferable facts.
- compat_needs: BEV grid.
- compat_conflicts: stochastic sampling vs a deterministic recursion.
- tried_before: none.
- verified: partly.

### CARD VM44 - mAS (Stability Under Scrutiny)
- papers: Stability Under Scrutiny: Benchmarking Representation Paradigms for Online HD Mapping (Shan et al., ICLR 2026, arXiv 2510.10660) https://arxiv.org/abs/2510.10660
- depth: M
- stage: E
- idea: A frame-pair stability benchmark. For an anchor frame t one frame t+k (k random in 1..M, M = 2) is drawn; predictions are matched to GT instances by Hungarian in each frame; predictions that map to the same GT instance in both frames are paired; the earlier one is moved by the ego pose into the later frame; presence, localisation and shape scores are combined into a per-instance stability.
- formulation: presence = 1 if both frames agree on the detection decision at threshold tau, else 0.5 (tau value not given); localisation: both polylines clipped to the range and sampled at N = 100 equidistant x values, Loc = beta * mean |y_{t+k}(x_i) - y_t(x_i)| with beta = 15 (the printed formula lacks the exponential the text mentions); shape = 1 - |kappa_{t+k} - kappa_t| / pi with kappa the mean turning angle; per instance Stability = Presence * [0.7 Loc + 0.3 Shape]; mAS = mean over classes of the class means (not an AP-style PR curve).
- input: per-frame predictions, GT instances, ego poses.
- output: scores (presence, localisation, shape, mAS).
- training: none.
- differentiable: no.
- sensors_data: nuScenes val (no geo split mentioned); uses GT for pairing.
- results: (nuScenes val, M = 2) MapTR (camera, GKT) mAP 44.1 / mAS 71.6; MapTR (C&L) 62.8 / 74.0; StreamMapNet 63.3 / 91.9; MapTracker (R50) 75.95 / 90.4; range of mAS 66.6 - 91.9 in one section, 71.6 - 91.9 in another. mAP and mAS are only loosely related; BeMapNet and PivotNet get 100 % presence by construction (dynamic vectorised sequence); temporal fusion helps MapTR with BEVFormer (71.3 -> 73.0) and hurts with GKT (71.6 -> 66.6); R18 -> R50 lowers MapTR localisation stability (75.0 -> 65.4); 42 models and variants over 10 families.
- compute: not applicable.
- code: toolkit announced at https://stablehdmap.github.io/ (release not checked)
- fit_to_us: reuse_idea - M - adapt as a temporal metric on our output: for each frame pair use ego motion, pair elements through GT, and compare presence (validity agreement), position offset along the polyline and turning-angle difference. Our "temporal consistency" metric in PROJECT_CONTEXT has no definition there; this gives one.
- compat_needs: per-frame predictions with identity via GT; ego pose.
- compat_conflicts: designed for ordered 3-class polylines (the x-sampling assumes a line that is a function of x within the range).
- tried_before: temporal consistency is listed among our metrics; its definition is not in PROJECT_CONTEXT.
- verified: partly (page truncated at 100 k of 134 k characters).

### CARD VM45 - C-mAP and G-mAP (consistency-aware and global-map metrics)
- papers: MapTracker (consistency-aware mAP, VM04) https://arxiv.org/abs/2403.15951 ; HisTrackMap (G-mAP, VM15) https://arxiv.org/abs/2503.07168
- depth: M
- stage: E, D
- idea: C-mAP: per-frame Chamfer matching as in the usual mAP, but a match is counted false positive if the GT track was earlier matched to a different predicted ID or an unmatched ancestor invalidates it; baselines without IDs get tracks by IoU of rasterised masks with a look-back window; computed on a "consistent GT" in which elements are merged over time. G-mAP: local GT vectors of N_seq frames are transformed to a global frame and rasterised; per-frame predictions are merged with MapTracker's merge function; polygons by rasterised AP, polylines by Chamfer AP with GT points from farthest-point sampling; average over the three categories.
- formulation: see idea; thresholds of the Chamfer AP as in the standard protocol; look-back window N and track-extraction details not read.
- input: per-frame predictions with IDs (or extracted tracks), consistent GT, ego poses.
- output: C-mAP, G-mAP.
- training: none.
- differentiable: no.
- sensors_data: nuScenes, AV2.
- results: nuScenes C-mAP MapTracker 69.1 vs StreamMapNet 56.4 (consistent GT); non-overlapping 32.5 vs 22.2; G-mAP nuScenes 50.2 (HisTrackMap) vs 49.4 (MapTracker). MapTCL (VM23) and PredMapNet (VM14) also report C-mAP only.
- compute: not applicable.
- code: https://github.com/woodfrog/maptracker (evaluation included, not checked)
- fit_to_us: reuse_idea - M - our pieces carry IDs in the GT piece graph, so an ID-consistency score (does the same lane keep the same output piece over time) is definable; merging multi-frame predictions into one global map to test long-horizon consistency is an option for the route task.
- compat_needs: element identity in output and GT.
- compat_conflicts: our output vertices have no persistent identity yet.
- tried_before: none.
- verified: partly.

### CARD VM46 - Geographically disjoint splits (StreamMapNet split, Near / Far Extrapolation)
- papers: Localization Is All You Evaluate: Data Leakage in Online Mapping Datasets and How to Fix It (Lilja et al., CVPR 2024) https://arxiv.org/abs/2312.06420 ; the new splits of StreamMapNet (VM01); the comparison in MapLightning (VM41)
- depth: M
- stage: D, E
- idea: Random scene splits of nuScenes and AV2 put most validation / test samples next to training samples, so models can recall the training-area map; geographically disjoint splits remove this.
- formulation: leakage measure = share of val / test samples within 5 m of a training sample: nuScenes 79.4 % (val) / 85.5 % (test), AV2 45.0 % / 41.9 % (Lilja Tab. 2). Lilja Near Extrapolation: regions defined by hand from map attributes, target about 70 / 15 / 15, zone classes (residential / commercial / industrial) and for nuScenes weather and time of day preserved, Boston and Singapore NOT separated, val carved from the original training data, test = original test; nuScenes 28 k / 6 k / 6 k samples, no buffer (filtering val / test within 60 m of training changed results negligibly). Far Extrapolation: whole cities as folds for cross-validation (Boston 55 % of nuScenes samples). StreamMapNet split (area, 30 m radius): nuScenes overlap 85 % -> 11 % (train 1.64 km2, val 0.54 km2), AV2 54 % -> 0 %, 700 / 150 scenes. MapLightning: the StreamMapNet split has 2.1 % and Near-Extrapolation 0.9 % overlap (5 m criterion) on nuScenes.
- input: frame positions of all splits.
- output: overlap shares, splits.
- training: none.
- differentiable: no.
- sensors_data: nuScenes, AV2.
- results: test mAP original -> Near (Lilja): VectorMapNet 54.8 -> 18.2, MapTR (R50 GKT) 59.9 -> 19.7, MapTRv2 72.7 -> 26.2, StreamMapNet 74.4 -> 27.9 (largest drop 46.5 points); AV2 MapTRv2 65.3 -> 55.2; HDMapNet IoU 38.8 -> 21.2; rankings largely preserved; lifting-method and depth-supervision conclusions weaken. Best published on the StreamMapNet split (60 x 30 m): MapLightning 45.8, Skeptic 45.4, PredMapNet 42.1, HisTrackMap 41.5, MapTracker 40.3, MambaMap 40.1 (3(a)-C).
- compute: not applicable.
- code: https://github.com/LiljaAdam/geographical-splits
- fit_to_us: reuse_idea - S - our split is scene-disjoint (212 / 22 / 21 scenes from the first three blobs) but NOT location-disjoint; with only two cities (a handful of map areas), VAL / TEST frames can lie within a few metres of training frames. Compute the 5 m overlap from nuScenes ego poses and report an overlap-filtered subset or a region split as a robustness check; the frozen foundation GaussianCaR may also have seen these scenes (not checkable here).
- compat_needs: ego poses (global) of all frames.
- compat_conflicts: with 255 scenes a region split is expensive.
- tried_before: none (split is scene-disjoint by design).
- verified: partly (HTML read; sample counts of the Far folds not read).

---

## 3. Cluster compatibility notes

### 3(a) Interface contracts

| item | literature (typical) | ours | mapping |
|---|---|---|---|
| BEV input | dense map from a TRAINABLE camera-to-BEV lift (GKT, LSS / BEVPoolv2, BEVFormer, IPM), 256 channels (128 in VectorMapNet); grids: 200 x 100 at 0.3 m (MapTR family, InteractionMap, PrevPredMap, HRMapNet, GaussianMap), 50 x 100 over 60 x 30 m i.e. 0.6 m cells (derived) in the StreamMapNet family (MapTracker, MemFusionMap, MambaMap), 400 x 200 at 0.15 m (HDMapNet, BeMapNet, DTCLMapper), 64 x 32 (PivotNet); range x +-15 m lateral by y +-30 m longitudinal ("60 x 30 m") or +-25 by +-50 m ("100 x 50 m") | frozen cached 39 channels (32 PCA + 7 map logits) at 200 x 200 cells, 0.5 m, +-50 m in BOTH axes | cropping to 120 x 60 cells reproduces the 60 x 30 m window; no listed method uses a square +-50 m window; cells are coarser than 0.3 m but finer than the 0.6 m (60 x 30 m) to 1.0 m (100 x 50 m, if the 50 x 100 grid is kept; derived) of the StreamMapNet family |
| element set | N = 50 - 100 fixed instance queries, each an ordered N_v = 20 polyline in normalised BEV coordinates, 3 classes (crossing, divider, boundary), centerline class only in MapTRv2 / SparseMeXt; BeMapNet piecewise Bezier <k, n>; PivotNet up to 10 / 2 / 30 pivots; VectorMapNet token sequence with EOS; HDMapNet dense raster + rule vectoriser | vertices + 128-d token + validity + successor edges (forks / merges) | polyline <-> graph adapter: path decomposition by graph traversal (MapTRv2 centerline) and arc-length resampling to N_v; an ordered point set cannot hold a fork except as two overlapping polylines |
| supervision | Hungarian instance matching + per-pair permutation minimum (MapTR), one-to-many auxiliaries (MapTRv2, PrevPredMap, DAMap), DP alignment (PivotNet), matching + raster Dice (MapVR), dense CE (HDMapNet) | point-to-polyline both ways, no matching | see 3(b) matching-free ingredients |
| temporal input | previous top-k queries / polylines (ego-warped), BEV memories (warped), instance masks, rasters of past output, memory tokens | previous set (positions, tokens, validity) ego-warped; input memory | table 3(a)-B |
| evaluation | Chamfer AP over 3 classes with 20 resampled points, thresholds {0.5, 1.0, 1.5} m ("easy") or {0.2, 0.5, 1.0} m ("hard"); C-mAP, G-mAP, mAS | pooled P and R of dense 0.5 m points at 0.5 / 1 / 1.5 m (F1), lanes found, split / merge F1 | thresholds coincide with the easy set but the protocol differs (instance matching ranked by score vs pooled points); numbers are not comparable |

3(a)-B. Temporal design space (what is carried, how it is aligned, selected, fused and trained):

| method (card) | carried | alignment | selection | fusion | prior source in training | first frame | extra loss |
|---|---|---|---|---|---|---|---|
| StreamMapNet (VM01) | queries + polylines + BEV | MLP[q, T]; points by T; BEV warp | top 33 of 100 by rank | concat after layer 1; GRU on BEV | own previous output, memory detached | learned queries | transformation loss 0.1 |
| SQD-MapNet (VM02) | + noised previous GT | T on GT | ATM threshold | extra DN queries | noised previous GT | normal DN | L_DN |
| PrevPredMap (VM03) | top-k vector predictions | ego transform | top 10 (nuScenes), 8 (AV2) | content + position queries | stored earlier predictions | single-frame mode (p = 0.5) | none |
| MapTracker (VM04) | vector + BEV latents (20 frames) | affine warp; PropMLP(quaternion, translation) | 4 memories by distance 1 / 5 / 10 / 15 m; thresholds 0.4 / 0.5 / 0.6 | cross-attention; conv residual | own outputs, clips of 5, pose noise | threshold 0.4 | transformation 0.1 |
| Sparse4D v2 / v3 (VM07, VM06) | (anchor, feature) | ego R, T + velocity | top 600 of 900 | concat; temporal cross-attention | own; v3 adds noised GT groups | single-frame layer | confidence decay x 0.6 |
| StreamPETR (VM08) | (dt, Q_c, Q_p, v, E) FIFO | explicit T + MLN | top 256 per frame, 4 frames | hybrid attention | own; first 6 of 8 frames detached | zero velocity, dt | none |
| TopoStreamer (VM09) | queries + BEV | MLP[q, pose]; BEV warp | top 30 % | replace lowest-confidence queries | previous GT by ID-following + DN | not described | streaming attribute constraints |
| FASTopoWM (VM10) | queries + BEV via world models | pose-conditioned transformers | about 30 % | GRU | self-supervised next-BEV MSE | fast stream | latent loss 0.3 |
| SparseDrive (VM11) | map instances (33 temporal of 100) | not described | not described | temporal cross-attention | own | not described | none |
| HRMapNet (VM13) | global raster of past output | global pose | S_th | concat + extra queries + prior embeddings | own, reset every epoch | empty map | none |
| PredMapNet / HisTrackMap (VM14, VM15) | per-instance decayed masks (0.9 / 0.95) | grid warp | 0.5 | mask-guided cross-attention | own | not described | future Chamfer (PredMapNet) |
| MemFusionMap (VM16) | 4 BEV features + overlap count | warp | none | concat + conv | own, detached | first BEV repeated | transformation 0.1 |
| InteractionMap (VM17) | BEV submaps | 4 x 4 warp | key frames every 5 m, N = 4 | GRU / stacking | own | 50 % single-frame epochs | none |
| MapExpert (VM18) | one fused BEV state | not specified | none | C_t = beta F_t + f(F_t, C_{t-1}) | own | not stated | none |
| MapUnveiler (VM20) | 96 memory tokens | none stated | token summariser | cross-attention | own | learnable embeddings | none |
| MambaMap (VM21) | 4 BEV + query frames | warp; Hungarian query reorder | window 4 | gated SSM | own, detached | not stated | none |
| L2G-Map (VM12) | global log-odds raster | vehicle pose | none | closed-form log-odds sum with distance weight | not learned | prior L_0 | none |

3(a)-C. Reported nuScenes numbers (camera only, mAP over crossing / divider / boundary, as reported by each paper, not re-run; thresholds {0.5, 1.0, 1.5} m unless stated; the old split leaks, see VM46):
- Old (overlapping) split, 60 x 30 m: MapTR-tiny 58.7 (110 ep); MapTRv2 61.5 (24 ep) / 68.7 (110 ep) / 73.4 (V2-99); StreamMapNet 60.4 - 63.4 (24 - 30 ep, depends on the re-run); SQD-MapNet 63.9; MapQR 72.6 (110 ep); GeMap 69.4; Mask2Map 71.6 (24) / 74.6 (110); InteractionMap 71.8 (24) / 76.0 (110); GaussianMap 70.5 (24); MapExpert 76.5 (100 ep); MapTracker 76.1 (consistent GT; 74.7 on the original GT, 72 ep); HisTrackMap 76.6; PredMapNet 76.9; Uni-PrevPredMap without prior 77.0 (72 ep); MapLightning 77.7 (paper notes memorisation).
- Location-disjoint (StreamMapNet split), 60 x 30 m: StreamMapNet 33.9 (34.1 - 35.2 in re-runs), SQD-MapNet 36.6 - 37.3, MapDiffusion 35.6, HRMapNet+StreamMapNet 37.1, MemFusionMap 38.0, AugMapNet 38.3, AugSQD 40.5, MapUnveiler 39.4, MapTracker 40.3, MambaMap 40.1, Uni-PrevPredMap (no prior) 39.7, HisTrackMap 41.5, MapExpert 42.0, PredMapNet 42.1, Mapping like a Skeptic 45.4, MapLightning 45.8; at 100 x 50 m: StreamMapNet 23.0, MapTracker 36.2, Skeptic 42.8, MapLightning 42.1.
- Near-Extrapolation split (Lilja): MapTRv2 26.2 / StreamMapNet 27.9 (test); in MapLightning's comparison MapTR 19.0, StreamMapNet 26.1, MapTracker 27.5, MapTRv2 26.7, AugMapNet 27.5, MapLightning 35.3.
- Current best in this cluster on location-disjoint data is therefore about 45 - 46 mAP against about 77 on the leaking split.

3(a)-D. Variable numbers of elements and of points per element:

| method (card) | number of elements | points per element |
|---|---|---|
| MapTR, MapTRv2, MapQR, GeMap, MGMap, StreamMapNet and its descendants (VM26, VM27, VM40, VM36, VM39, VM01) | fixed set of N = 50 - 100 queries; GT padded with a "no object" class; presence = class score above a threshold at inference; N matters (25 / 50 / 75 -> 41.6 / 50.3 / 52.2 mAP in MapTR) | fixed N_v = 20 after resampling (10 / 20 / 40 -> 48.0 / 50.3 / 50.0) |
| TopoStreamer, FASTopoWM (VM09, VM10) | fixed 200 queries, Hungarian, visualisation threshold 0.3 | fixed 10 per polyline |
| MapTracker, HisTrackMap, PredMapNet (VM04, VM15, VM14) | tracks with thresholds: MapTracker 0.4 on the first frame, then 0.6 for new and 0.5 for propagated elements; HisTrackMap and PredMapNet 0.4 birth / 0.5 track; 100 new-candidate embeddings per frame | fixed 20 |
| Sparse4D, StreamPETR (VM06 - VM08) | fixed 900 instances, top-k carried | not applicable |
| PivotNet (VM28) | up to 20 / 25 / 15 instances per class, validity from the pivot predictor | dynamic: per-slot pivot classifier (max 10 / 2 / 30), collinear points tied by linear interpolation |
| BeMapNet (VM29) | 60 instance queries (20 / 25 / 15), bipartite | piece count by an end-index softmax, Bezier <k, n> per class |
| VectorMapNet (VM31) | N_max = 100, GT padded | EOS token, variable length (maximum not stated) |
| HDMapNet (VM30) | data-dependent: DBSCAN on the instance embedding (eps 1.5, min_samples 50) | greedy tracing with a fixed 7-pixel step (1.05 m) |
| MapDiffusion (VM43) | fixed l queries, GT padded with Gaussian noise (mean 0.5, sd 0.25) | N_P (value not stated) |
| ours | variable graph, validity per vertex | about 22 vertices per 100 m (E94), step classes 2 / 4 / 6 m |

3(a)-E. Direction and start-point ambiguity, and whether matching is needed to resolve it:

| method (card) | handling | needs set matching? |
|---|---|---|
| MapTR / MapTRv2 (VM26, VM27) | permutation group (2 for polylines, 2 N_v for polygons); per matched pair the permutation with the lowest point-to-point cost is chosen and the direction loss uses that order; fixed order 44.4 vs 50.3 mAP | instance matching yes; the permutation minimum is local to a pair |
| MapTRv2 centerline (VM27) | direction given by the data, the group has one element | yes |
| StreamMapNet, MapTracker, SQD, PrevPredMap (VM01 - VM04) | inherit MapTR's group; the transformation loss needs none (identity of the carried query) | yes (main loss) |
| HDMapNet (VM30) | direction is a per-pixel classification over 36 bins, forward and backward both positive; instances by clustering | no |
| PivotNet (VM28) | first and last point fixed to the GT ends, dynamic-programming alignment of the rest | sequence alignment inside an instance, assignment across instances not detailed |
| BeMapNet, SparseDrive (VM29, VM11) | not addressed | yes |
| VectorMapNet (VM31) | vertex order encodes direction; no canonical-direction or start-point rule described | yes |
| MapVR (VM32) | distance-based rendering is order-independent (reader's inference) | yes (pairing) |
| Chamfer cost (MapTR ablation, VM26) | symmetric nearest-neighbour distance, order-free; 47.5 vs 50.3 mAP for the Manhattan point cost | yes |
| GeMap (VM36) | invariant to start point, translation, rotation; angle signs change under reversal | yes |
| TopoStreamer, FASTopoWM (VM09, VM10) | centerline direction fixed by the data | yes |

### 3(b) Known adapters and known conflicts

Adapters (name; source cards; effort):
- A1 polyline set <-> lane graph: path decomposition by graph traversal (MapTRv2 centerline, LaneGAP-style) one way, arc-length resampling to N_v = 20 the other way (VM27, VM26). S.
- A2 query <-> bilinear sample: StreamMapNet's multi-point attention samples at all N_p predicted points plus offsets, deformable attention at a reference point plus offsets, MGMap pools 5 x 5 ROI patches around each point; our fan of 27 bilinear samples + 7 cross-section points per step is the same operation with a fixed pattern (VM01, VM26, VM39). S.
- A3 Hungarian -> matching-free: (i) per-pair permutation minimum without set matching (VM26 inside a matched pair), (ii) threshold rule ATM with bidirectional Chamfer (VM02), (iii) identity inheritance / ID-following (VM04, VM09), (iv) identity assignment for noised queries (VM38), (v) raster losses over all outputs (VM33 BCE of the aggregated query raster, VM24 occupancy L1 across frames, VM32 soft rendering + Dice), (vi) soft quality targets from the distance to GT (VM34, VM17, VM06). Our point-to-polyline both ways (E120) is an instance of the same family. S - M.
- A4 vertex set <-> raster: soft line rendering exp(-D / tau) (VM32), thresholded masks (VM13, VM15, VM14); needed for any raster memory or raster consistency loss of our output. S - M.
- A5 token update by pose: MLP over [token, flattened T] (VM01), PropMLP with quaternion + translation (VM04), MLN affine modulation (VM08), pose-conditioned transformers (VM10). S.
- A6 BEV ego warp: 4 x 4 transform reduced to a 2D affine and bilinear sampling of the raster (VM01, VM04, VM16, VM17); zero padding resets visibility counts (VM16). S.

Known conflicts:
- K1 Hungarian-based losses sit under almost every temporal method (VM01 - VM04, VM06 - VM11, VM14 - VM18, VM20, VM21, VM25 - VM27, VM38 - VM40). Their temporal parts (warp, token update, memory selection, prior construction) do not depend on the matcher and are portable; their main losses are not.
- K2 Global-pose memories (VM13 HRMapNet, VM19 NMP, VM03 Uni-PrevPredMap, VM12 L2G-Map) assume an accurate global ego pose; our route task has "no pose given" and map offsets / pose jumps are injected synthetically, so only a rolling local memory built from inter-frame ego motion fits.
- K3 Camera-specific modules cannot run on cached BEV: perspective-view depth and segmentation heads (VM27, VM36, VM42), IPM-PE (VM29), probabilistic projection (VM22), 1D image-token mapper (VM41), Gaussian splatting from camera features (VM42), the PV branch of HisTrackMap's prior fusion (VM15).
- K4 Autoregressive token decoding (VM31) is not differentiable through the sequence.
- K5 Track identity (VM04, VM14, VM15, VM45) is undefined at forks and merges of a graph unless a rule is added.
- K6 Published training budgets (72 epochs in three stages on 8 GPUs for the trackers, 110-epoch schedules for the long MapTR-family runs) exceed our reference runs (E94 85 min, E120 70 min on one RTX 3060); gains are measured with a jointly trained BEV encoder, which we do not have (frozen foundation); MapDiffusion's frozen-encoder ablation lost 3.9 mAP (31.7 vs 35.6) (VM43).
- K7 Fixed query counts and 20 points per element (VM26, VM40) versus a variable graph with about 22 vertices per 100 m.

### 3(c) Combinations that the papers themselves demonstrate

- Host StreamMapNet: + SQD (VM02), + MapTCL (VM23), + AugMapNet (VM37, also on SQD-MapNet), + MapQR scatter-and-gather (VM40), + HRMapNet (VM13), + MemFusionMap / MambaMap (VM16, VM21), + MapDiffusion decoder (VM43), + DTCLMapper (VM24), + L2G-Map closed loop (VM12).
- Host MapTracker: + MapTCL (VM23), + PredMapNet, HisTrackMap, MapExpert, AMap teacher / student (VM14, VM15, VM18, VM25), + probabilistic projection (VM22).
- Host MapTRv2 / MapTR / MapQR: + HRMapNet (VM13), + MapGR (VM33), + DAMap (VM34, also on Mask2Map), + MapVR (VM32), + GaussianMap encoder with the MapQR + DAMap decoder (VM42), + Uni-PrevPredMap heatmap priors (VM03), + ADMap (VM35).
- Denoising + temporal: SQD (VM02), Sparse4D v3 (VM06), TopoStreamer (VM09), SparseMeXt (VM05), Mask2Map (VM38).
- Teacher / student across time: AMap (VM25); self-supervised next-frame prediction: FASTopoWM (VM10).

### 3(d) Observations that follow directly from the cards (no new claims)

- O1 Carried fraction: only 8 - 33 % of the previous set is carried (VM01 33 of 100, VM09 30 % of 200, VM03 k = 10 of 100, VM11 33 of 100; VM06 600 of 900 is the exception); TopoStreamer's ablation has a broad optimum at 30 % (10 / 30 / 50 / 75 % -> 34.8 / 36.6 / 34.3 / 32.9 mAP), PrevPredMap saturates near k = 12.
- O2 Pose: positions are transformed in closed form by T, the carried token is updated by a learned function of T; explicit compensation alone did not help in StreamPETR (early error propagation); pose noise during training only in MapTracker (Gaussian on the vector-module transform, sigmas not on the page); inference-time pose noise: HRMapNet 67.2 -> 63.8 mAP at sigma_t 0.2 m, sigma_rot 0.02 rad; FASTopoWM and others did not test pose error; fast single-frame fallback in FASTopoWM and dual-mode in PrevPredMap.
- O3 The source of the prior in training decides robustness: in SQD's own 24-epoch setting the single-frame baseline is 59.9, the plain self-fed temporal stream 59.2 (no gain), normal denoising on current-frame GT 62.7 and stream denoising on warped previous GT 63.9 (most of the gain comes from denoising, the stream-specific part adds 1.2); PrevPredMap uses stored real predictions; HRMapNet self-generated memory; AMap a future-aware teacher.
- O4 Memory selection by distance is mildly better than the last frames (MapTracker 76.1 vs 75.8 vs 75.0; InteractionMap 5 m key frames); memory length saturates at 4 (MemFusionMap, MambaMap N = 6 worse, MapTCL N_t = 5); MapExpert argues for a single recurrent state because stacking dilutes the current frame (weight 1/S).
- O5 Validity memory forms in the literature: additive bounded counter (HRMapNet +30 / -1 on uint8), decayed rasters (PredMapNet 0.9, HisTrackMap 0.95), decayed confidence of carried instances (Sparse4D v3, x 0.6), visibility count (MemFusionMap), learned GRU (StreamMapNet, NMP), confidence-weighted averaging of warped features (Skeptic), hand-specified log-odds with a distance weight on grids (L2G-Map); no vertex-level recursion with explicit log-odds exists in this cluster.
- O6 Validity / classification as quality: DAMap (exp(-L1)), InteractionMap (1 - Manhattan, edge cosine, GIoU), Sparse4D v3 (centerness, yawness) all replace the hard 0/1 target by a geometric quality; DAMap reports +2.4 mAP on a MapTRv2 host (24 ep), for the other two the gain of this component alone was not read; all need an assigned GT, which our point-to-polyline distance can supply.
- O7 Stability: native temporal models (StreamMapNet mAS 91.9, MapTracker 90.4) are far more stable than single-frame MapTR (71.6) but mAP and mAS are only loosely related (VM44); MapTCL adds +0.6 to +3.7 mAP and +0.5 to +6.2 C-mAP on its three temporal hosts and DTCLMapper +1.1 / +1.5 mAP on StreamMapNet / MapTracker; ADMap's jitter measure (mean chamfer of well-formed true positives) fell from 0.597 to 0.518 (MapTR) with a direction-difference loss.
- O8 Splits: on the leaking split many papers sit at 74 - 77 mAP; on location-disjoint splits they sit at 27 - 46 (VM46).

### 3(e) Which part of each method runs on a cached BEV raster (independent of the camera encoder)

| part | cards | blocker |
|---|---|---|
| decoder with hierarchical / instance queries over a BEV feature map (deformable or full attention) | VM26, VM27, VM40, VM36, VM39, VM34, VM17, VM05 (sparse variant uses images) | Hungarian; PV auxiliaries; deformable-attention op (mmcv in the original code, a pure-PyTorch implementation exists - from memory) |
| temporal modules on BEV features (warp + GRU / conv / SSM / recurrent state) | VM01, VM04, VM16, VM17, VM18, VM21, VM19, VM20 | need frame sequences and ego pose; MapTracker stage 1 trains the BEV encoder |
| temporal modules on element sets (carried queries / predictions / tokens) | VM01, VM03, VM06, VM07, VM08, VM09, VM10, VM11 | element identity and a fixed-size set |
| raster memories of past output | VM13, VM14, VM15, VM12 | rasteriser; global pose in VM13 / VM12 |
| training-only schemes (noised previous GT, consistency losses, distillation) | VM02, VM05, VM23, VM24, VM25 | GT of neighbouring frames, matching inside the vector consistency terms |
| dense heads + rule post-processing | VM30 | none; not differentiable |
| losses that need only outputs and GT | VM26 (permutation minimum), VM28 (DP alignment), VM29 (curve restoration), VM32, VM33, VM34, VM35, VM36 | assigned GT or a rasteriser |
| image-side parts | VM22, VM27 depth / PV heads, VM29 IPM-PE, VM41, VM42 | not usable (no images) |

### 3(f) Ingredient map for our three planned blocks

- Time recursion on the previous set (ego-warped positions, token update, selection): VM01, VM03, VM04, VM07, VM08, VM09, VM10, VM11 (what to carry), VM02, VM05, VM06 (what to train it on), VM10 (fallback when pose or history fails).
- Validity fusion over time: VM12 (log-odds with distance weight), VM13 (additive, bounded), VM14 / VM15 (decay), VM16 (visibility count), VM06 (decay of carried confidence), VM22 (confidence-weighted merge), VM18 (single-state recurrence).
- Matching-free losses and targets: VM26 (permutation minimum inside a pair), VM30 (dense), VM32 / VM33 / VM24 / VM23 (raster), VM34 / VM17 / VM06 (soft quality), VM02 (ATM), VM38 (identity), VM29 (curve restoration), VM35 (direction difference), VM36 (invariant shape terms).
- Evaluation: VM44 (stability), VM45 (consistency, merged map), VM46 (leakage check).

## 4. Reuse versus re-implement

Re-implement from the method section (small, no mmcv):
1. Pose-conditioned token update + identity-based transformation loss on the carried set (VM01 L_trans, VM04 PropMLP, VM08 MLN). S.
2. Dual-mode / mixed-source training of the recursion: stored own predictions + noised previous GT + single-frame mode (VM03, VM02, VM06, VM10 fallback). S - M.
3. Previous-GT warp-and-perturb with the four noise types (rotate, shift, scale, curvature) and a decay with the distance to the current GT (VM02, VM05); magnitudes must be chosen by us. S.
4. Distance-strided memory selection (VM04 1 / 5 / 10 / 15 m, VM17 5 m key frames). S.
5. Matching-free temporal consistency loss between the ego-warped previous prediction and the current one: our point-to-polyline loss both ways, plus a rasterised occupancy L1 / mask loss after warp (VM23, VM24). S.
6. Soft quality targets for the keep gate from the distance to GT (VM34, VM17, VM06). S.
7. Visibility / evidence-count channel and a log-odds validity memory with distance weight and decay (VM16, VM12, VM13). S - M.
8. Differentiable soft rasteriser exp(-D / tau) of the output graph for global raster losses (VM32, VM33). M.
9. Stability evaluation: adapt mAS and C-mAP to our pooled-point metric with GT-based pairing (VM44, VM45). M.
10. Geographic leakage check of VAL / TEST against TRAIN from ego poses (5 m rule), optionally an overlap-filtered subset (VM46). S.

Adapt code: none recommended. MapTR / MapTRv2 / MapQR / PivotNet / StreamMapNet-based trackers are built on mmcv / mmdet3d-style code (PivotNet states MMCV 1.7.1, Detectron2 and custom CUDA operators; the MapTR code uses scipy `linear_sum_assignment` and mmcv layers) and their value for us is in ideas, not in a port. HDMapNet is plain PyTorch + torch-scatter (GPL-3.0 stated) and could serve as the dense-head + rule-tracing baseline; its post-processing constants are listed in VM30.

Use as baselines (comparison, not integration): HDMapNet-style dense heads + DBSCAN + direction tracing (our rule system A is its descendant), a StreamMapNet / MapTRv2-style matched decoder on the cached BEV (already approximated by E31 / E32), and the published location-disjoint numbers (3(a)-C) as external reference points only.

## 5. Open questions and gaps

- SQD-MapNet's noise magnitudes (lambda 1 - 4), number of groups, attention mask and its new-split numbers are not on the pages read and no code exists; SparseMeXt's four noise types also have no magnitudes; only TopoStreamer (box shift 0.2, label flip 50 %), Mask2Map (class flip 0.2) and Uni-PrevPredMap (displacement uniform in [-6, 6] m) give numbers.
- No vertex-level or graph-level validity memory with Bayesian (log-odds) fusion was found; the closest are grid-level (L2G-Map, hand-specified, global pose) and decayed instance rasters.
- No Hungarian-free training of a temporal vector-map model was found: all temporal methods keep Hungarian for the main loss and use identity-based or threshold-based losses only for the temporal auxiliaries.
- No vector-map head was found that is evaluated on frozen or cached BEV features, except MapDiffusion's frozen-encoder ablation (-3.9 mAP); no radar-based vector-map method was found in this cluster (two searches).
- Lane-centerline GRAPHS with connectivity on nuScenes map-expansion data are covered only by the centerline classes of MapTRv2 / SparseMeXt (54.0 and 58.8 average mAP) and by the OpenLane-V2 topology work of other clusters; the three-class online-map benchmark (crossing, divider, boundary) is a different target from our lane graph.
- Pose-error robustness of temporal fusion is almost untested (HRMapNet, MapTracker noise); FASTopoWM lists it as future work; our synthetic pose jumps and map offsets have no published counterpart in this cluster.
- Temporal consistency metrics need element identity (C-mAP) or GT pairing (mAS); a metric for dense or graph outputs does not exist in this literature. The mAS formula as printed is inconsistent (Loc without the exponential mentioned in the text, two different mAS ranges), and its toolkit release was not checked.
- Our split is scene-disjoint, not location-disjoint: by the 5 m rule of Lilja et al. the overlap of VAL / TEST with TRAIN may be high on the Boston / Singapore maps; also the frozen foundation's own training scenes are unknown.
- Unverified or ambiguous items (flagged in cards): StreamMapNet's third loss weight, the last 4 k characters of the MapTRv2 page, the mAS page truncated at 100 k of 134 k characters, the MapLightning appendix, the MapTR assigner's default point cost in the code (Chamfer class) versus the paper's Manhattan cost (config not seen), the exact title and authors of several 2025 - 2026 arXiv papers.
- DN-DETR / DINO denoising defaults (the origin of the box-noise formula used by SQD-MapNet) were not read here; they belong to the detector cluster.

## 6. Search log

Searches (WebSearch, one line each):
1. BeMapNet End-to-End Vectorized HD-map Construction with Piecewise Bezier Curve arXiv
2. MGMap Mask-Guided Learning for Online Vectorized HD Map Construction arXiv
3. online vectorized HD map construction nuScenes 2026 arXiv state-of-the-art temporal
4. Stability under scrutiny online HD map construction temporal stability mAS metric
5. Localization Is All You Evaluate: Data Leakage in Online Mapping Datasets ... geographically disjoint splits nuScenes
6. HRMapNet historical rasterized map online vectorized HD map construction arXiv
7. vectorized HD map construction 2026 arXiv streaming query propagation consistency nuScenes new split
8. SQD-MapNet github code Stream Query Denoising ... repository
9. MapTracker github / PivotNet github / VectorMapNet code github / HDMapNet code github
10. MapUnveiler Unveiling the vector map clip-level tokens NeurIPS 2024 vectorized HD map temporal
11. MapVR online vectorized HD map construction differentiable rasterization NeurIPS 2023
12. Neural Map Prior for Autonomous Driving global neural map prior GRU update
13. DAMap Distance-aware MapNet for High Quality HD Map Construction arXiv
14. Mask2Map Vectorized HD Map Construction Using Bird's Eye View Segmentation Masks
15. ADMap Anti-disturbance framework for reconstructing online vectorized HD map arXiv jitter
16. MemFusionMap working memory fusion online vectorized HD map construction
17. AugMapNet latent BEV augmentation vectorized HD map construction
18. online HD map construction temporal ego-pose noise robustness previous frame memory misalignment pose error 2025
19. InteractionMap Improving Online Vectorized HDMap Construction with Interaction CVPR 2025
20. denoising training vectorized HD map construction noise polyline point jitter translation rotation noise magnitude
21. recursive Bayesian filter log-odds temporal fusion lane map elements online HD map existence probability Kalman
22. radar camera fusion online vectorized HD map construction nuScenes radar BEV
23. frozen BEV encoder lightweight map head vectorized HD map decoder on pretrained BEV features

Pages fetched (WebFetch; arXiv HTML / ar5iv unless noted): 2208.14437, 2308.05736, 2308.12570 (twice), 2401.09112 (three times), 2403.15951, 2507.00709, 2507.23325, 2308.16477, ar5iv 2306.09700, ar5iv 2206.08920, ar5iv 2107.06307, ar5iv 2305.14018, ar5iv 2311.11722, ar5iv 2303.11926, 2402.17430, 2312.03341, 2404.00876, 2602.16669, 2610.01905, 2510.10660, 2608.05209 (abs and HTML), 2407.17378, 2312.06420, 2409.00620 (twice), 2508.21689, 2507.20224, 2503.07168, 2405.05518, 2606.31177, 2510.06969, 2411.11002, ar5iv 2306.10502, ar5iv 2304.08481, 2507.21423, 2512.19150, 2504.06647 (HTML and abs), 2409.18737, 2503.21659, 2503.13430, 2407.13517, 2510.22675, 2505.08808, 2503.02578, 2405.19620, 2401.13172 (abs and HTML), 2412.12704; GitHub pages: hustvl/MapTR `decoder.py` and `maptr_assigner.py`, Tsinghua-MARS-Lab/HDMapNet (front page, `postprocess/vectorize.py`, `data/dataset.py`), woodfrog/maptracker, wenjie710/PivotNet, shuowang-ai/SQD-MapNet (404, a guessed URL); project page tsinghua-mars-lab.github.io/vectormapnet; ECCV page for SQD-MapNet.

To extend: search "DN-DETR / DINO denoising noise scale map", "Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention" (BEV feature reuse downstream), "InsMapper", "HIMap", "MapTRv2-CL", "HRDX composite score", and the supplements of SQD-MapNet and InteractionMap.
