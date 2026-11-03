# Literature overview - lane-centre graph, temporal fusion and route assignment from camera + radar

The entry point for people and for agents. It says what we are looking for and why, which papers were found and what each is in short, and where the detailed extraction of a paper (level 2) and of its code (level 3) lives. Tables marked *generated* are built from the notes and `papers.csv` (do not edit them; edit `papers.csv`). Folder map: [README.md](README.md). All papers: [CATALOGUE.md](CATALOGUE.md).

## 0. Quick routes

| I need ... | go to |
|---|---|
| the goal and what "good" means | section 1 |
| the papers for one topic | section 2, axis A1 - A9 (families, what we took, gap, core papers in one line); every single paper with link, location and relevance: [CATALOGUE.md](CATALOGUE.md) |
| what a paper does in 20 lines | its note `papers/<ID>_*.md` (level 1) |
| how to rebuild it (shapes, equations, loss weights, labels, inference rules) | `deep/<ID>_*.md` (level 2, core papers; rules in [deep/README.md](deep/README.md)) |
| how to port its code, licence | `code/<repo>.md` (level 3; process in [code/README.md](code/README.md)); all code extractions at a glance (licence, portability, reuse, effort, status): last section of [CATALOGUE.md](CATALOGUE.md) |
| technique ids that experiments use | [register.md](register.md) (K ids) |
| what is still open | section 3 |
| how to add a paper or run an extraction | section 5 |
| how the loop-16 builder fits the literature | [../experiment_execution/plans/Literature_Compatibility_Space_EN.md](../experiment_execution/plans/Literature_Compatibility_Space_EN.md) |

**For an agent that gets a literature task:** find the axis of the topic, read its axis section, grep its papers in [CATALOGUE.md](CATALOGUE.md) before searching the web (456 papers are already there), open `deep/` and `code/` where they exist, and mark every fact you add as checked / digest / memory. Do not download anything without the user's OK.

## 1. What we are looking for and why

### 1.1 Goal
A vehicle that should **navigate anywhere** is given a destination as a route on a standard-definition (SD) map, in practice an ADASIS-like horizon, and has **no HD map**. At every moment it must therefore know three things from its own sensors (six cameras, radar, no lidar): where the lane centre-lines around it are, how they connect (forks, merges, junctions), and which of them lead along the route. That is what a planner needs. Four capabilities follow, and each is a literature axis:

1. **Lane centre-line detection**, possibly **dense first** (heat map, direction field on the BEV grid) and then **sparse** (a graph with few vertices, edges and fork / merge states). Sparse, because assignment and planner need a graph; short, because every superfluous vertex is a cost and a false lane is worse than a missing short one.
2. **Temporal fusion of the detection results.** Per-frame graphs flicker and break under failures. Evidence has to be accumulated over frames on the level of the detections (vertices, edges, validity) with ego motion, not only on the level of features, and must survive a lost sensor.
3. **Logical assignment of the perceived lane graph to a map graph** (the SD route graph): which perceived lane belongs to which map road, which one continues on the route, which one must be taken for the next manoeuvre. The task is stated without a given pose: registration is part of it.
4. **Navigation context at the best level that is available.** Target: **lane level**, routing information for each perceived lane (this lane continues on the route, take this one to turn right in 80 m), obtained from perception plus the SD route, **without an HD map**. Worst case: **road level**, route attribute for the road or carriageway as a whole (the perceived lane set belongs to SD road X), and an explicit "cannot resolve the lane" instead of a wrong lane. The planner consumes the graph with this attribute.

### 1.2 The chain and the axes

```
 camera + radar --A1--> BEV features (frozen foundation, cached)
        |
        +--A2--> dense lane field (heat map, direction)
                   |
                   +--A3--> sparse lane graph (vertices, edges, forks, merges)
                              |   inside A3: A4 differentiable selection / hypotheses / training regime
                              +--A5--> temporal fusion of the graph detections (ego-warped memory, validity)
                                          |
 SD map route (ADASIS-like) --A6--> assignment: route attribute per lane / vertex  (fallback: per road)
                                          |
                                          +--A7--> planner (path points, route-aware lane tokens)

 A8 robustness (sensor failure, uncertainty) and A9 evaluation (metrics, data, splits) cut across all stages
```

| axis | question | name in the catalogue |
|---|---|---|
| A1 | which camera-radar BEV features already carry lane structure, can heads train on them frozen? | Sensing foundation |
| A2 | which dense intermediate (heat map, distance, direction, graph tensor) leads best to a graph? | Dense lane-centre field |
| A3 | how to get a short, correct graph with forks and merges: growth, set prediction, linking, readers, verification? | Sparse lane graph |
| A4 | how to make choosing among hypotheses trainable end to end without matching? | Differentiable selection, multi-hypothesis, training regime |
| A5 | how to fuse detections over time, on vertices and edges? | Temporal fusion of detections |
| A6 | how to attach the perceived graph to the SD route graph, at lane or at least road level? | Assignment to a map graph |
| A7 | what does a planner need besides geometry (route flag, confidence)? | Navigation context to the planner |
| A8 | what happens when a sensor fails or the map is wrong; how to express uncertainty? | Robustness, sensor failure, uncertainty |
| A9 | how to measure all of this fairly (metrics, splits, leakage)? | Evaluation, data, metrics |

### 1.3 What "good" means (standing rules of the project, see `../CLAUDE.md`)
- **Fully differentiable path:** no argmax, no matching, no rule step on the training path; selections with gradient through the selected items are allowed; Hungarian matching is not used in our losses.
- **Short and correct graphs:** better short lane graphs than false ones; sparsity is measured in vertices per 100 m (champion 13.5, control 21.8, Douglas-Peucker oracle 14.3) and enters the score S; Q is S without the sparsity term.
- **Quality and speed together:** every experiment reports ms per frame (idle GPU, batch 1, cached features); the chain runs in 31 ms.
- **Lane-level route attribute** measured as route success (coverage and precision >= 0.8 within 1.5 m) at map errors of 0 - 3 m; road-level fallback and abstention are open (section 3).
- State whether a head is pixel-based or graph-based.

### 1.4 Constraints that decide whether a paper is usable
- **Frozen cached BEV raster** of the foundation (GaussianCaR: 39 channels, 200 x 200 cells of 0.5 m, +-50 m). Perspective-image features, a trainable BEV encoder, lidar and end-to-end backbone training are not available: such papers contribute **head, decoder, loss or training ideas only**.
- **Environment:** WSL2, conda `gcar`, Python 3.12, PyTorch 2.8, one RTX 3060 12 GB, **no mmcv / mmdet / mmdet3d**. Nearly every lane or vector-map repository pins Python 3.8, PyTorch 1.9 and mmcv-full 1.x: ideas are reusable, code must be re-implemented.
- **Licences:** nuScenes is non-commercial; GPL-3 repositories and repositories without a licence file are read-only sources.
- **No downloads without the user's OK**; reading web pages is fine.
- Experiments so far use exact poses and a simulated ADASIS-like SD map; the task statement is "no pose given".

### 1.5 Where the project stands (details: `../HANDOVER.md`)
Loop 16 produced a champion builder (verified-ray hypothesis tree, junction head, time recursion, fork relocation; VAL S 0.746 against 0.725 for the control). Its gain comes entirely from sparsity; at equal sparsity it is 0.036 worse (junction recall J 0.265 against 0.408, lanes found 0.695 against 0.800). Next experiments aim at quality at <= 14.3 vertices per 100 m, and at the open items of section 3.

## 2. The literature by axis

How to read a block: **core papers** are the generated table (tier 1; each has, or will get, a deep and a code extraction); the text lists the other families by note id (links in [CATALOGUE.md](CATALOGUE.md)); **we took** names the K ids / experiments that used the literature; **gap** says what the literature did not give. Numbers are author-reported on mixed protocols and must not be compared across papers or with ours.

### A1 - Sensing foundation (camera + radar BEV)
**Families.** View lifting and BEV encoders (LSS SN23, BEVDepth SN24, BEVFormer SN22); camera-radar fusion (CRN SN05, RCBEVDet SN25, HyDRa SN27, Simple-BEV SN06); camera-radar map segmentation (BEVCar SN04, RESAR-BEV SN07, FIN SN35; the only camera-radar lane-topology result, TopoBDA SN09, sits in A3); Gaussian lift (GaussianCaR SN01, ours); adapting a frozen network (LP-FT SN28, adapters SN29, pre-training SN30).
**We took.** CRN (K60) in loops 1 - 3, then the map-pretrained GaussianCaR (K61) from E27 on: F1@1 0.624 -> 0.867 on VAL. Radar adds little to lane geometry.
**Gap.** No paper trains lane heads on frozen cached BEV features or measures the seen-to-unseen gap of a foundation (ours: 0.932 on its own training scenes, 0.820 on TEST).

<!-- BEGIN core:A1 -->
**Core papers (1):** SN01 GaussianCaR.
<!-- END core:A1 -->

<!-- BEGIN more:A1 -->
**Also worth using (1):** 2203.05482 model soups. Further: watch 5, context 20, probably not relevant 11 (trainable-encoder 5, out-of-scope 3, lidar 2, perspective-only 1). Statement and link for every paper: [CATALOGUE.md](CATALOGUE.md).
<!-- END more:A1 -->

### A2 - Dense lane-centre field
**Families.** Dense heads with vectorisation (HDMapNet VM30); a graph as a dense tensor (Sat2Graph GR09); a dense direction field as conditioning (Lane Field TP11); generative or refining priors on the lane mask (LaneDiffusion GR24, LSR-DM GR31); points from masks (TP26); rasters as feedback or loss (VM37, MapVR VM32, MapGR VM33).
**We took.** A small U-Net (K62) predicting a lane heat map (Gaussian, sigma 0.75 m) and travel direction; dense heat map + tracer beat a matched sparse decoder (+0.058 F1, +0.18 route F, E31 / E32).
**Gap.** The literature mostly goes straight to sparse set predictors; dense-then-sparse with gradient through the conversion is rare. A grid head with end-to-end sparsification is on the roadmap.

<!-- BEGIN core:A2 -->
**Core papers (4):** 2205.14756 EfficientViT; GR09 Sat2Graph; TP11 Lane Field; VM30 HDMapNet.
<!-- END core:A2 -->

<!-- BEGIN more:A2 -->
**Also worth using (2):** 1505.04597 U-Net / FCN dense head; 2006.04388 Generalized Focal Loss. Further: watch 18, context 13, probably not relevant 2 (aerial-only 1, lidar 1). Statement and link for every paper: [CATALOGUE.md](CATALOGUE.md).
<!-- END more:A2 -->

### A3 - Sparse lane graph (vertices, edges, junctions)
**Families.** Growth agents and tracers (RNGDet GR01, CenterLineDet GR02, iCurb GR03, RoadTracer GR04, DAGMapper GR05, DeH4R GR06); graph extraction from dense maps with link verification (SAM-Road GR10, Sat2Graph GR09); set / query predictors with Hungarian matching (MapTR SN16, MapTRv2 VM27, LaneSegNet TP10, VectorMapNet GR18); topology and linking (TopoNet TP09, TopoLogic TP01, TopoPoint TP02, LaneGAP TP04, TopoFG TP05, CoPo TP13, STSU TP30); sequence and generative graph models (SeqGrowGraph GR13, LaneGraph2Seq GR15, TopoGPT GR16; mostly unsuitable: tokens, argmax); readers and verification blocks used by the builder (Deformable DETR VS20, RAFT VS21, DINO VS19, Cascade R-CNN VS18, CLRNet VS26, L-CNN VS12).
**We took.** Key-point heads for split / merge (E44: F1 0.34 / 0.38 -> 0.55 / 0.61) and Bezier connectors (E53: 0.71 / 0.64); learned link heads and TopoLogic-style scoring stayed within 0.01 of the rules (E42 - E56); trace network E94 (F1@1 0.851); loop-16 builder = verified-ray hypothesis tree (K01 - K03, K30 - K51).
**Gap.** No growth paper passes gradient through the selected children, selects its step length by verifying intermediate samples, or reports fork resolution; link papers verify a *given* segment.

<!-- BEGIN core:A3 -->
**Core papers (5):** GR05 DAGMapper; GR02 CenterLineDet; GR10 SAM-Road / SAM-Road++ / LineGraph2Road; TP04 LaneGAP; TP01 TopoLogic.
<!-- END core:A3 -->

<!-- BEGIN more:A3 -->
**Also worth using (17):** 1612.00603 Chamfer distance as a point-set loss ...; 1712.01312 learned sparsity with hard-concrete gates ...; 1801.07372 numerical coordinate regression; 1803.03550 optimal polyline simplification; 2011.07057 learned edge keep gate with an ...; 2204.00408 hard-concrete masks plus a Lagrangian pull ...; 2512.10416 Beyond Endpoints; VS18 Cascade R-CNN; VS12 L-CNN; VS20 Deformable DETR; VS19 DINO; VS26 CLRNet; VS24 Sparse4D v1 / v3; VM32 MapVR; VS14 DeepLSD; TP07 TopoEnhance; VM34 DAMap. Further: watch 37, context 55, probably not relevant 26 (hungarian-or-tokens 9, trainable-encoder 5, out-of-scope 4, perspective-only 3, no-detail 2, aerial-only 1, lidar 1, compute 1). Statement and link for every paper: [CATALOGUE.md](CATALOGUE.md).
<!-- END more:A3 -->

### A4 - Differentiable selection, multi-hypothesis, training regime
**Families.** Winner-take-all family (rMCL MH01, aMCL MH02, relaxed WTA MH04, MCL MH06); anchor / slot labels (MultiPath MH10, MTR MH12) and quality-argmax labels (MH19, MH29); duplicate removal (MH31 - MH33); differentiable top-k, beams and particle filters (SOFT top-k MH34, entmax MH35, Gumbel top-k MH39, relaxed beam MH41, differentiable beam MH42, MH43 - MH45); beam-aware training and exposure bias (BSO MH49, GR27, scheduled sampling MH51); survival and hazard verification (VS01 - VS04).
**We took.** K38 (rMCL / aMCL) for training the children of a tip, relaxed WTA as the `mcl` switch, K41 hazard loss, K129 / K134 sparsity penalties.
**Gap.** No paper trains a b-ary geometric tree with a beam through the selection; Hungarian-free assignment for crowded configurations (parallel lanes, junction mouths) has no remedy except matching; a hazard with censoring is unprecedented in perception and the calibration of a product of conditionals is untested.

<!-- BEGIN core:A4 -->
**Core papers (2):** MH01 Resilient Multiple Choice Learning; VS01 Discrete-time hazard network with right ....
<!-- END core:A4 -->

<!-- BEGIN more:A4 -->
**Also worth using (6):** 2409.11172 Annealed Winner-Takes-All for Motion ...; 2606.26424 Rethinking Training & Inference for ...; MH04 Relaxed WTA; MH31 Relation Networks for duplicate removal; MH34 SOFT top-k by entropic optimal transport; MH02 Annealed Multiple Choice Learning. Further: watch 42, context 36, probably not relevant 6 (no-detail 3, out-of-scope 1, other 1, hungarian-or-tokens 1). Statement and link for every paper: [CATALOGUE.md](CATALOGUE.md).
<!-- END more:A4 -->

### A5 - Temporal fusion of detections
**Families.** Query propagation (StreamMapNet VM01, SQD-MapNet VM02, Sparse4D v3 VM06, StreamPETR VM08, TopoStreamer VM09, FASTopoWM VM10); tracking (MapTracker VM04); memories (HRMapNet VM13, Neural Map Prior VM19, MemFusionMap VM16, PredMapNet VM14, MambaMap VM21); evidence fusion (log-odds in L2G-Map VM12); consistency losses (MapTCL VM23); lane graph from video (TP31).
**We took.** Feature-level temporal fusion (E39 / E41: +0.008 F1, kept); loop-16 time recursion on the graph (ego-warped vertices and tokens as prior); loop-17 hold / gate memory (E193 - E212).
**Gap.** No validity memory with Bayesian (log-odds) fusion on vertices; no Hungarian-free temporal training; no streaming memory for a lane graph with explicit edges under modality failure; pose-error robustness of memories almost untested.

<!-- BEGIN core:A5 -->
**Core papers (4):** VM01 StreamMapNet; VM04 MapTracker; VM12 L2G-Map; VM09 TopoStreamer.
<!-- END core:A5 -->

<!-- BEGIN more:A5 -->
**Also worth using (5):** 2607.19781 WASABI; VM02 SQD-MapNet; VM03 PrevPredMap / Uni-PrevPredMap; VM16 MemFusionMap; VM23 MapTCL. Further: watch 24, context 17, probably not relevant 6 (out-of-scope 2, hungarian-or-tokens 2, superseded 1, perspective-only 1). Statement and link for every paper: [CATALOGUE.md](CATALOGUE.md).
<!-- END more:A5 -->

### A6 - Assignment to a map graph (SD route; lane level, road level as fallback)
**Families.** Lane-level assignment of an SD route (MAT / Online Navigation Refinement PL01: each perceived centre-line to one SD road, lane path derived afterwards); SD priors inside perception (SMERF PL43, P-MapNet PL42, SMART PL41, NavMapFusion C03, UMPE PL13, SCORE TP18, OpenLane-V2-OSM PL44); localisation to a navigation map for 'no pose given' (OrienterNet PL46, MapLocNet C27, HOLO C26, U-ViLAR C25); assignment tools (SuperGlue C30 with dustbin, LightGlue C31, weighted Procrustes C32); non-differentiable lane-level map matching (C33); noisy or changed maps (C05, C07).
**We took.** Route head by cross-attention from vertex tokens to SD raster tokens (E104b, K17; 0.69 M parameters), MAT as K75, OrienterNet as K99 context; asking the route head along the edges instead of at the vertices raised route success of the sparse graph from 0.798 to 0.888 (K185, control 0.885).
**Gap.** Published assignment is hard and road-level; nothing gives a soft per-vertex route probability, an explicit road-level fallback, or a map-trust / abstain output beyond training with noise (LightGlue's matchability and SuperGlue's dustbin are the closest tools).

<!-- BEGIN core:A6 -->
**Core papers (5):** C30 SuperGlue; PL46 OrienterNet; PL43 SMERF; PL01 Online Navigation Refinement; C03 NavMapFusion.
<!-- END core:A6 -->

<!-- BEGIN more:A6 -->
**Also worth using (1):** 2608.07835 SeqLoc. Further: watch 14, context 22, probably not relevant 2 (out-of-scope 1, perspective-only 1). Statement and link for every paper: [CATALOGUE.md](CATALOGUE.md).
<!-- END more:A6 -->

### A7 - Navigation context to the planner
**Families.** Planners that read lane tokens with an on-route flag (PlanTF PL03, PLUTO PL04, Diffusion Planner PL05); rule-based centre-line planner (PDM PL02); end-to-end planners with map queries (VAD PL08, SparseDrive PL10, UniAD PL11); lane-graph encoders (LaneGCN PL35, QCNet PL37); uncertainty into planning (UncAD PL09); goal and target-point biases (C18). Vision-language-action models were read and set aside: little average gain from language, non-LLM planners are on par on NAVSIM (PL20), cost and fragility are high.
**We took.** Planner consumer E117 (93 k parameters: edges + route probability + ego motion -> path; path error 0.34 m at 20 m). A planner loss through positions (E121) and into the route head (E119, E202) gave no gain.
**Gap.** No paper measures the closed-loop effect of a perceived lane graph with topology in a rule-based or nuPlan-style planner, and none back-propagates planning into topology.

<!-- BEGIN core:A7 -->
**Core papers (3):** PL02 PDM-Closed / PDM-Open / PDM-Hybrid; PL03 PlanTF; PL04 PLUTO.
<!-- END core:A7 -->

<!-- BEGIN more:A7 -->
**Also worth using (2):** 2206.02883 Lane-Level Route Planning for Autonomous ...; 2512.10660 route compliance metric and counterfactual .... Further: watch 13, context 24, probably not relevant 3 (trainable-encoder 1, out-of-scope 1, lidar 1). Statement and link for every paper: [CATALOGUE.md](CATALOGUE.md).
<!-- END more:A7 -->

### A8 - Robustness, sensor failure, uncertainty
**Families.** Post-fusion repair and gates between a frozen fusion and the head (PFS RB01, ModalPatch RB02, SafeMap RB03, RoboMap RB04, Grace-BEV RB06, MetaBEV RB07); corruption benchmarks (nuScenes-C RB17, MapBench RB18, MultiCorrupt RB19); test-time adaptation (RB22 - RB25); uncertainty outputs (Laplace per vertex RB29, evidential loss RB30).
**We took.** Our own failure series (E57, E193+: night, blur, cameras off, radar off) because published suites are for detection or map segmentation; head-side remedies restore only part of what the frozen foundation loses; gates and hold memory in loop 17.
**Gap.** [RB14](papers/RB14_gaussiancar-no-failure-evaluation-exists.md): no failure evaluation of GaussianCaR exists; no lane-graph robustness benchmark.

<!-- BEGIN core:A8 -->
**Core papers (4):** 2208.02814 Conformal Risk Control; RB04 RoboMap; RB01 Post Fusion Stabilizer; RB06 Grace-BEV.
<!-- END core:A8 -->

<!-- BEGIN more:A8 -->
**Also worth using (7):** 1706.04599 calibration and temperature scaling; 1906.02530 calibration under dataset shift; RB17 RoboBEV / nuScenes-C; RB19 MultiCorrupt; PL15 Producing and Leveraging Online Map ...; RB03 SafeMap; RB02 ModalPatch. Further: watch 20, context 16, probably not relevant 8 (trainable-encoder 5, perspective-only 1, out-of-scope 1, compute 1). Statement and link for every paper: [CATALOGUE.md](CATALOGUE.md).
<!-- END more:A8 -->

### A9 - Evaluation, data, metrics
**Families.** OpenLane-V2 data and metrics (TP29); fork / merge landmark and reachability metrics (RNTR GR14, collection GR28); stability and consistency (mAS VM44, C-mAP / G-mAP VM45, A-mAP PL18); location leakage and disjoint splits (VM46, SN03); visibility masks (SN10, SN11); planner-centric scores (NAVSIM PL20 / PL21, PKL PL23); input-swap protocols (PL24).
**We took.** Point F1 at 0.5 / 1 / 1.5 m, lanes found, J (split / merge F1 at 6 m), route success, planner path error, S / Q, paired scene bootstrap; scene-disjoint splits (TRAIN 8,414 / VAL 873 / TEST 833 frames).
**Gap.** Our splits are scene-disjoint but not location-disjoint (Lilja et al. measure per sample: on the original nuScenes split 79.4 % of val and 85.5 % of test samples have a training sample within 5 m, and MapTR falls from 48.8 to 19.0 on their geographically disjoint val; StreamMapNet's adopted split still overlaps 11 % of the val area): the audit is about three hours of numpy on ego poses and is still open. OpenLane-V2 scores are not comparable across devkit versions (the same model: TOP_ll 4.1 against 10.9). Fork / merge protocols use 0.5 - 5 m tolerances, our J uses 6 m: report a curve. No closed-loop or planner-aware map-quality metric for lane graphs.

<!-- BEGIN core:A9 -->
**Core papers (3):** GR14 RNTR; TP29 OpenLane-V2; VM46 Geographically disjoint splits.
<!-- END core:A9 -->

<!-- BEGIN more:A9 -->
**Also worth using (6):** PL19 Ego-status and open-loop critiques of ...; A03 MapBench; 2603.19852 Failure Modes for Deep Learning-Based ...; 2605.22578 Beyond Chamfer Distance; SN13 Calibration-only visibility mask for ...; VM44 mAS. Further: watch 9, context 12, probably not relevant 1 (out-of-scope 1). Statement and link for every paper: [CATALOGUE.md](CATALOGUE.md).
<!-- END more:A9 -->

### 2b. What the first full-text and code extraction found (P1: time and assignment, 2026-10-24)

Ten core papers (A5, A6) were extracted at level 2, nine repositories at level 3 (files in `deep/` and `code/`). All web-derived content is **digest** level (a summarising fetch tool; PDFs came back as binary), so numbers there are to be checked against the papers before they enter a report. What changed or became concrete:

- **Nobody carries a lane graph's topology across frames.** TopoStreamer ([VM09](deep/VM09_topostreamer.md)) recomputes the adjacency every frame (what is carried: 66 of 200 queries; its code is not released); MapTracker ([VM04](deep/VM04_maptracker.md)) matches ids one-to-one by mask IoU and does not model forks and merges; StreamMapNet's carry loss uses the matched GT of the previous frame in the code, so it inherits the Hungarian matching ([VM01](deep/VM01_streammapnet.md)); L2G-Map's log-odds fusion ([VM12](deep/VM12_l2g-map.md)) is hand-specified on a raster and has no repository. A hard-gated log-odds fusion of vertex validity already ran in loop 16 (E173, E176).
- **Cheap temporal pieces that transfer:** SE(2) warp of vertices, zero-initialised MLP([token, pose]) with residual (about 0.07 M parameters at 128-d, the extraction's own estimate), detached memory with a single-frame warm-up, pose-noise augmentation (MapTracker code: 0.08 rad, 0.25 m), a sweep of the carried share (TopoStreamer: best 30 %, 75 % was worse than no streaming in TOP), a forecast loss on the warped prior. MapTracker's stride rule adds only about +1 mAP at test time.
- **No published assignment method takes the route as input, and none has a lane-level flag, a road-level fallback or abstention.** MAT / OMA ([PL01](deep/PL01_online-navigation-refinement.md)) outputs, per perceived vector, a softmax over SD roads: the sum of that softmax over the route roads is a differentiable soft route probability (the road level), the rest has to be added by us; it never trains a relative SD shift. SMERF ([PL43](deep/PL43_smerf.md)) encodes the SD map as geometry only (11 points, 1 of 8 road classes). SCORE ([TP18](deep/TP18_score.md)) uses it as query seeds without a route bit. NavMapFusion ([C03](deep/C03_navmapfusion.md)) has no trust output; trust is implicit in corruption training, and a wrong map without that training is worse than none (mAP 17.8 or 20.1 against 22.4). Our map-fault training never drops the route polyline itself (`degrade_sd`, checked).
- **Abstention tool:** SuperGlue's dustbin ([C30](deep/C30_superglue.md)) gives an explicit no-match output, but its one-to-one marginals push the mass of many lanes of one road into the dustbin; row softmax with a bin column or a capacitated Sinkhorn are the fixes the extraction proposes.
- **Registration without a pose:** OrienterNet's pose-volume matching ([PL46](deep/PL46_orienternet.md)) can be rebuilt in pure PyTorch on our ego-centred BEV; lane-level assignment needs about 1 m lateral and 1 - 2 degrees heading; one frame is weak (15.8 % within 1 m on one of their sets), the product of volumes over a sequence reaches sub-metre accuracy (KITTI, 20 s: 79.7 %). That ties A5 to A6: temporal fusion of the registration.
- **Licences of the nine repositories:** only OMA-MAT is MIT (copyright Pointcept 2023). The rest is read only: GPL-3.0 (StreamMapNet, MapTracker with a non-commercial notice), non-commercial (SMERF, OrienterNet CC-BY-NC without version, SuperGlue), or there is no code (TopoStreamer README only, NavMapFusion README and GPL-3.0 LICENSE only, L2G-Map and SCORE no repository).
- **Splits:** StreamMapNet's nuScenes "new" split is the Roddick and Cipolla scene list, adopted from earlier work, and about 11 % of the validation area still overlaps. A disc-union overlap audit of our own splits is small work (the extraction's estimate: S).

Corrections these extractions made to level-1 notes are appended to the notes themselves (VM01, VM09, VM12, C03, section "Correction from the level-2 extraction").

### 2c. Second extraction round (dense field, graph, selection, consumers, metrics; 2026-10-25)

Seventeen more papers were extracted at level 2 (files in `deep/`), seven of them with a code extraction (`code/`; an eighth file, `topo-boundary`, is an unplanned GPL re-implementation found while searching for DAGMapper code). Again all web content is **digest** level; numbers marked "arithmetic" or "estimate" are the reviewing agent's own, not the paper's.

- **Dense to sparse.**
  - *Sat2Graph* ([GR09](deep/GR09_sat2graph.md)): the graph tensor has 19 channels at 1 m, is undirected and uses 6 world-fixed 60-degree slots. A lane fork opens at roughly 13 degrees (arithmetic), so both children fall into one slot; we would need heading-relative successor and predecessor slots. The decoding is all rule steps; it could seed our builder at fork mouths, not replace it.
  - *HDMapNet* ([VM30](deep/VM30_hdmapnet.md)): its dense-to-sparse conversion is a 7-step untrained rule chain (mask, DBSCAN, ridge thinning, greedy sort, tracing from a random ridge pixel) with about 95 vertices per 100 m (arithmetic) against our 13.5 and no fork handling. The idea that transfers is direction as bin classification, which can be multi-modal at forks.
  - *Lane Field* ([TP11](deep/TP11_lane-field.md)): an unsigned direction tensor with a closed-form, differentiable rendering of a graph into the field. It flags crossings but not Y-forks or merges, cannot replace our signed direction plus heat map, was never tested with a frozen encoder, and has no code yet.
  - *DAGMapper* ([GR05](deep/GR05_dagmapper.md)): the states are normal / fork / terminate; merge is just terminate. The part worth taking is the 1.5 m ground-truth-direction guard against runaway in a free unroll (effort S). Its training path is not matching-free.
  - *CenterLineDet* ([GR02](deep/GR02_centerlinedet.md)): the "multi-frame" gain is a scene-wide mean over past and future frames, so it is offline; training is cross-fitted. Cheap for our builder: carry only open ends and suppress seeds near the existing trail (our E173 carried all nodes and tied).
  - *LaneGAP* ([TP04](deep/TP04_lanegap.md)): `hustvl/LaneGAP` holds no code; the Graph2Path labels are in `hustvl/MapTR` (branch maptrv2, MIT), about 30 lines to re-implement; Path2Graph is a distance-only merge at 0.15 m; TOPO / JTOPO have no released implementation and a 0.45 m tolerance below our cell size, so report a tolerance curve. A per-path soft-min loss over our tree chains is sketched in the deep file.
  - *SAM-Road* ([GR10](deep/GR10_sam-road.md)): the edge head sees only the two endpoint features, and the set-context transformer over 16 candidates matters most. Reusable: directed breadth-first pair labels with stop nodes, random-score NMS that keeps junctions. MIT licence.
  - *TopoLogic* ([TP01](deep/TP01_topologic.md)): the "half value at about 2.8 m" is arithmetic from constants hard-coded in the code, which differs from the paper; at our 0.5 m vertices the bandwidth would need refitting. Our E133 used the unrefit width and tied in S. No licence.
  - *DeH4R* ([GR06](deep/GR06_deh4r.md)): expansion from dangling ends is a minor part (+0.19 TOPO-F1); the dense candidates and the neighbour-set head do the work. Take the mutual-agreement cost as a soft tip-to-child link score. Repository moved to `godx-7/DeH4R`.
- **Selection and verification.** *rMCL* ([MH01](deep/MH01_rmcl.md)): repository `Victorletzelter/code-rMCL` (MIT per the sidebar); the score target is 1 for every hypothesis that wins at least one target; all evidence uses at least 3 hypotheses and rMCL is worse than a single regressor on single-source inputs. Our `mcl` switch is the relaxed winner-take-all only; the score target, a separate validity head and a winner-share monitor are missing. *Hazard network* ([VS01](deep/VS01_discrete-time-hazard.md)): our `prev_ok` mask is the uncensored case of every formulation; the censoring switch equals the pycox "floor" rule; a per-metre hazard fixes unequal gaps; per-ring and product ECE and D-calibration are the missing evaluation.
- **Consumers and metrics.** *PDM-Closed* ([PL02](deep/PL02_pdm-closed.md)): the route is a hard set, a short graph gives a slow ego, not a crash; an evaluation-only test bed (Apache-2.0). *PlanTF* ([PL03](deep/PL03_plantf.md)): a soft route flag fits as a convex mixture of the two flag embeddings; ego kinematic inputs raise the open-loop score and lower the closed-loop score, a warning for E117. *PLUTO* ([PL04](deep/PL04_pluto.md)): no ablation of the route-restricted lines; E117 is within 0.025 m of its oracle-route headroom. *OpenLane-V2* ([TP29](deep/TP29_openlane-v2.md)): DET_l matches whole pieces by Frechet distance at 1 / 2 / 3 m, so a truncated lane costs a miss and a false positive; TOP_ll 4.1 against 10.9 is the devkit version (v1.0.0 against v2.1.0); centre-lines use 11 points. *RNTR* ([GR14](deep/GR14_rntr.md)): every graph node is a landmark in the code and matching is many-to-one, so our J is not comparable; our greedy one-to-one matching is nested in the tolerance, so one pass gives the whole 0.5 - 6 m curve; repository now `LogosIntelligenceLab/RoadNet` (MIT).
- **Location leakage** ([VM46](deep/VM46_geographical-splits.md)): see A9; 29 of our 43 VAL / TEST scenes lie in Lilja's training zone (digest-level lists).
- **Corrections to project documents** found on the way: K11 was wrong (fixed in the register); the Loop-16 Triage Plan treats arXiv 2411.16733 as SAM-Road++ (K85), the extraction says it is another paper; the old repository list said "licence not stated" for MapTR (MIT per the extraction); note C17 says OLS 44 where the paper has 42.

**Relevance review.** Every paper now carries a verdict (use, watch, context, not-relevant with a reason code) and a 3 - 4 sentence statement of what is new and how it touches our components (`papers.csv`, shown in [CATALOGUE.md](CATALOGUE.md)). Reviewers flagged 21 tier doubts; the parent session applied them in `papers.csv` (core papers moved out: BEVCar, DeH4R, SCORE; into core: Post Fusion Stabilizer as the closest match to our frozen-BEV interface).

### 2d. Search update (2026-10-25)

Four search agents looked for papers, code releases and news since the loop-17 reading; 92 new rows were added to `papers.csv` (none tier 1, 25 tier 2, 8 verdict "use"). Everything is **digest level**: abstract pages and repository pages read by the agents, not re-read by the parent session; arXiv search was rate-limited, so coverage is uneven. Nothing found for conformal prediction on maps, planner-aware map metrics, camera-radar lane graphs or a MapTracker successor.

- **Selection / training (A4, A9).** Metric and failure-mode work on multi-hypothesis output: Beyond Chamfer 2605.22578 (SOSPA, a set metric), Failure Modes 2603.19852, WTA as K-means / GMM 2606.26424; aWTA 2409.11172 (Apache-2.0) and ModeSeq 2411.11911 as winner-take-all variants; DP top-k 2601.21775 as a differentiable selection.
- **Dense field and graph (A2, A3).** MaGRoad 2512.10416 (MIT): edge score from path statistics (mean, std, soft-min) along the candidate edge, a cheap extra input for our ray verifier. WPRF 2607.07123 and TRACE 2610.22210: reconnection losses for broken thin lines (no code / after acceptance). clDice 2003.07311: soft-skeleton connectivity loss.
- **Temporal (A5).** WASABI 2607.19781 (lane-topology stabiliser with a 10-frame edge memory and toggle-rate / flicker metrics; rule-based, internal data, no code) is the closest published relative of our time recursion on explicit edges; copy the two metrics. Also: Beta-posterior evidence per BEV cell (PseudoMapLabeler 2608.12600) as a bounded alternative to log-odds; detection-level voxel fusion for map elements 2409.00593; AsyncBEV 2601.12994 (flow alignment under sensor time offset, code).
- **Assignment and consumers (A6, A7).** Odyssey 2610.06469: nuPlan benchmark with an SD route as the only navigation input (code not out). SeqLoc 2608.07835: entropy-tempered fusion of pose volumes over frames with an abstain signal. Lane-Level Route Planning 2206.02883: expected cost-to-go with lane changes, a soft lane-level route target. U-BEV 2310.13766: BEV with a differentiable template matcher on SD-map data. NudgeVAD 2605.24531: identity-initialised route conditioning of a frozen planner.
- **Robustness (A8).** Repair Before You Fuse 2609.35264 (zero-initialised repair modules on a frozen host); structured probabilistic mapping 2603.20076; MapStableTest / mAS toolkit (MIT per its README, numpy only).
- **Code releases and status changes.** Released: SeqGrowGraph (`MIV-XJTU/SeqGrowGraph`, no licence file, mmdet3d stack), TopoStreamer (LaneSegNet based, no licence file, py3.8 / torch 1.9 / mmcv-full 1.5.2, see [code/topostreamer.md](code/topostreamer.md)), DeH4R (`godx-7/DeH4R`, no licence file, weights on Hugging Face), MAT / ONR (ICLR 2026, MIT, dataset and checkpoint on Hugging Face). Still without code: Lane Field ("coming soon"), SCORE, HOLO, U-ViLAR v2, L2G-Map (v2 only), FASTopoWM, MapTCL, PredMapNet; NavMapFusion is still a README plus the GPL-3.0 licence.

### 2e. Foundation topics F1 - F3 (scouts, 2026-10-25)

Three scouts read abstract pages only (digest level; some repository URLs and kernel remarks are from memory and flagged in the rows) and added 84 rows (3 tier 1, 29 tier 2). Row texts start with `[F1]`, `[F2]` or `[F3]`.

- **F1 building blocks.** EfficientViT 2205.14756 (ReLU linear attention, global context from plain matmuls on the 200x200 grid; test in the dense head against a conv-only control at equal ms for far-range recall). Cheap per-vertex memory cells: minGRU 2410.01201, LRU 2303.06349; Gated DeltaNet 2412.06464 is the best memory over a token set but its kernels are custom. Pure-PyTorch long-range blocks for thin structures: SCNN, CCNet, VAN, DSCNet. Mask2Former 2112.01527 only as a masked cross-attention block (corridor read along a ray). No pure-PyTorch Mamba / VMamba scan within 25 ms was found; no ego-motion-aware SSM memory over a vertex set exists (a gap).
- **F2 training and losses.** Conformal Risk Control 2208.02814 (post-hoc threshold with a guarantee on an expected loss: candidate for the pruning and validity thresholds). Semi-supervised online mapping 2410.10279 (multi-frame fused teacher pseudo-labels: the template for the 628 ms teacher -> 25 ms student). Generalized Focal Loss 2006.04388 (calibrated quality score on the heat map), APML 2509.08104 and unbalanced optimal transport 2307.02402 (Sinkhorn point losses, soft assignment with a background column), robust multi-sensor map construction 2507.01484 (modality dropout that keeps the clean case). Not found: radar-camera BEV-lane distillation, a range-weighted loss for camera BEV.
- **F3 formulations.** PMB-NLL 2203.07980 (Bernoulli + Poisson set likelihood on the vertex set; code georghess/pmb-nll; needs a check whether it needs an assignment step), conditional marked point process 2506.21486 (likelihood-trained intensity field with empty-space probability), smoothed Viterbi 1802.03676 and Fenchel-Young losses 1901.02324 (differentiable chain choice without argmax gradient). Not found: a learned RFS filter with a Bernoulli existence update, a learned eikonal centre-line solver, a conformal method for polylines (gaps).
- **Deep reads of eight candidates** (files `deep/2*_*.md`, digest level: arXiv HTML and repository pages through a summarising tool, local code read directly). Worth an experiment: Conformal Risk Control 2208.02814 (calibrate the fork-pruning threshold on scene-level losses; the loss must be monotone with L(lambda_max) <= alpha, so use the miss relative to the unpruned tree; scene is the exchangeable unit, VAL the only calibration set), EfficientViT 2205.14756 (global-context layer in the seed / direction stage against a dilated-conv control at equal ms; prediction R@1 far +0.02), Generalized Focal Loss 2006.04388 (our dense-head loss is not minimised at p = y; calibration swap with temperature-scaling control). Watch only: semi-supervised online mapping 2410.10279 (gains at 2.5 - 25 % labels only, the teacher is an EMA, no code; fused-teacher raster from the 628 ms trace network as a three-arm test after a location-overlap check), PMB-NLL 2203.07980 (training loss needs a hard assignment; matching-free Poisson loss is our own sketch), APML 2509.08104 (released code differs from the paper; a position-only loss cannot create missing lanes). Low value: Gated DeltaNet 2412.06464 and minGRU 2410.01201 (our vertex rule is already a fixed minLSTM-like log-odds update; at 3 - 8 frames a learned cell initialised to reproduce it is the only arm worth trying).
- **Next scouts (not fetched yet):** Perceiver IO, RepLKNet / SLaK, DiNAT, Griffin, Hyena, GraphGPS, ConvGRU temporal grids (F1); EMA / mean-teacher dense self-distillation, heat-map label shaping, evidential or selective-prediction heads for graphs (F2); 2025-26 follow-ups to PMB-NLL (F3).

## 3. What the literature did not provide (open, ranked by use for the next loops)

1. A differentiable selection of hypotheses (tree / beam) with gradient to the dropped nodes for *geometric* nodes (only discrete or latent states are demonstrated). (A4)
2. **A route attribute that works at lane level and degrades to road level**, soft per vertex, with abstention when the map or the lanes cannot be trusted; assignment papers are hard and road-level, trust beyond noise training is missing. Building blocks found, none complete: MAT's road-pool softmax (soft route probability), SuperGlue's dustbin (abstain), OrienterNet's pose volume (registration), SMERF's SD-token encoder (section 2b). (A6)
3. Validity memory on vertices with Bayesian fusion and a temporal consistency metric for graph outputs (mAS / C-mAP need instance identity or GT pairing; MapTracker's C-mAP as coded only penalises id switches). (A5, A9)
4. Calibration of per-vertex and per-fork probabilities; nothing measures it. (A3, A4, A8)
5. A closed-loop or planner-aware map-quality metric for lane graphs. (A7, A9)
6. Dense-then-sparse with gradient through the conversion, and lane heads on frozen cached BEV features. (A1, A2)

## 4. Numbers (generated)

<!-- BEGIN stats -->
540 papers from 404 notes (without a note: 217).

| axis | core | relevant | background | not suitable | all |
|---|---|---|---|---|---|
| A1 Sensing foundation (camera + radar BEV) | 1 | 10 | 20 | 7 | 38 |
| A2 Dense lane-centre field | 4 | 15 | 19 | 1 | 39 |
| A3 Sparse lane graph (vertices, edges, junctions) | 5 | 50 | 68 | 17 | 140 |
| A4 Differentiable selection, multi-hypothesis, training regime | 2 | 29 | 58 | 3 | 92 |
| A5 Temporal fusion of detections | 4 | 19 | 31 | 2 | 56 |
| A6 Assignment to a map graph (SD route, lane / road level) | 5 | 12 | 25 | 2 | 44 |
| A7 Navigation context to the planner | 3 | 16 | 24 | 2 | 45 |
| A8 Robustness, sensor failure, uncertainty | 4 | 18 | 29 | 4 | 55 |
| A9 Evaluation, data, metrics | 3 | 10 | 17 | 1 | 31 |

Verdicts on relevance for our problem: use 78, watch 182, context 215, not-relevant 65, not yet rated 0.

Core papers: 31; deep extraction done for 26, still queued: 2205.14756, 2208.02814; not planned: SN01 (our own foundation, already integrated); RB04 (no repository in the note; search only when robustness work starts); RB06 (no repository in the note; search only when robustness work starts).
Code extractions: 17 files in `code/` (overview at the end of [CATALOGUE.md](CATALOGUE.md)).
<!-- END stats -->

Reading depth of the level-1 notes: loop-16 notes (301) are method level (M) or method + code page (M+C) for 284; loop-17 notes (103) were read through page digests (method sections queried, equations not read by the author). Where one paper has several notes, they look at it from different clusters; the longest loop-16 note is the home note.

## 5. How this folder works

**Three levels per paper.**

| level | where | content | who wrote it |
|---|---|---|---|
| 0 | this file, [CATALOGUE.md](CATALOGUE.md), `papers.csv` | every paper: link to the original, where it lives here, content, axis, tier, verdict and 3 - 4 sentences on relevance | `papers.csv` by hand / reviewing agents; CATALOGUE.md is generated |
| 1 | `papers/<ID>_*.md` (404 notes) | idea, formulation, training, results, fit to our project, conflicts; from method sections via page digests | loops 16 / 17 (research agents) |
| 2 | `deep/<ID>_*.md` | full-text extraction: architecture with shapes, equations, loss weights, labels, training and inference rules, numbers with table numbers, paper-versus-code, transfer to our pipeline | extraction agents, core papers first |
| 3 | `code/<repo>.md` | licence, stack, map of the code, key algorithms with file and function, config versus paper, label pipeline, metric code, port plan | extraction agents, repositories of core papers |

**Tiers** (column `tier`): 1 core = defines our design or is the nearest precedent / baseline on an axis, gets levels 2 and 3 (rules: [deep/README.md](deep/README.md), [code/README.md](code/README.md); open items in the generated statistics of section 4); 2 relevant = used in an experiment (has a K id) or a baseline / re-implementation candidate, level 1 is enough; 3 background = context; 4 not suitable = recorded with the reason so nobody reads it twice. Tiers and axes are set by rules in `tools/literature/catalogue.py`; all decisions (axis, tier, core list, extraction plan, verdict) are in `papers.csv` - change them there, not in the generated tables.

**Rules.**
- Every fact carries a mark: **checked** (page seen), **digest** (seen through a summarising tool, numbers may be mis-transcribed), **memory** (not seen). Papers that only the technique register knows (41) have no note yet; they appear in the catalogue with their K id.
- A technique gets a K id in [register.md](register.md) before an experiment uses it.
- No downloads, clones or installs without the user's OK (state file, source, size first). Reading pages is fine.
- GPL-3 / AGPL / no-licence code: read, describe, re-implement; never copy. nuScenes data: non-commercial.
- New paper: one note `papers/<ID>_<slug>.md` (fields as in the section "Note format" of README.md; ID = cluster prefix + number), add its row to `papers.csv`, then regenerate. A paper found by a search and not yet read needs only a row in `papers.csv` with title, url and idea.
- Regenerate and check after any change: `wsl python3 tools/literature/catalogue.py all` (render + check; selftest: `... selftest`).

**Code: select, pull, make readable, summarise, link** (details and gates: [code/README.md](code/README.md)).
1. *Select* only the most meaningful repositories (core paper, something we would really port: labels, losses, assembly rules, metrics, planner feature builders); the rest stays at level 2 and can be done later (column `extract` in `papers.csv`).
2. *Pull* in three steps of rising cost: L0 web read of README, LICENSE, tree and the key files (default, no approval); L1 sparse shallow clone of only the needed paths into a read-only folder in WSL (needs the user's OK per repository: name, commit, size); L2 full clone or release files only for parity tests (OK again). Foreign code is never imported or executed.
3. *Make readable:* `tools/literature/code_digest.py` generates a factual skeleton from a checkout (files, line counts, heavy-dependency flags, outline of classes and functions with line ranges, keyword index; no bodies, so it is safe for every licence); the narrative in `code/<repo>.md` then explains the key functions in own words (shapes, steps, config versus paper, label pipeline, gotchas).
4. *Summarise* in the port card at the top of the code file: licence, portable (pure-torch / mmcv / cuda-op ...), reuse (read-only / adapt / vendored / metric-only / idea-only), effort, port targets in our tree, status (read / ported / verified).
5. *Link:* the catalogue turns the port cards into the code table at the end of [CATALOGUE.md](CATALOGUE.md) and the `[code]` links of the paper rows; a port in `training/` or `evaluation/` cites the code file and the source file and line, gets a K id, and is marked `ported`, after a parity test `verified`.

**Provenance.** The original multi-paper research notes of the loops (lit7, lit8, lit_extra*, lit_p16*, lit16 clusters with `cards.csv`, lit17 A / B / C) were moved out of this folder and removed from the tree on 2026-11-03 (D49; only in the git tag `archive/docs-before-cleanup-2026-11-03`, the folder was called literature_sources and lay under the history of the experiment execution); the three local clones (CRN, GaussianCaR, BEVFusion) are listed in [code/README.md](code/README.md).

## 6. Related documents
- [../experiment_execution/plans/Literature_Compatibility_Space_EN.md](../experiment_execution/plans/Literature_Compatibility_Space_EN.md): how each technique family fits the loop-16 builder (stages, adapters, conflicts).
- [../experiment_execution/plans/Concept_Hypothesis_Tree_Lane_Graph_EN.md](../experiment_execution/plans/Concept_Hypothesis_Tree_Lane_Graph_EN.md), [Hypothesis_Tree_Spec_and_Tests_EN.md](../experiment_execution/plans/Hypothesis_Tree_Spec_and_Tests_EN.md), [Growth_and_Link_Architectures_EN.md](../experiment_execution/plans/Growth_and_Link_Architectures_EN.md), [Loop16_Triage_Plan_EN.md](../experiment_execution/plans/Loop16_Triage_Plan_EN.md): the design the literature was read against.
- [../HANDOVER.md](../HANDOVER.md), [../experiment_execution/PLAN.md](../experiment_execution/PLAN.md): state and next loop.
