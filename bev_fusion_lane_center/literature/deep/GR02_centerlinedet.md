# GR02 - CenterLineDet: full-text extraction

| field | value |
|---|---|
| paper | CenterLineDet: CenterLine Graph Detection for Road Lanes with Vehicle-mounted Sensors by Transformer for HD Map Generation. Zhenhua Xu, Yuxuan Liu, Yuxiang Sun, Ming Liu, Lujia Wang. ICRA 2023, arXiv 2209.07734 (v1 submitted 2022-09-16, v2 revised 2023-02-28). Pages opened: arxiv.org/abs/2209.07734, arxiv.org/html/2209.07734 (this is v2), the project page tonyxuqaq.github.io/projects/CenterLineDet/, the GitHub repository pages and raw files (see the code file). |
| axis, tier, K ids | A3 (sparse lane graph), tier 1 core; family of K03 (growth tracer / builder); related ids K23 and K50 (scheduled sampling / DAgger, restricted + free exploration) and K24 (log-odds fusion over time) |
| level-1 note(s) | [GR02](../papers/GR02_centerlinedet.md) (home note), [SN19](../papers/SN19_centerlinedet-bev-heat-maps-from.md) |
| code extraction | [centerlinedet](../code/centerlinedet.md) (GPL-3.0, read only) |
| sources read | arXiv HTML v2 in five separate queries (abstract page, section structure, III-A to III-C, IV-A to V, a literal re-read of the noise sentences of III-C2); the project page (no extra content: no ranges, no hyper-parameters, no supplement link); the repository READMEs and the files listed in the code file; the local files named in section 5 and 7. The supplementary document the paper refers to twice was **not found** (not on the arXiv page, not on the project page). Read 2026-10-25. |
| reading marks | Everything about the paper and the code is **digest**: every page came through the summarising fetch tool, no page was seen raw. Table I and II were cross-checked against the numbers in the level-1 notes (all values both list agree), which guards against transcription errors but is not an independent reading. Statements about our own code, plans and experiments are **checked** (local files read). Section 5 is my analysis, not paper content. Unit conversions (4 px per metre) and counts like 100/5 are my arithmetic and are marked. |
| extracted by, date | Claude (Sonnet 5.5) literature-extraction agent, 2026-10-25 |

## 1. Problem and claim
- Predict a **global lane centre-line graph** (vertices and edges in world coordinates, with forks and merges) for a whole nuScenes sequence from six cameras and lidar, to speed up HD-map creation (offline map building, not online driving).
- Two stages that are not trained end to end: a BEV network (HDMapNet or the authors' FusionNet) predicts a centre-line heat map and a candidate-start heat map; an imitation-learned agent then walks over the **predicted raster** (not over features) and builds the graph vertex by vertex with a DETR-like decision transformer.
- The claim that matters: graph traversal over the (multi-frame fused) heat map raises topology precision a lot (T-Pre 0.518 -> 0.782 with FusionNet) while recall barely moves (0.356 -> 0.409); pixel F1 moves little (single frame 0.719 -> 0.725). Recall stays bound by the BEV map (the paper's own limitation).
- Head type: the BEV head is **pixel-based** (heat maps), the agent is **graph-based** (vertices appended by a loop).

## 2. Method as built

### Pipeline / architecture (paper, with code values marked "code")
```
6 cams (+ lidar)  ->  BEV net (HDMapNet | FusionNet)      200 x 200 cells, +-25 m, 0.25 m (code defaults)
                        ch 0: centre-line heat map H_L      (sigmoid)
                        ch last: candidate-start map H_I    (sigmoid)
per scene: warp H_L of the frames to a common canvas, per-pixel mean over overlapping frames
           (paper: frames T-tau .. T+tau; code: ALL frames of the scene, see section 4)
           candidate map used by the agent (code) = fused H_L x single-frame H_I
per frame T, agent loop:
  seeds S = peaks of the candidate map + open ends of earlier frames inside the new patch
  pop a seed s (FIFO in code; the paper says the first seed is chosen at random)
  crop ROI around v_t from [fused H_L, history map M_E]            2 channels
  -> ResNet (conv1 replaced by a 2-channel conv, random init) -> DETR encoder 6 / decoder 6, d 256, 8 heads
  -> 10 queries, each: validity logits (2) + a point offset (2, tanh)    [+ ahead-path mask per query, training aux]
  -> N-hat valid vertices: 0 stop | 1 continue | >1 push all
```
- **BEV network.** HDMapNet: each camera feature map goes through a learned fully connected view transformer phi_i, aligned to the ego BEV with the extrinsics (Eq. 1). FusionNet adds a parameter-free inverse perspective mapping (IPM) as a shortcut and keeps phi_i as a residual branch; the camera BEV feature is the maximum over cameras (Eq. 2; the prose says "summation", the equation has a max over i, the paper does not resolve it). Lidar enters through a separate branch (model files for HDMapNet fusion / FusionNet exist in the repository; the paper text read does not describe the lidar branch). Channel counts, tensor shapes and the BEV losses are **not in the paper**; code: two supervised channels (segment, candidate), see the code file.
- **Historical maps.** M_E: binary map of the trajectory already traced, in the ego frame, 1 px wide; M_W: the same in world coordinates, kept so that M_E can be re-rendered consistently for each new frame (paper). Code: trajectories persist in world coordinates and are re-drawn into every frame's history map.
- **Agent step.** ROI R cropped around v_t from [H~_L, M_E]. The decision transformer returns up to N-hat next vertices with validity (paper: "N valid vertices"; the paper gives neither N-hat nor ROI size nor a threshold). Code: ROI_half_length argparse default 50 px (ROI 100 px) but every released shell script passes 100 (ROI 200 px = 50 m = the whole BEV canvas); 10 queries; the point head is an MLP (256, 256, 2, 3 layers) with tanh, scaled by ROI_half_length.
- **Inputs / outputs.** Input per step: 2 x ROI x ROI (fused centre-line probability, history trail). The **travel direction is not an input** (the BEV net may predict instance embedding and direction but the agent reads only channel 0 of the prediction and the history map, code). Output: a global graph in world coordinates; the code writes a pickle with a dict keyed by vertex coordinates and a list of neighbours, and a rasterised world map.

### Equations (numbered as in the paper)
- E1: F_bev^i = phi_i(F_Ii), i = 1..6 (HDMapNet view transformer per camera).
- E2: F_bev = max over i of { IPM(F_Ii) + phi_i(F_Ii) } (FusionNet).
- E3: V* = f(v_t, G*, M_E): next-step label vertices from the ground-truth graph G*, the current vertex and the history; f is modified from the labelling algorithm of RNGDet to handle complex topology. The paper gives no formula for f.
- No loss equation, no matcher equation, no equation for the fusion in the paper.
- Code-level definitions (not in the paper): label vector = (clean next GT vertex - noisy current vertex) / ROI_half_length, both in the ego canvas; decoded vertex = v_current + ROI_half_length x offset.

### Losses (code; the paper states none)
- Matcher: Hungarian (scipy) per image between the 10 queries and the k <= 10 labels of the step. Cost = 5 x Euclidean distance of the 2-D points (cdist, p = 2; the argparse help says "L1") + 1 x (minus the predicted probability of the target class); a giou cost is configured (weight 2) but never enters the matrix.
- Loss terms in the training script: cross-entropy on validity with the no-object class down-weighted by eos_coef (default 0.1, scripts 0.25) + 5 x L1 on the matched points + 1 x binary cross-entropy (pos_weight 5) of a per-query **ahead-path mask** (a 1-px polyline of the GT path between v_t and the target vertex, drawn on a 2 x ROI_half canvas around v_t). A focal term is computed in the criterion but not added to the sum. The weight dictionary in the criterion is hard-coded and not used.
- Class convention: 2 logits, class 0 = valid vertex (label value 0), last class = no object (unused slots carry label 1); the number of valid targets of a sample is stored as list_len.

### Labels / ground truth (GT graph, BFS expert, noise)
- **GT graph.** nuScenes map expansion, lane and lane_connector centre-lines discretised at 1 m (a modified devkit method get_graph); nodes with exactly two incident lanes are removed and their lanes concatenated; lanes with 10 or fewer vertices are dropped; one JSON per map location holds lanes (vertices, per-vertex orientation in degrees) and junctions (neighbouring lane ids). Per frame the lanes are clipped to the local BEV patch.
- **Expert.** The paper: "based on BFS" over G*, one sample per visited vertex. Code: no BFS routine; agents are objects in a list processed with a moving cursor, new agents are appended at intersections, which is a FIFO worklist (BFS-like). An agent starts at each clipped lane piece (more than 5 vertices) at its border end point, walks along the lane in steps of step_size (20 px = 5 m at 4 px/m, my conversion) towards the first GT vertex beyond the step; if the heading changes by more than 20 degrees inside the step range, the target is the vertex where it changes (shorter step in curves); if the remaining in-patch length is within the step, the target is the lane end. At a lane end inside the patch the agent switches to intersection mode: one new agent per untracked neighbouring lane, a **half step**, and one label per neighbour (a multi-target sample). Caps: 20 steps per agent, 100 steps per frame over all agents. Frames are sampled with sample_rate (default 3, scripts 1).
- **Noise injection.** Paper: "we add evenly distributed noises to the expert trajectory" (III-C2) and, in the paragraph of E3, noise is added to V* "when updating the graph"; the distribution is not defined and the paper never says that the labels stay clean (its stated purpose is robustness). Code (resolves it): the **executed path** is perturbed, the **labels stay clean**: the next position of the sampler agent is the clean expert vertex plus independent uniform noise in [-noise, noise] per coordinate (canvas pixels; noise default 3, scripts 4 = 1 m, my conversion), clamped to the canvas, halved by integer division on curves; the label at each step is computed from the noisy current position to the clean next GT vertex; the history map is drawn with the noisy path. So every sample is a "recovery" example: noisy state, clean target.
- **Seed targets for the BEV net.** Candidate map GT = discs of radius 2 px at the points where a clipped lane touches the patch border (the exclusion test keeps left / top / bottom border points and drops the right border; which ego direction these are is not verified); interior lane ends and junctions are not seeds. Centre-line GT = clipped lanes as 2-px polylines. The expert agents start exactly at the same border points.
- Training sets: the agent is trained on predictions of a BEV net that **never saw those scenes** (cross-fitting, see below).

### Training recipe
- Paper: BEV net one day; behaviour-cloning sampling about 5 h; CenterLineDet training one extra day; four RTX 3090 for training, one GPU for inference. No optimiser, learning rate, epochs or batch size in the paper (supplement not found).
- Code scripts: BEV net AdamW, StepLR(10, 0.1), gradient clip 5, BCE with pos_weight 2.13 on both channels, 50 epochs; the "for CenterLineDet" BEV net is trained on scene half train_1 (349 scenes) for 50 epochs (learning rate 1e-3 in the script), the agent on the other half train_2 (349 scenes) with the **held-out** predictions of that BEV net. Agent: AdamW, StepLR(20, 0.25), lr 1e-4, weight decay 1e-5 default, 100 epochs, batch size argument 128 on four GPUs (per-process or total is not verified), ResNet-101 trained from scratch, one random 90-degree rotation as the only augmentation, best checkpoint by F1 on 10 validation scenes (which the data README says are taken from the 148 test scenes).
- BC only: the loader raises unless training_mode is BC; the DAgger option named in the arguments is not implemented in the released code.

### Inference (the hidden rule steps)
- **Heat-map fusion.** Paper: neighbouring heat maps T-tau .. T+tau are warped with the ego poses, summed and averaged into H~_L (tau not given). Code: one script fuses **all frames of a scene** (past and future) by a per-pixel mean over overlapping footprints, only for the centre-line map; the candidate map handed to the agent is the fused centre-line map times the single-frame candidate prediction.
- **Seeds.** Connected components (8-neighbourhood) of the candidate map above binary_thr (default 0.5, scripts 0.05) with area above area_thr (default 4, scripts 8); centroid = seed. Seeds closer than filter_distance (default 15, scripts 10 px = 2.5 m) to any pixel of the history trail are dropped.
- **Dead-end seeds of the previous frame.** All traced instances persist in world coordinates. In each new frame, every instance whose first or last vertex lies inside the new patch and whose end is not flagged closed is queued again and continues from that end. An end stays open when the agent stopped with N = 0, when it hit the 40-step cap, or when it left the patch; an end is closed when it branched (N > 1) or was snapped to another instance. So the carried seeds are open ends only, not all nodes.
- **Queue.** FIFO; each popped instance is traced forward from its last vertex and backward from its first vertex, at most 40 network calls per direction, a counter of 200 per frame is checked only when popping (so it can be exceeded by up to 40).
- **Decision per step.** Validity = softmax probability of the valid class >= logit_thr 0.9 (the thresholded quantity is a probability, despite the name); the point must lie in the canvas, point forward (positive dot product with the previous move, first step exempt), be longer than 1 px; near-parallel candidates (unit vectors with dot product above 0.99) are thinned (the branch meant to keep the longer one looks dead). N = 0: stop (end stays open). N = 1: align, append, stop if aligned, if the 40 cap is reached or the point is within about 2 px of the canvas border (x < 5 not tested). N > 1: close this end, for each candidate align and draw it and spawn a new instance (queued).
- **Endpoint alignment.** A new vertex within world_alignment_distance (1 world unit = 4 px) of the first or last vertex of any other live instance is snapped to it and the instance stops: **merges are snapped only to end points**, never to the interior of another polyline. Instances whose end points lie within that distance are also de-duplicated by flags at the start of each frame.
- **Output.** World-coordinate polylines, rasterised at 4 px per metre; connected components below 100 px are removed; the graph is saved from the polyline vertices.

## 3. Numbers

### Table I (comparison; digest, nuScenes, 148 inference sequences; "Single" per frame, "Multi" scene-level after merging)
| approach | single P-Pre | P-Rec | P-F1 | multi P-Pre | P-Rec | P-F1 | T-Pre | T-Rec | T-F1 |
|---|---|---|---|---|---|---|---|---|---|
| HDMapNet | 0.805 | 0.649 | 0.709 | 0.714 | 0.665 | 0.685 | 0.517 | 0.354 | 0.400 |
| TopoRoad | 0.408 | 0.566 | 0.461 | 0.410 | 0.526 | 0.477 | 0.360 | 0.349 | 0.352 |
| FusionNet | 0.813 | 0.658 | 0.719 | 0.726 | 0.674 | 0.695 | 0.518 | 0.356 | 0.403 |
| CenterLineDet + HDMapNet | 0.785 | 0.675 | 0.711 | 0.700 | 0.713 | 0.699 | 0.768 | 0.403 | 0.511 |
| CenterLineDet + FusionNet | 0.811 | 0.675 | 0.725 | 0.732 | 0.708 | 0.714 | 0.782 | 0.409 | 0.518 |

Topology-level scores are reported for the multi-frame setting only. The F1 columns are not the harmonic mean of the printed averages (my arithmetic: 0.732 / 0.708 would give 0.720, printed 0.714), consistent with scores averaged per scene.

### Table II (ablation, multi-frame; digest)
| approach | P-Pre | P-Rec | P-F1 | T-Pre | T-Rec | T-F1 |
|---|---|---|---|---|---|---|
| CenterLineDet - no LiDAR | 0.719 | 0.683 | 0.698 | 0.721 | 0.365 | 0.466 |
| CenterLineDet - no camera | 0.631 | 0.576 | 0.605 | 0.620 | 0.338 | 0.394 |
| CenterLineDet (full) | 0.732 | 0.708 | 0.714 | 0.782 | 0.409 | 0.518 |

Removing the cameras hurts more than removing the lidar. There is no ablation of the agent itself (history channel, noise, multi-frame fusion, seeds from the previous frame, validity threshold, step size): the effect of each design choice named in this extraction is **unmeasured** in the paper.

### Compute
- CenterLineDet inference 41 min for 5,981 frames = 0.41 s per frame (2.43 Hz) on one RTX 3090 (training machine: i7-8700K, four RTX 3090); BEV net alone 13 min for the same frames; the paper itself notes the speed is below the 2 Hz key-frame rate and calls the task not online. Parameters and per-network-call time: **not found**; the released code has no timing code.
- Data: 698 training sequences, 148 for inference, about 40 frames each at 2 Hz (sequences without centre-lines or without a usable perspective transformation excluded from inference). Code README: 698 = 349 + 349 scenes (train_1, train_2); the 148 test scenes are the TopoRoad split.

### Caveats
- The multi-frame numbers are **offline and acausal**: the heat maps are averaged over the whole scene (code), the graph is accumulated over the scene and scored against a scene-level GT; our J, F1@x and S are per frame on a +-50 m window. No number can be compared with ours.
- The single-frame pixel gain of the agent is small (+0.006 with FusionNet, +0.002 with HDMapNet in P-F1); the topology gain is in precision.
- Tolerances: the paper gives none (delta, epsilon, epsilon-prime not stated). Code: pixel metric 3 px (0.75 m at 4 px/m, my conversion) on a skeletonised prediction; TOPO as in the code file (units unresolved).
- Checkpoint selection used 10 of the 148 evaluation scenes (data README).

## 4. Paper versus code
- **Fusion window:** paper T-tau .. T+tau, code all frames of the scene; the fused product (centre-line x candidate) feeding the seed detector is a code detail the paper does not mention.
- **First seed:** the paper says the agent starts from a randomly selected seed, code pops from the front of a list (FIFO).
- **Noise:** the paper's wording is ambiguous (trajectory vs V*); code perturbs the executed path and keeps labels clean.
- **Loss:** the paper has none; code uses Hungarian matching, CE + 5 x L1 + ahead-mask BCE; the giou weight is dead configuration; the ahead mask is an auxiliary output not seen used at inference.
- **Defaults vs scripts:** the argparse defaults differ from the values in all released scripts (ROI half length 50 -> 100, noise 3 -> 4, sample rate 3 -> 1, binary_thr 0.5 -> 0.05, area_thr 4 -> 8, filter_distance 15 -> 10, eos_coef 0.1 -> 0.25, lr 1e-3 -> 1e-4, backbone resnet50 -> resnet101, batch 1 -> 128). Unchanged: step_size 20, logit_thr 0.9, alignment 1, 10 queries, 6 + 6 layers, d 256.
- **TOPO:** the paper describes the metric in its own words (sub-graph around each GT vertex); the code is the marble / hole TOPO of the road-graph literature with one-to-one matching. The README calls the unit metres while the pickles seem to be in 4 px per metre canvas units (section 6).
- **DAgger:** named in the arguments, absent in the training code.

## 5. Transfer to our pipeline

| question | answer |
|---|---|
| what it needs from us | per frame: a continuous centre-line probability map in the ego frame (our dense head: heat map with Gaussian sigma 0.75 m on 200 x 200 cells of 0.5 m, +-50 m; no retraining of the foundation), a candidate-start map, ego pose (translation + yaw) for the world persistence, and a GT lane graph with junction adjacency for the sampler. No image features, no lidar, no SD map. |
| what it gives us | axis A3 baseline of the trace / builder family; ideas for A5 (carry only open ends) and A4 (training regime). Not a new module for our chain: it is slower and not differentiable. |
| conflicts | Hungarian matching on its training loss (our standing rule forbids it in our losses; allowed only if we agree that an external baseline may use it); thresholds, connected components, KD-tree seed filter, FIFO queue and endpoint snapping are rule steps; the BEV head is not trained through the agent; ResNet-101 on 200 x 200 for up to 200 calls per frame cannot meet the 31 ms budget; GPL-3.0 code (describe and re-implement, copy nothing); python-2 TOPO. |
| reuse mode | Idea only for the three ideas below; **re-implement from description** for a baseline. |
| effort | Baseline (my estimate): M, about 3 - 5 days of work plus 1 - 2 days of GPU with a small backbone (ResNet-18/34, not 101); L if faithful (ResNet-101, 100 epochs). The three ideas: S each. |
| expected gain and risk | Baseline: a published-style reference for "heat map + agent" on our frozen foundation; the nearest in-house analogue is E94 (seed head, fan sampling, teacher forcing, F1@1 0.851 on TEST), so the new information is only DETR set prediction with validity versus fan sampling plus the history channel; risk: it will be dense (see 5.3) and slow, so S and ms per frame will look bad unless a Douglas-Peucker pass is added and reported separately. Ideas: small, mostly inside the noise band on clean frames (E173 was a tie). |
| feeds which experiment / K id | proposed technique names, to be registered in `../register.md` **before** use: "open-end carry" (previous-frame seeds restricted to open tips plus trail suppression) and "expert-tip jitter" (uniform, curvature-halved, for teacher-forced warm start); related existing ids K23, K24, K50. |

### 5.1 What the agent needs from our BEV model, and how to supply it
1. **Centre-line map H_L.** Our dense head already gives it. Differences: their target is a 2-px-wide binary line trained with BCE (pos_weight 2.13), ours a Gaussian ridge with sigma 0.75 m; the agent has to be **retrained on our maps**, the released weights are useless (different resolution, different value profile, 2-channel input trained from scratch). Cell size: 0.5 m instead of 0.25 m, so their step of 20 px becomes 10 px for 5 m; ROI of 100 m (200 cells) if we keep "ROI = whole canvas", or a +-25 m crop (100 cells).
2. **Candidate-start map H_I.** Not optional: without it no instance starts in the first frame. We have no such channel in the dense head. Options: (a) one more output channel trained on discs at the border crossings of the GT lanes (their definition; it excludes interior starts, so lanes that begin inside the field of view are reached only by branching from another instance or by previous-frame ends); (b) our E94 seed head or peaks of the heat map (E94 used heat-map peaks as seeds); (c) cut the thresholded heat map ridge at the canvas border. Their code multiplies the fused centre-line map into the candidate map before thresholding (threshold then only 0.05); my reading is that this suppresses isolated candidate blobs away from any centre-line, the paper does not say why.
3. **Multi-frame fusion.** Their fusion is the plain mean of warped maps over the whole scene. For us the causal version is a warp of the previous fused map by `grid_sample` (our time recursion already warps by the ego motion). Cross-reference: the L2G-Map extraction ([VM12](VM12_l2g-map.md)) reports that a probability mean was worse than log-odds accumulation in a global-map fusion (digest there); option R of that file (log-odds on the dense heat map) is the stronger version of what CenterLineDet does, at effort S.
4. **Training data hygiene.** The agent must be trained on maps of scenes the map network never saw (their half / half split; the README states the reason). We already keep out-of-fold token files for the builder; a baseline needs out-of-fold dense-head maps for TRAIN (two folds of a small U-Net, cheap) because our foundation is frozen but our head is trained on TRAIN. This also matches our own seen-versus-unseen finding for the foundation (0.932 against 0.820): the published method treats seen-scene predictions as unusable training input.
5. **GT for the sampler.** A polyline graph with junction adjacency in the ego or world frame: we have the nuScenes lane / lane-connector centre-lines; the sampler needs them clipped to the patch, merged through degree-2 nodes and oriented.
6. **Poses.** Exact ego poses (we use them). The method needs a persistent world frame inside a scene, which clashes with the "no pose given" task statement; for the baseline use ground-truth poses and say so.

### 5.2 Blockers and evaluation
- **Python-2 container.** TOPO runs in a TensorFlow-1.15 image (Python 2 code, text-mode pickles, the road-graph packages rtree and hopcroftkarp); pixel F1 runs in Python 3. We need neither Docker nor Python 2: the metric logic is plain Python and would be re-implemented in Python 3 (rtree could be replaced by a KD-tree, Hopcroft-Karp by a bipartite matching routine in scipy). GPL and the third-party origin of the TOPO code mean re-implementation from the description, not copying. Installing rtree or hopcroftkarp needs the user's OK.
- **What the pixel F1 is:** nearest-neighbour precision and recall in both directions at 3 px on a skeletonised prediction against an unskeletonised GT mask, mean over images: the same kind of metric as our point F1 at 0.5 / 1.0 m. A threshold at 0.75 m would make it directly comparable inside our evaluation (our evaluation/ already has the machinery; per project rule extend it, do not write a private metric).
- **What TOPO is here:** marbles propagated from GT start points up to a distance, one-to-one matching with distance and angle tests, precision from the prediction side, recall multiplied by a coverage factor (digest); averaged with equal weight over scenes. Unit question open (section 6). Worth building only if a TOPO-style number is demanded; our J and path-level scores cover the same ground.
- **Speed.** Up to 200 network calls per frame plus CPU work (shapely, KD-tree, drawing) for 0.41 s per frame on a 3090; with 873 VAL frames this is minutes, so evaluation is not the blocker; the ms-per-frame rule means the baseline sits far outside the speed frontier and can only be a quality reference.

### 5.3 Which ideas are cheap to add to the verified-ray builder
| idea | what the code does | where it fits our builder | effort / expectation |
|---|---|---|---|
| **Open-end seeds of the previous frame** | Only ends that were open (N = 0, cap hit, left the patch) are re-queued when they lie in the new patch; ends that branched or snapped are closed; new seeds within filter_distance of the existing trail are dropped. | E173 / E176 already warp the previous frame's nodes (top 64, validity above 0.3, validity x 0.9) into extra seeds, a tie in S. The difference is the restriction to **open tips** (tips whose survival ended inside the field of view or at its border, which our hazard read-out already provides) plus trail suppression; the latter is what the merge radius 1.5 m and snap 1.0 m partly do. | S. Expect a tie on clean frames; possible gain in the far field and under failure frames, fewer duplicate seeds. Test it as a variant of the E173 arm with the loop-17 failure protocol. |
| **Clean labels + noisy executed path** | Executed path = clean vertex + uniform noise (1 m, half on curves), label = clean next vertex minus the noisy position, history drawn with the noisy path. | Our builder trains by free roll-out from step 1 with matching-free point-to-polyline losses, so its labels do not depend on any expert state and the trick is mostly built in. It matters for the **teacher-forced warm start** (factor F, K23) and for E94-type trainers: jitter the expert tip instead of replaying exact states. | S. Expect a small effect, bounded by the fact that free roll-out replaces it after the warm start. Check that p16 training has no tip jitter (a grep of the training folder found none in `p16_builder.py`). |
| **Step halving at junctions, snap to node** | The expert halves the step for the branches after a fork and ends a step exactly on the lane end (the junction node) when the remaining length is within the step; curves shorten the step where the heading changes by more than 20 degrees. | The expert never steps over a fork and starts the two children at half range, which keeps the children of a fork mouth (about 3.5 m apart) from collapsing into one hypothesis. Our junction head + fork relocation does this after the fact; as a label rule, let the shortest ray ring (4 m) carry the child targets whenever a GT fork lies within the step range of the tip. | S - M. Plausible help for junction recall J (0.265 against 0.408 at equal sparsity); untested. |
| curvature-triggered vertex (20 degrees) | vertex where the heading has changed by more than 20 degrees | A Douglas-Peucker-like placement rule; our champion (13.5 per 100 m) is already below the DP oracle (14.3). | nothing to take; useful as a reminder that their graph is dense. |

### 5.4 What not to take
- Direction-free input: the agent infers heading only from its own trail; our 2-channel travel direction is information it lacks.
- Merges snapped only to first / last vertices of other instances, no merge state, no junction head: its fork / merge quality is limited by construction (this is my reading of the digest; they report no fork / merge metric).
- Density: with a 5 m expert step the straight-lane graph has at least 20 vertices per 100 m (100 / 5, my arithmetic) before the extra curve vertices, i.e. as dense as our control (21.8) and far above the champion (13.5); a fair S comparison needs a simplified variant reported next to the raw one.
- Acausal scene-wide fusion and a scene-level metric: not our setting.

## 6. Not found / open
- The supplementary document (training hyper-parameters, N-hat, ROI, thresholds, tau); the paper states none of them.
- Per-call and per-frame timing split (BEV net, network calls, CPU work); number of calls per frame; parameter count.
- Whether TOPO distances are in metres or in canvas pixels: the README says propagation distance 100, interval 1, matching threshold 2 "meters" and an angle of 30 degrees, while the digest of the code shows pickles in 4 px per metre canvas units, default arguments of about 2.2 and 1.1 "metre" units, and angle tests of 0.04 and 0.29 (one minus the absolute cosine, about 15 and 45 degrees) with the 30-degree option commented out. If the pickles are not rescaled, the true tolerances are four times smaller. Not resolved without running or reading topo.py itself raw.
- How the shell-script batch size 128 maps to the effective batch with four processes; the number of training samples produced by the sampler; the BEV net checkpoint-selection details.
- Whether the ahead-path mask is used at inference (not seen); what the label function does at unusual topology (parallel lanes, merges into an interior point).
- The direction of the border sides used for candidate seeds (left / top / bottom kept, right dropped) in ego terms.

## 7. Our experiments that touch it
- E94 - E99: learned trace network (E94: 1.67 M parameters, seeds = heat-map peaks, fan of 27 samples, classes 2 / 4 / 6 m, F1 0.883 / 0.851 VAL / TEST, 22 vertices per 100 m, split / merge F1 0.36 / 0.46); the same interface family as CenterLineDet (raster in, local steps out) without DETR queries or a history channel (it uses a covered-crop raster). Seeds limited it (oracle seeds 0.921 against 0.908 F1@1).
- E31 / E32: dense heat map + tracer beat a matched sparse decoder (+0.058 F1, +0.18 route F), the same direction as CenterLineDet's precision gain.
- E41: temporal feature fusion +0.008 F1 (comparable to their heat-map averaging).
- E173 / E176 / E192: previous-frame nodes as seeds with hard-gated validity fusion: a tie in S (VAL250).
- Not tried: open-end-only carry, expert-tip jitter, junction-halved child targets, a baseline run of the CenterLineDet agent.

## Corrections to the level-1 note (GR02; SN19 where marked)
1. **Defaults are not the used values.** ROI 100 px, noise 3 px, every 3rd frame, seed threshold 0.5, area 4, filter distance 15, lr 1e-3, batch 1 are the argparse defaults; the released scripts use ROI half length 100 (ROI 200 px = the full 50 m canvas), noise 4, every frame, 0.05, 8, 10, lr 1e-4, batch 128, ResNet-101, eos_coef 0.25. Unchanged: 20 px step, 0.9 validity, alignment 1, 10 queries.
2. **Matcher weights:** the giou weight 2 is never used; the L1 cost is a Euclidean distance on 2-D points; the loss also has an ahead-path mask term (not mentioned), the final sum is CE + 5 L1 + BCE, the focal term is not summed. Hungarian matching is confirmed (scipy).
3. **Multi-frame fusion** is a scene-wide mean over all frames, not a +-tau window, only for the centre-line map; the agent's candidate map is fused map x single-frame candidate. The Table I multi numbers are therefore offline and acausal.
4. **"Seeds plus endpoints where the agent stopped"** is narrower in the code: any open end inside the new patch is re-queued (N = 0 stops, cap hits, patch exits), closed ends (branched or snapped) are not.
5. **Expert:** there is no BFS function; it is a FIFO worklist of agents; GT graph source and filters (lane + lane_connector at 1 m, degree-2 merge, more than 10 vertices) were missing. "100 steps per sample" is 100 steps per frame over all agents.
6. **Cross-fitting** (BEV net on train_1, agent on train_2) and the border-only candidate GT are missing in the note; 10 test scenes are used for checkpoint selection.
7. **"Differentiable? no ... matching"** stands, and the DAgger flag is confirmed unimplemented. The licence is **GPL-3.0** (the note says not noted).
8. The paper does not say that the labels stay clean; this comes from the code only. Where the note says the paper "omits ... loss equations": confirmed, it also omits all thresholds, tau and the metric tolerances.
9. SN19: "code not checked" is now checked; "ROI ... size not stated" is correct for the paper, the code has it (100 px or 200 px as above). Table I values in both notes were re-checked and agree.
