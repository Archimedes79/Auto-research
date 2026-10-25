# Cluster GR - iterative, agent-based and autoregressive growth of lane graphs and road graphs

Date: 2026-10-23. Agent: literature agent GR (reading only: WebSearch / WebFetch; nothing downloaded, no git / curl / pip). Project facts: `PROJECT_CONTEXT.md`; card format: `SCHEMA.md`. Earlier abstract-level review: `lit_p16_tree.md` (Q1).

---

## 1. Coverage statement

- **Searched:** growth / agent / autoregressive / sequence / diffusion methods for road graphs (aerial) and lane graphs (BEV, nuScenes, OpenLane-V2, Argoverse), 2018 - 2026, plus adjacent fields that grow trees by tracing (medical centerline trees) and the generic exposure-bias literature (DAgger, scheduled sampling, DART). 28 search queries and about 110 page fetches (log in section 6).
- **Read at depth M (method section, via arXiv / ar5iv HTML):** all 12 priority items: RNGDet, RNGDet++, CenterLineDet, iCurb, SeqGrowGraph, RNTR, LaneGraph2Seq, DAGMapper, RoadTracer, DeH4R, Sat2Graph (graph-tensor encoding), TopoGPT, Anagnostidis et al. (note: ICCV 2023; arXiv 2022). Widened at M: LaneGNN, LaneGraphNet, SAM-Road, SAM-Road++, LineGraph2Road, DOGE, LaneDiffusion, MapDreamer, SLEDGE, Topo2Seq, VectorMapNet, HDMapGen, Neural Turtle Graphics, Image-conditioned graph generation (Belli and Kipf), Ventura et al. (iterative road topology), PolyMapper, PAMR, LSR-DM, Trexplorer Super. At depth A: Trexplorer base paper (MICCAI page), Friman et al. (multiple-hypothesis vessel tracking), DAgger / scheduled sampling / DART abstracts.
- **Depth C (code pages):** README pages for RNGDet++, CenterLineDet, iCurb / Topo-boundary, RNTR / LaneGraph2Seq (one repo), DeH4R, Sat2Graph, SAM-Road, TopoGPT, RoadTracer, VecRoad, Trexplorer Super. Raw source files read for CenterLineDet (`config/get_args.py`, `CenterLineDet/sampler.py`, `CenterLineDet/inference.py`), iCurb (`config.yml`, `main_env.py`), RNGDet++ (`cityscale/main_train.py`) and RNTR / LaneGraph2Seq (`RoadNetwork-2.0.1/get_started.md`); these give hyper-parameters and dependencies the papers omit and are marked "(code)". The code was read as text only; nothing was run.
- **How to read the numbers:** WebFetch returns a condensed text made by a small model; tables with check-mark columns often arrived garbled. Where a row-to-variant mapping could not be recovered it is said so in the card; such numbers are given only as ranges or omitted.
- **Not readable / only depth A:** VecRoad (CVPR 2020: paper and supplement are PDF-only, 403 / binary) is depth A + C; the base Trexplorer paper (MICCAI 2024, no arXiv) is known only through its MICCAI listing page (abstract, reviews, rebuttal) and the Trexplorer Super text; SeqGrowGraph has no code page and its paper text lacks the training hyper-parameters; the Anagnostidis ablation table is garbled. PDFs fetched via WebFetch come back as binary, so PDF-only papers could not be read.
- **Not found:** no lane / road graph paper (2024 - 2026) that grows a graph with a fixed-shape b-ary tree or beam (the only beam decoder found is PolyMapper, 2018, over vertex sequences); no BEV grower trained with gradient through its own rollout except the short truncated-BPTT in DAGMapper (aggregated lidar-intensity BEV maps of highways); no grower that picks step length by sample-wise verification.
- **Our-environment note (what the code pages say):** RNGDet++ / CenterLineDet / iCurb-Topo-boundary run in Docker images with PyTorch (no mmcv / mmdet mentioned; CenterLineDet evaluation needs a python-2 image); DeH4R needs PyTorch + SAM2 and the SAM / SAM2 checkpoints; TopoGPT is plain PyTorch 2.1 + Hydra (no mm stack); Trexplorer Super is plain PyTorch 2.2; **RNTR / LaneGraph2Seq (one repo) need mmdet3d 1.4.0, mmcv >= 2.0.0rc4, mmdet >= 3.0.0 and mmengine** (`RoadNetwork-2.0.1/get_started.md`) - a blocker in our environment. Licences seen on README pages are mentioned in section 4 only; the EV cluster owns the licence table.

---

## 2. Cards (ordered by relevance to the verified-ray hypothesis tree)

### CARD GR01 - RNGDet / RNGDet++ (DETR-style set-of-next-vertices agent, behaviour cloning)
- papers: RNGDet: Road Network Graph Detection by Transformer in Aerial Images (Xu et al., IEEE T-GRS 2022) https://arxiv.org/abs/2202.07824 ; RNGDet++: Road Network Graph Detection by Transformer with Instance Segmentation and Multi-scale Features Enhancement (Xu et al., RA-L 2023) https://arxiv.org/abs/2209.10150
- depth: M+C
- stage: S4 (per-step hypotheses), S1 (seeds), T
- idea: An agent crawls the road graph from seed vertices. At each vertex it crops a window of image features, two segmentation maps (road, intersection / endpoint) and a raster of the graph built so far; a DETR transformer with 10 learned queries proposes up to 10 next vertices, each with a validity score. The count M of valid outputs decides: 0 = dead end (take next seed), 1 = move on, >1 = junction (all outputs go to a FIFO buffer). Trained only by behaviour cloning on a BFS expert whose executed path is perturbed by noise.
- formulation: L = L_seg + 5 L_coord + 1 L_valid (RNGDet). L_coord = L1 over bipartite-matched (prediction, GT) pairs; L_valid = BCE (matched = 1). RNGDet++ adds L_ins (BCE of a per-query mask of the road segment between v_t and the proposed vertex; weights garbled in the fetched text) and 4-level FPN features. Label rule on the GT graph after removing degree-2 vertices: road-segment mode, one target = GT point tau = 40 px ahead, or the nearest sharp turn, or a seed if one lies within tau; intersection mode (after a segment ended), one target per incident segment at tau' = 20 px. Agent executes target + noise (RNGDet: Gaussian, scale not stated; RNGDet++: evenly distributed); labels stay clean. ROI 256 px at 0.6 m/px (RNGDet) or 128 px at 1 m/px (RNGDet++). Seeds: local peaks of the intersection map I (RNGDet++: of a global heat map merged over ROIs), plus vertices created at junctions and candidates met on the way. Queries 5 / 10 / 20: APLS 62.78 / 67.88 / 65.31 (RNGDet Table III). Repo defaults (RNGDet++ README, code): move-ahead 30 px, sampling noise 8 px, validity threshold 0.75, seed peak threshold 0.7, candidate filter 30 px, merge ("alignment") distance 5 px. Step length is regressed, not fixed. Training script (`cityscale/main_train.py`): `ROI_SIZE` default 256 (the RNGDet++ paper text says 128; the shell scripts may override), 10 queries, 6 + 6 transformer layers, hidden 256, AdamW lr 1e-4 (backbone 1e-5), DETR matcher costs class 1 / L1 5 / giou 2, loss = CE + 5 L1 + segmentation (+ instance) terms, validation F1 at 3 px on skeleton points; the file contains no DAgger or rollout code.
- input: RGB ROI (+ S, I maps, + history raster 1 x L x L); `image_feat` + `bev_raster`-like history.
- output: up to 10 vertices (x, y) + validity per step; graph assembled by a FIFO candidate buffer. No stop or junction head: M comes from thresholding validity.
- training: behaviour cloning on about 300 K (RNGDet) / 400 K (RNGDet++) samples, 50 epochs, lr 1e-4, 4 x RTX 3090; regime `teacher_forcing` + noise; Hungarian-type matching. DAgger is named in the text and rejected for sampling cost.
- differentiable: no - validity threshold / M logic, FIFO control flow, bipartite matching, per-step re-cropping.
- sensors_data: aerial RGB. RoadTracer 40-city set (0.6 m/px, 4096 x 4096 tiles); City-scale (180 tiles, 2048 x 2048, 1 m/px, 144 / 9 / 27); SpaceNet (2,551 tiles of 400 x 400, 1 m/px).
- results: City-scale TOPO-F1 / APLS: RNGDet++ 78.44 / 67.76, RNGDet 76.87 / 65.75, Sat2Graph 76.26 / 63.14; SpaceNet: 82.51 / 67.73 vs RNGDet 81.13 / 65.61 (RNGDet++ paper tables). RNGDet R-101 APLS 67.88 on the RoadTracer set (Table I). RNGDet++ ablation (City-scale F1 / APLS; the two single-removal rows are the instance head and the multi-scale features, row assignment unclear): 77.94 / 66.36 and 78.01 / 66.94 vs full 78.44 / 67.76, i.e. small gains of 0.4 - 0.5 F1 and 0.8 - 1.4 APLS. DeH4R reproduces RNGDet++ as 78.44 / 67.72.
- compute: parameters not stated; one backbone pass per step: all test images City-scale 2.68 h (RNGDet) vs 3.85 h (RNGDet++) in the paper; DeH4R measures 136 min for RNGDet++.
- code: https://github.com/TonyXuQAQ/RNGDetPlusPlus
- fit_to_us: reuse_idea - effort S (label rule, two modes, queue logic, query-count ablation, instance-mask-ahead auxiliary target). adapt_code is not advised: per-step image ROI + backbone, bipartite matching.
- compat_needs: seed set from a heat map; history as raster or token path; simplified GT graph with BFS expert; K output slots with a validity score.
- compat_conflicts: Hungarian matching of K slots (our losses are matching-free); BC only, no gradient through the rollout; per-step crop vs cached pyramid (adapter: grid_sample of the f0 / f1 / f2 pyramid around v_t).
- tried_before: E94 - E99 (learned trace network, partial: F1@1 0.851 vs 0.867, seeds limited) is the same family; E31 / E32 (matched sparse DETR decoder dropped) used matching on the whole image, not per step.
- verified: checked (paper HTML, README); RNGDet++ ablation row mapping not recoverable.

### CARD GR02 - CenterLineDet (RNGDet agent over a predicted BEV raster, nuScenes)
- papers: CenterLineDet: CenterLine Graph Detection for Road Lanes with Vehicle-mounted Sensors by Transformer for HD Map Generation (Xu et al., ICRA 2023) https://arxiv.org/abs/2209.07734
- depth: M+C
- stage: S4, S1, S8 (seeds carried between frames)
- idea: Two stages. A BEV network (HDMapNet, or "FusionNet" = HDMapNet + a parameter-free inverse-perspective shortcut with lidar) predicts a centerline heat map and a candidate-seed heat map. Then an RNGDet-type agent walks over the *predicted raster* (not over features) and builds the graph. Heat maps of neighbouring frames are ego-warped and averaged; the dead ends of frame T-1 become seeds of frame T.
- formulation: Seeds S = peaks of H_I plus endpoints where the agent stopped in frame T-1, minus already explored ones. Agent input = ROI crop around v_t of [averaged H_L, history map M_E]; DETR-like decision transformer returns N-hat vertices with validity; N = 0 stop (endpoint becomes a seed for the next frame), N = 1 continue, N > 1 push all. The paper omits BEV extent, ROI, N-hat and loss equations. Code (`config/get_args.py`, `sampler.py`): BEV +-25 m at 0.25 m (200 x 200), ROI 100 px (= 25 m), expert step 20 px (= 5 m), BC noise 3 px (uniform, added to the executed path, halved on curves, labels stay clean), 10 queries, 6 + 6 layers, hidden 256, validity threshold 0.9, seed filter distance 15 px, endpoint alignment 1 m, lr 1e-3, 100 epochs, batch 1, DETR matcher cost weights (class 1, L1 5, giou 2). Sampler: label = (next GT vertex - v_t) / ROI_half; road mode target = first GT vertex beyond step size (curve if orientation changes > 20 deg within the step); intersection mode uses step / 2 and one target per neighbour; at most 20 steps per agent, 100 per sample; every 3rd frame sampled. A `training_mode` flag allows "DAgger" but the released sampler only contains the BC branch.
  Inference code (`inference.py`): seeds = centroids of connected components (area > 4 px) of the thresholded (0.5) candidate map, dropped if closer than 15 px to any stored past trajectory; validity = softmax score >= 0.9, in addition a vertex must point forward (dot product with the previous direction > 0), be longer than 1 px, and near-parallel duplicates (dot > 0.99) are removed (the removal branch looks like it never fires); N = 0 ends the direction, N = 1 appends (with snap-to-other-endpoint alignment, which stops the instance), N > 1 spawns one new instance per branch; FIFO queue; caps 40 steps per direction and 200 per frame; trajectories persist in world coordinates across frames and are re-rendered into each frame's history map, so open ends of earlier frames are continued when they lie in the new patch; the multi-frame heat-map fusion is done by a separate script (not in the agent code).
- input: ROI crop (100 x 100 px) with **two channels fed to the network: predicted centerline segmentation and history map** (code: channels [0, 2] of the stored tensor; the GT-segment channel is only used for labels); `bev_raster`.
- output: global graph from agent trajectories; up to 10 vertices per step.
- training: BC on a BFS expert (about 5 h sampling + 1 day training on 4 x RTX 3090); BEV net trained separately; `teacher_forcing` + noise; DETR matching.
- differentiable: no - two-stage (BEV net is not trained through the agent), thresholds, queue logic, matching.
- sensors_data: 6 cameras + lidar (FusionNet) or cameras only; nuScenes 698 training sequences, 148 inference sequences (5,981 frames, 2 Hz); BEV extent not in the paper.
- results: nuScenes multi-frame (Table I): CenterLineDet + FusionNet pixel-F1 0.714, topology-F1 0.518 (T-Pre 0.782, T-Rec 0.409) vs FusionNet alone 0.695 / 0.403; no lidar 0.698 / 0.466; no camera 0.605 / 0.394. Topology precision rises a lot, recall barely (0.356 -> 0.409): recall stays bound by the BEV map.
- compute: 0.41 s / frame (2.43 Hz, one RTX 3090), 41 min for 5,981 frames; BEV net alone 13 min.
- code: https://github.com/TonyXuQAQ/CenterLineDet ; https://tonyxuqaq.github.io/projects/CenterLineDet/
- fit_to_us: baseline + reuse_idea - effort M for a baseline (the agent only needs a heat map, so it can run on our dense head output; blockers: Docker, python-2 evaluation container, CPU-bound sampler); S for the ideas (previous-frame dead ends as seeds, clean labels + noisy executed path, step-halving at junctions).
- compat_needs: heat map (+ optional seed map), ego poses for warping, GT graph with BFS expert.
- compat_conflicts: DETR-style bipartite matcher (implied by the matcher cost weights in the config; the paper does not name it); BC only; not end to end; 5 m expert step vs our 2 / 4 / 6 m classes (adapter: step as class).
- tried_before: E94 - E99 same family (partial); E41 temporal feature fusion gave only +0.008 F1, comparable to their heat-map averaging.
- verified: checked (paper HTML, config and sampler source); paper and code differ in what they state.

### CARD GR03 - iCurb (DAgger with restricted and free exploration, dynamic expert labels)
- papers: iCurb: Imitation Learning-based Detection of Road Curbs using Aerial Images for Autonomous Driving (Xu, Sun, Liu, RA-L 2021) https://arxiv.org/abs/2103.17118
- depth: M+C
- stage: T, S4, L
- idea: A curb graph is grown vertex by vertex; each step predicts a displacement inside a small crop and a stop flag. Its contribution is the training: DAgger with a restricted run (follow the expert whenever the prediction strays more than 15 px) followed by free runs (always follow the prediction), with the expert label computed *after* the prediction as the nearest ground-truth point that lies ahead of the learner. This is a matching-free label that works for any off-track state.
- formulation: FPN (frozen, pre-trained) feature map F (8 channels) + binary history map g_t; the agent crops d x d around v_t, concatenates v_t and v_{t-1}; coordinate head -> displacement (scaled to [-1, 1]); stop head -> binary. Seeds Q: end vertices of the skeletonised curb mask + local maxima of an initial-vertex heat map (training: GT starts + Gaussian noise, scale not stated). Alg. 1: per image one restricted exploration, then N = 5 free explorations (paper); after each, train on the aggregated dataset D. Code (`config.yml`, `main_env.py`): crop 63 x 63 px (about 9.6 m at 15.2 cm / px), the expert candidate set = GT points inside the crop, with index >= the progress index ii and farther than 15 px from v_now, nearest one by KD-tree; restricted switch thr = 15 px; stop label = 1 when no candidate is left (end of instance) or no progress; circular DAgger buffer 2,048, batch 512, 3 epochs per round, lr 1e-4, README recommends 1 restricted + 1 free round (more free rounds cost time without clear gain), max 120 steps per instance. Loss: L1 between predicted and expert displacement (Eq. 2); stop loss and weights not stated; stop labels are rare (acknowledged).
- input: aerial RGB + infrared (4 ch), 1000 x 1000 px patches; `image_feat` crop + history raster.
- output: curb polyline graph; per step one displacement and a stop flag (no fork handling).
- training: `dagger` (restricted + free, aggregated D) with dynamic labels; no matching; BC with static labels shown to fail on recovery.
- differentiable: no - argmax over the crop, stop thresholding, DAgger loop (but each step is a plain regression).
- sensors_data: NYC planimetric aerial, 29,000 patches (18,000 segmentation pre-training, 10,000 agent training, 1,000 test).
- results: pixel F1 at 1 / 2 / 5 / 10 px: iCurb 0.175 / 0.556 / 0.883 / 0.930, connectivity CC 0.866; RoadTracer 0.101 / 0.334 / 0.640 / 0.736 (CC 0.767); DAGMapper 0.082 / 0.283 / 0.648 / 0.821 (CC 0.832) (Table II, authors' re-implementations). Ablation F1@5 px (Table I; ticks inferred from row order and text, the authors call free exploration critical): no history 0.818, no restricted exploration 0.865, no free exploration 0.750 (CC 0.739), no aggregation (SEARN-like) 0.865 (CC 0.877, higher than the full model's 0.866, not discussed). Does not win at 1 - 2 px.
- compute: 0.89 s / image inference; 22.38 s / image training (one GTX 1080 Ti).
- code: https://github.com/TonyXuQAQ/iCurb (data and the maintained code in https://github.com/TonyXuQAQ/Topo-boundary)
- fit_to_us: reuse_idea - effort S / M. The expert rule and the restricted / free switch can be applied to each tip of the tree during a free unroll; the removal of free exploration is the largest drop seen in this cluster.
- compat_needs: GT polyline with a progress index per instance (our piece graph in travel direction provides it); an expert that works on any state.
- compat_conflicts: single-branch agent (no fork handling, forks are separate instances); label rule assumes one GT chain per instance, forks need one progress index per branch.
- tried_before: none (E94 is teacher-forced; no DAgger / free-rollout variant in the verdict table).
- verified: checked (paper HTML, README, config, environment source).

### CARD GR04 - RoadTracer (fixed-step walker, dynamic oracle labels with map-matching)
- papers: RoadTracer: Automatic Extraction of Road Networks from Aerial Images (Bastani et al., CVPR 2018) https://arxiv.org/abs/1802.03680
- depth: M+C
- stage: S4, T, L
- idea: A CNN walks along roads in fixed 12 m steps: per step an action (walk / stop) and a direction from a discrete angle set; branches are explored one after the other through a stack. The training is the lasting idea: the oracle labels the states that the CNN itself reaches (dynamic labels), because a static oracle trajectory made the CNN drift off the road after small angle errors.
- formulation: Alg. 1: graph G and stack S start at v0; loop: CNN(G, S_top, window) -> (action, alpha); walk: u = S_top + D (cos a, sin a), D = 12 m, push u; stop: pop S. Merge rule: if u is within 3D (36 m) of an existing vertex v whose graph distance from S_top is at least 6D (72 m), add edge (u, v) and do not push (prevents small spurious loops). Outside the box = stop. Heads: 2-way softmax (walk / stop) + sigmoid over a angle bins (a not stated); walk if o_walk >= T (swept in evaluation), direction = argmax. Input window d x d (d not stated) of RGB + the current graph rendered as anti-aliased lines, 0.6 m / px. Dynamic labels: run the search with the CNN; at each step the oracle takes R = angles at S_top along which the GT has unexplored road: o*_stop = 1 if R empty, else o*_walk = 1 and a multi-hot over the nearest angle bins of all of R (all unexplored branches are targets, although only one is followed). Locating S_top in the GT: random walk of w = 10 vertices in G, map-matched to the GT with a Viterbi method; explored GT edges are remembered and avoided. Loss: cross-entropy on the action, MSE on the angle (static section; the dynamic section is not restated). Noise injection on static data: no meaningful gain.
- input: RGB + graph channel window; `image_feat` + rendered history.
- output: graph; per step one action and one direction (multi-hot target).
- training: dynamic labels from CNN-driven states (`dagger`-like, no mixing with expert actions), GT graph + map-matching; no matching of sets.
- differentiable: no - argmax over angles, threshold T, stack control.
- sensors_data: 40 cities in 6 countries, 60 cm / px, OSM ground truth, 25 cities train / 15 test.
- results: junction metric at 5 % average F_error: F_correct 0.58 vs 0.40 for a segmentation baseline (+45 % in the abstract); shortest-path agreement (SP) correct 0.72 vs 0.58 (segmentation) vs 0.21 (DeepRoadMapper, authors' re-implementation); TOPO better than segmentation for recall < 0.43 (Table 1 / figures).
- compute: not stated.
- code: https://github.com/mitroadmaps/roadtracer
- fit_to_us: reuse_idea - effort S. The multi-hot "all unexplored GT branches" target is a matching-free fixed-slot label for b children; map-matching plus an explored-edge memory handles revisits and off-GT states in free rollouts.
- compat_needs: GT graph with map-matching of arbitrary (off-road) states; an angle or sector discretisation.
- compat_conflicts: one branch followed per step at inference (no multi-hypothesis); fixed step; sequential stack.
- tried_before: none.
- verified: checked (ar5iv text, README).

### CARD GR05 - DAGMapper (conv-RNN lane-boundary walker trained through a 5-step unroll with Chamfer loss)
- papers: DAGMapper: Learning to Map by Discovering Lane Topology (Homayounfar et al., ICCV 2019; arXiv 2020) https://arxiv.org/abs/2012.12377
- depth: M
- stage: S4, T, L
- idea: Lane boundaries of highway forks and merges are unrolled as a DAG. Each node holds position, turning direction and a state {continue, fork, terminate}. Three convolutional-RNN heads (direction, position, state) read a bilinearly interpolated, rotated RoI of a BEV feature map around the parent. One of only two growers here that are trained through several of their own steps (the other: Trexplorer Super): argmax replaced by softmax, truncated back-propagation through 5 steps, symmetric Chamfer loss on dense samples of the produced polylines.
- formulation: p(G|D) = prod_i p(theta_i | .) p(x_i | theta_i, .) p(s_i | theta_i, .) (Eq. 3); inference = greedy argmax of theta, then x, then s. Loss L = 1 Chamfer + 100 cosine (direction) + 10 focal (state) + 10 l2 (distance transform, DT); the global network also outputs a thresholded inverse-DT image that is concatenated to the features and skeletonised for initial vertices (skeleton endpoints); boundaries missed by the graph are re-seeded from uncovered DT regions. Training: start from GT initial vertices with small Gaussian noise; GT states are fed to the heads; predicted direction is used unless the parent is > 30 px from the GT boundary (then GT direction); rollouts stop at the image edge, a GT terminate, or 30 steps; BPTT 5 steps; AMSGrad lr 1e-3, 8 GPUs, 12 h. Fork: the parent re-enters the queue to spawn the second child; merge = terminate (no explicit merge target). RoI size and test-time step cap not stated; step size comes from the position softmax inside the RoI.
- input: aggregated lidar-intensity BEV image, 5 cm / px, crops up to 8000 x 1200 px; `bev_raster` features (FPN, C channels not stated).
- output: DAG of boundary points with states.
- training: `teacher_forcing` partly (GT states, GT direction guard) + free rollout of predicted positions with `matching_free` Chamfer loss; softmax relaxation.
- differentiable: partial - arg max replaced by softmax in training only; whether gradients pass the RoI rotation / bilinear sampling into the direction head is not stated; test time is argmax.
- sensors_data: lidar intensity only; 3,000 images from 68 fork / merge areas on one North American highway (70 / 15 / 15 split by area), generalisation set 336 images from another highway.
- results: F1 at 2 / 3 / 5 / 10 px (5 cm px): 76.3 / 88.8 / 94.4 / 96.3; the DT baseline gives 66.9 / 86.4 / 94.9 / 97.0 (better at 5 and 10 px); 89 % of GT boundaries with correct topology; new highway F1@3px 69.0 (Tables 1 - 3). Ablations F1@2px (Table 2): adding the conv-RNN memory 69.5 -> 76.0 (without DT recovery) and 73.1 -> 76.3 (with it, = full model); no state head 70.0 vs 76.3 (full).
- compute: not stated beyond training time; one 1200 x 8000 image per GPU.
- code: none found
- fit_to_us: reuse_idea - effort M: the *training recipe* (short unroll, softmax relaxation of the selection, Chamfer rollout loss, GT guard against runaway states) is what our free-rollout, matching-free tree needs; fork / merge states per node are a simple event head (we have continue / end / fork / merge).
- compat_needs: a feature map the RoI can be sampled from; an initial-vertex source.
- compat_conflicts: the RoI is rotated by the *predicted* direction, so position loss depends on direction error (coupling); single child per step plus a fork queue.
- tried_before: E94 trace network (events continue / end / fork / merge) is close; no BPTT variant tried (E121 sent a planner loss through positions: no gain). E44 / E53 (learned split / merge key points, Bezier connectors) lifted split / merge F1 from 0.34 / 0.38 to 0.55 / 0.61 and 0.71 / 0.64 in system A, in line with DAGMapper's state-head ablation (about -6 F1@2px without it).
- verified: checked (paper HTML, two passes); gradient detail through the RoI marked not stated.

### CARD GR06 - DeH4R (decoupled hybrid: candidate vertices, adjacent-vertex prediction, expansion from dangling ends)
- papers: DeH4R: A Decoupled and Hybrid Method for Road Network Graph Extraction (Gong, Ji; accepted to IEEE T-GRS; arXiv 2508.13669, Aug 2025, v2 Mar 2026) https://arxiv.org/abs/2508.13669
- depth: M+C
- stage: S1, S4, S7
- idea: Splits graph growing into parts that run in parallel on one backbone pass: (1) candidate vertices from keypoint / sampling-point / road masks, (2) for every candidate a decoder predicts up to 10 adjacent vertices (relative coordinates + road / non-road), (3) an initial graph by mutual matching of the predicted neighbours, (4) a graph expander that treats all degree-1 vertices as new seeds, inserts or merges predicted vertices, and repeats for at most 3 rounds.
- formulation: Backbone SAM2 Hiera-B+ -> FPN (1/4, 1/8, 1/16, 1/32). AVP: for each candidate an l x l (l = 3) grid of sample points per scale, bilinear `grid_sample`, scales summed -> f_roi; 3 transformer decoder layers (hidden 128, 8 heads, FFN 256), N_q = 10 learned queries -> p_i in R^4 (2D offset + 2 class logits); overlapping patches are averaged query-wise (queries keep a fixed order). Decoding: candidates v_n, v_c connect if their predicted neighbours point at each other; cost z = s1 + s2 + w (1 - cos th1) + w (1 - cos th2), w = 10 (taken from Sat2Graph). Expansion: valid vertex absent from the graph is inserted with its edge; if within D_merge = 10 px of an existing vertex it is merged; stop when no degree-1 vertex remains or after 3 rounds (Table V: F1 81.14 at 1 round, 81.26 at 6 - 7). Loss L = BCE(three masks) + CE(class) + 10 L2(coordinates), Hungarian matching of predicted to GT adjacent vertices; validity threshold 0.5 (CityScale) / 0.45 (SpaceNet). Training samples = each GT vertex in a patch plus its direct neighbours, N(0,1) noise on adjacent vertices, random 90-degree rotations; no rollout training, exposure bias not discussed.
- input: RGB patch 512 x 512 (CityScale) / 256 x 256 (SpaceNet), 1 m / px; `image_feat` pyramid.
- output: graph; per candidate up to 10 neighbours.
- training: `teacher_forcing`-like (GT vertices as candidates) + `hungarian_set`; SAM2 backbone fine-tuned at 0.1 x lr.
- differentiable: no - NMS for candidates, Hungarian, handcrafted decoding and expansion rules; each module is a plain regression head.
- sensors_data: aerial RGB; CityScale (144 / 9 / 27 tiles), SpaceNet (2,040 / 127 / 382).
- results: CityScale TOPO-F1 / APLS / IoU 81.21 / 72.38 / 55.79 vs RNGDet++ 78.44 / 67.72 / 45.61 and SAM-Road 76.20 / 66.54 / 33.87 (reproduced, Table I); SpaceNet 86.04 / 73.37 vs SAM-Road 82.72 / 71.10. Modes (CityScale): decoding only 81.02 / 71.72, expansion only 79.18 / 68.70, both 81.21 / 72.38 (Table IV; row mapping inferred). ROI 1x1 / 3x3 / 5x5 / 7x7 / 9x9: F1 81.07 / 81.21 / 80.96 / 81.00 / 80.91 (Table VI) - a single sample point almost suffices with a strong backbone. Time 13.34 min vs RNGDet++ 136.0 min (CityScale).
- compute: not stated (parameters); 4 GPUs for training.
- code: https://github.com/7777777FAN/DeH4R
- fit_to_us: reuse_idea - effort M. Parallel expansion of all dangling ends in a few rounds (a level-synchronous wavefront like our ring wavefront), grid-sampled multi-scale ROI features (same operation as our bilinear fan), merge-by-radius with insertion, and the "expansion cannot remove wrong edges" caveat. Backbone and image domain not reusable.
- compat_needs: candidate set; matching of GT neighbours (Hungarian) - replace by sector slots; a merge radius.
- compat_conflicts: Hungarian set loss; hand rules for decoding; expansion seeds only from degree-1 vertices (cannot revisit wrong edges).
- tried_before: E94 - E99 (trace network, similar events) partial; E42 / E45 / E46 / E51 / E56 (learned link heads) within 0.01 of the rules, dropped - DeH4R's initial-graph decoding is a hand rule on predicted neighbours.
- verified: checked (paper HTML twice, README).

### CARD GR07 - Trexplorer / Trexplorer Super (recurrent DETR tree tracker, chained sub-trajectory training)
- papers: Trexplorer: Recurrent DETR for Topologically Correct Tree Centerline Tracking (Naeem et al., MICCAI 2024; no arXiv found) https://research.chalmers.se/en/publication/544069 ; Trexplorer Super: Topologically Correct Centerline Tree Tracking of Tubular Objects in CT Volumes (MICCAI 2025) https://arxiv.org/abs/2507.10881
- depth: M (Super) / A (base paper: MICCAI page with abstract, reviews and rebuttal)
- stage: S4, S5, T
- idea: Medical analogue of RNGDet in 3D: from a root point a DETR decoder tracks a vessel / airway tree level by level (breadth-first); every patch emits the next nodes of all active branches, branches stop at predicted end nodes and split at bifurcations. The base model adapts the TrackFormer idea (queries carried from step to step) and gives each bifurcation a *fixed number of extra queries* (nb = 2 on the synthetic set per the rebuttal; should be at least the maximum bifurcation degree) that attend to each other and are penalised by the Hungarian loss for following the same branch. Super adds chained training over several sub-trajectories, so that the query outputs of one patch feed the next patch during training, and target augmentation; this removes duplicate branches and premature stops.
- formulation: base model (as described in the Super paper): nodes [x, y, z, r], about one voxel from the parent, labelled end / intermediate / bifurcation; each patch tracks 9 steps; new patches start at endpoints with a past-trajectory token (up to 10 previous positions); at a bifurcation the branch halts and 26 object queries are assigned (some track the new branches, the rest are discarded); up to 196 concurrent tokens; Hungarian loss + self-attention among queries against duplicates. Super: Super Trajectory Training = 54-node trajectories cut into six overlapping 10-node sub-trajectories with linked nodes, query outputs passed along the chain; Focal Cross Attention = large high-resolution region extracted, cross-attention restricted to a small focal window; Target Augmentation = bifurcation offsets drawn from a Laplace distribution scaled with the radius + Gaussian noise (sigma 0.025) on other points. Inference reuses the query outputs patch to patch from the root.
- input: 3D CT patches + past-trajectory token; `image_feat` (volumes).
- output: tree graph with radius; per patch the next nodes of all branches.
- training: `hungarian_set` with chained multi-patch training (`free_rollout`-like unrolling through query states; whether gradients pass between sub-trajectories is not stated); about 2 M iterations on 4 A100.
- differentiable: partial - the query state chain is trainable across sub-trajectories (stated as fed forward); Hungarian matching, end / bifurcation thresholds and patch placement are discrete.
- sensors_data: CT; synthetic vessels (368 / 32 / 100), ATM'22 airways (220 / 16 / 60), Parse 2022 arteries (72 / 8 / 20), 0.5 mm.
- results: point-F1 (1.5 voxel, one-to-one) Trexplorer Super 77.83 / 60.45 / 39.46 vs Trexplorer 39.40 / 3.34 / 10.01 on synthetic / ATM'22 / Parse; branch-F1 77.12 / 41.15 / 23.46; Betti-0 and Betti-1 errors 0 for both Trexplorer models (Tables 2 - 3). Ablation: chained trajectories most important (row mapping ambiguous).
- compute: not stated.
- code: https://github.com/RomStriker/Trexplorer-Super ; base model https://github.com/RomStriker/Trexplorer
- fit_to_us: reuse_idea - effort M. Evidence that (a) chained multi-step training of a level-synchronous tree tracker cures duplicates and early stops, (b) reserving a fixed block of queries for a bifurcation (b slots per tip that must separate by loss and self-attention) is workable - the closest published analogue of a fixed-shape b-ary expansion, (c) a metric without linear assignment (1.5 voxel one-to-one) exists.
- compat_needs: recurrent query state; patch-wise features (our cached pyramid replaces the patch); Hungarian loss.
- compat_conflicts: Hungarian matching; 3D medical domain; whether gradient crosses sub-trajectories unclear.
- tried_before: none.
- verified: partly (Super paper read; base paper known from the Super text and listings).

### CARD GR08 - LaneGNN (successor sub-graph per pose, rollout with Dijkstra pruning and aggregation)
- papers: Learning and Aggregating Lane Graphs for Urban Automated Driving (Buechner, Zuern, Todoran, Valada, Burgard, arXiv 2302.06175, Feb 2023; venue CVPR 2023 from memory, not on the arXiv page) https://arxiv.org/abs/2302.06175
- depth: M
- stage: S4, S6, S7, S8
- idea: Instead of one vertex, a GNN predicts the whole *successor lane graph* reachable from a virtual agent pose (inside a 256 x 256 aerial crop, 15 cm / px, oriented by the pose). A rollout moves the pose forward along the best edge, transforms each prediction to world coordinates and merges it into a global graph; many start poses run in parallel and overlapping predictions are aggregated by weighted node updates.
- formulation: Nodes = 400 Halton points filtered by an ego-lane mask; edges = pairs within [d_min, d_max]; GNN with causal message passing (predecessor and successor messages summed separately), L = 6 steps; edge features = geometry + a 4 x 32 x 32 oriented image patch (ResNet-18); outputs node, edge and terminal-node scores. Targets: node score (1 - d)^8 (d = distance to GT), binary edges from minimum-cost paths start -> endpoints; BCE sums; Adam 1e-3, 100 epochs, batch 2; pose noise sigma 5 px / 0.3 rad as the only exposure-bias remedy. Rollout per step: predict -> Dijkstra prune (paths to terminals, scores zeroed) -> world transform -> Laplacian smoothing -> merge (lateral distance a_thresh = 20 px; weights = merge counts) -> step forward one edge; depth of successor tree capped at 10 (loops); exploration depth first with confidence threshold; termination by steps (36), branches (4), branch age (12) or revisited pose; 178 start poses from predicted lane starts; unvalidated splits / merges and parallel branches shorter than 6 edges are removed.
- input: aerial RGB crop + predicted lane / ego-lane masks (PSPNet-152 regressors).
- output: successor sub-graph per step; global lane graph after aggregation.
- training: `teacher_forcing` with pose noise (single-step supervision, rollout untrained); `matching_free` BCE on node / edge scores.
- differentiable: no for the rollout (Dijkstra, argmax pose choice, merge rules); inside one step yes (GNN).
- sensors_data: aerial (Urban Lane Graph dataset derived from Argoverse 2 maps, six cities); no vehicle sensors.
- results: Successor-LGP: TOPO P / R 0.600 / 0.699, SDA20 0.227, IoU 0.347 (skeleton baseline 0.597 / 0.613, SDA20 0.020); Full-LGP (Miami): TOPO P / R 0.481 / 0.670 and GEO P / R 0.649 / 0.689 vs naive aggregation 0.366 / 0.654 and 0.523 / 0.727 (Tables 2, 3). A lower a_thresh = 10 raises recall, lowers precision and Graph IoU by about 3 points.
- compute: about 1 s per rollout step (RTX 3090), 7 min per 10^6 m^2 tile with parallel runs.
- code: dataset and code announced at https://urbanlanegraph.cs.uni-freiburg.de (page not reachable during this session)
- fit_to_us: reuse_idea - effort M. A step that returns a *sub-graph with terminal scores* rather than one vertex is the same idea as our per-level children + rescoring; the aggregation of overlapping predictions with weights and lateral-only updates is a ready recipe for tree -> DAG and for time fusion.
- compat_needs: start poses; candidate node / edge proposals; a merge threshold.
- compat_conflicts: Dijkstra pruning and rule-based aggregation are discrete; single-step supervision.
- tried_before: E42 - E56 (learned link / path repair within 0.01 of rules, dropped) is the closest; temporal fusion E41 small.
- verified: checked (full HTML text including supplement).

### CARD GR09 - Sat2Graph (graph-tensor encoding: a graph as a dense multi-channel raster)
- papers: Sat2Graph: Road Graph Extraction through Graph-Tensor Encoding (He et al., ECCV 2020; arXiv 2007.09547) https://arxiv.org/abs/2007.09547
- depth: M+C
- stage: S6, S7, L (dense alternative to growth)
- idea: One forward pass of a plain segmentation network predicts a tensor that encodes the graph; no agent, no recurrence. Each cell stores a vertex score and, for each of D_max = 6 angular sectors, an edge score and the offset to the neighbour vertex in that sector. Decoding = local maxima + sector offsets + nearest-vertex matching. The sector slots give a fixed number of outgoing-edge slots per vertex without any matching.
- formulation: tensor (W/lambda) x (H/lambda) x (1 + 3 D_max), lambda = 1 m -> 19 channels. Encode: segments are interpolated to < d = 20 m (15 - 25 m best); stacked roads shifted so overlapping vertices are > 5 m apart; each edge encoded once from one endpoint, at most one edge per 60 degree sector. Loss: sum over cells of CE(p_v) + [vertex present] * sum_i (CE(p_ei) + L2(offsets)); edge terms only where GT has a vertex (long segments are not label-consistent). Decode: vertices = maxima above p_thr; for vertex v and sector i choose the nearest candidate u by d = ||(v + offset_i) - u|| + w cos_dist(offset_i, u - v), w = 100 (F1 75.87 - 76.72 for w from 1 to 150), reject beyond 15 m. Backbone DLA, receptive field 352 px, 300 k iterations, lr 1e-3 halved every 50 k, one V100.
- input: aerial RGB, 1 m / px; `image_feat`.
- output: `graph(nodes+edges)` from a dense tensor; undirected, direction as attribute.
- training: `matching_free` dense supervision; no rollout, no exposure bias.
- differentiable: partial - the network is differentiable; decoding (maxima, nearest matching) is not.
- sensors_data: 20 US cities (OSM, 720 km^2, 144 / 27 / 9 tiles of 2 km), SpaceNet 2,549 tiles.
- results: City-scale TOPO-F1 / APLS 76.26 / 63.14 (DLA) vs RoadTracer 66.16 / 57.29; SpaceNet 80.97 / 64.43; the encoding lifts a U-Net from 70.36 to 76.40 TOPO-F1 (Table 1 and text). Stacked roads: precision 83.11 %, recall 49.81 %. Failure modes: stacked roads in complex scenes, missing small roads under wide highways; the RNGDet text adds (second-hand) that it cannot separate segments with a small intersection angle and is weaker on curves - relevant to fork resolution.
- compute: not stated; training on one V100.
- code: https://github.com/songtaohe/Sat2Graph
- fit_to_us: reuse_idea - effort S / M: (a) angular-sector slots as a *matching-free* way to give each tip b children (our fan has 9 bearings; 2 children per tip = 2 of the sectors), (b) the "edge terms only at vertex cells" mask, (c) baseline for a "spray + snap + link" dense route.
- compat_needs: dense head with extra channels (vertex + 6 x 3); decoding rules.
- compat_conflicts: non-iterative, so it does not give hypotheses with verification; undirected.
- tried_before: E31 / E32 (dense grid + tracer won over a matched sparse decoder); dense heat + direction head already exists (2.7 M U-Net).
- verified: checked (paper HTML, README).

### CARD GR10 - SAM-Road / SAM-Road++ / LineGraph2Road (vertex mask NMS + pairwise edge classifier with interior samples)
- papers: Segment Anything Model for Road Network Graph Extraction (Hetang et al., CVPRW 2024) https://arxiv.org/abs/2403.16051 ; Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method (SAM-Road++, Yin et al., arXiv 2411.16733; CVF CVPR 2025 listing seen in a search result) https://arxiv.org/abs/2411.16733 ; LineGraph2Road: Structural Graph Reasoning on Line Graphs for Road Network Extraction (2026) https://arxiv.org/abs/2602.23290
- depth: M+C (SAM-Road README); M (the other two)
- stage: S6, S3, S2, T
- idea: Not growers but the strongest "link" counter-design on the same task. Vertices come from road / intersection masks by NMS; every vertex looks at its 16 nearest neighbours within 64 px; a small transformer scores each pair from features sampled at both ends. SAM-Road++ adds two things that matter here: (1) *node-guided resampling* - training pairs use the vertices that NMS would choose at inference (distribution match, an exposure-bias fix for link heads); (2) *extended-line samples* - mask values sampled along the extension of both ends (n = 15, 8 px) and along the segment (m = 20) are fed to the classifier (occlusion, ray-style verification). LineGraph2Road builds edge features from the two endpoints + 4 interior bilinear samples and runs a graph transformer on the line graph (edges as nodes).
- formulation: SAM-Road: pair input = bilinear features at source and target + relative offset -> 3 self-attention layers -> sigmoid; positive = immediate BFS neighbours; losses BCE (masks and edges, equal weights); teacher forcing on GT graphs subdivided and NMS'd to mimic inference. Inference by 16 x 16 sliding windows with averaging. LineGraph2Road: Coupled NMS radii 8 / 16 px, candidate pairs within 64 px, 3-layer graph transformer (4 heads, hidden 128), loss BCE masks + 0.1 BCE edges.
- input: aerial RGB 512 x 512 (City-scale) or 256 x 256 (SpaceNet), 1 m / px; `image_feat` pyramid (SAM encoder).
- output: `graph(nodes+edges)`; edge probabilities.
- training: `teacher_forcing` / `matching_free` BCE; one RTX 4090.
- differentiable: partial - edge scoring is differentiable; NMS for vertices and thresholds are not.
- sensors_data: aerial; CityScale (27 test tiles) and SpaceNet; SAM-Road++ adds a global-scale set.
- results: SAM-Road CityScale 77.23 TOPO-F1 / 68.37 APLS, 4.6 min vs RNGDet++ 231.0 min (Table 2 of SAM-Road). SAM-Road++ ablation (rows match the SpaceNet numbers; dataset not stated in the table; APLS / F1): base 71.64 / 80.52; resampling only 71.90 / 81.77; extended-line only 73.22 / 80.89; both 73.44 / 81.57 (Table 4). LineGraph2Road CityScale 83.77 / 70.40, SpaceNet 84.24 / 73.94 (best variants). SAM-Road ablations: no transformer in the topology decoder 73.75 / 59.39, no intersection mask 71.94 / 64.62, A* instead of learned edges 69.21 / 63.32.
- compute: edge decoder small; a 4 x 4 window stride cuts time to 2.9 min with F1 77.00.
- code: https://github.com/htcr/sam_road ; https://github.com/earth-insights/samroadplus ; https://github.com/wzzzzzzy/LineGraph2Road
- fit_to_us: reuse_idea + baseline - effort S: interior / extension samples along a candidate segment as classifier input is the clearest published support for verifying long rays by intermediate samples; node-guided resampling is the cheap exposure-bias fix for any pairwise head.
- compat_needs: candidate vertex set (NMS), a pairwise head, a distance cap (64 px).
- compat_conflicts: replaces growth by linking (so it is the alternative family, not a part of the tree); SAM encoder not available (frozen BEV pyramid instead).
- tried_before: E42 / E45 / E46 / E51 / E56 (learned link heads, TopoLogic-style scoring, path repair) were within 0.01 of the rules and dropped; E113 / E116 (junction / piece verifier on the learned graph) lifted split / merge F1 from 0.385 / 0.428 to 0.437 / 0.504 - a verifier trained on the trace network's own proposals, which is what node-guided resampling would formalise.
- verified: checked (three paper HTML pages, README).

### CARD GR11 - Anagnostidis et al. (edge-sequence agent trained with MuZero-style RL, tree search at inference)
- papers: Mastering Spatial Graph Prediction of Road Networks (Anagnostidis, Lucchi, Hofmann, ICCV 2023; arXiv 2210.00828, Oct 2022) https://arxiv.org/abs/2210.00828
- depth: M
- stage: S4 (search), T
- idea: Given fixed candidate key points (from a segmentation detector), an agent adds edges by pointing at key points (two pointers = one edge, or EOS). Because the evaluation metrics (APLS, path / junction / sub-graph F1) are non-smooth, the agent is fine-tuned with a learned-model RL scheme (MuZero-style) whose reward is the *change in a weighted metric*; at inference a Monte-Carlo tree search over learned dynamics is run. The search tree is a DAG because the graph embedding is permutation invariant.
- formulation: Policy = pointer network over N_V key points + EOS (dot product); representation = autoregressive transformer (two stacks of 16 self-attention layers, ResNet-18 / 50 features sampled at the key points); dynamics = LSTM predicting next hidden state and reward (up to 90 x fewer FLOPs than the decoder); value with a 601-bin support. Reward = delta of 0.35 APLS + 0.25 path-F1 + 0.25 junction-F1 + 0.15 sub-graph-F1; early training targets the graph from the post-processed segmentation mask. Pre-training: cross-entropy on a fixed autoregressive order (the ARM), RL with lower lr on pre-trained modules; 5-step unroll of the dynamics, Dirichlet root noise, re-analysis of old episodes, Ray workers. Search: PUCT-style selection with learned prior, simulation count at evaluation not stated (10 only in an illustration).
- input: aerial image (300 x 300 resized; SpaceNet crops 400 x 400 with 100 px overlap) + key-point set; `image_feat` + `point_set`.
- output: sequence of edges (pairs of key-point indices).
- training: `next_token` pre-training then `rl` (reward from GT graph metrics); no matching.
- differentiable: no - sampling of actions, reward is a non-smooth metric; the learned dynamics model is differentiable internally.
- sensors_data: aerial; SpaceNet, DeepGlobe, a synthetic city set (CityEngine).
- results: GT key points, SpaceNet APLS: random 0.008, classifier 0.310, GCN 0.488, ARM 0.743, ours 0.778; fuzzy key points SpaceNet APLS 0.6523 vs best baseline 0.6422, path / junction / sub-graph F1 0.7431 / 0.7792 / 0.7922 (Tables 1 - 2). Order dependence of the supervised sequence model: perplexity 0.733 bits / edge in its training order but 30.84 in a random order, the RL model 4.636 / 4.565 (order-invariant graph embedding). Ablation (garbled in the fetched text): removing AR pre-training or visual key-point features costs on the order of 12 - 15 points, removing the tree search about 1 - 2, cross-attention to the image is not significant.
- compute: trainable on a single GPU from the ARM start (stated); total compute not stated.
- code: none found
- fit_to_us: reuse_idea - effort L (RL) / S (idea): the only search over graph-growth hypotheses in this cluster; its gain is the smallest effect of all components, which argues that for us a beam only pays if the per-step scorer is already good. The reward shaping by metric deltas is an alternative to a matching-free geometric loss.
- compat_needs: fixed key-point candidate set (our heat-map peaks); GT graph for rewards.
- compat_conflicts: RL / sampling excludes gradients through selections; cannot add or delete key points.
- tried_before: none.
- verified: partly (ablation and Table 2 columns garbled; ICCV PDF unreadable).

### CARD GR12 - DOGE (differentiable Bezier graph optimisation with topology adaptation)
- papers: DOGE: Differentiable Bezier Graph Optimization for Road Network Extraction (arXiv 2511.19850, 2025) https://arxiv.org/abs/2511.19850
- depth: M
- stage: S5, S7, L
- idea: No network is trained for the graph. A fine-tuned SAM2 gives a road mask; a graph of nodes and cubic-Bezier edges is rendered differentiably (DiffVG) and its positions, widths and curve parameters are optimised by Adam to cover the mask, with interleaved discrete operators that add, merge, split and prune nodes and edges. Matching-free, label-free at the graph level; test-time optimisation.
- formulation: loss = 1.0 cover (L2 between union of rendered edges and mask) + 0.3 overlap (sum of renders above 1) + 0.012 G1 continuity at degree-2 nodes + 6e-3 offset + 6e-3 spacing (control points near 1/3, 2/3); 300 outer iterations, rendering 512 x 512. TopoAdapt: add edges where mask is uncovered (random small edges), merge nodes closer than 4 m (midpoint), snap node onto a nearby edge and split it (T-junction), collinear merge of degree-2 nodes with refit, prune short or thin edges.
- input: road probability mask; `bev_raster`.
- output: `bezier` graph.
- training: none for the graph (per-image optimisation, `matching_free`).
- differentiable: yes inside the optimisation (rendering); the topology operators are discrete.
- sensors_data: aerial; CityScale and SpaceNet.
- results: SpaceNet TOPO-F1 / APLS 84.58 / 73.48 (SAM-Road++ 81.57 / 73.44), CityScale 80.59 / 70.24 (80.01 / 68.34); ablation SpaceNet: no TopoAdapt 82.80 / 69.45, no overlap loss 83.38 / 65.95, L2 only 70.94 / 37.10 (Tables 1, 3). Compact: 14.0 edges / km vs 52.2 for the GT.
- compute: one RTX 4090; run time not stated.
- code: none found (announced)
- fit_to_us: reuse_idea - effort M: node merge / T-junction split / collinear merge are a ready operator set for tree -> DAG (S7) and a matching-free coverage loss for E120-type refinement; DiffVG is an external CUDA dependency (a torch distance-field renderer would replace it).
- compat_needs: a mask or heat map as target; spatial grid for neighbour queries.
- compat_conflicts: test-time optimisation (300 iterations per image) is far from a feed-forward real-time block.
- tried_before: E120 (position refinement with point-to-polyline loss both ways, no matching): +0.012 F1@1, kept optional - same family of objective.
- verified: checked (paper HTML).

### CARD GR13 - SeqGrowGraph (lane graph as a chain of adjacency-matrix expansions)
- papers: SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions (Xie et al., ICCV 2025; arXiv 2507.04822) https://arxiv.org/abs/2507.04822
- depth: M
- stage: S4, S6 (token sequence)
- idea: The directed lane graph grows by one *junction vertex* per step; at step n the adjacency matrix expands from n x n to (n+1) x (n+1), so a step emits the new vertex position plus its incoming and outgoing links with a quadratic Bezier control point for each connecting centerline. Vertices are visited in DFS order; a 6-layer transformer decoder generates the token sequence from LSS BEV features. Handles loops and two-way single-lane roads because the graph is directed rather than a DAG.
- formulation: S_n = S_{n-1} + (v_n + F_n + T_n) with F_n / T_n the incoming / outgoing centerlines to earlier vertices; tokens: coordinates 0 - 200 (0.5 m bins), node indices 200 - 350, Bezier control coordinates 350 - 570, special tokens to / SEP / EOS / BOS / N/A = 571 - 575. Loss = token NLL, weight 2 on point-coordinate tokens, 1 elsewhere (weights 1 / 2 / 3 / 4 give L-F 54.7 / 56.4 / 55.3 / 55.9). Ordering ablation (landmark-F / reachability-F): DFS 56.4 / 67.8, BFS 55.1 / 67.7, coordinate 53.7 / 64.4 (junctions sorted by x, y; key order not stated), centre-out 52.4 / 63.7 (junctions by increasing distance from the graph centre); random order not tested. Not stated (checked in three passes): DFS root, tie-breaking, how generation stops, decoding strategy, optimiser, learning rate, image size, BEV grid size, augmentation, post-processing, parameters, speed, limitations.
- input: 6 (nuScenes) or 7 (AV2) cameras; LSS BEV encoder with ResNet-50 pre-trained on centerline segmentation; BEV x in [-48, 48] m, y in [-30, 30] m (nuScenes). `image_feat` -> `token_sequence`.
- output: `token_sequence` -> adjacency + Bezier centerlines.
- training: `next_token` with GT prefix (teacher forcing implied, not named), 400 epochs nuScenes / 50 AV2 on 4 A100.
- differentiable: no - tokenisation (0.5 m bins), argmax / sampling at decoding.
- sensors_data: cameras; nuScenes, Argoverse 2.
- results: nuScenes default split landmark P / R / F 63.6 / 50.8 / 56.4, reachability 75.5 / 61.4 / 67.8 vs RNTR 57.1 / 42.7 / 48.9 and 63.7 / 45.2 / 52.8, LaneGraph2Seq 46.9 / 43.7 / 45.2, TopoNet 52.5 / 47.1 / 49.6 (reachability-R 10.8); Argoverse 2 landmark-F 56.4, reachability-F 55.7 (Tables 1 - 2). A re-segmentation study splits GT centerlines at junctions and optionally every 20 / 30 / 40 m (junction-level landmark-F 59.5 - 60.4, Table 4; different GT, not comparable with Table 1; which version the main table uses is not stated).
- compute: not stated.
- code: none found
- fit_to_us: not_suitable as a block (tokenised autoregression, junction-only vertices); reuse_idea for metrics (landmark / reachability) and the junction-level graph definition.
- compat_needs: a sequence order (DFS); coordinate quantisation.
- compat_conflicts: no gradient through tokens; vertices only at junctions (we need dense vertices).
- tried_before: none.
- verified: partly (training details absent in the paper text).

### CARD GR14 - RNTR (RoadNet sequences: autoregressive, semi-autoregressive and non-autoregressive decoding)
- papers: Translating Images to Road Network: A Sequence-to-Sequence Perspective (Lu et al., ICCV 2023; TPAMI 2025; arXiv 2402.08207) https://arxiv.org/abs/2402.08207
- depth: M+C
- stage: S4, S6, E (landmark / reachability metrics), B (TIT)
- idea: The road network (a DAG of landmarks: crossroads, stop lines, fork and merge points) is serialised as an integer sequence; three decoders: AR (token by token), SAR (a DETR key-point decoder first, then sub-trees rooted at key points decoded in parallel, 6 x 18 tokens), NAR (SAR fine-tuned with masked-LM at 90 % mask, 3 mask-predict iterations). Also introduces landmark and reachability metrics and "topology-inherited training" (distilling camera BEV towards lidar-based features).
- formulation: Graph -> forest: each merge point gets clones of its parents except the first; DFS order; six integers per vertex-edge pair: x, y, category (Ancestor / Lineal / Offshoot / Clone), parent index, two Bezier control offsets (+10); vocabulary 576 (coordinates 0 - 199, category 200 - 249, parent 250 - 349, control 350 - 569, special 570 - 573). AR loss = class-weighted MLE; SAR key-point loss = Hungarian (class + L1); NAR = masked-token likelihood. Sequence augmentation hurt; sequence noise padding (noise vertices with a noise class) is kept because AR models stop early. DFS tie-break: random or coordinate-based (numbers not stated). Training 300 epochs, lr 2e-4, flip / rotation / scaling of BEV features.
- input: LSS BEV (128 x 352 images, BEV x in [-48, 48], y in [-32, 32] m), ResNet-50 or VoVNetV2; pre-trained BEV encoder frozen; `image_feat` -> `token_sequence`.
- output: integer sequence -> `graph`.
- training: `next_token` (AR), `hungarian_set` (SAR key points), masked-LM (NAR).
- differentiable: no - tokenisation, argmax / mask-predict, Hungarian key points.
- sensors_data: 6 cameras; nuScenes (default and a non-overlapping split), Argoverse; optional SD map (SD-RNTR).
- results: nuScenes multi-camera default split landmark-F / reachability-F: AR 54.1 / 61.3 at 0.1 FPS, SAR 56.0 / 65.7 at 0.6 FPS (6 x), NAR 55.1 / 65.1 at 5.5 FPS (47 x); non-overlapping split AR 36.5 / 40.9, SAR 36.9 / 43.0; with TIT AR 39.1 / 49.8 (Tables IV - V). Mask ratio 90 %: 1 / 3 / 6 iterations 59.0 / 60.2 / 60.3 landmark-F (VoVNetV2 row). Removing intra-sequence attention drops reachability-F from 66.0 to 56.3.
- compute: AR 37.25 M parameters, NAR 35.90 M; NAR about 6 GB at batch 2.
- code: https://github.com/fudan-zvg/RoadNet
- fit_to_us: reuse_idea - effort S: SAR = a *fixed-shape set of sub-trees rooted at key points* decoded in parallel with intra / inter-sequence attention is the closest published structure to our fixed-shape tree; the metrics are reusable.
- compat_needs: key-point decoder (matching); sequence order; BEV encoder.
- compat_conflicts: tokens and Hungarian key points.
- tried_before: none.
- verified: checked (paper HTML up to appendix; README).

### CARD GR15 - LaneGraph2Seq (vertex-centric DFS plus edge-partition sequence, classifier-free guidance)
- papers: LaneGraph2Seq: Lane Topology Extraction with Language Model via Vertex-Edge Encoding and Connectivity Enhancement (Peng et al., AAAI 2024; arXiv 2401.17609) https://arxiv.org/abs/2401.17609
- depth: M+C
- stage: S4, S6
- idea: A lane graph is written as a vertex sequence (junction points in DFS order) followed by an edge sequence in which each edge is [child index, Bezier midpoint x, y] and a Split token closes the children of each parent. A transformer decoder generates it from BEV features; at inference classifier-free guidance (alpha_c = 4) contrasts a vertex-conditioned and an unconditioned stream to improve connectivity.
- formulation: sequence padded to 200 vertex + 400 edge tokens; class-weighted log-likelihood with lowered weights of EOV / Split / EOE (against early termination; values not stated); <N/A> excluded from the loss; AdamW lr 2e-4, 300 epochs, 8 V100; 6 decoder layers (8 layers slightly better in Table 3: M-F 74.0 vs 73.3). Ordering ablation M-F / Detect / C-F (VoVNetV2, nuScenes): DFS 73.3 / 68.7 / 67.6, BFS 72.1 / 67.2 / 67.2, coordinate 69.8 / 61.7 / 61.2, random 70.2 / 61.3 / 62.1. Logits: uncond + alpha_c (cond - uncond). Nucleus sampling mentioned, parameters not stated.
- input: LSS BEV (ResNet-50 or VoVNetV2, pre-trained on centerline segmentation), x in [-48, 48] m, y in [-32, 32] m; `image_feat` -> `token_sequence`.
- output: `token_sequence` -> DAG.
- training: `next_token`, teacher forcing implied; BEV augmentation.
- differentiable: no - tokenisation, argmax / sampling.
- sensors_data: 6 / 7 cameras; nuScenes, Argoverse 2.
- results: nuScenes M-P / M-R / M-F 64.6 / 63.7 / 64.1, connectivity C-F 63.2 (R-50); VoVNetV2 73.3 / 67.6; AV2 61.7 / 60.1 (Table 1). Failure: premature termination before all edges are emitted.
- compute: not stated.
- code: https://github.com/fudan-zvg/RoadNet
- fit_to_us: not_suitable as a block; reuse_idea: ordering result (DFS best), reduced weights of terminating tokens, the observation that early termination is the dominant AR failure.
- compat_needs: sequence order; quantised coordinates.
- compat_conflicts: tokens; sampling.
- tried_before: none.
- verified: checked.

### CARD GR16 - TopoGPT (map-only pre-trained autoregressive lane prior plus flow-matching perception adapter)
- papers: Generative Lane Topology Reasoning via Autoregressive Model with Geometry Prior (Fu et al., ECCV 2026; arXiv 2606.31814) https://arxiv.org/abs/2606.31814
- depth: M+C
- stage: S4, T, B
- idea: Two phases. (1) Pre-train a decoder-only transformer on 3.3 M map scenes (no sensors) with next-token prediction over tokenised lane graphs, conditioned on a rasterised copy of the graph (scene tokens), augmented by flips / translations / rotations and by *condition degradation* (random lanes removed from the raster). (2) A perception adapter turns multi-view BEV features into scene-token-like features by deterministic flow matching (Euler steps; gradients flow through the unrolled steps), then the decoder is fine-tuned with LoRA.
- formulation: each centerline = K sampled points fit by least squares with a cubic Bezier (4 control points, ends fixed); coordinates quantised to 64 bins per axis over +-52 m (x) and +-32 m (y); token = Q(y) * 64 + Q(x), 4 tokens per lane; lanes sorted by bounding box [x_min, y_min, x_max, y_max]; <SOS>, <SEP>, <EOS>; no connectivity tokens - topology by endpoint distance < 0.5 m at inference; lanes split only at merges and forks. Decoder: LlamaGen-like with M-RoPE; sizes 24.7 M / 45.9 M / 91.3 M. Loss: CE (teacher forcing) + flow MSE + alignment MSE (weights not stated); greedy decoding; 16 epochs pre-training, 24 epochs fine-tuning, 8 H20 GPUs.
- input: BEV features from 6 - 7 cameras (ResNet-50 + Simple-BEV style lifting); `image_feat` / `map_raster` scene condition.
- output: `token_sequence` of Bezier lanes.
- training: `next_token` + `teacher_forcing`; pre-train / fine-tune; flow-matching alignment.
- differentiable: partial - adapter path yes (unrolled ODE steps); token generation no.
- sensors_data: cameras; OpenLane-V2 subsets A and B; pre-training maps from large motion datasets (2.07 M + 1.01 M + 0.25 M scenes).
- results: OpenLane-V2 subset A M-P / M-R / M-T 45.6 / 42.6 / 24.4, APLS 28.7 vs TopoPoint 42.7 / 34.0 / 16.6, 17.4; subset B 43.1 / 39.2 / 23.4, APLS 29.5 (Table 1). From scratch (no pre-training): 7.4 / 7.0 / 0.8; frozen / LoRA / full fine-tune 44.3 / 45.6 / 45.3 M-P (Tables 2 - 3). Flow steps 0 / 3 / 6 / 9: M-P 43.1 / 44.3 / 45.6 / 45.4.
- compute: 24.7 - 91.3 M parameters; latency not stated (sequential decoding acknowledged as a limit); early errors propagate.
- code: https://github.com/buaa-colalab/TopoGPT
- fit_to_us: reuse_idea - effort L: a map-only geometric prior (condition degradation = train on partial graphs) as a rescoring prior for children; not the decoder.
- compat_needs: large map corpus (not in our environment; no downloads without asking); BEV features for the adapter.
- compat_conflicts: tokenisation; sequential decoding; no gradient to geometry.
- tried_before: none.
- verified: checked (paper HTML, README).

### CARD GR17 - Topo2Seq (training-only autoregressive topology decoder as regulariser)
- papers: Topo2Seq: Enhanced Topology Reasoning via Topology Sequence Learning (Yang et al., AAAI 2025; arXiv 2502.08974) https://arxiv.org/abs/2502.08974
- depth: M
- stage: L, T
- idea: A DETR lane-segment decoder (LaneSegNet style, 200 queries) and a Pix2Seq-style autoregressive *topology sequence decoder* share the BEV features; only the first is used at inference. The sequence decoder receives the predicted key points as unordered prompts and must reconstruct the graph, which forces the BEV features to carry topology. Randomising the prompt order matters.
- formulation: sequences as in RoadNet (six integers per edge: x, y, category, connection index, 2 Bezier controls; DFS from the front-right point; noise edges marked <ncls>); prompts 201 + edges 601 = 802 tokens; L = 0.025 L1 + 1.5 cls + 3.0 seg + 0.1 lt + 5.0 top + 1.0 seq. Ablation TOPls-ls: ordered GT prompts 25.1, randomised GT prompts 27.8, predicted key points early in training 26.5, full setup 29.9. BEVFormer encoder, BEV x in [-50, 50] m, y in [-25, 25] m on a 200 x 100 grid; 14.7 FPS inference.
- input: BEV features; `queries`.
- output: lane segments + adjacency (inference); sequence only in training.
- training: `hungarian_set` (main decoder) + `next_token` auxiliary.
- differentiable: yes for the auxiliary path (it only adds a loss); the sequence is never sampled at inference.
- sensors_data: 7 cameras, OpenLane-V2 subset A (subset B not stated).
- results: lane segment 48 epochs mAP 37.7 / TOPlsls 29.9 vs LaneSegNet 36.4 / 27.3; centerline TOPll 30.0 vs 27.3 (Table, subset A).
- compute: no inference cost.
- code: none found
- fit_to_us: reuse_idea - effort M: an auxiliary order-agnostic path-reconstruction loss from the cached pyramid costs nothing at inference; our tree could receive a training-only DFS-sequence head on its node tokens.
- compat_needs: key-point prompts from the main decoder.
- compat_conflicts: Hungarian main decoder; token loss needs quantisation.
- tried_before: none.
- verified: checked (paper HTML).

### CARD GR18 - VectorMapNet (detector plus autoregressive polyline generator; exposure-bias fine-tuning)
- papers: VectorMapNet: End-to-end Vectorized HD Map Learning (Liu et al., arXiv 2206.08920; ICML 2023 from memory) https://arxiv.org/abs/2206.08920
- depth: M
- stage: S4, T
- idea: A DETR-style detector places map elements as keypoint sets (bbox corners by default); a second stage generates each polyline's vertices autoregressively as discrete coordinate tokens conditioned on keypoints, class and BEV features. Teacher forcing uses GT keypoints; an optional two-stage fine-tuning feeds predicted keypoints to remove the train-test mismatch.
- formulation: 100 element queries (6 layers, hidden 256, deformable cross-attention); generator: next-token NLL on a 200 x 100 grid of 0.3 m cells, EOS token; detector loss with Hungarian matching; Gaussian noise on generator input vertices with probability 0.3; 110 epochs, batch 32, AdamW. Discrete vs continuous (Gaussian mixture) vertex modelling: Chamfer mAP 32.5 vs 26.5, Frechet mAP 50.4 vs 41.9 (Table 6; the appendix quotes other numbers).
- input: BEV features (camera, lidar or fusion); output: polylines.
- training: `hungarian_set` detector + `next_token` generator; `teacher_forcing` with a fine-tune on predicted inputs.
- differentiable: no - tokenisation, argmax.
- sensors_data: nuScenes, Argoverse 2; BEV about 60 x 30 m at 0.3 m.
- results: nuScenes mAP camera 40.9 -> 46.0 with the predicted-keypoint fine-tune; fusion 45.2 -> 53.7; HDMapNet fusion 31.0 (best baseline); no temporal consistency, hallucination in occluded areas (stated).
- compute: latency not stated.
- code: project page https://tsinghua-mars-lab.github.io/vectormapnet (repo not checked)
- fit_to_us: reuse_idea - effort S: size of the exposure-bias gain by retraining the second stage on the first stage's own output (+5.1 / +8.5 mAP) is a data point for our cascade.
- compat_needs: detector keypoints; quantisation.
- compat_conflicts: tokens; Hungarian.
- tried_before: none.
- verified: checked (paper HTML).

### CARD GR19 - Iterative deep learning for road topology (best-first tracing with a local heat map)
- papers: Iterative Deep Learning for Road Topology Extraction (Ventura, Pont-Tuset, Caelles, Maninis, Van Gool; arXiv 1808.09814, Aug 2018; no venue on the page) https://arxiv.org/abs/1808.09814
- depth: M
- stage: S4, S1
- idea: A global segmentation network gives the start and a cost map; a local hourglass network sees a 64 x 64 patch centred on the query point and outputs a heat map of *border locations connected to the centre*, which makes several branches per patch natural. Candidates above a confidence threshold go into a bag; the most confident one is visited next (best-first), joined to its predecessor by Dijkstra over the segmentation probability map.
- formulation: training on 130 random 64 x 64 patches per image centred on road pixels; targets Gaussian peaks at skeleton intersections with the patch border square; visited set blocks re-expansion; near-visited points are discarded or connected but not expanded; when the bag is empty a new start is chosen if far from explored areas and confident.
- input: 64 x 64 patch; output: heat map of connected border points; `bev_raster` / `scores`.
- training: `matching_free` heat-map regression, `teacher_forcing` (patches from GT).
- differentiable: no (bag / Dijkstra), but each step is a dense heat map (no set matching).
- sensors_data: Massachusetts roads (1500 x 1500 tiles), DRIVE retina.
- results: Massachusetts F^C 74.4 / F^R 81.6 vs best VGG threshold 49.3 / 72.4; DRIVE F^C 85.5 vs 79.8; patch-level F 84.5 on 700 validation patches.
- compute: not stated. code: none found
- fit_to_us: reuse_idea - effort S: best-first order by confidence (instead of FIFO / DFS) and heat-map-valued branch targets (no assignment).
- compat_needs: confidence queue.
- compat_conflicts: sequential.
- tried_before: heat-map seeds in E94.
- verified: checked (paper HTML).

### CARD GR20 - VecRoad (point-based iterative exploration with flexible step size)
- papers: VecRoad: Point-Based Iterative Graph Exploration for Road Graphs Extraction (Tan et al., CVPR 2020) https://mmcheng.net/vecroad/
- depth: A+C
- stage: S4
- idea: Each next move is one *point* that encodes direction and step length together (variable step), guided by jointly predicted road-segmentation and junction cues so that the walk stays aligned; segmentation network and exploration are trained together with summed equal-weight losses (author comment on the project page).
- formulation: not read (method section PDF-only). Second-hand from the RNGDet text: next-step vertices are extracted as local peaks of a predicted heat map, step size is flexible, the pipeline is two-stage (not end to end), and it may fail for vertices that lie very close together; APLS 65.69 in RNGDet's Table I (RoadTracer set).
- input/output/training/differentiable: not read; training: segmentation network and exploration trained together with summed, equally weighted losses (author comment on the project page); dynamic-label or DAgger use not confirmed.
- sensors_data: aerial, RoadTracer-set (pixel F and junction F compared against RoadTracer).
- results: none seen from the paper itself (APLS 65.69 second-hand, see above).
- compute: not stated.
- code: https://github.com/tansor/VecRoad (PyTorch >= 0.4.1, `infer.py`, `train.py`)
- fit_to_us: reuse_idea - effort S - variable step sizes as a *point* is the same as our step class + residual; details unread.
- compat_needs / compat_conflicts: unknown.
- tried_before: step class {2, 4, 6} m in E94.
- verified: partly (project page, README, abstract-level text).

### CARD GR21 - Image-conditioned graph generation (Graph Generative Transformer, StreetMover distance)
- papers: Image-Conditioned Graph Generation for Road Network Extraction (Belli, Kipf, arXiv 1910.14388, 2019; no venue seen) https://arxiv.org/abs/1910.14388
- depth: M
- stage: S4, E, L
- idea: A self-attentive decoder emits the graph node by node in BFS order: each step outputs the node coordinates (tanh) and a soft adjacency vector over the M frontier slots (sigmoid), conditioned by attention on a 900-d image vector. Trained by teacher forcing; evaluated with a StreetMover distance - a Sinkhorn approximation of the Wasserstein distance between point clouds sampled on the graph edges - which needs no matching and no post-processing.
- formulation: 12 blocks, d = 256, 8 heads; loss 0.5 BCE(adjacency) + 0.5 MSE(coordinates); BFS from the top-left node with children ordered clockwise relative to the incoming edge; M = frontier width at the 99th percentile (value not stated); greedy thresholding at 0.5 (beam search not tested).
- input: 64 x 64 grayscale semantic segmentation (900-d encoder code); `map_raster`. output: `graph`.
- training: `teacher_forcing`; `matching_free` loss, ordered by a fixed convention.
- differentiable: no at inference (threshold), yes in the loss.
- sensors_data: Toulouse OSM graphs (111,034 tiles of about 110 m).
- results: StreetMover 0.0158 (GGT) vs 0.0289 (RNN), 0.0245 (modified GraphRNN), 0.0192 (no context attention); fails on cluttered graphs and with noisy input segmentations (no pre-training).
- compute: one GTX 1080 Ti, 200 epochs. code: none found
- fit_to_us: reuse_idea - effort S: StreetMover / Sinkhorn-OT between sampled graph points as a matching-free metric or loss; frontier-width idea.
- compat_needs: fixed node order. compat_conflicts: toy-scale input, no real sensor.
- tried_before: point-to-polyline both ways (E120) is a cheaper cousin.
- verified: checked.

### CARD GR22 - HDMapGen (hierarchical autoregressive graph generation of lane maps)
- papers: HDMapGen: A Hierarchical Graph Generative Model of High Definition Maps (Mi et al., CVPR 2021; arXiv 2106.14880) https://arxiv.org/abs/2106.14880
- depth: M
- stage: S4, X
- idea: Unconditional generation of lane maps: a coarse key-point graph (endpoints and intersections, about 30 % of the points) is grown node by node in DFS order from a random start; for each new node the model predicts its coordinates (Gaussian mixture) and, in one shot, its edges to all previous nodes (mixture of Bernoulli); a local decoder then fills each edge with a fixed-length coordinate sequence plus validity mask. A recurrent graph-attention network (GRAN style) with a GRU update is the backbone.
- formulation: three global-decoder variants (coordinate-first, topology-first, independent); teacher forcing; NLL + BCE + MSE; temperature scaling of the mixture (tau about 0.2 bound for quality); Adam 1e-4, one V100. Plain per-point graph "fails completely" at this task.
- sensors_data: Argoverse Miami (12,000 maps) and Pittsburgh (5,000) at 200 m x 200 m; in-house data.
- results: connectivity Frechet 0.17 vs 0.50 (SketchRNN conditional); generation latency 0.20 s per 200 m map (PlainGen 0.89 s, SketchRNN 2.28 s); metrics are distributional (MMD of degree / spectrum, Frechet). Limitations: unsmooth local graphs, cut-offs and dead ends in the sequence baseline, no cycles discussed.
- code: none found
- fit_to_us: reuse_idea - effort S: two-level (key points then curves) with one-shot edge prediction to all previous nodes; no unconditional generation need.
- compat_needs: DFS order. compat_conflicts: sampling; unconditional.
- tried_before: none. differentiable: no. training: `teacher_forcing`, `next_token`. compute: 0.20 s / map. verified: checked.

### CARD GR23 - Neural Turtle Graphics (queue-based autoregressive road layout, closure by proximity)
- papers: Neural Turtle Graphics for Modeling City Road Layouts (Chu et al., ICCV 2019; arXiv 1910.02055) https://arxiv.org/abs/1910.02055
- depth: M
- stage: S4, S7
- idea: Generation pops an unvisited "active node" from a queue; a bidirectional GRU encodes the incoming paths, a GRU decoder emits the *outgoing nodes* one by one as relative offsets (dx, dy) plus a continue flag; outgoing nodes are sorted counter-clockwise during training so no assignment problem arises; a new node within 5 m of an existing one is not queued but joined to it, which closes cycles. For aerial parsing the CNN likelihood is multiplied with the learned prior and nodes are sampled until the maximum new-node probability falls below 0.05.
- formulation: offsets discrete at 1 m, clipped to +-100 m (city generation), polar coordinates aligned to the previous edge for parsing; K random incoming paths per active node during training; teacher forcing with CE; Adam 1e-3.
- sensors_data: OSM (17 cities), SpaceNet.
- results: SpaceNet APLS 74.97 (NTG-I with DLA + STEAL) vs 71.04; unseen-city 57.89 vs 56.15; FID-based city metrics.
- code: none found (stated as not open in HDMapGen)
- fit_to_us: reuse_idea - effort S: counter-clockwise ordering of children as a way to make b outputs order-fixed without matching; closure-by-proximity rule (5 m) as the first dedup rule; prior x likelihood product as a rescoring form.
- differentiable: no (sampling). training: `teacher_forcing`. tried_before: none. verified: checked.

### CARD GR24 - LaneDiffusion (diffusion of a lane prior at the BEV-feature level)
- papers: LaneDiffusion: Improving Centerline Graph Learning via Prior Injected BEV Feature Generation (Wang et al., ICCV 2025; arXiv 2511.06272) https://arxiv.org/abs/2511.06272
- depth: M
- stage: B, S5
- idea: A diffusion model (ResShift-style, starts near the original BEV feature) generates "prior-injected" BEV features from the plain ones; the prior-injected target is made by encoding GT centerlines (20 points each, 6-layer transformer) and cross-attending them in the BEV constructor; at inference no GT is needed, the generated features are refined by concat + encoder-decoder and a CGNet decoder produces centerlines and topology. Three training stages (prior module, diffusion, decoder).
- formulation: T = 15 steps (5 / 15 / 30: TOPO-F1 44.8 / 46.8 / 46.0), kappa = 2.0, geometric schedule p = 0.3; Swin-UNet denoiser predicting x0; loss w_t ||f - x0||^2; sampling run 3 x and averaged; BEV feature 256 x 200 x 100, range x +-15 m, y +-30 m.
- sensors_data: cameras; nuScenes, Argoverse 2.
- results: nuScenes point-level GEO / TOPO / JTOPO F1, APLS, SDA: 58.9 / 46.8 / 38.8 / 37.1 / 10.6 vs CGNet 54.7 / 42.2 / 34.1 / 30.7 / 8.8; segment-level TOPll 3.4 vs 1.3 (Tables 1 - 2).
- differentiable: yes in the network; sampling of the diffusion is stochastic.
- code: none found
- fit_to_us: reuse_idea (not growth) - effort L: a learned BEV-feature "restoration" stage before the heads; a different cluster should judge.
- tried_before: E39 / E41 refinement and temporal fusion small gains.
- verified: checked.

### CARD GR25 - MapDreamer and SLEDGE (latent diffusion over lane tokens with a connectivity decoder)
- papers: MapDreamer: Aerial Imagery Conditioned Latent Diffusion for Lane-Level Map Generation (arXiv 2607.01370, 2026) https://arxiv.org/abs/2607.01370 ; SLEDGE: Synthesizing Driving Environments with Generative Models and Rule-Based Traffic (arXiv 2403.17933, 2024; authors and ECCV 2024 venue from memory) https://arxiv.org/abs/2403.17933
- depth: M
- stage: S4, X
- idea: MapDreamer denoises one latent per lane (VAE of 20-point centerlines + four adjacency matrices: successor, predecessor, left, right), conditioned by cross-attention on frozen aerial features; a cardinality head predicts the lane count, "ghost" latents pad a variable count to fixed shape; global maps by raster-order tiling with boundary tokens. SLEDGE generates a rasterised latent map with a DiT, vectorised by a transformer decoder with N <= 30 polylines of 20 points and an existence score; adjacency recovered by linking endpoints within 1.5 m whose orientations differ by < 60 deg.
- formulation: MapDreamer: DDPM 100 steps or DDIM 20 steps; losses eps-prediction + focal existence + cardinality CE; local tiles 76.8 m at 512 px; results local GEO / TOPO F1 at 1.2 m: 0.640 / 0.562 vs BGFormer 0.458 / 0.345; fixed 64 slots without cardinality: precision collapses (0.498 / 0.484); 326 ms (DDPM) / 73 ms (DDIM) per tile on a V100 (240 M parameters; small 31 M).
- sensors_data: aerial + Argoverse 2 lane graphs (MapDreamer), nuPlan (SLEDGE).
- differentiable: yes in training; stochastic sampling.
- code: https://github.com/autonomousvision/sledge ; MapDreamer none found
- fit_to_us: reuse_idea (ghost / padding for variable counts; endpoint-linking adjacency rule 1.5 m / 60 deg) - effort S; generative approach itself not suitable.
- tried_before: none. verified: checked (MapDreamer text truncated at the appendix; some percentage claims in the text do not match its tables).

### CARD GR26 - LaneGraphNet (adaptive resampling of the ground truth as a matching-free anchor loss)
- papers: Lane Graph Estimation for Scene Understanding in Urban Driving (Zuern, Vertens, Burgard, arXiv 2105.00195, 2021; venue RA-L 2021 from memory) https://arxiv.org/abs/2105.00195
- depth: M
- stage: L, S6
- idea: One-pass lane-graph estimation from lidar / camera BEV layers with a Graph R-CNN (nodes = lane anchor points, edges = connections, direction by a separate per-pixel direction net with 18 bins of 20 deg). The useful piece: *adaptive resampling* - each predicted anchor is projected orthogonally onto its nearest ground-truth lane segment, so position along the lane does not enter the loss.
- results: APLS 0.176, F1 0.574, IoU 0.420 on NuScenesGraph (Boston-Seaport, 256 x 256 crops, 0.2 m / px); nearest-neighbour connections baseline 0.088 / 0.448 / 0.298; direction accuracy 93.5 % for the direction net vs 50.8 % from the graph network. Failure: opposite lanes with the same direction, complex intersections ill-formed.
- differentiable: yes (single pass). code: none found.
- fit_to_us: reuse_idea - effort S: a projection-based loss tolerant to the along-lane shift is the matching-free loss most growers lack (cf. point-to-polyline in E120).
- tried_before: E120 same principle. verified: checked.

### CARD GR27 - Exposure-bias toolbox (DAgger, scheduled sampling, DART)
- papers: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (Ross, Gordon, Bagnell, AISTATS 2011) https://arxiv.org/abs/1011.0686 ; Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks (Bengio, Vinyals, Jaitly, Shazeer, arXiv 2015) https://arxiv.org/abs/1506.03099 ; DART: Noise Injection for Robust Imitation Learning (Laskey et al., CoRL 2017) https://arxiv.org/abs/1703.09327
- depth: A
- stage: T
- idea: Three generic remedies for the gap between teacher-forced training and free-running inference: DAgger trains a single policy on states visited by the learner (labels from an expert), scheduled sampling gradually replaces ground-truth previous inputs by the model's own outputs, DART adds noise to the executed demonstration to widen the state distribution without extra on-policy rollouts. Growers in this cluster use exactly these: RNGDet / CenterLineDet = DART-like noise, iCurb / RoadTracer = DAgger-like dynamic labels, VectorMapNet = second-stage retraining on predicted inputs.
- formulation: not read beyond abstracts. results: DART reports up to 3 x faster than DAgger on Humanoid with only 5 % loss of supervisor reward during training (abstract).
- fit_to_us: reuse_idea - effort S; compat: all three need an expert / label function defined on arbitrary states (available in our case from the GT lane graph by nearest-point rules; see GR03, GR04).
- tried_before: none (E94 teacher-forced). verified: partly (abstract pages).

### CARD GR28 - Metrics used by growth papers (TOPO, APLS, junction metric, SDA, landmark / reachability, StreetMover, Betti)
- papers: RoadTracer (junction metric, TOPO, SP); RNTR / SeqGrowGraph (landmark and reachability); LaneGNN (GEO / TOPO P-R, SDA20 / SDA50, graph IoU); Trexplorer Super (point, branch, Betti); Belli and Kipf (StreetMover); TopoGPT (M-P / M-R / M-T, G-IoU, SDA, G-F1, T-F1, JT-F1, APLS); links in the cards above.
- depth: M
- stage: E
- idea (definitions as read): Junction metric (RoadTracer): a junction = vertex with >= 3 edges; each GT junction matched to an inferred one; f_correct = share of GT edges captured around it, f_error = share of inferred edges not in the GT. TOPO: simulate driving from seed locations for a fixed distance, compare reachable destinations. SP: fraction of sampled origin-destination pairs whose shortest paths agree within 5 %. Landmark P/R (RNTR): nearest-landmark match within 0.5 - 5 m thresholds; reachability P/R: path Chamfer distance < 0.5 - 2.5 m for matched landmark pairs within 5 edges. SDA20 / SDA50 (LaneGNN): split detection accuracy within 20 / 50 px. Point-level F1 with one-to-one matching within 1.5 voxels and branch-F1 (>= 80 % coverage), Betti-0 / Betti-1 errors (Trexplorer Super) avoid linear-assignment metrics. StreetMover: Sinkhorn-Wasserstein between sampled point clouds.
- fit_to_us: reuse_idea - effort S: for fork / merge evaluation use landmark P/R for out-degree-2 / in-degree-2 vertices (cf. lit_p16_tree, item 12) and report SDA; our split / merge F1 at 6 m already is a landmark metric.
- tried_before: split / merge F1 at 6 m (project metric). verified: checked.

### CARD GR29 - Multiple-hypothesis template tracking (classical precedent for a verified hypothesis beam)
- papers: Multiple hypothesis template tracking of small 3D vessel structures (Friman et al., Medical Image Analysis 2010) https://lmi.bwh.harvard.edu/publications/multiple-hypothesis-template-tracking-small-3d-vessel-structures
- depth: A
- stage: S4, S3
- idea (from the abstract only): several hypothetical vessel trajectories are tracked at the same time so that low-contrast passages can be crossed; each trajectory is scored by a vessel template model that also gives the centerline; fast enough for interactive use; evaluated on 32 coronary CT angiography datasets with GT centerlines.
- formulation / input / output / training / differentiable / results / compute: not read (abstract page only; hypothesis tree, scoring and pruning details not on the page).
- code: none found
- fit_to_us: reuse_idea (conceptual) - effort not assessed: the nearest classical analogue of "b hypotheses per tip, each verified against a template, weak ones pruned".
- tried_before: none. verified: partly (abstract page only).

### CARD GR30 - PAMR (segment-wise autoregressive mapping with a persistent map cache; vision-language decoder)
- papers: PAMR: Persistent Autoregressive Mapping with Traffic Rules for Autonomous Driving (arXiv 2509.22756) https://arxiv.org/abs/2509.22756
- depth: M
- stage: S8, X
- idea: A vision-language model (Qwen2-VL-2B backbone) reads up to 10 front-view images plus quantised pose tokens and writes lane vectors and traffic rules as one token sequence for a 22.4 m x 22.4 m map segment. Segments are chained: after each segment the vector points near its border are projected into the next segment and used as its initialisation (map-rule cache, 10 % overlap; the succeeding prediction wins in the overlap).
- formulation: tokens: `[lane] ... [/lane]` with quantised coordinates, `[rule]` templates; deterministic parser; prompts (coordinate, box or ROI features) select the target sign. Loss, teacher forcing, decoding: not stated. Training: batch 256, AdamW, lr 2e-5, 20 epochs, 32 H20 GPUs. Ablation on the MapDRv2 test set: map-rule cache off / on 41.24 / 71.32 (HMA; the on-value equals the full model's HMA precision, so the printed column may be precision); segment width 122 / 224 / 448 px: 53.60 / 70.37 / 59.64; overlap 5 / 10 / 15 %: 58.20 / 70.37 / 51.82; 5 / 10 / 15 frames: 64.66 / 70.37 / 64.68.
- input: perspective images (644 x 364), not BEV; `image_feat` + pose tokens. output: `token_sequence`.
- training: `next_token`; no matching. differentiable: no (tokens).
- sensors_data: front camera sequences, MapDRv2 (re-annotated lane geometry plus rules); topology (forks, merges) not described.
- results: HMA F1 70.37 vs 65.29 / 64.20 for two MapDR baselines; rule-extraction F1 83.39 (single-task baseline 89.30).
- code: project page https://miv-xjtu.github.io/PAMR/ (release not verified)
- fit_to_us: reuse_idea - effort S: border-point hand-over between consecutive windows and its large measured effect (cache on / off) is the only AR-growth data point on *temporal / spatial carry-over*; the model class is not suitable.
- compat_needs: ego poses; window overlap. compat_conflicts: VLM, perspective input, token decoding.
- tried_before: temporal carry-over by ego-warped previous set is in the spec (S8); E41 temporal feature fusion +0.008 F1.
- verified: checked (paper HTML; table columns garbled, so the cache number is flagged).

### CARD GR31 - LSR-DM (conditioned diffusion refinement of a lane mask before graph extraction)
- papers: Lane Graph Extraction from Aerial Imagery via Lane Segmentation Refinement with Diffusion Models (Ruiz et al., arXiv 2405.00620; Remote Sensing 2025) https://arxiv.org/abs/2405.00620
- depth: M
- stage: S5
- idea: A D-LinkNet gives a lane mask; a DDPM conditioned on the aerial patch and on the initial mask refines it (DDIM, 25 steps, started from the initial mask with partial noise), then thresholding, thinning and graph extraction (no training) yield the lane graph. Recall rises, precision stays.
- formulation: denoising network sees image (3-layer CNN) + time embedding; target = GT mask; T = 1000 training steps, sigmoid schedule; sampling starts after 50 % of the forward steps (best: GEO F1 0.839, TOPO F1 0.775; 90 % collapses, 100 % = pure noise gives TOPO 0.562); steps 10 - 1000 give almost the same (GEO F1 0.838 - 0.841); diffusion run at 256 x 256.
- input: aerial RGB + mask; `bev_raster`. output: refined mask -> graph by skeleton rules. training: `denoising`, matching-free.
- differentiable: yes inside the net; graph extraction rule-based.
- sensors_data: aerial, 12.5 cm / px, 35 tiles (24 train / 11 test), undirected, no intersections.
- results: TOPO F1 0.774 vs 0.713 (reproduced baseline), TOPO recall 0.789 vs 0.673, GEO F1 0.841 vs 0.813 (Table 1); mask-conditioned DDIM only 0.784 / 0.703.
- code: none found
- fit_to_us: reuse_idea - effort M: a cheap, few-step refinement of the dense head output before seeding; relevant to the seed-limited E94 (192 peaks). Not growth.
- compat_needs: initial mask as condition. compat_conflicts: stochastic sampling; 11 test tiles only.
- tried_before: E39 cascade refiner (+0.001 F1) found little.
- verified: checked (paper HTML).

### CARD GR32 - PolyMapper (ConvLSTM vertex-sequence decoder with beam search; edge check by heat-map integral)
- papers: Topological Map Extraction from Overhead Images (PolyMapper; Li et al., arXiv 1812.01497, 2018; authors and ICCV 2019 venue from memory, not on the fetched page) https://arxiv.org/abs/1812.01497
- depth: M
- stage: S4, S3, S7
- idea: (The vertex-by-vertex ConvLSTM with beam search follows the Polygon-RNN family used for object outlines, from memory.) A VGG backbone gives a boundary / centerline heat map B and a candidate-keypoint mask V at 1/8 resolution; a multi-layer ConvLSTM (4 layers for roads) walks over vertices on a 28 x 28 grid, each step choosing the next vertex (or an explicit end symbol) from the previous two vertices, so the traversal direction is unique. Roads are linearised into polygon sequences by a maze "wall follower" rule (turn right at intersections, turn around at dead ends; every segment is visited twice). Decoding uses beam search (width 5); after decoding every predicted segment is scored by the integral of B along it and dropped below 0.7.
- formulation: start candidates for roads lie on the image border, the best one starts the outer polygon, inner polygons are seeded from segments passed only once; per-step classification trained with cross-entropy (softmax over the grid is the natural reading, not stated explicitly); max sequence length 30; Adam, batch 4, lr 1e-4; 300 x 300 patches, city scale by 50 % overlapping patches, splicing, small-loop removal and duplicate merging.
- input: aerial patch features + B + V + the first one or two vertices; `image_feat`, `bev_raster`. output: ordered vertex sequence -> graph.
- training: `teacher_forcing` implied (not named), no RL; `next_token`-like CE.
- differentiable: no - beam / argmax over grid, wall-follower traversal.
- sensors_data: aerial; crowdAI buildings, RoadTracer road set (25 / 15 cities), own multi-city set.
- results: RoadTracer set: SP +-10 % 61.1 vs RoadTracer 61.8, AP85 65.5 vs 64.9, AR85 84.2 vs 85.3 (accuracy on par) with far fewer vertices: 31,749 vertices / 35,998 edges vs 271,244 / 281,518 for RoadTracer; 0.29 s per road patch.
- compute: single GPU, 12 h training for roads.
- code: none found
- fit_to_us: reuse_idea - effort S: (a) the only beam-search decoder over a graph-growth sequence found in this cluster (aerial, sequence of vertices, not lane hypotheses), (b) edge validation by the *integral of a heat map along the candidate segment* is a direct, sample-along-the-line precedent for ray verification, (c) explicit vertex choice yields sparse graphs (compare our 22 vs 14.3 vertices per 100 m).
- compat_needs: heat map B along segments; keypoint mask V.
- compat_conflicts: sequential traversal order (wall follower) and softmax over a coarse grid.
- tried_before: none (E94 compactness issue noted in the project context).
- verified: checked (paper HTML).

---

## 3. Cluster compatibility notes

### (a) Interface contracts: what flows in and out in the literature, and how it maps to our pipeline

| aspect | in the literature | our side (cached pyramid, vertex / token / validity / edge output) | adapter |
|---|---|---|---|
| per-step input | image or BEV *crop* around the tip (RNGDet 256 / 128 px; CenterLineDet 100 px = 25 m; iCurb 63 px = 9.6 m; LaneGNN 256 px = 38 m; Ventura 64 px) plus a *raster of the graph built so far* | no crop; bilinear `grid_sample` of f0 / f1 / f2 at ray samples; ancestors as tokens | DeH4R does the same operation (l x l sample grid per scale, scales summed); history raster -> ancestor tokens or rasterised path channel |
| per-step output | up to N = 10 vertices + validity from DETR slots (RNGDet, CenterLineDet, DeH4R; Trexplorer: a block of extra queries per bifurcation); one displacement + stop (iCurb, RoadTracer); direction / position / state (DAGMapper); sector slots (Sat2Graph) | b children per tip (position, heading, step class, validity, event) | replace Hungarian by fixed sector or angle-bin slots, or nearest-GT winner-takes-all |
| seeds | peaks of a heat map (RNGDet, CenterLineDet, iCurb, DeH4R, Ventura), skeleton endpoints of DT (DAGMapper), start poses (LaneGNN), previous frame's dead ends (CenterLineDet) | 192 peaks of the dense heat map (limited E94) | previous-frame dangling ends are a ready S8 rule |
| ordering | FIFO (RNGDet, CenterLineDet), stack / DFS (RoadTracer), best-first (Ventura), level-synchronous rounds (DeH4R <= 3, Trexplorer), DFS tokens (SeqGrowGraph, LaneGraph2Seq, RNTR: DFS best in ablations) | ring wavefront (level-synchronous) | DeH4R / Trexplorer are the precedent for batched rounds |
| fork / merge | M > 1 outputs queued (RNGDet family); fork / terminate state (DAGMapper); clones (RNTR); merge by radius (RoadTracer 36 m with graph distance >= 72 m; RNGDet 5 px; CenterLineDet 1 m; DeH4R 10 px; NTG 5 m; DOGE 4 m) | events continue / end / fork / merge + merge pointer; tree -> DAG by clustering | radius rules are a shared convention; DOGE operators and LaneGNN weighted aggregation for DAG |
| loops | RoadTracer: edge to an existing vertex only if its graph distance is >= 6 steps; NTG: join when a new node is within 5 m of an old one; LaneGNN: successor tree depth capped at 10 (loops give infinite weight otherwise); SeqGrowGraph: directed adjacency covers loops and two-way lanes; RNTR / LaneGraph2Seq / DAGMapper: DAG, no loops (RNTR unrolls SD-map cycles by duplicating a vertex); RNGDet / DeH4R: no explicit loop logic (history raster, merge radius only) | tree -> DAG by clustering (merges are allowed, loop handling not specified) | merge radius + minimum graph distance (RoadTracer) is the cheapest guard against tiny loops |
| dead ends / stop | validity below threshold (RNGDet family, DeH4R), stop head (iCurb, RoadTracer), terminate state (DAGMapper), end node (Trexplorer), N = 0 and the open end becomes a next-frame seed (CenterLineDet), EOS tokens (sequence family; premature stops are the main failure) | event head (end) + survival of the last verified sample | no paper separates a real lane end from a censored ray (range limit, occlusion); re-expansion from degree-1 vertices (DeH4R) is the common cure |
| step length | fixed 12 m (RoadTracer); regressed, label from tau = 40 / 20 px (RNGDet); expert 20 px = 5 m (CenterLineDet code); position softmax in RoI (DAGMapper); edges < 20 m (Sat2Graph) | classes {2, 4, 6} m + residual; rays to 24 m | none found that chooses the step by verification |
| labels in rollout | static expert (RNGDet family, BC); dynamic (RoadTracer, iCurb); GT-guarded free rollout with Chamfer (DAGMapper) | free rollout, matching-free | iCurb rule and RoadTracer multi-hot are directly implementable on the nuScenes piece graph |

### (b) Known adapters and known conflicts

- **Hungarian matching** appears in RNGDet, RNGDet++, CenterLineDet (matcher costs in code), DeH4R, Trexplorer, RNTR key points, VectorMapNet detector. All of them are set-prediction heads; matching-free variants in the cluster: angular sectors (Sat2Graph D_max = 6), angle-bin multi-hot (RoadTracer), counter-clockwise order (NTG), heat-map-valued targets (Ventura), dense Chamfer / projection losses (DAGMapper, LaneGraphNet).
- **Token sequences** (SeqGrowGraph, RNTR AR, LaneGraph2Seq, TopoGPT, VectorMapNet generator) break differentiability at quantisation and argmax / sampling; the only way to use them is as *training-only* auxiliary targets (Topo2Seq: +2.7 TOPls-ls from randomised-order prompts) or as a prior.
- **RL / search** (Anagnostidis) needs reward from GT graph metrics and excludes gradients through selection.
- **Per-step backbone recomputation** (RNGDet) is irrelevant for us because the pyramid is cached; the matching adapter is a local `grid_sample`.
- **Image domain**: all aerial methods (RGB, 0.6 - 1 m / px, 100 - 2000 m tiles) need only their *head* ideas; BEV methods (CenterLineDet, SeqGrowGraph, RNTR, LaneGraph2Seq, TopoGPT) show the same heads on camera BEV.
- **Link-based vs growth**: SAM-Road family reaches RNGDet++ accuracy 14 - 50 x faster, DeH4R's hybrid is the best of both; this is the evidence behind the project's "spray, snap, link" alternative (E42 - E56 found the rules already decide).

### (c) Combinations demonstrated by the papers themselves

- RNGDet agent + segmentation-derived seeds (RNGDet, CenterLineDet); agent + temporal heat-map averaging and dead-end carry-over (CenterLineDet).
- DAgger-style rollout labels + frozen pre-trained FPN + heat-map seeds (iCurb); dynamic oracle + segmentation-peak seeds (RoadTracer).
- Hybrid: candidate detection + neighbour prediction + initial graph + expansion in rounds (DeH4R: 81.21 vs 81.02 decode-only, expansion-only 79.18).
- Pairwise link head + interior / extension samples + node-guided resampling (SAM-Road++).
- Teacher-forced AR prior + learned perception adapter (TopoGPT), AR sequence head as training-only regulariser (Topo2Seq), autoregressive pre-training + RL fine-tune + tree search (Anagnostidis).
- Chained multi-step training of a level-synchronous DETR tracker (Trexplorer Super); truncated BPTT with Chamfer loss (DAGMapper).
- Prior x likelihood product for aerial parsing (NTG).

### (d) Comparison matrix of the growers

| method | per step | seeds | order | regime | dedup / merge | differentiable break | key number |
|---|---|---|---|---|---|---|---|
| RoadTracer | walk / stop + angle bin (multi-hot target), D = 12 m | segmentation peaks | stack, DFS | dynamic labels | 3D / 6D graph-distance rule | argmax, T | junction F_correct 0.58 at 5 % error |
| RNGDet(++) | <= 10 vertices + validity | I-map peaks | FIFO | BC + noise | 30 px filter, 5 px align | threshold, Hungarian | APLS 67.76 City-scale |
| CenterLineDet | <= 10 vertices | H_I peaks + last frame ends | FIFO | BC + noise (clean labels) | 15 px, 1 m | two-stage, DETR matcher | T-F1 0.518 nuScenes |
| iCurb | displacement + stop | skeleton ends + heat peaks | per instance | DAgger | none | argmax, stop | F1@5px 0.883 |
| DAGMapper | direction, position (softmax), state | DT skeleton ends | queue | TF + free rollout, BPTT 5 | DT recovery | argmax at test | F1@3px 88.8 |
| DeH4R | <= 10 neighbours per candidate | masks | rounds <= 3 | TF | 10 px merge | NMS, Hungarian | TOPO-F1 81.21 |
| LaneGNN | successor sub-graph | start poses | DFS, depth 10 | TF + pose noise | a_thresh 20 px | Dijkstra | TOPO R 0.670 Full-LGP |
| Trexplorer Super | all active branches, level by level | root | BFS | chained TF | query self-attention | Hungarian | branch-F1 77.12 synthetic |
| Anagnostidis | pointer to key points (edge) | key-point set | any | NTP + RL | duplicates harmless | sampling | APLS 0.778 (GT key points) |
| PolyMapper | next vertex on a 28 x 28 grid or end symbol, beam width 5 | border start + keypoint mask | wall follower | CE per step | loop removal, merge duplicates | beam / argmax | SP +-10 % 61.1 with about 8.5 x fewer vertices than RoadTracer |
| token family | tokens | n/a | DFS best | NTP | clones | tokenisation | L-F 56.4 (SeqGrowGraph) |

### (e) What is reusable for a fixed-shape hypothesis tree (b >= 2 children per tip, beam, long-ray verification, free rollout, matching-free)

1. **Labels for free rollouts without matching exist and are cheap**: iCurb's rule (nearest GT point ahead of a progress index, farther than a minimum distance, restricted / free switch at 15 px, stop when nothing is left ahead) and RoadTracer's oracle (multi-hot over all unexplored GT branches, map-matched position with explored-edge memory). Both give a label for any tip position, including off-lane tips, and need no assignment.
2. **Exposure-bias evidence with sizes**: removing free exploration costs the most in iCurb (F1@5px 0.883 -> 0.750; the authors call it critical); VectorMapNet's retraining on predicted keypoints adds 5.1 (camera) and 8.5 (fusion) mAP; SAM-Road++'s node-guided resampling adds about 1.25 F1 (SpaceNet rows, dataset inferred); RNGDet/CenterLineDet rely on noise only. DAgger-type loops need 1 - 5 free passes (iCurb README: 1 is enough).
3. **Fixed child slots without Hungarian**: sector encoding (6 x 60 deg, one edge per sector) and multi-hot angle bins are the published matching-free precedents; DETR query count 10 was best among 5 / 10 / 20 (junctions rarely exceed about 5 arms), so b = 2 - 3 slots per tip with sector assignment is plausible but untested in the literature.
4. **Verification by intermediate samples**: SAM-Road++ (extension and segment samples: APLS +1.6) and LineGraph2Road (endpoint + 4 interior samples) show that sampling features along the candidate segment helps a pairwise classifier; PolyMapper scores each predicted segment by the integral of a heat map along it (threshold 0.7); RNGDet++ adds a per-query mask of the segment ahead as supervision. No grower derives the *step length* from such samples.
5. **Dedup / merge radius conventions** (5 - 15 px at 1 m / px, 1 - 5 m in metric units; RoadTracer additionally requires a graph distance of at least 6 steps to avoid tiny loops) - a ready rule set for the tree -> DAG step, with DOGE's operator list (merge < 4 m, T-junction split, collinear merge, prune) and LaneGNN's weighted lateral aggregation.
6. **Level-synchronous batching works**: DeH4R expands all degree-1 vertices in parallel, at most 3 rounds (F1 81.14 at 1 round vs 81.26 at 6 - 7); Trexplorer assigns a block of extra queries at each bifurcation (nb = 2 on its synthetic data per the rebuttal; the Super paper's description lists 26 object queries) and tracks 9 steps per patch; both prefer rounds to a sequential queue.
7. **Search over hypotheses has small measured gain**: Anagnostidis' tree search is the smallest ablation component (about 1 - 2 points); the only beam decoder found is PolyMapper's (width 5 over a ConvLSTM vertex sequence, aerial, with no ablation of the width seen); no lane-graph grower uses a beam. The beam in our design is therefore untested territory; its value will depend on the rescoring quality.
8. **Early termination and duplicates are the dominant failure modes** of growers (RNTR, LaneGraph2Seq, Trexplorer, Anagnostidis ARM); known cures: re-expansion from dangling ends (DeH4R), reduced weights of end tokens (LaneGraph2Seq), noise padding (RNTR), chained training (Trexplorer Super), previous-frame dead ends as seeds (CenterLineDet).
9. **Short-unroll gradients are possible**: DAGMapper trains 5 steps with softmax relaxations and a Chamfer loss; TopoGPT unrolls ODE steps with gradient. Nothing in the cluster back-propagates through a *selection of b children*; that remains our own design question (selection with gradient through the selected items).
10. **Carry-over between windows / frames**: CenterLineDet (dead ends of frame T-1 become seeds; neighbouring heat maps warped and averaged), LaneGNN (weighted aggregation of overlapping successor graphs), PAMR (border points handed to the next segment; its ablation shows a very large effect). These are the S8 precedents for growth-type blocks.

### (f) Pairs worth testing (derived from the cards and the project context; these are design suggestions, not findings of the papers)

| pair | what it would test | needs from our side |
|---|---|---|
| iCurb restricted / free label rule x fixed-shape tree | exposure bias of free rollout without matching (GR03, GR27) | nearest-GT-ahead label per tip with a progress index per branch |
| RoadTracer multi-hot unexplored branches x b child slots | matching-free child assignment (GR04) vs sector slots (GR09) vs nearest-GT winner-takes-all | angle / sector discretisation of the 9 bearings |
| DAGMapper short unroll + Chamfer x E94 / tree | gradient through 3 - 5 levels with softmax-relaxed selection (GR05) | soft selection of children; point-to-polyline loss from E120 |
| DeH4R rounds x E94 output | re-expansion from dangling ends, merge radius, 1 - 3 rounds (GR06) as a recall fix for seed-limited traces | degree-1 detection, merge rule |
| SAM-Road++ extension / segment samples x ray verification | whether per-sample survival beats one score from interior samples (GR10) | same rays, single-score head as baseline |
| node-guided resampling x junction / piece verifier | exposure-bias fix for pairwise heads (GR10) | verifier trained on the tree's own proposals |
| DOGE operators x tree -> DAG | merge < ε, T-junction split, collinear merge (GR12), compared with LaneGNN weighted aggregation (GR08) | clustering radius, edge-split rule |
| CenterLineDet dead-end carry-over x time recursion | previous-frame open ends as seeds (GR02, GR30) | ego-warp of vertex set |
| Topo2Seq training-only sequence head x tree nodes | free topology regulariser (GR17) | DFS serialisation of the tree |
| landmark / reachability + SDA metrics x our split / merge F1 | comparable fork / merge evaluation (GR28) | vertex out- / in-degree labels |

### (g) Scales of the published designs in metres (for choosing our step classes and window)

| design | step / spacing | window | merge / closure radius |
|---|---|---|---|
| RoadTracer | 12 m fixed | d x d at 0.6 m/px (d not stated) | 36 m with graph distance >= 72 m |
| RNGDet | label tau = 40 px (about 24 m) road mode, 20 px (about 12 m) at junctions | 256 px = about 154 m | not stated (5 px align in repo, 1 m/px) |
| RNGDet++ | repo move-ahead 30 px = 30 m | 128 px = 128 m | 5 m alignment, 30 m candidate filter |
| CenterLineDet (code) | 5 m (half at junctions) | 25 m window on a 0.25 m grid | 1 m alignment, 15 px = 3.75 m seed filter |
| iCurb | continuous displacement inside crop | 63 px = 9.6 m | none |
| LaneGNN | one edge per step | 256 px at 15 cm = 38 m | lateral 20 px = 3 m |
| DeH4R | neighbours predicted per candidate (no step) | 512 / 256 px tiles, 3 x 3 ROI | 10 m |
| Sat2Graph | edges < 20 m (15 - 25 m best) | 352 px | 15 m decode cap |
| NTG / DOGE | offsets clipped to +-100 m / curves | 300 m / 512 px | 5 m / 4 m |
| ours (E94) | classes 2 / 4 / 6 m + residual; rays to 24 m | +-50 m BEV | merge pointer |

### (h) Reported run times (for the RTX 3060 budget)

CenterLineDet 0.41 s / frame (agent) on one RTX 3090; iCurb 0.89 s / image; LaneGNN about 1 s per rollout step; RNGDet++ 3.85 h for City-scale test tiles vs DeH4R 13.3 min and SAM-Road 4.6 min; RNTR AR 0.1 FPS, SAR 0.6, NAR 5.5; Topo2Seq 14.7 FPS (decoder removed); HDMapGen 0.20 s per 200 m map; MapDreamer 73 - 326 ms per 76.8 m tile (V100). Sequential per-vertex growth costs seconds per frame, which the level-synchronous rounds of DeH4R / Trexplorer avoid.

---

## 4. Reuse versus re-implement (given: no mmcv / mmdet, cached features, RTX 3060, WSL env `gcar`)

- **Re-implement (small, from method sections):** iCurb dynamic-label rule + restricted / free schedule (GR03); RoadTracer multi-hot unexplored-branch target and merge rule (GR04); DAGMapper short-unroll Chamfer training (GR05); DeH4R round-based expansion with radius merge and l x l grid-sampled ROI features (GR06); Sat2Graph sector slots and "edge terms only at vertex cells" mask (GR09); SAM-Road++ extended-line samples and node-guided resampling (GR10); DOGE node-merge / split / collinear operators for tree -> DAG (GR12); landmark / reachability and SDA metrics (GR28).
- **Adapt code (only for baselines, effort M):** CenterLineDet agent on our dense head's heat map (GPL-3.0 on its README; Docker and a python-2 evaluation container are the blockers; it needs no mmcv); RNGDet++ sampler / labels (GPL-3.0, "academic use only" per README) for the label logic.
- **Use as comparison only:** RNGDet++, SAM-Road / SAM-Road++ (MIT on the SAM-Road README), DeH4R (licence not stated on its README), TopoGPT (Apache-2.0, PyTorch 2.1, plain PyTorch), RNTR / LaneGraph2Seq (MIT on the README, but the code needs mmdet3d 1.4.0 + mmcv 2.x + mmdet 3.x + mmengine: not runnable in `gcar` without adding that stack), Trexplorer Super (licence not stated). These need sensor images or aerial RGB and cannot take our cached pyramid.
- **Not suitable:** tokenised AR decoders and RL (GR11, GR13 - GR16, GR18) as blocks; diffusion generators (GR24, GR25).

---

## 5. Open questions and gaps

- No paper trains a tree / beam of growth hypotheses with gradient through the selected children; DAgger and BC dominate, and the single differentiable rollout (DAGMapper) does not say how gradients pass the rotated RoI.
- No grower selects its step length by verifying intermediate samples; link papers (SAM-Road++, LineGraph2Road) do verify a *given* segment, growers regress or fix the step.
- No comparison of FIFO vs DFS vs best-first vs level-synchronous order for geometric growers; only token-sequence papers ablate order (DFS best).
- How many of RNGDet's / CenterLineDet's gains come from the history raster is not isolated for BEV (iCurb: removing history costs 0.883 -> 0.818 F1@5 px, inferred).
- CenterLineDet: BEV extent, ROI, step and noise appear only in the code; the DAgger flag exists, but the released sampler is BC only - unclear whether the paper used DAgger variants (paper says BC).
- RoadTracer's window size d and angle count a, VecRoad's method details, SeqGrowGraph's DFS root and decoding, Anagnostidis' simulation count and ablation columns: not found.
- Fork resolution (minimum divergence angle, minimum separation of children): none of the growers reports it; RNGDet uses tau' = 20 px (12 m) after a junction, CenterLineDet halves the step at junctions.
- Radar / camera-radar BEV: none of the growers uses radar; CenterLineDet's no-lidar ablation (0.698 / 0.466) shows the agent still works on camera-only heat maps.
- Evaluation: the growth papers' metrics (TOPO, APLS, pixel F1) have no per-fork resolution analysis; landmark / reachability from RNTR are the closest.

---

## 6. Search log

WebSearch queries (standard unless marked extended):
1. RNGDet++ github TonyXuQAQ road network graph detection transformer instance segmentation code
2. SeqGrowGraph github code lane graph sequence growth ICCV 2025
3. autoregressive road network graph generation 2025 centerline lane graph transformer BEV iterative vertex (extended)
4. lane graph generation diffusion model centerline topology BEV 2025 2026 (extended)
5. road network graph extraction aerial images 2026 iterative graph growing vertex-by-vertex new method arXiv (extended)
6. SAM-Road++ OR SAM-Road road graph extraction improved decoupled graph growing arXiv 2025
7. HDMapGen hierarchical graph generative model high definition maps autoregressive lane graph
8. lane graph next-step OR next vertex prediction BEV centerline agent imitation learning nuScenes 2024 2025 road graph exploration (extended)
9. VecRoad point-based iterative graph exploration road graphs extraction flexible step length segmentation cues temporal cues
10. Mamba OR state-space road network graph extraction aerial vertex edge 2025 topology
11. Monte Carlo tree search OR beam search road network OR centerline graph extraction learned policy branching tracing
12. arXiv 2026 centerline graph BEV autoregressive OR sequential lane graph generation sparse tokens successor prediction nuScenes OpenLane-V2 (extended)
13. road graph extraction satellite graph growing OR iterative OR autoregressive 2026 arXiv CityScale SpaceNet APLS state-of-the-art (extended)
14. Neural Turtle Graphics for Modeling City Road Layouts arXiv Chu Fidler ICCV 2019
15. vessel centerline tracing deep learning branches multiple hypotheses tree tracking CNN direction bins reinforcement learning
16. reinforcement learning agent road graph extraction aerial images tracing policy reward 2023 2024 2025 vertex-by-vertex
17. TD-Road top-down road network extraction OR Cascaded graph road graph extraction CityScale SpaceNet 2024 graph transformer
18. Trexplorer recurrent DETR centerline tree tracking 3D volumes MICCAI 2024 arXiv
19. TopoRoad road topological lane graph centerline graph ego vehicle camera BEV baseline CenterLineDet comparison
20. lane centerline graph tree structured decoder branching hierarchical queries BEV online mapping successor children prediction
21. lane graph level-by-level OR breadth-first parallel expansion frontier all endpoints simultaneously road graph growing transformer batch
22. language model OR vision-language lane topology generation autoregressive next-token map elements vectorized HD map 2025 MapAR OR MapGPT OR "map as language"
23. Xu Zhenhua Liu Ming imitation learning lane centerline graph BEV journal 2024 2025 CenterLineDet extension DAgger road network
24. diffusion road network graph extraction aerial image conditioned vertices edges denoising graph 2025 2026 road topology
25. end-to-end differentiable road graph extraction no post-processing graph decoding transformer 2025 satellite lane graph differentiable
26. large language model OR multimodal LLM generate lane-level vector map centerline graph from BEV or surround images 2025 2026 MapLM OR LaneLLM
27. hierarchical OR coarse-to-fine lane graph generation transformer keypoints then curves BEV online HD map centerline 2025 autoregressive keypoint
28. Anagnostidis Mastering Spatial Graph Prediction of Road Networks github code release MuZero road graph

WebFetch (reading) targets, grouped: arXiv HTML / abstract pages 2202.07824, 2209.10150, 2209.07734, 2103.17118 (twice), 2507.04822 (three passes), 2402.08207 (two passes), 2401.17609, 2012.12377 (two passes), 2508.13669 (two passes), 2007.09547, 2606.31814, 2210.00828 (two passes), 2511.19850, 2511.06272, 2607.01370, 2302.06175 (two passes), 2105.00195, 2403.16051, 2411.16733, 2602.23290, 2507.10881, 1910.14388, 2106.14880, 1910.02055 (ar5iv), 2206.08920, 2403.17933, 2502.08974, 1808.09814, 1011.0686, 1506.03099, 1703.09327, 2409.12409; ar5iv 1802.03680 (two passes); project pages tonyxuqaq.github.io (RNGDet, CenterLineDet), mmcheng.net/vecroad; further abstract / HTML pages: 1812.01497 (PolyMapper), 2509.22756 (PAMR), 2405.00620 (LSR-DM, abstract + v1 HTML), arXiv abstract pages used for venue checks (2302.06175, 2007.09547, 2508.13669, 2103.17118, 2403.16051, 2209.07734, 2507.04822), 2607.21281 (not a grower), MICCAI page of Trexplorer, Harvard LMI page of Friman et al., ar5iv 2209.10150 (ablation table); GitHub README / source: RNGDetPlusPlus (README, cityscale/main_train.py), CenterLineDet (README, config/get_args.py, CenterLineDet/sampler.py, CenterLineDet/inference.py), Topo-boundary (README, iCurb README, config.yml, main_env.py), RoadNet (RNTR / LaneGraph2Seq: README, RoadNetwork-2.0.1/get_started.md), DeH4R, Sat2Graph, sam_road, TopoGPT, roadtracer, VecRoad, Trexplorer-Super. Failed or unusable: arxiv.org/pdf versions (binary), CVF PDFs (403 / binary), RNGDet GitHub (404), urbanlanegraph.cs.uni-freiburg.de (connection refused), Springer (redirect), Google search (consent redirect).
