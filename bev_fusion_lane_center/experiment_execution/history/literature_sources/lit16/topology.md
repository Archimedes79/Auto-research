# Cluster TP - lane topology and lane-segment perception (OpenLane-V2 / nuScenes)

Date 2026-10-23. Card prefix TP. Schema: `SCHEMA.md`; project facts: `PROJECT_CONTEXT.md`. Reading only (WebSearch / WebFetch); nothing downloaded.

## 1. Coverage statement

- **What was searched:** 27 web searches (queries in section 6) and about 95 page fetches: arXiv HTML pages (method sections with equations and ablation tables, two-part reads for long pages), GitHub README pages and source files of the head modules, the OpenLane-V2 repository page, its metric-fix issue (#76) and its evaluation file, the OpenDriveLab challenge-2025 page. All page text arrived through WebFetch, which returns a digest of the page; equations are therefore paraphrased, and typos / ambiguities that the digest flagged are repeated in the cards. Where the released code differs from the paper text (TopoLogic, TopoPoint, TopoFG, CGNet, UniTopo) the cards say so ("CODE AS READ"). Only numbers printed in the primary paper are used (a third-party survey table lists different numbers for the same models, e.g. TopoLogic DET_l 34.4 instead of 29.9, and is not used).
- **Depth M (method section read, equations and tables):** all 12 priority papers: TopoLogic, TopoPoint, TopoFG, UniTopo, CGNet, LaneSegNet, TopoNet, TopoMLP, Topo2Seq, TopoBDA (both halves of the page), LaneGAP, SMART. Further papers read at M: TopoEnhance (Oct 2026), Lane Field (Oct 2026), CoPo (KDD 2026), RATopo, T2SG, TopoStreamer, FASTopoWM, LaneDiffusion, TopoGPT (ECCV 2026), SeqGrowGraph, LaneGraphNet, Reusing-Attention, RoadPainter, TopoMask, Topo2D, SCORE (SD-map stack), and three challenge reports (2023 first place; 2024 LGmap and MapVision).
- **Depth C (code page read):** LaneSegNet (README); TopoLogic (README, `topologic_head.py`, `sgnn_decoder.py`, `relationship_head.py`); TopoPoint (README, `topopoint_head.py`, dataset file); TopoFG (README, `topofg_head.py`); UniTopo (README, `unitopo_head.py`); CGNet (README, `CGNet_head.py`); LaneGAP (landing page only, almost empty) and the MapTR README; OpenLane-V2 (README, issue #76, `evaluate.py`). Not read (URL only): TopoMLP, TopoNet, T2SG, RATopo, TopoStreamer, FASTopoWM, STSU, SMERF code.
- **Challenge reports:** 2023 OpenLane Topology first place (arXiv 2306.09590) and 2024 Mapless Driving first place (LGmap) and MapVision read. 2nd / 3rd place of 2023 and other 2024 teams: not found (only a reproducibility report of LaneSegNet exists, not opened). 2025: the Autonomous Grand Challenge 2025 page lists three tracks (NAVSIM v2 end-to-end driving, a world-model challenge, AgiBot World) and no mapless / lane-topology track, so no 2025 report exists for this task.
- **Newest work found (2026):** UniTopo (TCSVT, May), CoPo (KDD, v4 Jul), TopoGPT (ECCV, Jun), TopoEnhance (arXiv Oct 12), Lane Field (arXiv Oct 18). No lane-topology paper of 2026 was found that uses radar.
- **Not opened:** OpenReview / CVF / proceedings pages, supplementary PDFs (CGNet supplement, TopoLogic appendix), configs and bbox coders of the released repositories (thresholds, loss classes), SEPT (arXiv 2505.12246, SD-map lane segments) and Chameleon (arXiv 2503.07485, VLM-assisted topology; abstract only), "Prior Based Online Lane Graph Extraction from Single Onboard Camera Image" (arXiv 2307.13344, search hit only), TSTGT, HDMapNet (numbers taken from LaneGAP's tables).
- **Metric caveat (important for every number below):** OpenLane-V2 changed the TOP metric after TopoMLP showed a loophole (devkit v1.1 / v2.1 versus v1.0 / v2.0, issue #76). Old numbers (TopoNet TOP_ll 4.1, TopoMLP 7.2) and v2.1 numbers (10.9, 21.7 for the same models) are not comparable. Unless stated, tables below are subset A (Argoverse 2), ResNet-50, 24 epochs, v2.1 metrics. Subset B = nuScenes.

### Index

| id | technique | depth | role for us |
|---|---|---|---|
| TP01 | TopoLogic (endpoint-distance prior + query similarity) | M+C | link scoring prior |
| TP02 | TopoPoint (endpoint queries, point-lane attention / GCN, snapping) | M+C | explicit vertex detection |
| TP03 | CGNet (junction heat-map queries, Bezier-space continuity, iterative adjacency) | M+C | nuScenes centerline graph |
| TP04 | LaneGAP (paths instead of pieces, Path2Graph) | M+C(partial) | tree -> graph conversion |
| TP05 | TopoFG (keypoint queries, boundary-point topology, matching-free denoising target) | M+C | vertex-like queries |
| TP06 | UniTopo (connected lane as the unit) | M+C | link-free topology |
| TP07 | TopoEnhance (post-hoc GAT on link scores, noise simulation) | M | verifier / calibration |
| TP08 | TopoMLP (MLP link head, oracle-topology experiment, metric loophole) | M | link-head baseline |
| TP09 | TopoNet (scene-graph GNN, vertex-level TOP) | M | baseline, metric |
| TP10 | LaneSegNet (lane segments, lane attention) | M+C | sampling pattern |
| TP11 | Lane Field (dense direction tensor field, FiLM, tangent-stretched sampling) | M | dense-head idea |
| TP12 | RATopo (denser topology supervision by one-to-many assignment) | M | label construction |
| TP13 | CoPo (geometry-biased attention, relational contrastive loss) | M | link loss |
| TP14 | T2SG (distance bias in self-attention, counterfactual layer) | M | geometry bias |
| TP15 | TopoBDA (Bezier deformable attention, instance-mask loss, radar) | M | radar evidence |
| TP16 | Topo2Seq (training-only sequence decoder) | M | aux regulariser |
| TP17 | SMART (SD + satellite map prior, frozen) with SMERF | M | route / SD prior |
| TP18 | SCORE (stack of SD map, one-to-many, denoising, GeoDist, temporal) | M | evidence of stacking |
| TP19 | TopoStreamer (stream queries) | M | temporal |
| TP20 | FASTopoWM (fast-slow, latent world model) | M | temporal |
| TP21 | LaneDiffusion (diffusion in BEV feature space) | M | feature refinement |
| TP22 | TopoGPT (generative, 3.3 M map pre-training) | M | prior (not suitable) |
| TP23 | SeqGrowGraph (vertex graph as adjacency growth) | M | vertex-graph view |
| TP24 | LaneGraphNet (anchor nodes + edges from BEV, nuScenes map expansion) | M | closest in setting |
| TP25 | Reusing attention (attention-derived topology, SD distillation) | M | light link head |
| TP26 | RoadPainter and TopoMask (mask-derived points, soft-argmax) | M | differentiable extraction |
| TP27 | Topo2D (2D lane priors for query initialisation) | M | not suitable |
| TP28 | Challenge winners 2023 / 2024 (tricks) | M | recipe list |
| TP29 | OpenLane-V2 data and metrics | M | evaluation |
| TP30 | STSU (query-based centerline graph, pairwise association classifier) | M | historical baseline |
| TP31 | Online lane graph from video (early BEV warp + max) | M | temporal evidence |

## 2. Cards

### CARD TP01 - TopoLogic (geometric endpoint distance + query similarity)
- papers: TopoLogic: An Interpretable Pipeline for Lane Topology Reasoning on Driving Scenes (Fu et al., NeurIPS 2024) https://arxiv.org/abs/2405.14747
- depth: M+C
- stage: S6 (L)
- idea: The link score between lane pieces is the sum of two terms: a geometric term that maps the distance from the END of piece i to the START of piece j through a learned monotone decay, and a semantic term (sigmoid of a dot product of two MLP-embedded queries). The fused N x N matrix also drives a GNN that updates the queries after every decoder layer. The geometric term can be added to an already trained detector without retraining.
- formulation: d_ij = |end_i - start_j| (end point of lane i, start point of lane j, each from 11 predicted 3D points; unit not stated, metres presumed); D = {d_ij}; G_dis(i,j) = exp(-d_ij^alpha / (lambda * sigma)), sigma = std(D), alpha and lambda learnable scalars (1 x 1), init alpha 0.2, lambda 2 (TopoPoint's text writes the two inits the other way round). G_sim = sigmoid(MLP1(Q) MLP2(Q)^T), Q = 200 x 256 lane queries, 3-layer MLPs. G = l1 G_dis + l2 G_sim (l1 = l2 = 1, learnable). GNN: aggregates neighbour queries with G (layer count, normalisation, skip not stated). L = L_det + L_top; L_det = focal (weight 1.5) + L1 (0.025); L_top acts only on G_sim (loss type and weight not named); G_dis has no topology loss, its two scalars learn through the GNN / detection path. Positive (i,j) iff end of i meets start of j (directed). Class imbalance not discussed. Head: Deformable-DETR, 200 queries, 11 x 3 points, supervision on every decoder layer. Alternatives tested for the mapping f: Gaussian exp(-x^2/2), 2/(1+e^x), tanh form. CODE AS READ (sgnn_decoder.py, topologic_head.py) differs from the paper text: d = sum over x, y, z of |end_i - start_j| (L1, not Euclidean) on DENORMALISED coordinates (metres) of the current decoder layer, point coordinates DETACHED (no gradient from topology into the points); mapping exp(-d^P / w) with learnable P (init 2) and w (init 10) and no alpha; sigma = std(D) is computed but never used; adjacency for layer l+1 = lamda_1 * sigmoid(relation branch).detach() + lamda_2 * topo (lamda init 1, learnable), diagonal of the distance term zeroed, no row normalisation or A + A^T; GNN layer: forward messages with adj and backward messages with adj^T (separate weights), edge weight 0.5, concat with the lane-traffic GNN output, Linear 2C -> C, residual; layer 0 gets zeros. The inference score in topologic_head.py hard-codes P = 2, w = 11.5275, lamda = 1: a Gaussian-type kernel with half value at an L1 distance of 2.8 m (d = 1 m -> 0.92, 3 m -> 0.46, 5 m -> 0.11), i.e. a wide tolerance for noisy query endpoints. Targets (topologic_head.py + relationship_head.py): GT adjacency indexed by the Hungarian assignment (pos_inds, pos_assigned_gt_inds) into a Q x Q zero matrix; ALL other pairs, including every pair with an unmatched query, are negatives (full supervision, no sampling, no pos_weight); the relation head embeds both query sets with 3-layer MLPs to 256-d and scores a pair by the DOT PRODUCT (no pair-concatenation classifier; symmetric when the weights are shared); the loss is mmdet FocalLoss (sigmoid, gamma 2, alpha 0.25) on labels passed as 1 - target because mmdet treats label == num_classes as background (so the inversion is a convention, not a semantic flip). Imbalance is handled only by the focal terms. The same relation-head file family is used by TopoPoint, TopoFG and UniTopo (TopoNet lineage).
- input: lane queries 200 x 256 and predicted points 200 x 11 x 3 of each decoder layer; BEV 200 x 100 (BEVFormer).
- output: directed N x N link probabilities; lane-traffic topology via the TopoNet path.
- training: DETR-style set loss with Hungarian matching (from memory, as TopoNet; not restated on the page); 24 epochs, AdamW 2e-4 cosine, R50 + FPN, images 1550 x 2048, batch 2 on 8 RTX 3090.
- differentiable: partial - the scores are differentiable in queries and points; the label assignment between queries and ground truth is a matching step.
- sensors_data: camera; OpenLane-V2 subset A (Argoverse 2) and B (nuScenes); BEV +-50 x +-25 m in a 200 x 100 grid.
- results: subset A v2.1: TOP_ll 23.9 vs TopoNet 10.9, OLS 44.1 vs 39.8, DET_l 29.9 (Table 1). Topology approach (Table 4, TOP_ll): MLP 10.8, similarity 12.9, GeoDist 20.1, both 23.9. Mapping function (Table 3, TOP_ll): tanh 15.1, sigmoid 19.1, Gaussian 21.7, learned exp 23.9. Post-hoc GeoDist (Table 5): TopoNet TOP_ll 10.9 -> 22.3, SMERF 15.4 -> 26.2, LaneSegNet TOP_lsls 25.4 -> 29.6. Subset B: TOP_ll 21.6, OLS 42.3 (no ablation on B). With SD map OLS 47.5.
- compute: not stated.
- code: https://github.com/Franpin/TopoLogic (README: Python 3.8, CUDA 11.1, PyTorch 1.9.1; no licence line; ResNet-50 checkpoint on Hugging Face)
- fit_to_us: reimplement - effort S - the mapping needs piece end / start points; in our vertex graph edges share vertices by construction, so it applies only where a link or merge is a decision (tip to existing vertex, DAG clustering).
- compat_needs: directed pairs with a defined "end" and "start" position; labels defined by endpoint coincidence.
- compat_conflicts: relies on accurate endpoints (authors: geometry depends heavily on detection accuracy); the tolerance learned for noisy query endpoints (about 3 m) is far wider than our vertex error (0.43 - 0.54 m), so the form transfers but not the bandwidth; original code is a PyTorch 1.9.1 / mm-series code base (the SGNN decoder builds on mmcv layers, from memory) and cannot run in our environment as is.
- tried_before: E45 (TopoLogic-style scoring) and E42 / E46 / E51 / E56 (learned link heads): within 0.01 of the rules, dropped.
- verified: checked (paper and the two code files read; paper-versus-code differences listed above)

### CARD TP02 - TopoPoint (endpoint detection coupled to lane tokens)
- papers: TopoPoint (Fu et al., arXiv 2025, arXiv:2505.17771; the page title was not seen) https://arxiv.org/abs/2505.17771
- depth: M+C
- stage: S1 S6
- idea: The de-duplicated start / end points of all ground-truth centerlines are made a separate detection target with their own 200 point queries (= the vertices of the piece graph). Points and lanes live in one 6-layer decoder; a self-attention with an additive distance bias couples them, a bipartite GCN passes features along the predicted point-on-lane incidence, and at inference lane endpoints within 1.5 m of a detected point are averaged with it.
- formulation: queries N_p = 200 (3D learnable reference points), N_l = 300, N_t = 100. PLMSA: logits of concat[Q_p; Q_l] get the additive bias [[0, M_pl], [M_pl^T, M_ll]], M = f_map(D), f_map(x) = exp(-x^p / (lambda * sigma)) (the TopoLogic mapping); D_ll = sum |end_i - start_j| over predicted lanes of the previous layer; D_pl = min(sum |p_i - start_j|, sum |p_i - end_j|); no bias between points. PLGCN: A_pl = l1 G_pl + l2 M_pl; Q_p += GCN(Q_l, A_pl); Q_l += GCN(Q_p, A_pl^T); GCN(X, A) = sigmoid(norm(A) X W). Heads: G_pl = sigmoid(MLP(Q_p) MLP(Q_l)^T) (point lies on lane), G_ll and G_lt analogous. PLGM (inference only): keep points and lanes with score > 0.3; for each point collect lane endpoints within delta = 1.5 m; replace them by E = (P + sum of matched endpoints) / (n + 1) (greedy, no Hungarian). Loss: lt, lp, ll weights 1, topology weights 5 (lpl missing); focal + L1; DET_p = mean AP over thresholds {1, 2, 3} of point distance. CODE AS READ (topopoint_head.py; the data pipeline, transformer and relation heads are in other files): points and lanes are matched by the same HungarianAssigner (cost cls 1.0 + L1 5.0 + GIoU 2.0); classification loss in this file is softmax cross-entropy with background weight 0.1 and L1 regression weight 5.0 (not focal; topology loss classes not in the file); GT endpoints: in the dataset file (used for evaluation) the first and last point of every GT centerline (11-point version) are collected and de-duplicated by EXACT equality (torch.unique; OpenLane-V2 pieces share identical coordinates at joints), no class label, single-lane endpoints are kept (a filter keeping only points shared by several lanes is commented out); the training-side point labels and the point-lane adjacency come from the pipeline, not visible; point-lane target = GT point-lane adjacency scattered into a (points x lanes) zero matrix by the Hungarian indices and supervised on the FULL matrix, while the lane-lane loss is computed on the matched sub-matrix only (valid-only supervision, the variant RATopo later shows to be better than full supervision); defaults num_query = num_pts_query = 100 (config values not seen). The inference snapping in get_lanes does NOT use detected points: it keeps lanes with score > 0.3, scores links by exp(-d^2 / 11.5275) with d = L1 distance end_i -> start_j, keeps links above 0.7 (d below about 2.0 m), then runs two passes that replace the end of lane i and the starts of its linked lanes by their arithmetic mean (second pass on the transposed links averages starts with ends), and recomputes the topology; the 1.5 m point threshold of the paper text does not appear in this file.
- input: BEVFormer BEV 200 x 100 x 256, front-view features for traffic elements.
- output: 200 3D points, 300 lanes (11 x 3), incidence N_p x N_l, adjacency N_l x N_l, lane-traffic matrix.
- training: end-to-end, 24 epochs, AdamW 2e-4 cosine, R50, multi-view x0.5 resolution, 8 V100; assignment method not stated (DETR-type from memory).
- differentiable: partial - set assignment (not stated) and the PLGM snapping (post-process, no gradient).
- sensors_data: camera; OpenLane-V2 A and B; BEV 200 x 100.
- results: subset A ablation (TOP_ll / DET_p / OLS): reproduced baseline 23.4 / 44.5 / 43.4; +FVScale 23.8 / 44.8 / 46.0; +PLMSA 27.2 / 49.8 / 47.6; +PLGCN 28.0 / 51.8 / 48.3; +PLGM 28.7 / 52.6 / 48.8. Main: A OLS 48.8, TOP_ll 28.7 (TopoLogic 44.1, 23.9); B OLS 49.2, TOP_ll 28.3, DET_p 45.1 (TopoLogic 42.3, 21.6, 39.2). Query counts (N_p, N_l) 100/200 -> 200/200 -> 200/300 -> 300/300: DET_p 49.7, 51.8, 52.6, 51.4.
- compute: not stated.
- code: https://github.com/Franpin/TopoPoint (README: PyTorch 1.9.1, CUDA 11.1; no licence line; no checkpoints; data per OpenLane-V2 v1.0.0, metrics v2.1.0)
- fit_to_us: reuse_idea - effort M - principle (explicit vertices, distance-biased coupling of vertex and edge tokens, snapping) maps onto the key-point head and the DAG clustering; the query decoder itself does not.
- compat_needs: lane pieces with start / end; a point detector; incidence labels; Hungarian-type matching for the point set (not stated).
- compat_conflicts: needs a joint token decoder with self-attention (not a dense heat map); thresholds 0.3 / 1.5 m fixed; authors: relies on accurate calibration.
- tried_before: E44 (learned key points) kept; E113 / E116 (junction diagnosis, verifier) partial.
- verified: checked (paper and head file; digest flagged typos: Eq. 4 and 8, lambda_pl missing, init order of alpha / lambda ambiguous; paper text and released head file differ on the snapping step)

### CARD TP03 - CGNet (junction-aware queries, Bezier-space continuity, iterative adjacency; nuScenes centerline graph)
- papers: Continuity Preserving Online CenterLine Graph Learning (Han et al., ECCV 2024) https://arxiv.org/abs/2407.11337
- depth: M+C
- stage: S6 S1 L
- idea: Three additions to a MapTR-style centerline-segment decoder on nuScenes: (JAQ) a junction heat-map branch whose features enhance the positional part of the queries; (BSC) a training-only head that decodes the Bezier curve of a CONNECTED PAIR of segments from the concatenated embeddings (continuity at joints); (ITR) an adjacency matrix refined once per decoder layer by GCN + GRU.
- formulation: N = 50 queries (content Q^c + positional Q^p), each one segment of n = 20 ordered points; 6 deformable layers; GKT BEV. JAQ: F_jp = conv-BN stack of F_bev; heat map Z, focal loss with GT junction map dilated by radius R = 9; Q~p = LinearAttention(Q^p, F_jp, F_jp), concatenated with Q^c. BSC: for connected matched pairs (E_i, E_j): E^ = B(E_i concat E_j), B = pseudo-inverse of the Bernstein matrix (m x 2n); MLP predicts control points (default 4); L_bezier = L1 to least-squares-fitted control points of the joined GT polylines; removed at inference. ITR: Lambda^l = GCN(E^l, A^(l-1)); A^l = MLP(GRU(Lambda^l, Lambda^(l-1))); A^0 = 0; 6 iterations, weights shared (pair scoring inside the MLP not detailed). Loss: 2 L_cls + 5 L_poly + 1 L_topo + 0.005 L_dir + 0.01 L_bezier + 0.1 L_ja; Hungarian (focal + L1); L_topo = BCE averaged over the T^2 adjacency entries, no class balancing. CODE AS READ (CGNet_head.py): hierarchical MapTR-style point queries (num_query = vectors x points per vector, instance + point embedding); junction GT = the dataset's line.key_points rendered as 1 in a BEV-sized map, dilated with a 9 x 9 elliptical kernel (cv2), CornerNet-style modified focal loss (exponents 2 and 4), loss weight 0.1 (which points are "key points" is decided in the dataset code, not visible); topology branch per decoder layer: vertex feature = mean of the point embeddings of a query, one GNN step (adjacency and its transpose, edge weight 0.8), GRU across layers, then TopologyHead = per-vertex MLP + concatenation of every pair + association classifier + sigmoid (a pairwise MLP; the GRU does not output the matrix directly); loss = nn.BCELoss on the submatrix of MATCHED queries only (sm_pred[pos_inds][:, pos_inds] against gt[pos_gt_inds][:, pos_gt_inds]), weight 1, i.e. valid-pairs-only supervision (the setting RATopo later shows to be better than full supervision); BSC: all matched pairs (i, c) with target 1: concatenate the point embeddings of both queries (2n of them), multiply by the 4 x 2n pseudo-inverse Bernstein matrix (cubic, 4 control points) to get 4 control-point embeddings, MLP(C, C/2, 2) -> control points; inference returns the adjacency as is (threshold in the bbox coder, not seen).
- input: six cameras -> BEV 100 x 200 cells (0.3 m; x in [-15, 15] m, y in [-30, 30] m).
- output: 50 segments (20 points) + 50 x 50 adjacency probabilities; junction heat map (aux).
- training: Hungarian one-to-one; 24 epochs (ablations) or 110; AdamW 6e-4 cosine, batch 32, A100.
- differentiable: partial - Hungarian assignment; BSC and JAQ are differentiable.
- sensors_data: camera; nuScenes and Argoverse 2 centerline graphs (point-level metrics GEO / TOPO / JTOPO F1, APLS, SDA; segment-level IoU, mAPcf, DET_l, TOP_ll).
- results: nuScenes 24 ep (GEO F1 / TOPO F1 / JTOPO F1 / APLS / SDA): CGNet 54.7 / 42.2 / 34.1 / 30.7 / 8.8, MapTR 53.3 / 39.7 / 32.4 / 26.8 / 7.6, TopoNet 50.8 / 39.6 / 31.9 / 26.2 / 5.6; 110 ep CGNet 63.9 / 53.2 / 43.3 / 41.4 / 14.5. Module ablation (TOPO / JTOPO / APLS / SDA): base 39.7 / 32.4 / 26.8 / 7.6; +JAQ 40.5 / 33.1 / 28.5 / 7.7; +BSC 41.4 / 33.4 / 29.1 / 8.3; +ITR 40.1 / 32.7 / 28.6 / 7.8; all 42.2 / 34.1 / 30.7 / 8.8. Control points 4 / 6 / 8: TOPO 42.2 / 40.8 / 40.4. ITR iterations 1 / 3 / 6: 39.6 / 41.3 / 42.2. Dilation R 7 / 9 / 11: 40.4 / 42.2 / 40.9.
- compute: 38.8 M parameters, 241.9 GFLOPs, 11.2 FPS (RTX 3090); MapTR baseline 36.0 M, 216.7 G, 12.7 FPS.
- code: https://github.com/XiaoMi/CGNet (README: Apache-2.0, bundles an mmdetection3d folder, compiled GKT op; Python 3.8)
- fit_to_us: reuse_idea - effort S - junction heat map as feature enhancement is the principle of our key-point head; the Bezier-continuity auxiliary is the principle of our connectors (E53). The decoder and adjacency GRU are tied to a 50-query set.
- compat_needs: a set of matched segment embeddings (pairs S) for BSC; junction GT points; segments with ordered points.
- compat_conflicts: Hungarian one-to-one matching; N = 50 queries on 30 x 60 m only (cannot be assumed at +-50 m); BSC needs pairs of pieces, not vertices.
- tried_before: E44 (learned key points) and E53 (Bezier connectors): kept, large gain (split / merge F1 0.34 / 0.38 -> 0.71 / 0.64).
- verified: checked (supplement not opened; image resolution and pair-scoring details not stated; the paper does not say which points count as junctions (fork, merge, endpoint), whether R = 9 is in cells, or whether the heat map is output at inference; JTOPO radius not given; SDA = junction localisation within radius R)

### CARD TP04 - LaneGAP (path-wise modelling, Path2Graph)
- papers: Lane Graph as Path: Continuity-preserving Path-wise Modeling for Online Lane Graph Construction (Liao et al., ECCV 2024) https://arxiv.org/abs/2303.08815
- depth: M+C (landing page only, no technical content)
- stage: S7 S6 E
- idea: Instead of predicting pieces (segments between junctions) and then linking them, predict whole root-to-leaf PATHS of the directed lane graph with a MapTR-like decoder and recover the graph with a simple merge step; forks and merges then appear where paths share vertices, and no link head exists.
- formulation: Graph2Path (labels): find roots (in-degree 0) and leaves (out-degree 0), DFS to every root-to-leaf path. Path2Graph (inference): discretise each predicted path to points; add points as vertices and consecutive points as edges; merge vertices of different paths closer than 0.15 m (= the graph interpolation step); the merged vertex inherits predecessors and successors. Decoder: MapTR-style, 6 deformable layers, learnable path queries (count not stated), polyline default with 30 points (coordinates normalised to (0,1)); Bezier alternative B = Gamma x V (Bernstein matrix). Loss: Hungarian; L = L_cls (focal) + L_path (sum of L1 over matched points) + L_auxseg (CE on BEV segmentation, train only), unit weights as written. Metrics: TOPO (precision / recall / F1 of the directed graph; vertices every 0.15 m, match threshold 0.45 m, traversal radius 7.5 m) and Junction TOPO (same, restricted to sub-graphs around GT junctions).
- input: six cameras, ResNet-50 + GKT, BEV 200 x 100 at 0.3 m.
- output: set of ordered paths -> directed graph.
- training: Hungarian one-to-one; 110 epochs on nuScenes (AdamW 6e-4 cosine, batch 32, 8 RTX 3090), 24 epochs for ablations.
- differentiable: partial - Hungarian; Path2Graph is a rule (merge by distance), no gradient.
- sensors_data: camera (also LiDAR-only and fusion variants); nuScenes x in [-15, 15], y in [-30, 30] m; Argoverse 2; OpenLane-V2 (mIoU).
- results: nuScenes (Junction F1 / TOPO F1, directed): HDMapNet 0.507 / 0.513, STSU 0.419 / 0.395, MapTR 0.496 / 0.465, LaneGAP 0.564 / 0.529 (110 ep, 15.6 FPS). 24 ep ablations: aux seg 0.529 / 0.450 -> 0.539 / 0.461; polyline points 10 / 30 / 40 -> junction F1 0.532 / 0.539 / 0.557; Bezier points 3 / 5 / 10 -> 0.481 / 0.510 / 0.501. Modality 24 ep (junction / TOPO): vision 0.539 / 0.461, LiDAR 0.600 / 0.509, vision + LiDAR 0.615 / 0.531. Post-processing on GT inputs (Table 10): pixel-wise (HDMapNet type) 0.773 / 0.788, piece and path 1.000 / 1.000 (lossless). Large range 50 x 100 m: junction 0.392, TOPO 0.388. Planning with VAD: average L2 0.70 m (LaneGAP), 0.74 (MapTR), 0.78 (no lane graph).
- compute: 35.9 M parameters, 16.5 FPS network only / 15.6 FPS end-to-end.
- code: https://github.com/hustvl/LaneGAP (landing page only, Apache-2.0; the path-wise centerline configs are announced in the news of https://github.com/hustvl/MapTR, branch `maptrv2`, MIT per its README page, mmdetection3d based; branch not opened)
- fit_to_us: reuse_idea - effort S - Path2Graph is exactly our tree -> DAG conversion by clustering (0.15 m) and the Junction-TOPO metric is a ready junction-level evaluation next to our split / merge F1; Graph2Path gives path-wise labels from our piece graph.
- compat_needs: ordered point sequences per path; a merge threshold; paths covering every edge.
- compat_conflicts: number of root-to-leaf paths grows multiplicatively with forks (path count per scene not reported; unknown at +-50 m); a path view duplicates shared edges, so per-edge validity / route attributes need de-duplication after merging.
- tried_before: none (the tree -> DAG step of the hypothesis tree is the analogue; see `Hypothesis_Tree_Spec_and_Tests_EN.md`).
- verified: checked

### CARD TP05 - TopoFG (keypoint queries, boundary-point topology, matching-free denoising target)
- papers: Fine-Grained Representation for Lane Topology Reasoning (Xu et al., AAAI 2026) https://arxiv.org/abs/2511.12590
- depth: M+C
- stage: S1 S6 L
- idea: One query per lane KEYPOINT (11 per lane) instead of one per lane, built from a predicted BEV lane mask (where) and an ordinal in-lane embedding (which keypoint). Links are scored only between the last keypoint of lane i and the first of lane j. A training-only denoising branch gives an adjacency target that does not depend on the matching.
- formulation: HPE: learnable queries + Mask2Former-style module -> BEV mask M; weights A = M * I[M <= tau] + alpha * I[M > tau] (tau 0.3, alpha 1.0); Q_pos = norm(P^T A) + Q_R (P = sine-cosine grid encoding); Q_seq = MLP(PE(index)) + Q'; fine-grained query Q_F(i,t) = Q_pos(i) + F(Q_seq(t)). RFD: 6 layers; inter-instance then intra-instance self-attention; deformable cross-attention with reference points sampled from the predicted mask (rule not given). RBTR: r_{i->j} = concat(f_i^end, f_j^start); S = sigmoid(MLP(r)); geometric topology = TopoLogic mapping of boundary-point distances; adjacency = S + geometric. Hungarian one-to-one matching places the GT adjacency into the query-index matrix (target moves with the matching); denoising: G = 5 groups of noisy GT-derived queries with the fixed block-diagonal GT adjacency; inference uses the vanilla queries only. Individual loss terms and weights are not reported. CODE AS READ (topofg_head.py): 11 keypoints per lane via nn.Embedding(11) + a sequence positional encoding, fine-grained queries = instance query (detached) + per-keypoint embedding, flattened to lanes x 11; positional map = sine grid encoding weighted by the mask probability where it is below 0.3, normalised by the maximum; reference points sampled from the mask thresholded at 0.3 (sampling rule in a file not shown); mask loss = Dice + a per-pixel term with its own matching of queries to GT masks (types set in the config); lane losses = softmax CE (class weight 1.0, background 0.1) + L1 (5.0) with the same Hungarian costs as TopoLogic (1.0 / 5.0 / GIoU 2.0, GIoU never used in the loss); DN losses weighted 1.0, topology DN target = the same GT adjacency repeated on the diagonal blocks of the G groups; at inference the lane-lane score is the TopoLogic form: L1 distance last keypoint of i -> first keypoint of j, exp(-d^2 / 11.5275), lamda 1 + 1, sigma unused. So the released method is TopoLogic plus keypoint queries, mask prior and denoising; the boundary-point MLP and DN loss type sit in config files not seen.
- input: multi-scale BEV (grid not stated), 200 lanes x 11 keypoints.
- output: 200 x 11 keypoints (3D lanes) and an N x N adjacency.
- training: Hungarian for vanilla queries, fixed target for denoising queries; 24 epochs, AdamW 2e-4 cosine, 500 warm-up steps, 8 A100, R50 + FPN; input 800 x 1024 (A, 7 views) / 480 x 800 (B, 6 views).
- differentiable: partial - Hungarian for vanilla queries; the denoising branch needs no assignment.
- sensors_data: camera; OpenLane-V2 A and B; lane-segment table also reported.
- results: ablation subset A (OLS / DET_l / TOP_ll / TOP_lt): TopoLogic 44.1 / 29.9 / 23.9 / 25.4; +HPE 45.4 / 31.3 / 26.1 / 26.6; +RFD 45.8 / 31.8 / 26.8 / 27.7; +RBTR 48.0 / 33.8 / 30.8 / 30.9. RBTR parts (TOP_ll): neither 26.8, boundary topology 28.3, denoising 29.0, both 30.8. Main: A 48.0 / 33.8 / 47.2 / 30.8 / 30.9; B 45.4 / 30.0 / 53.0 / 27.2 / 21.7 (OLS / DET_l / DET_t / TOP_ll / TOP_lt). Lane segment: mAP 34.4, TOP_ls 31.3.
- compute: not reported.
- code: https://github.com/GXmmm18/TopoFG (README: PyTorch 1.9.1, CUDA 11.1; no licence line; subset-A checkpoint on Google Drive; builds on TopoLogic / TopoNet / MapTR per acknowledgements)
- fit_to_us: reuse_idea - effort M - a keypoint-per-token query set resembles our vertex tokens; the ordinal embedding, inter / intra attention and the fixed denoising adjacency are small, matching-free ideas.
- compat_needs: ordered keypoints along every lane; a mask prior; deformable attention over BEV.
- compat_conflicts: vanilla branch needs Hungarian; mask supervision unspecified; mm-lineage code.
- tried_before: E44 / E53 (key points, connectors) kept; the dense heat map as spatial prior already exists in the dense head.
- verified: partly (paper + head file read; loss types of topology / mask / DN, noise scale, mask GT rasterisation width and the reference-point sampling rule are in files or configs not seen; Table 4 checkmarks ambiguous)

### CARD TP06 - UniTopo (connected lanes as unified queries)
- papers: Unified Modeling of Lane and Lane Topology for Driving Scene Reasoning (Li et al., IEEE TCSVT 2026) https://arxiv.org/abs/2605.08911
- depth: M+C
- stage: S6 S4 T
- idea: Topology is not derived from pairs of detected pieces. A CONNECTED LANE (end of piece i joined to start of piece j, resampled to N_P points) is a geometric object like any other lane, predicted by its own 300 "connection queries" through the same lane head. A link then exists where a connection query matches a pair of piece queries, found by nearest average L1 distance of the first / second half to the pieces.
- formulation: lane queries 300 and connection queries 300 (random learnable, no anchors), shared head (existence + N_P x 3 regression). GT connected lane = merge of the two GT pieces at the junction, resampled; Hungarian matching per query group with the same cost as the loss; training uses K = 6 replicated query groups (one-to-many style, shared at inference). TAM (topology-aware attention): D = min(f_g(L, Lc_first), f_g(L, Lc_second)), f_g = average L1; S = sigmoid(MLP(D)); A = softmax(f_q(Q) f_k(Qc)^T / sqrt(C) + log S); Q~ = LN(A f_v(Qc) + Q). Pair score: Q_ll_ij = concat(MLP(Qc_ij + Q_i), MLP(Qc_ij + Q_j)); G_ij = sigmoid(MLP(Q_ll_ij)); unmatched pair uses only the two lane features. Loss L = 1.0 L_lane + 1.0 L_traffic + 5.0 L_ll + 5.0 L_lt (focal for topology; lane cls 1.5, reg 0.025). CODE AS READ (unitopo_head.py): 200 lane queries per group (default), 11 points per lane (33 values for 3D); the connection GT is built by concatenating the previous lane with the next lane minus its first point and keeping every second point (11 points); connection labels are all class 0; Hungarian assigner with focal cost 1.5 and L1 cost 0.025; the pairing of a connection query to its two pieces is done in the loss with torch.cdist (p = 1, sum not average) between 6 sub-sampled lane points and the first 6 / last 6 points of the path, taking the minimum for each half; lane classification focal (alpha 0.25, gamma 2); the relation loss classes are in the relation-head file (not seen); the relation losses of the one-to-many group are multiplied by zero in the code (the extra groups supervise detection only); the inference decoding (score and adjacency thresholds) is in the bbox coder (not seen); GeoDist is not in the head file.
- input: ResNet-50 / FPN, 3-layer BEVFormer-style encoder, BEV 200 x 100 over x +-51.2 m, y +-25.6 m.
- output: piece polylines, connected-lane polylines, link probabilities (edge threshold not stated).
- training: Hungarian per group; 24 epochs (48 for the long Swin-B run), AdamW 2e-4, batch 8, 8 A100.
- differentiable: partial - Hungarian and the argmin that pairs a connection query with two pieces (hard selection).
- sensors_data: camera; OpenLane-V2 A and B; lane segment too (TOP_lsls 31.4, mAP 34.4 vs LaneSegNet 25.4 / 32.6).
- results: subset A ablation (TOP_ll / TOP_lt / DET_l / DET_t / OLS): TopoNet 10.9 / 23.8 / 28.6 / 48.6 / 39.8; matched-parameter baseline 12.4 / 26.0 / 30.0 / 50.4 / 41.6; +connection queries 23.3 / 27.9 / 29.4 / 49.4 / 45.0; +TAM 25.0 / 28.8 / 30.2 / 51.1 / 46.2; +groups 29.6 / 31.2 / 29.8 / 49.3 / 47.3; +GeoDist (TopoLogic) 30.1 / 31.4 / 31.4 / 52.4 / 48.7. Queries 200 / 300 / 400: TOP_ll 27.0 / 30.1 / 31.6. Groups 4 / 6 / 8 / 10: 28.4 / 30.1 / 29.8 / 30.0. BEV: 1 layer 200 x 100 26.2, 3 layers 100 x 50 29.2, 3 layers 200 x 100 30.1. Subset B R50: TOP_ll 31.8, OLS 47.6. Swin-B 48 ep: A TOP_ll 35.1, OLS 51.0; B 40.7, 54.6.
- compute: 67.70 M parameters, 77.05 GFLOPs, 4.21 FPS (TopoNet 65.96 M, 63.07 G, 4.57 FPS).
- code: https://github.com/homothetic/UniTopo (README: Python 3.8.10, CUDA 11.8; no licence line; built on TopoNet, credits TopoMLP / TopoLogic; weights for A and B)
- fit_to_us: reuse_idea - effort L - "predict the turning movement (two consecutive edges through a vertex) as one object" is the opposite of pairwise link scoring; it is the closest published relative of growing edges with fork / merge events, but requires a query decoder with Hungarian groups.
- compat_needs: pair-level GT objects; query groups; hard pairing step.
- compat_conflicts: Hungarian matching; 600 + 100 queries; code is a TopoNet derivative.
- tried_before: none directly; E42 .. E56 (pairwise heads) point the same way (pairwise scoring adds nothing over geometry).
- verified: checked (paper + head file; N_P = 11 from the code, decoder depth and the edge threshold are in files / configs not seen)

### CARD TP07 - TopoEnhance (post-hoc graph refinement of link scores, noise-simulating training)
- papers: Refine Connections, Close the Gap: A Reliable Enhancement Framework for Driving Scene Topology (Wang et al., Bosch / MIT, arXiv 2610.11058, 12 Oct 2026) https://arxiv.org/abs/2610.11058 (the digest calls the method "TopoEnhance")
- depth: M
- stage: S6 L E
- idea: Raw pair scores of lane topology methods are badly calibrated, so the thresholded edge set (tau = 0.5) is poor even when TOP (a ranking metric) looks good. A small heterogeneous graph-attention network is trained to re-score ALL pairs of detected nodes from node geometry (and traffic-element embeddings), on training data in which ground-truth nodes are corrupted by simulated false positives and jitter (adaptive curriculum). It is applied after any trained detector without retraining it.
- formulation: nodes: lanes (3D polylines) and traffic elements (2D boxes + DINOv2 embedding), candidate edges = all pairs (the detector's adjacency is not used). Forward process: false positive v_fp = v + eps with dist > delta, eps ~ N(0, sigma_fp^2 I); jitter v' = v + eps with dist <= delta; edges touching a false positive get label 0; curriculum: sigma_tp rises from 1e-5 toward delta, sigma_fp falls from 10 toward delta. Reverse: two-layer GATv2 (hidden 64, 8 heads, dropout 0.1) with relation set {ll, lt}; refined score s~ = sigmoid(cos(h_u, h_v)). Fusion: s_final = w_r s~ + (1 - w_r) s_raw (w 0.6 for lane-traffic, 0.6 .. 0.9 for lane-lane). Loss: BCE per relation, normalised by the number of candidate edges of that relation (so the larger lane-lane set does not dominate); no Hungarian in training. New metric TJS = |E* intersect E^| / |E* union E^| on thresholded edges; Upper-Bound = oracle edges between true-positive detections.
- input: detected lanes (polylines), traffic elements, raw pair scores.
- output: calibrated pair scores and thresholded edge set.
- training: noise-simulation on GT graphs, AdamW 1e-3, batch 64, cosine to 1e-4, one H200 about 16 h; one model per dataset.
- differentiable: yes (small GNN) - the threshold at inference is a rule.
- sensors_data: any detector output; OpenLane-V2 A and B (v2.1 metrics).
- results: TOP_ll before -> after (upper bound), subset A: TopoNet 10.9 -> 21.8 (24.4), TopoMLP 21.6 -> 24.4 (25.0), TopoLogic 23.9 -> 24.5 (26.0), SMART 37.0 -> 40.1 (40.9). TJS_ll (A): TopoNet 16.3 -> 32.9 (44.3), TopoMLP 18.5 -> 59.7 (69.7), TopoLogic 21.0 -> 43.9 (50.9), SMART 38.1 -> 87.7 (92.9). Subset B: TopoNet TOP_ll 6.7 -> 19.5, TJS_ll 9.0 -> 42.3. Appendix ablations (subset A, TopoNet host, TOP_ll / TOP_lt): plain BCE 12.1 / 22.0, BCE + focal 22.2 / 22.0, per-relation normalised BCE 21.8 / 25.8 (so the normalisation, not the focal term, is what matters); without geometric jitter 21.7 / 25.4; traffic-element embedding backbone irrelevant (DINOv2-S 20.3 .. ViT-L 21.84, ResNet-50 21.8). Ablations of the false-positive synthesis, curriculum, fusion weight and GAT depth were not found in the part read.
- compute: GAT about 0.1 ms per sample in the main text; stress test 200 lanes + 100 traffic elements = about 60,000 candidate pairs: 6.43 ms on one H200 (message passing 5.2 ms), +0.3 MB memory (the appendix breakdown does not sum exactly).
- code: none found for TopoEnhance (only TopoPoint's repo is linked as a baseline)
- fit_to_us: reuse_idea - effort M - train the junction / piece verifier (E113 / E116) and link head on SIMULATED corruptions of ground-truth vertex sets (false vertices, jitter, curriculum) with per-relation normalised BCE; add TJS-like edge-set Jaccard to our metrics.
- compat_needs: a set of nodes with geometry; pairwise candidate edges; a threshold.
- compat_conflicts: refines only pairs, not geometry; evaluated on query-decoder detections with many false positives (our learned graph has 1.8 x too many junctions, similar regime).
- tried_before: E113 / E116 (junction / piece verifier on the learned graph): partial (split / merge 0.385 / 0.428 -> 0.437 / 0.504).
- verified: partly (appendix read from offset 100000: loss, jitter and backbone ablations seen; definitions of TJS / Upper-Bound and the false-positive / curriculum ablations not found in the digest)

### CARD TP08 - TopoMLP (MLP link head; oracle-topology experiment; metric loophole)
- papers: TopoMLP: A Simple yet Strong Pipeline for Driving Topology Reasoning (Wu et al., ICLR 2024) https://arxiv.org/abs/2310.06753 ; The 1st-place Solution for CVPR 2023 OpenLane Topology in Autonomous Driving Challenge https://arxiv.org/abs/2306.09590
- depth: M
- stage: S6 E
- idea: A strong 3D lane detector (PETR style, learnable 3D anchors, each a 4-control-point Bezier) and two plain MLP topology heads on the last-layer queries. The paper's main message is that topology quality is bounded by detection quality: giving the true topology to the predicted lanes barely changes the score.
- formulation: lane queries Q^L = 300 learnable 3D anchor points -> 256-d by position encoding + linear; 6 decoder layers; offsets and class by MLPs; 11 points per lane for training. LL head: an MLP embeds the predicted lane coordinates and adds them to the query features; features are repeated along two axes and concatenated to N x N x 2C; three linear layers with ReLU give N x N logits G_LL. LT head: lane query + MLP of the 3 x 3 view-transform matrix, concatenated with traffic queries, MLP. Loss: L = L_detl + L_dett + L_topll + L_toplt; lane focal 1.5 + L1 0.2; LL topology: focal weight 5 plus an extra L1 term (0.1) on matched lane points; LT focal 0.5. Matching only for detection, reused for topology. Metric loophole (old metric): unmatched false positives get default confidence 1.0 and rank above true positives; pushing scores to 0 / 1 raised TopoMLP's TOP_ll from 7.2 to 19.0 (TopoNet 4.0 to 11.5); the paper proposes weighting each precision term by N_TP / (N_TP + N_FP). The devkit fix (issue #76, .1 releases) was triggered by this paper but is implemented differently as far as the code digest shows (unmatched predicted lanes are dropped from the matrix, see TP29).
- input: multi-view images -> backbone -> PETR-style 3D position-embedded features (no BEV encoder); lane range x +-51.2, y +-25.6, z [-8, 4] m.
- output: up to 300 lanes + N x N topology matrix.
- training: Hungarian; 24 epochs (48 for one run), AdamW 2e-4 cosine, batch 8, 8 A100, 1550 x 2048 downsampled x0.5; HSV and GridMask augmentation; optional YOLOv8 proposals for traffic elements.
- differentiable: partial - Hungarian assignment.
- sensors_data: camera; OpenLane-V2 A and B.
- results: oracle-topology table (v1 metric, subset A, TOP_ll predicted -> with GT topology): R50 7.2 -> 7.5, VoV 7.9 -> 8.5, Swin-B 9.5 -> 10.0; OLS 38.2 -> 38.3 / 40.1 -> 40.8 / 42.2 -> 43.1. Ablations (R50, DET_l / TOP_ll / OLS): queries 200 / 300 / 500: 28.2 / 6.1 / 36.9, 28.3 / 7.2 / 38.2, 27.9 / 7.3 / 38.0; control points 3 / 4 / 5: 26.6 / 7.0 / 37.3, 28.3 / 7.2 / 38.2, 27.8 / 6.6 / 37.1; LL head without position input 6.9 (OLS 37.9), without the L1 term 6.5 (OLS 37.5). Subset A OLS: R50 38.2, with YOLOv8 41.2, Swin-B 42.2 (24 ep) / 43.7 (48 ep); subset B R50 38.7. In the v2.1 table of other papers TopoMLP has TOP_ll 21.7, OLS 44.1.
- compute: not stated.
- code: https://github.com/wudongming97/TopoMLP (not opened)
- fit_to_us: baseline - effort S - the head is a plain pair MLP; the oracle experiment is the argument why the link head stays small.
- compat_needs: query embeddings and predicted coordinates per lane; Hungarian matching for labels.
- compat_conflicts: PETR-style 3D position embedding needs perspective features (not available on a cached BEV raster).
- tried_before: E42 (learned link head) within 0.01 of rules, dropped.
- verified: checked

### CARD TP09 - TopoNet (scene-graph GNN with lane-lane and lane-traffic relations; TOP metric)
- papers: Graph-based Topology Reasoning for Driving Scenes (Li et al., arXiv 2304.05277) https://arxiv.org/abs/2304.05277
- depth: M
- stage: S6 E
- idea: Lane and traffic-element queries are nodes of a scene graph; a GNN with relation-type-specific weights passes messages along the previous layer's predicted adjacency (self loop, predecessor, successor channels), and an MLP head scores ordered pairs.
- formulation: BEV 200 x 100 (BEVFormer-like, 3 layers, no temporal input); 200 centerline queries (11 x 3 points) and 100 traffic queries; initial adjacency T_ll^0 = I; T_ll^i = beta (A + A^T) + I, beta 0.6; GCN Q' = sigma(T Q W); knowledge-graph variant: K_ll = stack(A, A^T, I), per-class weights W_ll(c), messages weighted by traffic-element class scores; update Q_l~ = Q_l + downsample(ReLU(concat(Q_l', Q_l''))). Topology head: Q~a' = MLP_a(Q~a), Q~b' = MLP_b(Q~b) (128 ch), conf = sigmoid(MLP_top(concat)), separate MLP per relation (asymmetric for lane-lane). Loss L_top = 5 focal(lane-lane) + 5 focal(lane-traffic); focal as the only imbalance remedy (alpha, gamma not given); Hungarian matching assigns GT topology to matched query pairs; supervision every layer.
- input: BEV features, query sets.
- output: N x N and N x M adjacency probabilities.
- training: 24 epochs, AdamW 1e-4 cosine, batch 8, 8 A100, 2048 x 1550 x0.5; range x +-50, y +-25 m.
- differentiable: partial - Hungarian.
- sensors_data: camera (front view for traffic); OpenLane-V2 A and B.
- results: ablations (old metric, subset A; DET_l / TOP_ll / OLS): MLP baseline 25.7 / 4.0 / 34.6; +scene graph 27.7 / 3.7 / 35.0; +knowledge graph 28.5 / 4.1 / 35.6. GNN layers 1 / 2 / 3 / 4: OLS 35.6 / 35.3 / 28.3 / 22.6 (TOP_ll collapses to 0.5 and 0.0 at 3 and 4 layers). beta 0.5 / 0.6 / 0.7: 35.4 / 35.6 / 35.1. v2.1 values quoted elsewhere: TOP_ll 10.9, OLS 39.8 (A); TOP_ll 6.7 (B). Metric: TOP = vertex-level mAP - for each GT vertex the predicted neighbours are ranked by confidence and precision is accumulated over hits, normalised by the true neighbour count; edges with confidence > 0.5 count as positive.
- compute: not stated (62.6 M parameters in Reusing-Attention's table).
- code: https://github.com/OpenDriveLab/TopoNet (not opened)
- fit_to_us: baseline - effort S - GNN message passing on a link head added nothing measurable in its own ablation (TOP_ll 3.7 -> 4.1) and breaks at depth 3.
- compat_needs: per-lane query vectors; predicted adjacency from the previous layer.
- compat_conflicts: set-prediction pipeline; deeper GNN diverges.
- tried_before: E46 / E51 (node-conditioned head, path repair): within 0.01 of rules, dropped.
- verified: checked

### CARD TP10 - LaneSegNet (lane segments, lane attention, identical initialisation)
- papers: LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving (Li et al., ICLR 2024) https://arxiv.org/abs/2312.16108
- depth: M+C
- stage: S2 S6 D
- idea: The output unit is a LANE SEGMENT (centerline + left and right boundary + boundary types + successor graph), which unifies centerline and map-element tasks. Cross-attention is replaced by "lane attention": the reference points are spread along the predicted segment and each head owns one region; the first decoder layer starts all heads from one shared reference point.
- formulation: segment = 3 lines x 10 3D points; boundaries V_left = V_center + V_offset, V_right = V_center - V_offset; types non-visible / solid / dashed. Lane attention: LaneAttn(q, p, B) = sum_m W_m [ sum_k a_{m,k} W'_m Bilinear(B, p_m + dp_{m,k}) ], M = 8 heads, K = 32 offsets, reference points 4 along each boundary (n_ref = 8); offsets and weights by linear layers from q. Identical initialisation: in layer 1 every head uses the same reference point from the learnable positional query; later layers use the previous prediction. Topology: q'_i = MLP_pre(q_i), q'_j = MLP_suc(q_j), A_ij = sigmoid(MLP_top(concat)) from query features only. Mask branch: S = sigmoid(E_mask (x) B). Loss L = 0.025 L_vec + 3.0 L_seg(CE + dice) + 1.5 L_cls + 0.01 L_type + 5.0 L_top; Hungarian (or Sinkhorn); topology focal. CODE AS READ (lane_attention.py): the kernel is the ordinary multi-scale deformable-attention op (BEVFormer's fp32 CUDA wrapper on GPU, fallback multi_scale_deformable_attn_pytorch from mmcv.ops, which is grid_sample based); reference points are an input that is broadcast over the 32 sampling points of a head, so the "lane attention" novelty sits in the decoder that builds per-head reference points from the predicted segment (laneseg_decoder.py: per layer a centerline regression and an offset regression, boundaries = centerline +- offset, each boundary sub-sampled at indices [0, 3, 6, 9] of its 10 points, giving 4 left + 4 right = 8 reference points (x, y only), detached between layers, refinement additive in logit space; the assignment of the 8 points to the 8 heads happens inside the attention module, not seen); offsets are linear outputs divided by the feature-map size, initial offset bias = 8 radial directions scaled by point index, zero weights; attention weights softmax jointly over levels and points; value projection, output projection, dropout + residual.
- input: R50 + FPN, 3 BEVFormer layers, BEV 200 x 100 (+-50 x +-25 m); 200 queries; 6 decoder layers.
- output: 200 lane segments + adjacency.
- training: Hungarian; 24 epochs, AdamW 2e-4 cosine, batch 8, 8 V100; 14.7 FPS on A100.
- differentiable: partial - Hungarian.
- sensors_data: camera; OpenLane-V2 subset A only (lane-segment labels exist only there; nuScenes and Waymo lack them, per the paper).
- results: formulation ablation (old metric): centerline only DET_l 22.9 / TOP_ll 3.7, lane segment 27.2 / 6.6. Cross-attention designs (mAP / TOP_lsls): deformable (instance) 28.7 / 6.9, deformable (hierarchical) 29.1 / 5.2, multi-point 24.6 / 6.9, lane attention 32.6 / 8.1. Decoder rows: heads-to-regions + identical init 32.6 / 8.1 vs 28.8 / 5.6 without either (the text credits identical init with +3.5 mAP). Mask supervision +2.0 AP_ls. Paper TOP_lsls 8.1 is v2.0 metric; v2.1 value 25.4 (README: mAP 33.5, TOP_lsls 25.4).
- compute: 14.7 FPS (A100); tiny variant mAP 28.5 at 16.2 FPS; 9.4 G memory (README).
- code: https://github.com/OpenDriveLab/LaneSegNet (README: Apache-2.0; Python 3.8, CUDA 11.1, PyTorch 1.9.1, mmcv-full 1.5.2, mmdet 2.26.0, mmsegmentation 0.29.1, mmdet3d 1.0.0rc6; subset A configs and checkpoint)
- fit_to_us: reuse_idea - effort M - sampling reference points along the element and one region per head matches our ray / ring sampling; the attention op itself is plain deformable attention and has a pure-PyTorch form (grid_sample), so only the decoder logic that places per-head reference points needs re-writing; the repository as a whole cannot be used (exact mm-series pins incompatible with PyTorch 2.8).
- compat_needs: predicted element geometry per layer to place reference points; a query decoder.
- compat_conflicts: Hungarian; boundaries only exist in subset A; adapting needs mmcv ops (deformable attention) or a plain-PyTorch grid_sample version.
- tried_before: E31 / E32 (sparse DETR-style decoder with matching vs dense grid): dropped.
- verified: checked

### CARD TP11 - Lane Field (dense BEV direction-tensor field as conditioning and sampling guide)
- papers: Learning Lane-Structured Conditioning Fields for Online HD Mapping (arXiv 2610.25284, 18 Oct 2026; authors not seen) https://arxiv.org/abs/2610.25284
- depth: M
- stage: S2 B X
- idea: A dense BEV field R (H x W x 5) with a SIGN-FREE direction tensor M = sum_i I_i t_i t_i^T (xx, xy, yy), a proximity P and a confidence C, supervised from analytic ground truth. It (a) modulates the BEV features SPADE-style and (b) stretches the tangential component of deformable sampling offsets along the local lane direction. Confidence C = anisotropy of the tensor is low where lane directions cross, so guidance is switched off at intersections.
- formulation: per segment i (endpoints A_i, B_i, tangent t_i, mean width w_i): I_i(p) = (sigma_i sqrt(2 pi)/2) exp(-d_perp^2 / (2 sigma_i^2)) [erf(u1) - erf(u0)], sigma_i = 0.5 w_i; M = sum_i I_i t_i t_i^T; P = sum_i I_i; C = Delta / tr, Delta = sqrt((Mxx - Myy)^2 + 4 Mxy^2). Targets: P_gt = P / (P + 1), M_gt = M / (tr M + eps) * sigmoid(k (P_gt - 0.3)), k = 20. FiLM: F_out = gamma * F_in + beta with gamma, beta from convs on the field. Guidance: s_new = r + [1 + g_C g_U g_H (Delta_par - 1)] s_par t^ + s_perp n^ with Delta_par = 4.0 and sigmoid gates g_C = sigmoid(10 (C - 0.3)), g_U (uncertainty), g_H (per head). Loss 5.0 Smooth-L1 on tensor + 2.0 L1 on finite differences + 1.0 Laplacian beta-NLL on proximity (beta 0.5). Matching: not described.
- input: BEV features (plug-in), field head on the same grid.
- output: field tensor; modulated features; guided sampling offsets.
- training: with the host model's loss; OpenLane-V2 24 epochs; AV2 / nuScenes with MapTRv2 (24 epochs nuScenes, +-30 x +-15 m).
- differentiable: yes - the whole field path; ablation shows detaching it from the BEV encoder loses accuracy.
- sensors_data: camera; OpenLane-V2 subset A (BEV 100 x 200, +-50 x +-25 m), Argoverse 2, nuScenes; SD-map variant with fixed sigma.
- results: LaneSegNet host (mAP / AP_ls / AP_ped / TOP_lsls): baseline 32.1 / 32.2 / 31.9 / 25.5; +guidance 33.2 / 33.4 / 32.9 / 26.8; +modulation 34.9 / 35.7 / 34.0 / 27.9; both 36.1 / 35.8 / 36.4 / 28.4; with SD map 36.8 / 36.6 / 37.0 / 29.7. Gradient flow (mAP): stop-gradient in and out 34.1, fully attached 36.1. Plug-ins: MapTRv2 nuScenes 55.8 -> 57.7 mAP; TopoPoint centerline TOP_ll 30.0 / OLS 43.8 (its own table).
- compute: +7.2 M parameters (4.8 head, 2.4 SPADE); LaneSegNet 14.6 -> 13.9 FPS, 1.3 -> 1.5 TFLOPs.
- code: project page https://andberto.github.io/LaneField/ (no repository link)
- fit_to_us: reuse_idea - effort M - our dense head already predicts heat map + direction (2 ch); the tensor form removes the sign ambiguity of a direction vector, gives a free junction-confidence channel and can condition the cached features before the trace network; the tangent-stretched sampling is the same idea as our long, narrow forward ray.
- compat_needs: a dense head on the BEV grid and supervision from piece geometry (analytic, no matching).
- compat_conflicts: the reported gain needs the field gradient to reach the BEV encoder (frozen in our setting, so only the modulation of the cached features is possible); uses width annotations (we have no lane width in the centreline labels except the map expansion polygons).
- tried_before: none for the tensor form (dense head heat + direction exists; E44 key points).
- verified: partly (last 2.3 k characters of the page unread; matching not described)

### CARD TP12 - RATopo (denser topology supervision through one-to-many assignment)
- papers: RATopo: Improving Lane Topology Reasoning via Redundancy Assignment (arXiv 2508.15272) https://arxiv.org/abs/2508.15272
- depth: M
- stage: L S4 T
- idea: In one-to-one matched training only the few matched queries carry topology labels, so the topology head sees very few valid pairs. Three changes (all but one training-only): reorder each decoder layer to cross-attention before self-attention, add parallel cross-attention blocks, and supervise topology also on an auxiliary one-to-many assignment (K = 3 predictions per GT lane).
- formulation: layer CA -> SA -> FFN; M = 4 parallel cross-attention blocks with independent weights, merged by a linear layer; one-to-many matching per block (cost not described), K = 3. Label under redundant assignment: Z_rs = G_pq if r in the K predictions of GT lane p and s in those of lane q, else 0. L = L_det(o2o) + L_topo(o2o) + 2.0 L_topo(o2m); topology weights 5.0, focal.
- input: BEV 200 x 100, 300 lane + 100 traffic queries, R50, image 1024 x 775.
- output: unchanged host outputs (the o2m branch is dropped at inference).
- training: 24 epochs, AdamW 2e-4, batch 8, 8 A100.
- differentiable: partial - matching steps as in the host.
- sensors_data: camera; OpenLane-V2 A and B; lane segments (LaneSegNet).
- results: with TopoLogic (subset A, TOP_ll / TOP_lt / DET_l / OLS): original 23.9 / 25.4 / 29.9 / 44.1; valid-pairs-only supervision 27.2 / 28.2 / 30.1 / 46.4; +reorder 28.2 / 27.9 / 30.8 / 46.7; +lane o2m 30.5 / 31.5 / 30.3 / 48.0; full 32.2 / 33.9 / 31.8 / 49.0. Positives per lane 1 / 2 / 3 / 4 / 5: TOP_ll 30.4 / 32.2 / 32.2 / 31.7 / 30.8. Naive o2m gives nothing (27.0), group-DETR style 29.8. Hosts: TopoNet TOP_ll 10.9 -> 29.5 (A), 6.7 -> 33.3 (B); TopoLogic B 21.6 -> 37.3; LaneSegNet TOP_lsls +5.9 (baseline row ambiguous).
- compute: reorder and o2m add no inference cost; the parallel cross-attention blocks add parameters (no numbers).
- code: https://github.com/homothetic/RATopo
- fit_to_us: reuse_idea - effort S - confirms that link-head quality is limited by the NUMBER of labelled valid pairs; in our matching-free setting every predicted vertex can be paired with the ground truth through geometry, which is the dense end of this spectrum.
- compat_needs: any query-based host with a topology head.
- compat_conflicts: concept lives inside Hungarian assignment; nothing to port except the label-density argument.
- tried_before: none (E42 .. E56 used matching-free labels).
- verified: checked

### CARD TP13 - CoPo (geometry-biased attention and relational contrastive loss)
- papers: Geometry-Guided Representations for Coherent Lane and Traffic Topology Reasoning in Driving Scenes (CoPo, KDD 2026, arXiv 2506.13553 v4) https://arxiv.org/abs/2506.13553
- depth: M
- stage: S6 L S2
- idea: Geometry enters three places: an attention bias from endpoint distance and relative angle between lane pairs, deformable reference points sampled along each lane's Bezier curve, and a distance embedding in the lane-lane head. A contrastive loss with hard negatives is added to the focal topology loss.
- formulation: Dist(l_i, l_j) = min over x, y in {start, end} of ||p_i^x - p_j^y|| (symmetric); Angle(l_i, l_j) (formula not given); Geometry = GE(Dist || Angle) (sinusoidal encoding + linear); Attention = softmax(QK^T / sqrt(d) + Geometry) V. Curve-guided cross-attention: K = 11 points sampled on the cubic Bezier as reference points, one shared query predicts offsets and weights. L2L head: G = (MLP1(Q) + PE_pre) * (MLP2(Q) + PE_suc), DistEmbed(i,j) = MLP(P_i^end - P_j^start) (directed), T = MLP(G + DistEmbed). Relational contrastive loss L_con = log(1 + sum_{v+} sum_{v-} exp(v- - v+)); connected pairs positive, hard negatives = top-n by logit (n not stated), groups of one positive and three negatives; L_relation = 5 L_focal + 0.1 L_con (training only).
- input: BEVFormer BEV 100 x 200, 300 lane queries, 6 layers, R50, 1024 x 800.
- output: lane Beziers, N x N lane-lane and lane-traffic matrices.
- training: 24 epochs, AdamW 2e-4, batch 8, 8 RTX 4090; matching not described (annotated connected lanes share endpoints).
- differentiable: partial - as host.
- sensors_data: camera; OpenLane-V2 A and B.
- results: ablation subset A (DET_l / DET_t / TOP_ll / TOP_lt / OLS): baseline 27.7 / 50.1 / 21.8 / 28.4 / 44.5; +geometry self-attention 28.1 / 48.8 / 24.7 / 28.3 / 44.9; +curve-guided cross-attention 32.7 / 50.1 / 26.8 / 29.9 / 47.3; +L2L head 33.7 / 48.0 / 28.5 / 29.7 / 47.4; +L2L +L2T 33.9 / 50.3 / 28.6 / 30.6 / 48.2; +contrastive 33.8 / 50.9 / 29.2 / 32.2 / 48.9. TopoLogic without its GeoDist L2L term: TOP_ll 11.6, OLS 40.4 (vs 23.9 / 44.1). Adding the fixed TopoLogic mapping on top of CoPo's own geometry encoding does not help: + GDT_L2L TOP_ll 28.7 (-0.5), + GDT_lane 28.4. Main A: OLS 48.9, B: 49.7 (TOP_ll 29.2 / 31.8). Bezier comparison: BeMapNet 46.2, BezierFormer 46.5, CoPo 48.9 OLS. Breakdown by distance: DET_l near (< 25 m, 45 % of GT) 34.4, far 30.2. Authors' future work: temporal continuity.
- compute: latency 159.7 -> 162.2 ms, model 218.7 -> 224.8 MB.
- code: https://doi.org/10.5281/zenodo.20455469 (Zenodo release as stated in the paper)
- fit_to_us: reimplement - effort S - symmetric endpoint-distance + angle bias and a hard-negative contrastive loss are both matching-free once positives are defined by geometry.
- compat_needs: link candidates with positions and headings; a labelled positive set.
- compat_conflicts: hard-negative count n unspecified; training-only loss tuned for 300-query sets.
- tried_before: E45 (TopoLogic-style scoring) and E42 .. E56: no gain over rules.
- verified: partly (appendix read; hard-negative count n, temperature, Angle formula and K / M / N of the deformable attention are not in the text read)

### CARD TP14 - T2SG / TopoFormer (distance bias in self-attention; counterfactual layer)
- papers: T2SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving (Lv et al., CVPR 2025) https://arxiv.org/abs/2411.18894
- depth: M
- stage: S6
- idea: Lane features are mixed by self-attention whose logits get an additive spatial prior from the distance end_i -> start_j; a "counterfactual intervention" replaces the learned attention by zeros and trains on the difference of factual and counterfactual edge predictions.
- formulation: A_SPM[i,j] = Norm(1 / (d(e_i, s_j) + eps)), d = L1, Norm divides by the mean inverse distance (asymmetric); A^l = softmax(X W_Q (X W_K)^T / sqrt(d) + A_SPM); GSA(X) = A^l X W_V; residual + FFN. CIL: counterfactual attention CSA = softmax(A_bar + A_SPM) X W_V with A_bar = zeros; E_TIE = E[E_A - E_A_bar]; edge loss = focal on E_TIE; inference uses the factual E_A. Edge head: two 3-layer MLPs for start / end embeddings, MLP_edge on the concatenation, sigmoid. Lane-traffic link: fixed rule (lane class equals traffic class).
- input: BEVFormer-style BEV 200 x 100, 200 lane queries, 11 points per lane, R50, 1550 x 2048.
- output: lane polylines and class (10 classes), adjacency.
- training: two stages (detector, then T2SG), AdamW 2e-4 cosine, 16 V100, batch 16; epochs and loss weights not given; no matching description for topology.
- differentiable: yes for the bias; partial overall (DETR assignment).
- sensors_data: camera; OpenLane-V2 A and B.
- results: ablation subset A (DET_l / DET_t / TOP_ll / TOP_lt / OLS): no A_SPM 34.1 / 47.3 / 21.6 / 29.1 / 45.4; multiplicative 34.7 / 46.3 / 22.9 / 29.0 / 45.7; additive (final) 34.7 / 48.2 / 24.1 / 29.5 / 46.3. Counterfactual layer: none 32.2 / 47.0 / 22.2 / 28.6 / 44.9; mean / random / zero attention OLS 45.2 / 45.6 / 46.3. Main: A 46.3 OLS (TopoLogic 44.1), B 47.5 (42.3).
- compute: no runtime reported.
- code: https://github.com/MICLAB-BUPT/T2SG
- fit_to_us: reuse_idea - effort S - an inverse-distance bias on attention logits between vertex tokens (fork / merge candidates) is a one-line addition; the counterfactual layer shows only +0.1 .. 1.1 OLS over simple variants.
- compat_needs: start / end coordinates per token.
- compat_conflicts: none beyond the query-set host.
- tried_before: E45 (distance-based scoring) no gain over rules.
- verified: checked (digest: OLS equation as printed does not reproduce reported values; thresholds and weights not stated)

### CARD TP15 - TopoBDA (Bezier deformable attention, instance-mask loss, radar and SD fusion)
- papers: TopoBDA: Towards Bezier Deformable Attention for Road Topology Understanding (Kalfaoglu et al., arXiv 2412.18951; Neurocomputing 2026 per the earlier review) https://arxiv.org/abs/2412.18951
- depth: M
- stage: S2 L B
- idea: Each Bezier control point of a centerline query acts as its own attention head and reference point, so sampling follows the curve; an auxiliary instance-mask head supervised by rasterised centerlines (and a mixed mask / L1 matcher) improves thin-curve detection; a one-to-many auxiliary decoder repeats the GT R times in training only.
- formulation: S(t) = sum_n B_{n,N}(t) c_n with N = 3 (4 control points; 8 gave marginal gain); BDA(q, V, C) = sum_n sum_k A_{n,k} W_n V(c_n + dp_{n,k}) (K per point not stated); control points refined per layer in inverse-sigmoid space. Mask: E_mask = MLP(query), P_mask = F_BEV . E_mask accumulated across layers, GT = centerline raster of width W (value not given), training only. Matching cost lambda_reg L_reg + lambda_mask L_mask + lambda_cls L_cls (reg 3, mask BCE 5, dice 5, cls 2; matcher reg 5); classification alpha 0.1 for matched. Topology: embeddings of lane and traffic queries projected by MLPs, concatenated, MLP (as TopoNet), L_total = L_l + L_t + L_ll + L_lt. Radar: points (x, y, z, features) assigned to the nearest voxel and concatenated with camera voxel features, 2D conv to BEV.
- input: BEV 200 x 104 cells of 0.5 m, Z = 20 voxels over -10 .. 10 m; 200 queries, 10 decoder layers, 6 encoder layers.
- output: Bezier centerlines + topology.
- training: AdamW 3e-4, batch 8, polynomial decay, warm-up 1000 iterations, grad clip 35; 24 epochs for R50 comparisons.
- differentiable: partial - mask-L1 mix matcher (assignment).
- sensors_data: camera, + LiDAR, + SD map (subset A), + radar (subset B nuScenes); OpenLane-V2.
- results: subset A (Swin-B, DET_l / TOP_ll / OLS): baseline 37.0 / 29.0 / 43.6; +instance-mask loss 40.7 / 32.4 / 46.6; +mask-L1 matcher 40.8 / 32.9 / 48.0. Attention: SA 41.0, MA 42.6, single-point deformable 44.1, multi-point (4) 47.4, multi-point (16) 47.5, Bezier 48.0 OLS. Sensors (R50, DET_l / TOP_ll / OLS), subset A: camera 38.9 / 29.4 / 44.1; +SD map 42.7 / 35.7 / 50.1; +LiDAR 46.5 / 36.7 / 52.0; +LiDAR + SD 52.0 / 40.0 / 56.0. Subset B: camera 45.1 / 35.6 / 49.9; +RADAR 49.4 / 38.6 / 54.8; +LiDAR 57.5 / 46.0 / 61.6; radar on top of LiDAR no gain. Main (C): A OLS 51.7, B 54.3.
- compute: decoder 10 layers: Bezier 18.47 ms (torch) / 8.07 ms (ONNX), RTX A6000.
- code: project page https://artest08.github.io/TopoBDA.github.io/ (no repository found)
- fit_to_us: reuse_idea - effort M - radar evidence is directly relevant (camera + radar: DET_l +4.3, TOP_ll +3.0, OLS +4.9 on nuScenes); a mask head on rasterised centrelines is a matching-free dense auxiliary.
- compat_needs: Bezier or ordered control points as reference points; centreline raster for the mask loss.
- compat_conflicts: control-point count tied to head count of attention (changing it changes channel layout); Hungarian-type matcher.
- tried_before: E31 / E32 (sparse decoder with matching): dropped.
- verified: partly (page read in two parts; epochs of the main run and Table S7 not seen; control points 4 (S.5) versus 8 (conclusion) inconsistent in the paper)

### CARD TP16 - Topo2Seq (training-only topology sequence decoder)
- papers: Topo2Seq: Enhanced Topology Reasoning via Topology Sequence Learning (Yang et al., AAAI 2025) https://arxiv.org/abs/2502.08974
- depth: M
- stage: T L
- idea: A Pix2Seq-style autoregressive decoder reads the shared BEV features and predicts the lane graph as a token sequence from a prompt of unordered keypoints; it is discarded at inference. Its loss is meant to improve the shared BEV representation, so the lane-segment decoder gains accuracy at no inference cost.
- formulation: lane-segment decoder = LaneSegNet (6 layers, 200 queries, lane attention); topology A = sigmoid(MLP(concat(MLP(Q), MLP(Q)))). Sequence decoder: causal self-attention + cross-attention to BEV; each edge = 6 integers [x, y, cls (ancestor / lineal / offshoot / clone), con (parent index), bx, by (intermediate cubic-Bezier control point)]; sequence length 802 = 201 keypoint prompt + 601 edge tokens; keypoint prompt in RANDOM order (not the edge order), ordering of edges by depth-first search from the right-front; noise edges with a <ncls> token; max-likelihood loss with class weights. Vocabulary size, bins and depth not stated. L = 0.025 L1 + 1.5 L_cls + 3.0 L_seg + 0.1 L_lt + 5.0 L_top + 1.0 L_seq.
- input: BEV 200 x 100 (+-50 x +-25 m); 10 points per centerline.
- output: unchanged lane segments + adjacency at inference.
- training: matching not stated; 24 + 24 style schedules (inconsistent in the text), 4 A100, batch 4; subset A only.
- differentiable: partial - decoder training uses teacher forcing; the benefit passes only through the shared BEV encoder.
- sensors_data: camera; OpenLane-V2 subset A (27 k train / 4.8 k val frames).
- results: ablation (mAP / AP_ls / AP_ped / TOP_lsls): ordered GT keypoint prompts 32.5 / 31.5 / 33.5 / 25.1; random order 34.9 / 33.4 / 36.3 / 27.8; early interaction 33.8 / 32.3 / 35.4 / 26.5; full (predicted keypoints, extra 24 epochs) 37.7 / 36.9 / 38.5 / 29.9. Versus LaneSegNet at 24 ep: +1.8 AP_ls, +1.5 TOP_lsls, centerline OLS* 42.7 vs 40.7, TOP_ll 27.0 vs 25.3 (the earlier review's "+2.0 / +2.7" came from a third-party summary and differs).
- compute: 14.7 FPS (unchanged).
- code: none found in the paper
- fit_to_us: not_suitable - the claimed mechanism is an auxiliary gradient into a TRAINABLE BEV encoder, which is frozen in our pipeline; effort would be L for no expected gain.
- compat_needs: trainable shared encoder; sequence tokenisation.
- compat_conflicts: frozen foundation; argmax-free token decoding not needed.
- tried_before: none
- verified: checked (digest: Eq. 6 garbled, schedule description inconsistent, epochs unclear)

### CARD TP17 - SMART (SD + satellite map prior, frozen) and SMERF
- papers: SMART: Advancing Scalable Map Priors for Driving Topology Reasoning (Ye et al., ICRA 2025) https://arxiv.org/abs/2502.04329 ; SMERF: Augmenting Lane Perception and Topology Understanding with Standard Definition Navigation Maps (Luo et al., arXiv 2311.04079) https://arxiv.org/abs/2311.04079
- depth: M (SMART and SMERF)
- stage: S9 B
- idea: A map-prior network is trained offline to predict lane graphs from SD-map polylines and a satellite crop, supervised by HD maps of a large geo-referenced set; it is then frozen and its BEV features replace (or are cross-attended by) the learnable BEV queries of an online topology model.
- formulation: SD map = m polylines of N = 11 points with attributes -> sinusoidal embedding + linear -> 6-layer transformer encoder with validity mask; satellite: Mapbox raster zoom 20 (about 0.11 m / pixel), rotated and cropped to ego, resized 500 x 250, ResNet-50 (last three stages). A learnable BEV map (200 x 100 x 256) cross-attends first to SD, then to satellite features. Pre-training: about 200 k scenes / 810 k HD maps from the Argoverse 2 motion-forecasting set (full set 199,908 scenes), 8 epochs, 8 A100 about 2 days. Injection: TopoNet (BEV queries replaced by prior features) and TopoMLP (extra cross-attention aligning prior with perspective features); decoder 200 queries + simplified GCN; focal + L1.
- input: SD polylines, satellite raster, ego pose (GPS + heading; error handling not described).
- output: BEV prior features (and offline lane graph).
- training: two stages; prior frozen during online training; 24 epochs.
- differentiable: yes to the online model; the prior is frozen.
- sensors_data: OpenLane-V2 (no A / B split reported), 100 x 50 m window.
- results: online OLS / TOP_ll: TopoNet 39.8 / 10.9 -> 51.1 / 27.5; TopoMLP 44.1 / 21.7 -> 53.1 / 37.0. Modality OLS: SD only 44.1, satellite only 48.1, both 51.1. Data scale offline TOP_ll 16.2 / 28.9 / 31.9 (base, 18x, 40x). Geo-disjoint split OLS 26.9 -> 33.6. SMART alone offline DET_l 37.9, TOP_ll 31.9, 44.3 ms (TopoNet offline 172.6 ms). Sensor reduction: 40 % of sensor data with the prior comparable to full data without (figure only). SMERF (encoder reused by SCORE, MapVision, Reusing-Attention): OSM polylines per frame (M polylines, N = 11 points each, sinusoidal position encoding d = 32, T = 1000, one-hot road type), 6-layer transformer encoder trained jointly with the topology model (no extra loss), fused by cross-attention (BEV queries attend to SD tokens) after every spatial cross-attention of the BEV encoder; GPS pose given as a 3-DoF transform, noise not handled. Subset A (DET_l / TOP_ll / OLS): TopoNet 28.2 / 4.1 / 34.5 -> +SMERF 33.4 / 7.5 / 39.4; weaker baseline 17.0 / 2.3 / 30.2 -> 26.8 / 3.9 / 34.8. Encoder ablation (OLS): raster SD map 30.9, VectorNet-style 31.2, SMERF polylines 34.8 (xy position encoding is worth +2.3, normalisation +1.6); heads 4 / 8 / 16: 34.8 / 34.5 / 33.9; gain concentrated on far lanes (25 - 50 m: drop close to far 3.4 points without, 0.8 with SD map) and intersections; geo-disjoint split 16.9 -> 22.1.
- compute: SMART alone 44.3 ms vs 172.6 / 328.2 ms for the online models (offline settings, see above).
- code: SMART project page https://jay-ye.github.io/smart (no repository); SMERF https://github.com/NVlabs/SMERF (not opened)
- fit_to_us: reuse_idea - effort L - our route head already cross-attends vertex tokens to SD-map raster tokens; SMART adds large-scale offline pre-training of the prior and shows the benefit grows with data (TOP_ll 16.2 -> 31.9); no satellite data in our setting.
- compat_needs: ego pose / localisation; large HD-map corpus for pre-training.
- compat_conflicts: assumes GPS pose without error (our registration problem E40 / E48 is not addressed); satellite imagery.
- tried_before: E40 / E48 (map prior at the grid, learned registration): kept; E102 .. E104 (route head, cross-attention kept).
- verified: checked (SMART, SMERF; SMERF encoder width H and polyline count M not stated, BEV size not stated)

### CARD TP18 - SCORE (stack: SD map + one-to-many + denoising + GeoDist + temporal)
- papers: Coherent Online Road Topology Estimation and Reasoning with Standard-Definition Maps (Bonn, arXiv 2507.01397; project page https://www.ipb.uni-bonn.de/html/projects/score) https://arxiv.org/abs/2507.01397
- depth: M
- stage: S6 S8 S9 T
- idea: A challenge-style stack on LaneSegNet whose incremental table shows that SD-map fusion, hybrid queries, one-to-many supervision, denoising, a geometric link post-process, resampling and ConvGRU temporal fusion add up.
- formulation: SD queries: n_sd = 50 reference points from SD edge midpoints (OSM maps), initial p = MLP(q_pos) + p_sd, attention mask so SD queries do not leak into others, SMERF-style cross-attention. Hybrid queries = SD-enhanced + base + denoising (60 DN queries, noise U(-1,1) scaled by lambda_dn, groups = floor(n_dn / N_GT), masked attention). Topology head: A^S = sigmoid(MLP(pair features)); distance score A^D = 2 / (1 + exp(d / alpha)), alpha 2.5; A = min(A^S + beta A^D, 1), beta 0.8. Temporal: ConvGRU on BEV with ego-motion compensation after 30 single-frame epochs + 30 temporal epochs. Hungarian (LaneSegNet costs); o2m weight 1.0.
- input: BEVFormer BEV (size not stated), 200 base queries.
- output: lane segments + topology.
- training: AdamW cosine, 8 V100, batch 8, images at 50 %.
- differentiable: partial - Hungarian; distance score is a differentiable closed form.
- sensors_data: camera + OSM SD map; OpenLane-V2 subset A (subset B lacks lane segments).
- results: incremental (DET_ls / DET_ped / TOP_ll): LaneSegNet 32.3 / 32.9 / 25.4; +SMERF 35.8 / 41.7 / 31.3; +SD queries 37.3 / 41.2 / 31.8; +OSM 38.2 / 42.0 / 32.4; +one-to-many 39.8 / 43.5 / 34.8; +denoising 39.8 / 44.1 / 34.6; +topology post-processing 39.8 / 44.1 / 36.2; +resampling 40.0 / 45.1 / 36.2; +temporal 44.0 / 47.3 / 40.0. Final val: OLUS 54.9 (DET_ls 44.0, TOP_ll 40.0, TOP_lt 39.1); test OLUS 55.8. Geo-disjoint split drops DET_ls to 23.8 (LaneSegNet 19.3). Traffic elements: YOLOv9 61.4 > DN-DETR + YOLO 57.0 > DN-DETR 53.0.
- compute: 8 V100 training; no inference numbers.
- code: none (project page only)
- fit_to_us: baseline - effort S - gives a stacking order and magnitudes (temporal +4.0 DET_ls and +3.8 TOP_ll; GeoDist-style post-process +1.6 TOP_ll).
- compat_needs: LaneSegNet-type host.
- compat_conflicts: one test split with geographic overlap inflates numbers.
- tried_before: E39 / E41 (temporal fusion small gain), E40 (map prior).
- verified: checked

### CARD TP19 - TopoStreamer (streaming query propagation for lane segments)
- papers: TopoStreamer: Temporal Lane Segment Topology Reasoning in Autonomous Driving (Yang et al., arXiv 2507.00709) https://arxiv.org/abs/2507.00709
- depth: M
- stage: S8 L
- idea: Previous-frame top-k queries are mapped to the current frame by an ego-motion MLP and replace the lowest-confidence queries; supervision of these stream queries uses instance IDs for the next frame's targets; a dynamic boundary positional encoding and lane denoising are added.
- formulation: BEV fusion F~ = GRU(Warp(F_{t-1}, Psi), F_t); Q_t^S = MLP(concat(Q_{t-1}, Psi)) + Q_{t-1}; 200 queries of which 30 % (66) are stream queries replacing the lowest-confidence ones in layer 1; DN queries 240 (not propagated). Stream losses on centerline coordinates, classes, boundary offsets and BEV mask (kappa weights 1.0, 0.01, 1.0, 1.0, 0.025, 1.0, 3.0); matched stream queries tracked by OpenLane-V2 instance IDs ("lossless supervision"); adjacency from predecessor / successor MLPs, threshold 0.5. DBPE: sinusoidal PE of 8 boundary reference points recomputed per layer. DN: box shift factor 0.2, label flip 50 %, topology DN focal.
- input: BEV 200 x 100 (+-50 x +-25 m), 7 cameras 2048 x 1550.
- output: lane segments + adjacency.
- training: Hungarian; first 12 epochs single-frame; 24 epochs; AdamW 2e-4; V100.
- differentiable: partial - Hungarian, top-k selection of stream queries.
- sensors_data: camera; OpenLane-V2 subset A (subset B via pseudo-segments from centerlines + 4 m width).
- results: module ablation (mAP / TOP_lsls / Acc_b): LaneSegNet 32.6 / 25.4 / 45.9; +DBPE +DN 34.5 / 26.7 / 47.6; +stream + DBPE 35.6 / 27.8 / 49.5; full 36.6 / 28.5 / 50.0; without ID tracking 34.4 / 28.0 / 49.5. Stream fraction 10 / 30 / 50 / 75 %: mAP 34.8 / 36.6 / 34.3 / 32.9. StreamMapNet 20.3 mAP, SQD-MapNet 26.0 (their numbers). Subset B centerline OLS 42.6 vs LaneSegNet 38.7.
- compute: 13.6 FPS vs 14.7.
- code: https://github.com/YimingYang23/TopoStreamer
- fit_to_us: reuse_idea - effort S - our time recursion already warps the previous set; the numbers say the propagated share matters (30 % optimum) and that instance identity gives cleaner supervision.
- compat_needs: instance IDs or mask matching for stream supervision.
- compat_conflicts: assumes known relative pose; slower.
- tried_before: E39 / E41 (temporal feature fusion +0.008 F1): consistent with small gains on a strong single-frame result.
- verified: checked

### CARD TP20 - FASTopoWM (fast-slow lane segment topology with latent world models)
- papers: FASTopoWM: Fast-Slow Lane Segment Topology Reasoning with Latent World Models (arXiv 2507.23325) https://arxiv.org/abs/2507.23325
- depth: M
- stage: S8 T
- idea: A slow branch propagates the previous frame's queries and BEV features through learned latent world models conditioned on the relative pose; a fast single-frame branch shares the same layers and heads, so the system falls back to single-frame output when pose is missing.
- formulation: Q~^(T-1) = MLP([Q^(T-1), Psi]), F~^(T-1) = MLP([F^(T-1), Psi]); Q~^T = QueryWorldModel(Q~^(T-1)) (transformer blocks), F~^T = BEVWorldModel(F~^(T-1)) (temporal self-attention, average-pooled 200 x 100 -> 100 x 50 and upsampled), 2 layers each; stream BEV fused by GRU. Losses: L_bev = ||F~^T - F^T||_2 (target = current extracted feature, no labels), L_query = coordinate L1 + class focal / CE + mask CE / dice with GT warped from T-1; L_slow = L_ls + 0.3 L_latent, L_fast = L_ls, total = both; Hungarian on both branches. 30 % of 200 queries reserved for propagation.
- input: BEV 200 x 100, 10 points per line.
- output: lane segments + adjacency (head not described).
- training: 24 epochs, first 12 single-frame, AdamW 2e-4, V100.
- differentiable: partial - Hungarian.
- sensors_data: camera; OpenLane-V2 A (lane segment) and B (centerline).
- results: subset A (temporal mAP / TOP_lsls): baseline stream 34.0 / 28.1, full 37.4 / 29.6; single-frame branch 29.8 / 23.4 -> 34.1 / 26.6. Action condition (temporal): none 36.5 / 28.5, trajectory 31.2 / 24.5, relative pose 37.4 / 29.6. World-model architecture: linear 35.2, MLPs 36.2, transformer 37.4 mAP. Subset B centerline OLS: LaneSegNet 38.7, fast 41.8, slow 46.3 (TOP_ll 33.0). With GeoDist: TOP_lsls 31.6 (slow).
- compute: slow 11.4 FPS, fast 14.0 FPS.
- code: https://github.com/YimingYang23/FASTopoWM
- fit_to_us: reuse_idea - effort M - two lessons: pose conditioning must use the RELATIVE pose (trajectory conditioning hurts by 5 mAP), and a feature-prediction loss against the next frame's own features needs no labels.
- compat_needs: relative pose; a trainable BEV feature stream.
- compat_conflicts: pose-noise robustness untested (we have synthetic pose jumps); BEV world model needs pooling for memory.
- tried_before: E39 / E41 (temporal feature fusion +0.008 F1); pose jumps are a stated failure mode in our task.
- verified: checked

### CARD TP21 - LaneDiffusion (diffusion in BEV feature space with a lane prior)
- papers: LaneDiffusion: Improving Centerline Graph Learning via Prior Injected BEV Feature Generation (Wang et al., ICCV 2025, arXiv 2511.06272) https://arxiv.org/abs/2511.06272
- depth: M
- stage: S5 B T
- idea: A diffusion model generates a "lane-prior-injected" BEV feature from the ordinary BEV feature, conditioned on it; the GT lane prior is used only to build the training target, and the downstream CGNet decoder works on the generated feature. Residual shifting (ResShift) is used instead of noise-to-data diffusion.
- formulation: prior = GT centerlines (M x 20 points) -> sinusoidal embedding + 6-layer transformer -> cross-attended into the BEV constructor (LPIM). Diffusion: x0 = prior-injected feature, xc = original feature, x_res = xc - x0; q(x_t | x0, xc) = N(x0 + eta_t x_res, kappa^2 eta_t I), kappa 2.0, T = 15 steps, schedule exponent p = 0.3; denoiser = Swin-UNet predicting x0; loss w_t ||f(x_t, xc, t) - x0||^2. Sampling: start from xc + noise, 15 steps, run 3 times and average. Refinement LPR fuses generated and original feature (concat + encoder-decoder). Three training stages: (I) LPIM + CGNet losses 24 ep, (II) frozen LPIM, train diffusion 24 ep (lr 6e-5), (III) frozen diffusion, train decoder 24 ep.
- input: BEV 256 x 200 x 100 (x +-15, y +-30 m); host CGNet.
- output: refined BEV feature for the lane decoder.
- training: three stages, 8 V100, batch 2 per GPU.
- differentiable: partial - stage-wise; the sampler is a rule-free iterative network, no gradient to the BEV encoder in stage III.
- sensors_data: camera; nuScenes and Argoverse 2.
- results: nuScenes 24 ep (GEO F1 / TOPO F1 / JTOPO F1 / APLS / SDA): CGNet 54.7 / 42.2 / 34.1 / 30.7 / 8.8 -> 58.9 / 46.8 / 38.8 / 37.1 / 10.6; segment level DET_l 22.0 -> 28.8, mAPcf 35.2 -> 41.6. Steps 5 / 15 / 30: TOPO 44.8 / 46.8 / 46.0 (sampling paradigm). Refinement: none 43.9, concat + FC 45.6, concat + encoder-decoder 46.8.
- compute: about 45 denoiser evaluations per sample; no latency reported.
- code: not found on the page
- fit_to_us: reuse_idea - effort L - the idea "train a generator that maps ordinary BEV features to GT-lane-conditioned features" can operate on cached 200 x 200 features (the foundation stays frozen), but 45 Swin-UNet passes on a 3060 is heavy; closest in spirit to our cascade refiner (E39).
- compat_needs: GT lane prior at training; a feature-space refiner.
- compat_conflicts: compute; only one host (CGNet) tested.
- tried_before: E39 (cascade refiner +0.001 F1): small.
- verified: checked

### CARD TP22 - TopoGPT (generative lane graph with map pre-training)
- papers: Generative Lane Topology Reasoning via Autoregressive Model with Geometry Prior (TopoGPT, Fu et al., ECCV 2026, arXiv 2606.31814) https://arxiv.org/abs/2606.31814
- depth: M
- stage: S6 T X
- idea: Lanes become tokens (each lane a cubic Bezier = 4 control points, each control point one joint (x, y) bin index), the lane set is sorted and generated by a decoder-only transformer pre-trained on 3.3 M map-only scenes; a flow-matching adapter aligns BEV tokens to the scene-token space for the camera version.
- formulation: Q(v) = floor(v / vmax (N_bins - 1)) + 1 with N_bins = 64 over the BEV range; token tau_i = Q(y_i) N_bins + Q(x_i); lanes sorted by [xmin, ymin, xmax, ymax]; tokens <SOS>, <SEP>, <EOS>; sequence 5M + 1 tokens for M lanes; lanes split at forks and merges so the graph is implied by shared endpoints; edges at inference from a 0.5 m endpoint-distance threshold. Pre-training data: Waymo Motion 2.07 M, nuPlan 1.01 M, Argoverse 2 Motion 0.25 M scenes; teacher forcing, cross-entropy; model sizes 24.7 / 45.9 / 91.3 M. Adapter: ResNet-50 BEV encoder, DiT predicting a velocity field (flow matching), L_align MSE to the frozen scene tokens, LoRA on the decoder; greedy decoding, Euler integration of the adapter ODE (6 steps best).
- input: rasterised lane image (pre-training) / BEV tokens (fine-tuning) as a prefix.
- output: token sequence -> Bezier lanes -> graph.
- training: pre-training 16 epochs on 8 H20; fine-tuning 24 epochs.
- differentiable: no - discrete tokens, greedy decoding; adapter flow is differentiable.
- sensors_data: camera; OpenLane-V2 A and B; range +-52 x +-32 m, 64 bins.
- results: subset A lane-level M-P / M-R / M-T 45.6 / 42.6 / 24.4 (best prior TopoPoint 42.7 / 34.0 / 16.6); point-level G-F1 60.3, T-F1 47.4, JT-F1 35.7, APLS 28.7; subset B 43.1 / 39.2 / 23.4. Ablations: from scratch 7.4 / 7.0 / 0.8; frozen 44.3 / 42.1 / 23.7; LoRA 45.6 / 42.6 / 24.4; pre-training augmentations none 41.0 -> both 45.6 (M-P).
- compute: latency not reported (authors name sequential decoding as a cost).
- code: project page https://buaa-colalab.github.io/topogpt_page (no repository)
- fit_to_us: not_suitable - argmax / token decoding, no differentiable path; only the idea of a map-only pre-trained prior over lane-graph layouts is of interest.
- compat_needs: tokenisation, large map corpus.
- compat_conflicts: end-to-end differentiability; latency; error propagation.
- tried_before: none (token approaches ruled out in `lit_p16_tree.md`).
- verified: checked

### CARD TP23 - SeqGrowGraph (vertex graph as growing adjacency + geometry matrix)
- papers: SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions (Xie et al., ICCV 2025) https://arxiv.org/abs/2507.04822
- depth: M
- stage: S7 S6
- idea: The lane graph is stored as vertices (junction-like key points, incl. forks and merges) and directed edges (centerlines); an adjacency matrix grows from n x n to (n+1) x (n+1) per step in depth-first order together with a geometry matrix holding one quadratic-Bezier control point per edge.
- formulation: A(i,j) = 1 iff a directed centerline runs from v_i to v_j; fork / merge = several non-zero entries in a row / column; loops and bidirectional lanes representable (A(i,j) and A(j,i) both 1). Step S_n = S_{n-1} + (v_n + F_n + T_n) (F incoming, T outgoing). Tokens: node coordinates 0-200 (0.5 m bins), node indices 200-350, control-point coordinates 350-570, <to> <SEP> <EOS> <BOS> <N/A>; point-position token loss weight 2. Model: LSS BEV (R50, pre-trained on centerline segmentation), 6-layer transformer decoder; negative log-likelihood given previous GT tokens. Ordering ablation (landmark F / reachability F): centre 52.4 / 63.7, coordinate 53.7 / 64.4, BFS 55.1 / 67.7, DFS 56.4 / 67.8.
- input: LSS BEV; nuScenes x [-48, 48], y [-30, 30] m.
- output: token sequence -> graph.
- training: teacher forcing, 400 epochs nuScenes, 50 on Argoverse 2, 4 A100.
- differentiable: no (sequence decoding).
- sensors_data: camera; nuScenes default and PON splits, Argoverse 2.
- results: nuScenes default L-F / R-F 56.4 / 67.8 (LaneGAP / RNTR 53.2 / 47.5 best baselines); PON split 37.7 / 46.7; centerline metrics M-F 70.6, C-F 65.8; length-based re-segmentation changes the GT (not comparable).
- compute: latency not reported.
- code: none in the paper
- fit_to_us: reuse_idea - effort S (representation only) - it states precisely how the OpenLane-V2 piece graph relates to a vertex graph (key vertices + edge geometry), which is the conversion we need for evaluation.
- compat_needs: junction vertices; edge geometry.
- compat_conflicts: autoregressive decoding (not differentiable); sequence length scaling.
- tried_before: none
- verified: checked

### CARD TP24 - LaneGraphNet (anchor nodes + pairwise edges from nuScenes BEV)
- papers: Lane Graph Estimation for Scene Understanding in Urban Driving (Zuern, Vertens, Burgard, arXiv 2105.00195) https://arxiv.org/abs/2105.00195
- depth: M
- stage: S6 L
- idea: The closest setting to ours (nuScenes map expansion as ground truth, BEV raster input): nodes are lane anchor points (at most 2 m apart), edges are node pairs "directly reachable"; Graph R-CNN proposes nodes and classifies pairs as connected, a separate segmentation net gives direction.
- formulation: BEV 256 x 256 px at 20 cm (51.2 m), 4 modalities (LiDAR intensity, RGB, 14 semantic classes, detected vehicle boxes); one feature extractor per modality, concatenated after conv 4; region proposal network (nodes), relationship proposal network (edges); node label single class "anchor point"; edge label binary has-connection (undirected). Adaptive resampling: each predicted anchor is projected orthogonally onto its nearest GT segment (t_proj = l1 + (l2 - l1)(t^ - l1)) before labelling, which removes the penalty for position along the lane (matching-free label). Direction: PSPNet, 18 bins of 20 deg + background, per-pixel CE; edge direction = the sign with larger dot product to the mean predicted angle. Post-processing: drop anchors < 0.5, edges < 0.2, edges longer than 1/4 of the image, isolated anchors. Class balance: most edges are 0; no balancing described.
- input: BEV raster (Boston-Seaport only, random 256 x 256 crops; test region about 211 x 211 m).
- output: undirected graph, then directed.
- training: details (optimiser, epochs) not stated.
- differentiable: partial - proposals and NMS inside Graph R-CNN.
- sensors_data: LiDAR + RGB + semantics from nuScenes; one city.
- results: APLS / F1 / IoU: skeleton regression 0.033 / 0.202 / 0.114, nearest-neighbour anchors 0.088 / 0.448 / 0.298, LaneGraphNet 0.176 / 0.574 / 0.420; direction accuracy 93.5 % (LaneDirNet) vs 50.8 % (Graph R-CNN itself).
- compute: not stated.
- code: not checked
- fit_to_us: baseline - effort M - the weak numbers (APLS 0.18) show how hard pairwise edge scoring from raster features was before 2022; the projection-based label is a matching-free label recipe.
- compat_needs: BEV raster; anchor sampling.
- compat_conflicts: complex intersections give ill-formed graphs (authors); one-city evaluation.
- tried_before: E42 .. E56 (link heads on raster-derived nodes): no gain over rules.
- verified: checked

### CARD TP25 - Reusing attention for one-stage topology (attention-derived links) with SD-map distillation
- papers: Reusing Attention for One-stage Lane Topology Understanding (Li et al., IROS 2025, arXiv 2507.17617) https://arxiv.org/abs/2507.17617
- depth: M
- stage: S6 T
- idea: Instead of a GNN, the pair feature for the topology MLP is built from the queries and keys of ALL decoder self-attention layers (stacked across layers, EGTR style), so the link head reads relations the decoder already computed. An SD-map teacher distils BEV features into a camera-only student.
- formulation: per layer l: projected Q^l, K^l of the lane (CL) and traffic (TE) decoders, stacked to N x L x d/2; pair resource R_ij = concat[Q_i, K_j] over layers plus the last layer; lane-lane head reads R[CL, CL]; gated sums + MLPs. Distillation: L_BEV = ||F_student - F_teacher||^2 (teacher = SMERF converted, frozen; weighting not given). Matching and losses as TopoNet (not restated).
- input: BEV (TopoNet's), 2 decoders.
- output: N x N lane-lane and lane-traffic topology.
- training: 24 epochs, AdamW 1e-4, 4 GPUs batch 1 each.
- differentiable: partial - as host.
- sensors_data: camera (student), + SD map (teacher); OpenLane-V2 A and B.
- results: subset A student 40.6 OLS / TOP_ll 11.0 vs TopoNet 39.8 / 10.9; teacher 44.3 OLS / 17.1 vs SMERF 43.0 / 16.2; distillation +0.7 OLS (39.9 -> 40.6); extra feature interactions hurt (44.3 -> 42.9). Subset B no distillation 37.0 vs TopoNet 36.0. Speed +17 %, parameters 49.4 M (student) vs 62.6 M (TopoNet).
- compute: see above.
- code: https://github.com/Yang-Li-2000/one-stage.git
- fit_to_us: reuse_idea - effort S - a link head that reads the existing token attention instead of adding a GNN; gain is small and the paper's own interaction experiment is negative.
- compat_needs: multi-layer self-attention with queries / keys accessible.
- compat_conflicts: tied to a decoder with N-query self-attention.
- tried_before: E46 (node-conditioned head): no gain.
- verified: checked

### CARD TP26 - RoadPainter and TopoMask (points from masks; soft-argmax extraction)
- papers: RoadPainter: Points Are Ideal Navigators for Topology transformER (Ma et al., ECCV 2024) https://arxiv.org/abs/2407.15349 ; TopoMask: Instance-Mask-Based Formulation for the Road Topology Problem via Transformer-Based Architecture (arXiv 2306.05419, 2023 challenge entry) https://arxiv.org/abs/2306.05419
- depth: M
- stage: S2 S5 X
- idea: A coarse polyline per query is turned into a BEV mask by a dot product with BEV features, and a second point set is read out of the mask with column-wise soft-argmax; both point sets are averaged. TopoMask drops Bezier / polyline regression altogether and extracts points from instance masks.
- formulation: RoadPainter: 150 real + 150 virtual queries (virtual ones supervised only through masks, masked self-attention so real queries ignore them), 4 decoder layers, K = 11 points; mask M_i = B . Q'_i; column sampling C_(i,j) = [0..H-1]^T softmax(M_i(:, j)) (one expected row per column, differentiable), row-wise set for vertical lines; existence P = sigmoid(phi1(M)), direction D = sigmoid(phi2(Q')); keep the set with more valid points (> 0.5), remove outliers (nearest neighbour > 1.5 m), resample to K, average with coarse points. Topology = pairwise MLP over E_i = psi1(Q_i) + psi2(l_i). Loss weights top 5, cls 1.5, det 0.025, mask 1, mask-point 7. TopoMask: Mask2Former on BEV (200 x 104), direction label up / down / left / right, 11 points via expectation + second-order polynomial fit; relations with Sinkhorn normalisation.
- input: BEV (RoadPainter grid not stated; TopoMask 200 x 104).
- output: polylines + adjacency.
- training: Hungarian (separate for real / virtual sets); 24 epochs.
- differentiable: yes for the soft-argmax readout; partial overall.
- sensors_data: camera (+ optional SD map); OpenLane-V2.
- results: RoadPainter subset A (DET_l / TOP_ll / OLS): baseline 26.9 / 7.7 / 37.2; +point-to-mask 28.1 / 7.9 / 37.6; +mask-to-point 30.7 / 7.9 / 38.9; +SD 36.9 / 12.7 / 42.6 (old metric); 6.5 FPS. TopoMask: masks vs Bezier OLS 36.0 vs 31.7 (F-score 45.2 vs 36.1); challenge 4th in OLS (39.2), 2nd in centerline F1 (45.9).
- compute: RoadPainter 6.5 FPS (RTX 3090, FP32).
- code: none found
- fit_to_us: reuse_idea - effort S - column / row soft-argmax is a differentiable way to read polylines from a dense map, relevant for a gradient path from vertex positions back to the heat map.
- compat_needs: per-instance masks.
- compat_conflicts: cannot read vertical lines with one pass; needs instance separation.
- tried_before: E120 / E121 (position refinement; planner loss through positions): small.
- verified: checked

### CARD TP27 - Topo2D (2D lane priors initialise 3D lane queries)
- papers: Enhancing 3D Lane Detection and Topology Reasoning with 2D Lane Priors (Topo2D, arXiv 2406.03105) https://arxiv.org/abs/2406.03105
- depth: M
- stage: S1
- idea: A per-view 2D lane detector (hierarchical instance + point queries) provides the appearance and position (lifted along camera rays) for the 3D lane queries and 2D lane features enter the topology head.
- formulation: 20 instance + 11 point queries per view, 6 layers; PETR-style lifting of (u, v) over predefined depths, MLP to a 3D positional embedding; topology MLP over concatenated pooled 2D + 3D lane embeddings plus the embedded 3D coordinates; weights lane cls 2, reg 5, topology 5, GIoU 2.
- input: perspective images only (no BEV encoder).
- output: 3D lanes + topology.
- training: bipartite matching (named, not Hungarian explicitly); 24 epochs, AdamW 3e-4, batch 8.
- differentiable: partial - matching.
- sensors_data: camera; OpenLane-V2 subset A only (B has no height).
- results: subset A TOP_ll 22.3 vs TopoNet 10.9, OLS 44.5; ablation: 3D query init from 2D 28.8 / TOP_ll 22.4, random 24.7 / 19.1.
- compute: not stated.
- code: https://github.com/homothetic/Topo2D
- fit_to_us: not_suitable - needs perspective image features and per-view 2D detectors; the principle (good query initialisation) corresponds to our seeds.
- compat_needs: image features.
- compat_conflicts: cached BEV only.
- tried_before: E31 / E32.
- verified: checked

### CARD TP28 - Challenge winners (2023 OpenLane Topology, 2024 Mapless Driving): what they report
- papers: The 1st-place Solution for CVPR 2023 OpenLane Topology in Autonomous Driving Challenge (arXiv 2306.09590) https://arxiv.org/abs/2306.09590 ; LGmap: Local-to-Global Mapping Network for Online Long-Range Vectorized HD Map Construction (arXiv 2406.13988; claims first place, Mapless Driving 2024) https://arxiv.org/abs/2406.13988 ; MapVision: CVPR 2024 Autonomous Grand Challenge Mapless Driving Tech Report (arXiv 2406.10125) https://arxiv.org/abs/2406.10125
- depth: M
- stage: T B X
- idea: What winners do is mostly not a better link head: stronger detector (big backbone, long schedule, ensembling), better auxiliary detectors, SD-map fusion, temporal fusion, a topology head trained last on frozen detectors.
- formulation: 2023 (test OLS 0.55 vs 0.47 of second place): PETRv2 lane detector with 300 queries, 3D Bezier control points, backbones ResNet-50 / VoV / ViT-L, BDA + HSV augmentation; YOLOv8-x traffic detector (mosaic / mixup, foreground-class reweighting 2, resampling rare classes x5-20, pseudo-labels, multi-scale TTA 0.7-1.4, no horizontal flip); two MLP topology heads trained 10 epochs on frozen detectors; no ensembling. 2024 LGmap (test UniScore 0.66): symmetric view transformation (BEVFormer + LSS with lidar depth supervision, SD-map polylines by cross-attention), hierarchical temporal fusion (streaming then stacking; N frames sampled from M previous), ped-crossing resampling (4 corners + 6 points per edge, 8 instead of 40 permutations), TopoMLP-style lane-lane head, ViT-L pre-trained on nuScenes map construction, 36 + 48 epochs, input 1536 x 1536, BEV 100 x 200, ensemble of three lane models (ViT and InternImage-XL at scales 0.5 / 0.75 / 1) plus YOLOv8 + YOLOv9 for traffic elements. 2024 MapVision (test OLUS 0.58): BEVFormer + SD map (OSM) encoder after SMERF with auto-encoder pre-training, InternImage-L, LDTR head, P2P IoU loss, YOLOX for traffic elements, topology fine-tuned (lr 4e-4) with the rest frozen, decoupled area head.
- input: multi-view images (+ SD map in 2024).
- output: lane segments / centerlines + topology.
- training: as above.
- differentiable: partial - Hungarian host models; ensembling is outside the network.
- sensors_data: camera, SD map; OpenLane-V2 (2023 subset A / B; 2024 lane segment subset A + traffic elements).
- results: 2023 validation: lane backbone DET_l ResNet-50 18.15, VoV 21.01, ViT-L 28.11 (20 ep) -> 34.16 (48 ep) -> 35.28 with BDA; traffic DET_t 65.32 -> 79.89 (aug +3.77, TTA to 79.89); topology Table 4 (old metric): DET_l 28.11 / TOP_ll 14.54 -> DET_l 35.28 / TOP_ll 23.01 (topology tracks detector quality). 2024 LGmap: BEVFormer 40.36 mAP, LSS 32.57, both 40.89, +augmentation 43.75; temporal none 52.93, streaming 56.61, streaming-stacking 57.13; lane-segment ensemble (test, row mapping ambiguous): single 48.67 / 40.04 / 48.27 (DET-l / TOP-ll / TOP-lt) -> three models 50.74 / 46.32 / 53.59. MapVision cumulative OLUS (validation): ResNet-50 0.36, +map encoder 0.40, +InternImage-L 0.45, +map pre-train 0.46, +LDTR head 0.48, +P2P IoU loss 0.50; ensemble step: +YOLOX 0.57, +topology fine-tune 0.58, +decoupled training 0.59; test OLUS 0.58 (DET_l 0.39, DET_a 0.40, DET_t 0.80, TOP_ll 0.38, TOP_lt 0.48).
- compute: 16 A800 (LGmap), no inference numbers.
- code: none for LGmap / MapVision; the 2023 solution is TopoMLP (https://github.com/wudongming97/TopoMLP)
- fit_to_us: reuse_idea - effort S - recipe items that apply to a frozen-foundation setting: train the link / verifier head last on frozen detections; ensemble of models; temporal streaming-then-stacking; resampling of rare classes (forks / merges).
- compat_needs: multiple detectors or checkpoints for ensembling.
- compat_conflicts: ViT-L / InternImage backbones and 16-GPU schedules are out of reach; most gains come from the foundation, which we already have.
- tried_before: E27 (foundation choice) map-pretrained kept.
- verified: partly (LGmap row mapping and MapVision step values inconsistent in the digest; no 2025 report exists)

### CARD TP29 - OpenLane-V2: data and metrics
- papers: OpenLane-V2: A Topology Reasoning Benchmark for Unified 3D HD Mapping (Wang et al., NeurIPS 2023 D&B, arXiv 2304.10440 v3) https://arxiv.org/abs/2304.10440 ; repository https://github.com/OpenDriveLab/OpenLane-V2 (issue #76 for the metric fix)
- depth: M
- stage: D E
- idea: Two subsets (A: Argoverse 2, B: nuScenes), 1,000 scenes each (700 / 150 / 150 train / val / test segments), 2 Hz keyframes; per frame about 24 - 26 centerlines (90 % of frames > 10 centerlines), average 1.8 - 1.9 connections per centerline (up to 7 at complex junctions); range x +-50 m, y +-25 m; each lane stored with 201 points and evaluated with 11. Topology is a per-frame directed adjacency matrix: (i,j) positive iff the end point of lane i meets the start point of lane j; lanes with a single predecessor / successor are merged, lanes are split only at intersections, forks and merges (= a junction-vertex graph with whole pieces as edges). Subset B has z = 0.
- formulation: DET_l = mean AP over discrete Frechet-distance thresholds {1, 2, 3} m, relaxed with distance to ego; DET_t IoU 0.75 over 13 attributes; TOP_ll / TOP_lt = vertex-level mAP on matched graphs: predictions are matched to GT by Frechet distance (per threshold 1 / 2 / 3 m); per GT vertex the candidate neighbours with confidence > 0.5 are ranked, precision is accumulated at every hit and divided by the true neighbour count (0 if one side is empty, 1 if both are); averaged over rows and columns (directed for lane-lane), thresholds and samples. Master-branch code as read through a digest (`_mAP_topology_lclc`): predicted lanes that match no GT are dropped from the matrix (no dummy vertices), cells of unmatched GT lanes get the fallback (1 - gt) x (0.5 + eps), i.e. 0 where an edge exists (a miss) and just above the cut-off where none exists; there is no explicit N_TP / (N_TP + N_FP) factor. Consequence: TOP measures link quality among MATCHED lanes only; false-positive lanes are punished by DET_l, not by TOP. OLS = mean of DET_l, DET_t, sqrt(TOP_ll), sqrt(TOP_lt) (sqrt on the 0-1 scale, reproduced numerically by several papers). Versions: devkit v1.0 / v2.0 (loophole: unmatched predictions get confidence 1.0) versus v1.1 / v2.1 (fixed; README recommends the .1 metrics; lane-segment tasks TOP_lsls and the OLUS score only in v2.x). Other nuScenes centerline-graph metrics: CGNet / LaneGAP / LaneDiffusion use point-level TOPO F1, JTOPO F1 (junction sub-graphs), APLS, SDA with BEV 30 x 60 m; RNTR / SeqGrowGraph use landmark and reachability P/R/F; TopoGPT G-F1, T-F1, JT-F1.
- input: n/a
- output: n/a
- training: n/a
- differentiable: n/a
- sensors_data: subset A 7 cameras 2048 x 1550 (Miami, Pittsburgh, Detroit, Washington D.C., Austin, Palo Alto; night 3.0 %, rain 1.1 %); subset B 6 cameras 1600 x 900 (Boston 55 %, Singapore 45 %; night 11.7 %, rain 17.4 %). Licence of the data CC BY-NC-SA 4.0, code Apache-2.0 (README).
- results: TopoNet baseline (old metric) A: OLS 35.4, DET_l 29.2, TOP_ll 4.1; B: 33.2, 24.3, 2.5.
- compute: n/a
- code: see papers: devkit in the repository.
- fit_to_us: reuse_idea - effort S - contract: our piece graph (nuScenes map expansion, travel direction) contracts to an OpenLane-V2 style junction-vertex graph by merging degree-(1,1) vertices; DET_l / TOP evaluation could then be run on our output as an external comparison; the TOP definition (ranking of neighbours) is not what our split / merge F1 measures.
- compat_needs: piece polylines with 11 points; adjacency by end / start coincidence.
- compat_conflicts: TOP depends on the metric version; lane-segment labels exist only for subset A; train / val geographic overlap inflates results (SMART, SCORE show 20 - 30 % drops on a geo-disjoint split).
- tried_before: none
- verified: partly (the evaluation file of the master branch was read through a digest, version of that file not confirmed as v2.1; supplements not read)

### CARD TP30 - STSU (first query-based centerline graph with a pairwise association classifier; nuScenes)
- papers: Structured Bird's-Eye-View Traffic Scene Understanding from Onboard Images (Can et al., ICCV 2021, arXiv 2110.01997) https://arxiv.org/abs/2110.01997
- depth: M
- stage: S6 L
- idea: A DETR-like transformer predicts centerlines as Bezier curves from ONE front camera and a pairwise MLP over per-curve association features gives the directed connectivity; this is the origin of the "adjacency between queries" formulation that all later piece-graph methods inherit (CGNet copies its BCE loss).
- formulation: 100 centerline queries (plus 100 object queries), each a quadratic Bezier curve (3 control points); each detected centerline gets a delta-dimensional association feature (delta < C); for every ordered pair (i, j) the two features are concatenated and passed through an MLP with sigmoid, giving P(i connects to j) (direction through the order of the pair). GT: x links to y when the start of y coincides with the end of x. The association step is trained only on Hungarian-matched detections; no class-imbalance measure, loss type not named. No cycle or acyclicity post-processing; junction points are averaged only for visualisation.
- input: image 448 x 800, DeepLabv3+ backbone (Cityscapes pre-trained), BEV region x in [-25, 25] m, forward 1 - 50 m at 25 cm.
- output: Bezier centerlines + directed association matrix (and BEV objects).
- training: Hungarian matching with cross-entropy for detection and L1 for geometry; about 11 FPS for the chosen model.
- differentiable: partial - Hungarian matching.
- sensors_data: camera (front only); nuScenes only (not evaluated on Argoverse in this paper).
- results: nuScenes lane graph (M-Pre / M-Rec / Detect / C-Pre / C-Rec / C-IOU): STSU 60.7 / 54.7 / 60.6 / 60.5 / 52.2 / 38.9; polyline variant with estimated start points 54.7 / 51.2 / 40.5 / 58.4 / 16.3 / 14.6. Later reproductions on OpenLane-V2 give TOP_ll 2.9 (A, old metric as quoted by others) and on nuScenes 30 x 60 m TOPO F1 0.395 (LaneGAP table).
- compute: about 11 FPS.
- code: https://github.com/ybarancan/STSU
- fit_to_us: baseline - effort S - historical reference for the pair classifier; the connectivity IoU of about 39 shows the difficulty of learned association from single-frame features.
- compat_needs: Bezier curves with endpoints; Hungarian matching.
- compat_conflicts: front camera only; Hungarian.
- tried_before: E42 (learned link head).
- verified: checked

### CARD TP31 - Online lane graph from onboard video (early BEV aggregation)
- papers: Online Lane Graph Extraction from Onboard Video (Can et al., ITSC 2023, arXiv 2304.00930) https://arxiv.org/abs/2304.00930
- depth: M
- stage: S8
- idea: Temporal evidence for the STSU-type lane graph is added early: the backbone features of past and future frames are warped into the reference frame's ground-plane BEV grid by a homography (flat-ground assumption, known poses and calibration) and combined by a per-cell maximum, so any number of frames works; the rest of the pipeline is the single-frame model.
- formulation: per frame, backbone features -> warp to the reference ground-plane BEV grid; binary BEV mask removes warping artefacts; residual block before the max; frames stacked in the batch dimension; per-cell max beat the mean. Training with 3 frames (one past, one future, the reference), test frames 2 s apart; frames sampled within a 4 s window as augmentation. Lane graph as STSU: Bezier centerlines (Q x 3 x 2), incidence matrix A[x,y] = 1 iff the end of x matches the start of y, association feature per query. Baselines: STETR (temporal sinusoidal embeddings in the transformer input) and PostMerge (merge per-frame estimates after inference).
- input: video frames of one front camera, ego poses; BEV x in [-25, 25] m, forward 1 - 50 m at 25 cm; DeepLabv3+ (Cityscapes).
- output: Bezier centerlines + connectivity.
- training: details (optimiser, epochs, loss weights) not reported.
- differentiable: partial - Hungarian matching as STSU; the warp and max are differentiable.
- sensors_data: camera (front); nuScenes and Argoverse.
- results: M-F / Detect / C-F, nuScenes: STSU (1 frame) 56.7 / 59.9 / 55.2; TPLR 58.2 / 60.2 / 55.3; PostMerge (3) 57.3 / 59.9 / 56.0; STETR (3) 57.2 / 60.0 / 61.2; this method 1 frame 58.3 / 60.0 / 61.0, 3 frames 58.8 / 60.3 / 61.3. Argoverse: 3 frames 58.2 / 62.2 / 58.1 vs 1 frame 57.9 / 62.1 / 57.8. Speed 10 FPS (1 frame), 3 FPS (3 frames).
- compute: see above.
- code: none ("will be made public")
- fit_to_us: reuse_idea - effort S - independent evidence that explicit multi-frame fusion adds little to a lane-graph metric once a single-frame model is good (+0.5 M-F, +0.3 C-F), like our E41 (+0.008 F1); our cached features are already ego-warpable.
- compat_needs: relative poses of all frames, flat-ground warp.
- compat_conflicts: pose error not analysed (odometry assumed exact); flat-ground assumption.
- tried_before: E39 / E41 (temporal feature fusion).
- verified: checked

## 3. Cluster compatibility notes

### (a) Interface contracts: what flows in and out in this literature, and how it maps to ours

| item | literature (OpenLane-V2 methods) | ours |
|---|---|---|
| BEV input | trainable BEV encoder (BEVFormer / LSS / GKT) from perspective images; 200 x 100 cells for +-50 x +-25 m (0.5 m), or 100 x 200 cells at 0.3 m for 30 x 60 m (nuScenes graph papers) | cached frozen raster pyramid 200 x 200, 0.5 m, +-50 m, 39 ch + 3 dense-head ch; no gradient into the encoder |
| unit of perception | query = one lane PIECE (11 3D points, or 4 Bezier control points, or lane segment 3 x 10 points) between junctions; 200 - 300 queries, set prediction with Hungarian matching; TopoFG: one query per keypoint | vertex tokens: position + 128-d token + validity + edges; edges = steps of 2 / 4 / 6 m |
| connectivity | N x N matrix between pieces: (i,j) = end of i meets start of j; produced by pair MLP (TopoNet, TopoMLP, RoadPainter, LaneSegNet), geometry mapping (TopoLogic), GCN + GRU (CGNet), or avoided (LaneGAP paths, UniTopo connected lanes) | edges between vertices; fork = out-degree 2, merge = in-degree 2 |
| labels for connectivity | GT adjacency re-indexed by the Hungarian assignment of queries (TopoNet, TopoLogic, CGNet, LaneSegNet); matching-free alternatives: TopoFG denoising block-diagonal target, TopoEnhance (all pairs of corrupted GT nodes), LaneGraphNet (project predicted anchor to GT segment) | no matching; geometric association (point-to-polyline) |
| geometric prior | endpoint distance end_i -> start_j through exp(-d^a / (l s)) (TopoLogic, TopoPoint, TopoFG, UniTopo, FASTopoWM, SCORE), additive inverse-distance attention bias (T2SG), min-over-endpoints + angle bias (CoPo) | rules decide merges today (E45: scoring within 0.01) |
| outputs consumed downstream | post-hoc threshold 0.5 on the matrix; no path / planner use in the OpenLane papers (LaneGAP alone reports VAD planning: L2 0.70 m vs 0.78 m without lane graph) | route head + planner consumer (E104b, E117) |

**Translation piece graph <-> vertex graph.** The OpenLane-V2 piece graph is the LINE GRAPH of a junction-vertex graph: pieces are edges, a piece-adjacency entry (i,j) means "piece i ends at the vertex where piece j starts". Fork at vertex w: the incoming piece has two successors (a row with two ones); merge at w: two pieces share a successor (a column with two ones). TopoPoint's point-lane incidence G_pl is exactly the vertex-edge incidence matrix, so the two views convert both ways: A_piece = End_incidence x Start_incidence^T. SeqGrowGraph states this vertex-plus-edge-geometry form explicitly (vertices = key junction nodes, quadratic Bezier per edge). Our graph has dense vertices (22 per 100 m); contracting all degree-(1,1) vertices gives the OpenLane-V2 piece graph, which allows external DET_l / TOP evaluation. Note the data fact behind every link head: the label is "endpoint coincidence", so a link is determined by geometry once the endpoints are right.

### (b) Known adapters and known conflicts

- **Adapters:** (1) Bezier <-> polyline: Bernstein matrix and its pseudo-inverse (CGNet), least-squares fit with pinned end points (TopoGPT, CGNet). (2) Query <-> bilinear sample: lane attention / Bezier deformable attention put reference points along the element (LaneSegNet 8, CoPo 11 along the curve, TopoBDA one head per control point); on our side these are bilinear `grid_sample` calls at ray points. (3) Path <-> graph: Graph2Path (DFS root-to-leaf) and Path2Graph (merge vertices within 0.15 m) of LaneGAP; the same merge is our tree -> DAG step. (4) Piece graph <-> vertex graph via incidence matrices (above). (5) Pair-score <-> edge set: threshold 0.5 (all), TopoEnhance GAT re-scoring, UniTopo / LaneGAP avoid it. (6) Soft-argmax readout of points from a dense map (RoadPainter / TopoMask column sampling).
- **Conflicts:** (1) Hungarian matching and fixed-size query sets (all DETR-derived heads: TopoNet, TopoLogic, TopoPoint, CGNet, LaneSegNet, UniTopo, RATopo, TopoStreamer, FASTopoWM) versus our matching-free losses: only the head formulas and geometric priors transfer, not the training recipe; TopoFG's fixed denoising adjacency and RATopo's label-density argument are the two matching-aware ideas that translate. (2) Trainable BEV encoder: Topo2Seq's auxiliary sequence loss, Lane Field's gradient into the encoder (36.1 vs 34.1 mAP when detached) and LaneDiffusion's three-stage scheme rely on it; our foundation is frozen. (3) Perspective-image components: lane-traffic topology, Topo2D, PETR-style position embeddings of TopoMLP, camera-matrix embeddings. (4) Code base: every repository found (LaneSegNet, TopoLogic, TopoPoint, TopoFG, UniTopo, CGNet) pins PyTorch 1.9.1 / CUDA 11.1-11.8 and mm-series code (LaneSegNet README: mmcv-full 1.5.2, mmdet 2.26.0, mmdet3d 1.0.0rc6); none runs in the `gcar` environment as is. (5) Autoregressive token decoders (TopoGPT, SeqGrowGraph, Topo2Seq decoder): no differentiable path.

### (c) Combinations the papers themselves demonstrate

- GeoDist (TopoLogic mapping) added post hoc to TopoNet (+11.4 TOP_ll), SMERF (+10.8), LaneSegNet (+4.2 TOP_lsls), FASTopoWM (TOP_lsls 29.6 -> 31.6), UniTopo (+0.5 TOP_ll); used inside TopoPoint, TopoFG, CoPo, SCORE.
- TopoLogic + RATopo (TOP_ll 23.9 -> 32.2), TopoLogic + TopoFG components (-> 30.8), TopoNet + RATopo (10.9 -> 29.5).
- TopoEnhance post-hoc on TopoNet, TopoMLP, Topo2D, TopoLogic, SMART, TopoPoint.
- SD-map priors with TopoNet and TopoMLP (SMART), with LaneSegNet (SCORE: stacking table), distilled into a camera-only student (Reusing attention).
- LaneSegNet + temporal (TopoStreamer, FASTopoWM), + dense field (Lane Field), + junction / continuity modules (CGNet is a MapTR-style host instead).
- Sensors: camera + radar on TopoBDA's nuScenes subset (OLS 49.9 -> 54.8), camera + LiDAR (LaneGAP, TopoBDA); no paper combines radar with SD map or temporal fusion.

### (d) What the literature predicts for our earlier results

- **E42 / E45 / E46 / E51 / E56 (learned link heads, TopoLogic-style scoring, node-conditioned head, path repair: within 0.01 of the rules).** Predicted. (i) The label is endpoint coincidence, so geometry determines it: in TopoLogic's ablation a plain pair MLP gives TOP_ll 10.8 against 20.1 for the distance mapping alone, and the learned semantic term adds 3.8 on top (23.9). (ii) TopoMLP's oracle experiment shows the link head is not the bottleneck: true topology on predicted lanes changes TOP_ll by only +0.3 .. +0.6 (old metric). (iii) Where learned heads gain a lot (UniTopo +10.9 from connection queries, RATopo +8.3, TopoNet + GeoDist +11.4), the baseline pair head saw few labelled pairs or no endpoint geometry; with dense matching-free labels and geometry already in the input, little is left. (iv) Possible exception: TopoEnhance shows that thresholded edge sets of query decoders are poor (TJS_ll 16 - 21 % on subset A) and a corruption-trained re-scorer lifts them to 33 - 60 % (TopoNet 16.3 -> 32.9, TopoMLP 18.5 -> 59.7), with the per-relation normalised BCE alone worth +9.7 TOP_ll over plain BCE; this regime (many false candidate nodes) resembles our learned graph with 1.8 x too many junctions, so the verifier (E113 / E116) is the place where learned re-scoring can still matter.
- **E44 / E53 (key points + Bezier connectors lift split / merge F1 0.34 / 0.38 -> 0.55 / 0.61 -> 0.71 / 0.64).** Predicted in direction: explicit junction / endpoint supervision helps in every paper that tried it (CGNet JAQ +0.8 TOPO F1 and +1.7 APLS; BSC continuity +1.7 TOPO F1 and +2.3 APLS; TopoPoint endpoint coupling +3.4 TOP_ll and +5.0 DET_p; TopoFG keypoint queries; LaneGAP path modelling junction F1 0.564 vs 0.496 for pieces). Published gains are 1 - 5 points on query-decoder baselines; ours is larger because the dense baseline had no junction concept. Published junction F1 values (0.43 - 0.63) use other protocols (30 x 60 m, camera only) and are not comparable to our 6 m split / merge F1.
- **Seeds limited E94 (192 peaks).** Consistent with TopoPoint's query-count ablation (100 -> 200 point queries: DET_p 49.7 -> 51.8; 300 not better) and UniTopo (200 / 300 / 400 lane queries: TOP_ll 27.0 / 30.1 / 31.6): endpoint / lane recall is a main limiter.
- **E31 / E32 (matched sparse decoder dropped for the grid).** The literature reports sparse decoders at TOP_ll 20 - 30 and DET_l 30 - 35 (OpenLane-V2) with trainable encoders; no paper compares a dense-grid tracer against them, so there is no contradiction.
- **E39 / E41 (temporal fusion +0.008 F1) and E40 / E48 (map prior).** Literature gains are larger (temporal +3 - 4 mAP and +3 - 4 TOP; SD / satellite priors +6 - 12 OLS), but baselines are weaker single-frame query models; for a strong map-pretrained foundation smaller gains are plausible, but no paper tests that.

### (e) Evidence ranking: what matters (OpenLane-V2 subset A, R50, 24 epochs, TOP_ll unless stated)

1. Detector / BEV quality sets the ceiling (TopoMLP oracle; challenge 2023: TOP_ll 14.5 -> 23.0 when DET_l 28.1 -> 35.3; UniTopo BEV depth 1 layer 26.2 vs 3 layers 30.1).
2. Endpoint geometry in the link score: GeoDist alone 20.1 vs MLP 10.8; post hoc +11. Not additive: once a model has its own geometry encoding the fixed mapping adds nothing (CoPo -0.5, UniTopo +0.5, FASTopoWM +2.0 TOP_lsls on a weaker base), so geometry priors replace, not stack with, each other.
3. Explicit endpoint / keypoint tokens and denoising adjacency target: TopoPoint +4.9 (23.8 -> 28.7), TopoFG +6.9 (23.9 -> 30.8).
4. Supervision density of valid pairs: RATopo +8.3 (23.9 -> 32.2).
5. Unified connected-lane objects: UniTopo +10.9 over a matched-parameter baseline.
6. Priors (SD / satellite): +6 .. +12 OLS; temporal: +3 .. +4 mAP; camera + radar: +4.9 OLS (nuScenes).
7. Small or negative: GNN depth (TopoNet 3+ layers collapse), counterfactual layer (+0.1 .. +1.1 OLS), extra feature interactions (negative), more Bezier control points than 4 (negative in CGNet, TopoMLP, LaneGAP at 10).

### (f) Suggested compatibility flags against our blocks (derived from the card fields; `-` = not applicable)

Blocks: B1 dense head (heat map + direction on cached BEV), B2 seeds / key points, B3 growth / hypothesis tree, B4 tree -> DAG merge, B5 link / verifier head, B6 position refinement, B7 time recursion, B8 route / SD-map head.

| card | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 |
|---|---|---|---|---|---|---|---|---|
| TP01 GeoDist mapping | - | - | adapter (score tip -> candidate vertex) | adapter (soft merge weight) | direct (prior term) | - | - | - |
| TP02 TopoPoint | - | adapter (point queries vs heat-map peaks; incidence labels) | - | direct (endpoint averaging) | adapter (distance bias between vertex and edge tokens) | - | - | - |
| TP03 CGNet | direct (junction heat-map branch) | direct | - | - | - | adapter (Bezier-space continuity as auxiliary) | - | - |
| TP04 LaneGAP | - | - | direct (paths = root-to-leaf paths of the tree) | direct (Path2Graph merge, 0.15 m) | - (no link head) | - | - | - |
| TP05 TopoFG | adapter (mask prior) | adapter (ordinal keypoint tokens) | - | - | adapter (matching-free denoising adjacency target) | - | - | - |
| TP06 UniTopo | - | - | conflict (query groups + Hungarian) | - | conflict (link-free design needs a set decoder) | - | - | - |
| TP07 TopoEnhance | - | - | - | - | direct (post-hoc; train on corrupted GT vertex sets) | - | - | - |
| TP11 Lane Field | adapter (tensor output instead of 2-ch direction) | adapter (confidence channel as junction cue) | adapter (tangent-stretched sampling) | - | - | - | - | - |
| TP13 CoPo | - | - | - | adapter (endpoint + angle bias) | direct (contrastive loss) | - | - | - |
| TP12 RATopo | - | - | - | - | conflict (concept lives inside Hungarian assignment) | - | - | - |
| TP19 / TP20 temporal | - | - | - | - | - | - | adapter (stream share, relative pose conditioning) | - |
| TP17 SMART / SMERF | - | - | - | - | - | - | - | adapter (polyline encoder vs our raster tokens); conflict (satellite data, error-free GPS) |
| TP21 LaneDiffusion | adapter (feature refiner on cached features); conflict (compute) | - | - | - | - | - | - | - |
| TP26 soft-argmax readout | - | - | - | - | - | adapter (differentiable read-out of points) | - | - |

Pairs between literature techniques that the papers combine without conflict: TP01 + TP05 (TopoFG uses the TopoLogic mapping), TP01 + TP14 / TP13 (geometry bias; redundant, not additive), TP12 + TP01 (RATopo on TopoLogic), TP19 + TP10 (stream queries on lane attention), TP17 + TP18. Unknown (untested anywhere): any of these on a frozen cached BEV raster; geometry-bias attention together with a matching-free label; TP07 on a vertex graph.

### (g) Observations for the design under test (hypothesis tree and spray-and-link), from the cards above

1. **A tree path is LaneGAP's primitive.** Path-wise modelling beats piece-wise modelling in the paper's own comparison on nuScenes (TOPO F1 0.529 vs 0.465, junction F1 0.564 vs 0.496 against MapTR; 30 x 60 m, camera only, 110 epochs), and Path2Graph is a pure distance merge (0.15 m, equal to the vertex spacing of their graph). The merge radius must therefore be set relative to vertex spacing and vertex error; ours (steps 2 / 4 / 6 m, vertex error 0.43 - 0.54 m) needs its own calibration, and a soft merge weight with a learned bandwidth (TopoLogic form) is the published alternative to a hard radius.
2. **Learned link scorers get a wide kernel because query endpoints are noisy** (half value at about 2.8 m in the released TopoLogic / TopoPoint / TopoFG code; threshold 0.7 in TopoPoint's snapping, i.e. about 2 m). A system with sub-metre vertices should not copy the bandwidth; this is consistent with rules already deciding in our E45.
3. **Junction recall is bought with explicit junction supervision and token coupling**, in this order of size: coupling of point and lane tokens through a distance-biased attention (TopoPoint: TOP_ll +3.4, DET_p +5.0), keypoint-level queries plus matching-free denoising (TopoFG: +6.9 TOP_ll with all modules), junction heat map (CGNet: +0.8 / +0.7 TOPO / JTOPO F1), Bezier-space continuity of connected pairs (CGNet BSC: +1.7 TOPO F1, +2.3 APLS).
4. **Training signals for fork / merge links that avoid label permutation:** fixed denoising adjacency (TopoFG), corruption of ground-truth nodes with false positives and jitter plus per-relation normalised BCE (TopoEnhance: +9.7 TOP_ll over plain BCE in its own ablation), hard-negative contrastive loss with one positive and three negatives (CoPo, +0.6 TOP_ll in its ablation), valid-pairs-only supervision versus all pairs (RATopo +3.3 TOP_ll).
5. **Temporal:** propagate a fraction of the previous set (30 % of queries optimal; 10 / 50 / 75 % worse), condition on the RELATIVE pose (trajectory conditioning hurts by 5 mAP), supervise the propagated set with identity-preserving targets (+2.2 mAP vs none), train the temporal stage after 12 single-frame epochs; gains +3 - 4 mAP on weak single-frame baselines; with an early BEV warp + per-cell max on a front-camera lane-graph model the gain is only +0.5 M-F and +0.3 C-F (TP31), the same order as our E41.
6. **Sensors:** camera + radar gives +4.3 DET_l, +3.0 TOP_ll, +4.9 OLS on nuScenes (TopoBDA); radar on top of LiDAR gave nothing; vision + LiDAR +0.07 TOPO F1 over vision in LaneGAP. Our foundation GaussianCaR (arXiv 2602.08784, ICRA 2026) is the only camera + radar BEV model of this kind seen in the search (nuScenes IoU 57.3 vehicles, 82.9 roads, 50.1 lane dividers); no paper builds a lane graph on top of a camera-radar BEV map output.
7. **Evaluation:** TOP is computed on matched lanes only (false-positive lanes are punished by DET_l alone), so TOP cannot judge spurious branches; the published graph-level metrics closest to ours are LaneGAP's TOPO / Junction-TOPO F1 (0.45 m match, 7.5 m traversal radius) and TopoEnhance's edge-set Jaccard.
8. **Dense raster methods lose structure in vectorisation, not in perception:** with perfect pixel-wise input the HDMapNet-type post-processing reaches only 0.773 / 0.788 (junction / TOPO F1) while piece and path representations are lossless (LaneGAP, Table 10); this supports a learned tracer instead of heuristic vectorisation of the dense head.
9. **Same ground-truth source (from memory):** the nuScenes-based sets (OpenLane-V2 subset B, STSU, CGNet, LaneGAP) take their centerlines from the nuScenes map expansion like our labels; their clipping (30 x 60 m or 100 x 50 m), merging of single-successor lanes and range differ from ours (+-50 x +-50 m).

### (h) State-of-the-art snapshot as printed in the primary papers (not a ranking: epochs, backbones, extra inputs and test protocols differ)

| task / setting | method (source card) | numbers |
|---|---|---|
| centerline, subset A, R50, 24 ep, v2.1 (OLS / TOP_ll) | TopoNet 39.8 / 10.9; TopoMLP 44.1 / 21.7; TopoLogic 44.1 / 23.9; T2SG 46.3 / 24.1; TopoFG 48.0 / 30.8; UniTopo 48.7 / 30.1; TopoPoint 48.8 / 28.7; CoPo 48.9 / 29.2; RATopo on TopoLogic 49.0 / 32.2 | TP09, TP08, TP01, TP14, TP05, TP06, TP02, TP13, TP12 |
| centerline, subset A, larger or extra inputs | UniTopo Swin-B 48 ep 51.0 / 35.1; TopoBDA camera 51.7 / 27.6, + LiDAR 56.4 / 35.5, + LiDAR + SD map 58.4 / 38.5; SMART on TopoMLP (SD + satellite) 53.1 / 37.0 | TP06, TP15, TP17 |
| centerline, subset B (nuScenes), R50, 24 ep | TopoLogic 42.3 / 21.6; TopoFG 45.4 / 27.2; UniTopo 47.6 / 31.8; TopoPoint 49.2 / 28.3; CoPo 49.7 / 31.8; RATopo on TopoLogic 50.3 / 37.3; TopoBDA 54.3 / 34.0; FASTopoWM slow 46.3 / 33.0 | TP01, TP05, TP06, TP02, TP13, TP12, TP15, TP20 |
| lane segment, subset A (TOP_lsls / mAP) | LaneSegNet 25.4 / 32.6; Topo2Seq 26.9 / 33.6 (24 ep), 29.9 / 37.7 (48 ep); TopoLogic 30.8 / 33.2; TopoFG 31.3 / 34.4; UniTopo 31.4 / 34.4; TopoStreamer 28.5 / 36.6; FASTopoWM slow 29.6 / 37.4 (31.6 / 37.2 with GeoDist); Lane Field on LaneSegNet 28.4 / 36.1 | TP10, TP16, TP01, TP05, TP06, TP19, TP20, TP11 |
| lane segment + SD map + temporal + stack, subset A | SCORE: DET_ls 44.0, TOP_ll 40.0, OLUS 54.9 (val), 55.8 (test) | TP18 |
| nuScenes centerline graph, 30 x 60 m, camera (TOPO F1 / JTOPO F1 / APLS) | CGNet 24 ep 42.2 / 34.1 / 30.7, 110 ep 53.2 / 43.3 / 41.4; LaneDiffusion 24 ep 46.8 / 38.8 / 37.1 | TP03, TP21 |
| nuScenes centerline graph, LaneGAP protocol (junction F1 / TOPO F1) | LaneGAP 110 ep 0.564 / 0.529; MapTR 0.496 / 0.465; HDMapNet 0.507 / 0.513 | TP04 |
| generative, OpenLane-V2 (lane-level M-P / M-R / M-T, subset A) | TopoGPT 45.6 / 42.6 / 24.4 | TP22 |
| challenge test scores | 2023 OLS 0.55 (TopoMLP team); 2024 UniScore 0.66 (LGmap), OLUS 0.58 (MapVision) | TP28 |

## 4. Reuse versus re-implement (given: no mmcv / mmdet, cached features, RTX 3060)

- **Re-implement in plain PyTorch (small, days at most):** TopoLogic mapping as a soft merge weight in the tree -> DAG step and as a bias on vertex-token attention (TP01, TP14, TP13: S); CoPo hard-negative contrastive loss for link or verifier heads (TP13: S); TopoEnhance-style corruption-trained verifier with per-relation normalised BCE and an edge-set Jaccard metric (TP07: M); LaneGAP Path2Graph merge and Junction-TOPO evaluation (TP04: S); column soft-argmax readout (TP26: S); piece-graph contraction for OpenLane-V2-style external evaluation (TP29, TP23: S).
- **Reuse the principle, new implementation (M):** keypoint-token queries with ordinal embedding and fixed denoising adjacency (TP05); distance-biased coupling of vertex and edge tokens with snapping (TP02); dense direction-tensor field as the dense-head output and as FiLM conditioning of the cached features (TP11); stream-query share and relative-pose conditioning for the time recursion (TP19, TP20).
- **Baseline only:** TopoMLP / TopoNet pair heads (TP08, TP09), SCORE stacking order (TP18).
- **Not suitable here:** Topo2Seq (needs trainable encoder), Topo2D (perspective features), TopoGPT and SeqGrowGraph decoders (token decoding), LaneDiffusion at full scale (45 U-Net passes), SMART pre-training (satellite data, GPS pose).
- **Adapt their code:** nothing; every repository is mm-series PyTorch 1.9.1 code; heads are short enough to rewrite.

## 5. Open questions and gaps

- No paper tests a link head or topology scorer on a cached frozen BEV raster, with matching-free dense labels, or on a vertex graph; the transfer of all numbers above is an extrapolation.
- Path counts of Graph2Path at +-50 m (many forks) are not reported by LaneGAP; the path view may explode at large range.
- Calibration of edge scores (TopoEnhance) is argued through TJS but not measured with a calibration metric; no paper evaluates calibration of vertex or fork / merge probabilities.
- Radar: the only lane-topology evidence is TopoBDA's nuScenes subset (camera + radar +4.9 OLS); no radar-specific design, no fusion of radar with SD map or temporal context.
- Class balance: all link losses are focal / BCE with no explicit re-balancing; only TopoEnhance (per-relation normalisation) and CoPo (hard negatives) address it; fork / merge rarity is never analysed.
- Metric comparability: TOP depends on devkit version; graph-level metrics (TOPO / JTOPO) use 30 x 60 m ranges; no published result at +-50 x +-50 m on nuScenes.
- Matching-free training for lane-topology decoders exists only as denoising (TopoFG), one-to-many (RATopo, UniTopo groups, SCORE) or contrastive (CoPo); no paper trains a vertex graph without any assignment.
- Not read: supplements (CGNet, TopoLogic), configs / bbox coders of the repositories (thresholds, relation-loss classes of TopoPoint / TopoFG / UniTopo), SEPT (arXiv 2505.12246), Chameleon (VLM, abstract only), TSTGT, 2nd / 3rd place 2023 reports; whether the master-branch `evaluate.py` equals the released v2.1 metric.
- Released code differs from the paper text in several places (TopoLogic: learnable w, P instead of alpha, lambda and no use of sigma; TopoPoint: snapping through lane links, no point detections; CGNet: BCE on the matched sub-matrix only). Reproducing a published number from the paper alone is therefore uncertain; the cards list what the code actually does.

## 6. Search log

1. TopoLogic lane topology GeoDist code Franpin github OpenLane-V2 NeurIPS 2024
2. OpenLane-V2 lane segment topology 2026 arXiv (extended)
3. Autonomous Grand Challenge 2024 OpenLane Topology track technical report lane topology winner
4. Mapless Driving challenge CVPR 2024 Autonomous Grand Challenge OpenLane-V2 lane segment solution technical report 1st place
5. CVPR 2025 Autonomous Grand Challenge Mapless Driving OpenLane-V2 lane segment perception technical report
6. "Mapless Driving" 2025 challenge lane segment OpenLane-V2 report arXiv topology solution OLUS (extended)
7. opendrivelab challenge2025 Autonomous Grand Challenge tracks list mapless OpenLane
8. lane topology reasoning 2026 arXiv centerline lane graph online BEV OpenLane-V2 new state of the art (extended)
9. T2SG lane topology reasoning traffic scene graph OpenLane-V2 arXiv
10. TopoStreamer temporal lane segment topology reasoning arXiv 2507.00709
11. arXiv 2026 online lane graph construction nuScenes centerline topology camera BEV junction-aware new method (extended)
12. arXiv 2026 lane segment perception topology reasoning OpenLane-V2 subset_A TOP_lsls state-of-the-art improves LaneSegNet (extended)
13. RoadPainter Points are Ideal Navigators for Topology transformER arXiv ECCV 2024 lane topology
14. Reusing Attention for One-stage Lane Topology Understanding arXiv 2507.17617
15. STSU Structured Bird's-Eye-View Traffic Scene Understanding from Onboard Images centerline graph Bezier association matrix
16. CVPR 2023 OpenLane Topology challenge technical report 2nd place solution lane topology OpenLane-V2 centerline
17. diffusion denoising lane topology centerline OpenLane-V2 generative lane graph BEV
18. centerline graph learning nuScenes JTOPO F1 TOPO F1 APLS SDA online lane graph outperform CGNet LaneGAP (extended)
19. SeqGrowGraph lane topology nuScenes Argoverse2 adjacency matrix expansion results CGNet comparison
20. radar camera fusion online HD map centerline lane topology nuScenes radar BEV lane graph
21. TopoMLP github wudongming97 README requirements mmdet3d PETR openlanev2 devkit (no README returned)
22. online lane topology improves end-to-end planning lane graph input planner 2025 2026 OpenLane-V2 nuScenes closed-loop (nothing found beyond LaneGAP + VAD)
23. lane topology reasoning vision-language model OR large language model prior OpenLane-V2 TOP_ll 2026 (found Chameleon, abstract only)
24. "Map Element Bucket" OpenLane-V2 challenge 2024 technical report team solution lane segment pedestrian crossing road boundary SD map (no further report)
25. outperforms LaneDiffusion CGNet centerline graph nuScenes Argoverse2 lane graph construction 2026 arXiv junction TOPO (nothing newer found)
26. online lane graph vertex detection edge classification BEV heatmap junction nodes camera nuScenes 2025 graph neural network centerline topology (extended; found Can et al. video and object-lane clustering papers, no heat-map / junction-node method)
27. lane topology BEV radar-camera fusion lane centerline connectivity 4D radar OpenLane OR nuScenes topology reasoning (no radar lane-topology paper besides TopoBDA; hit arXiv 2602.08784 = GaussianCaR, our foundation)
28. Fetched pages (not searches), arXiv HTML (also 2304.00930 and abs 2602.08784): 2405.14747 (three times), 2505.17771, 2511.12590, 2605.08911 (offsets 0 and 100000), 2407.11337 (twice), 2312.16108, 2304.05277, 2310.06753 (twice), 2502.08974, 2412.18951 (offsets 0 and 100000), 2303.08815, 2502.04329, 2306.09590, 2406.13988, 2406.10125, 2508.15272, 2411.18894, 2507.00709, 2507.23325, 2610.11058 (offsets 0 and 100000), 2610.25284, 2506.13553 (offsets 0 and 100000), 2606.31814, 2507.17617, 2407.15349, 2306.05419, 2507.01397, 2406.03105, 2511.06272, 2507.04822, 2504.01989 (survey), 2304.10440, 2311.04079, 2110.01997 (ar5iv), 2105.00195 (ar5iv); abs pages 2503.07485, 2311.04079.
29. Fetched code pages: Franpin/TopoLogic (README, tree listings, `topologic_head.py`, `sgnn_decoder.py`, `relationship_head.py`), Franpin/TopoPoint (README, listings, `topopoint_head.py`, `openlanev2_subset_A_dataset.py`), GXmmm18/TopoFG (README, listings, `topofg_head.py`), homothetic/UniTopo (README, listings, `unitopo_head.py`), XiaoMi/CGNet (README, listings, `CGNet_head.py`), hustvl/LaneGAP and hustvl/MapTR (README), OpenDriveLab/LaneSegNet and OpenDriveLab/OpenLane-V2 (README, issue 76, `openlanev2/centerline/evaluation/evaluate.py`); opendrivelab.com/challenge2025.
