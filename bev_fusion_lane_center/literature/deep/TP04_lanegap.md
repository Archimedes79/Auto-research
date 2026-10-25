# TP04 - LaneGAP: full-text extraction

| field | value |
|---|---|
| paper | Lane Graph as Path: Continuity-preserving Path-wise Modeling for Online Lane Graph Construction. Bencheng Liao, Shaoyu Chen, Bo Jiang, Tianheng Cheng, Qian Zhang, Wenyu Liu, Chang Huang, Xinggang Wang (HUST, Horizon Robotics). ECCV 2024. arXiv 2303.08815 (cs.CV; cs.RO). v1 2023-03-15, v2 2023-12-17, v3 2024-08-15 (current; this extraction reads v3). Links opened: arxiv.org/abs/2303.08815, arxiv.org/html/2303.08815v3, arxiv.org/html/2303.08815v1, github.com/hustvl/LaneGAP (page, issues #4 and #5, README, LICENSE), github.com/hustvl/MapTR (branch `maptrv2`: tree pages, LICENSE, issues #27 #100 #184 #193), github.com/songtaohe/Sat2Graph (metrics/topo), arxiv.org/html/2507.04822 (SeqGrowGraph, only for its LaneGAP row). |
| axis, tier, K ids | A3 (sparse lane graph), tier 1 core; K12 (path-level supervision instead of pieces); touches K47 (J metric) and K38 (rMCL) in section 5 |
| level-1 note(s) | [TP04](../papers/TP04_lanegap.md) (reading depth there: landing page only; several statements are now out of date, see "Corrections to the level-1 note" at the end of section 4) |
| code extraction | [lanegap](../code/lanegap.md): the repository `hustvl/LaneGAP` holds a README, a LICENSE and images, **no code**; the path extraction (Graph2Path) is in `hustvl/MapTR` branch `maptrv2`; Path2Graph and the TOPO / Junction-TOPO metric are **not released** |
| sources read | arXiv abstract page and the HTML of v3 (all sections 1 - 6 and appendices 0.A - 0.E, nine separate queries: structure, sections 1 and 3, section 4 + 5.1 + 5.2, sections 5.3 - 5.6, appendix tables, 5.6 text + references, section 3.3 + Algorithm 2, 5.4 + FPS + Table 9, v1 text); GitHub pages and raw files listed above; local files `literature/papers/TP04_lanegap.md`, `literature/register.md`, `experiment_execution/STATE.md` (line 36, E46), `experiment_execution/plans/Experiment_Table_EN.md`, `Literature_Compatibility_Space_EN.md`; all read 2026-10-25 |
| reading marks | every statement about the paper and its repositories is **digest**: all pages came through a summarising fetch tool (no page seen raw, the PDF was not read, figures were not seen). Numbers were transcribed by that tool; where two queries returned the same table the values agreed (Tables 1, 3 - 5, 10, 13, 14). Statements about our own files are **checked** (files read). Section 5 is my analysis, not paper content. Nothing here is **memory** except where a sentence says so |
| extracted by, date | Claude (Sonnet 5.5) literature-extraction agent, 2026-10-25 |

## 1. Problem and claim
- Predict an ego-centred, directed lane-centre-line graph online from six cameras (optionally LiDAR) and compare three decompositions of the graph: pixel-wise (HDMapNet: segmentation + direction map, then skeletonise and trace), piece-wise (STSU, a MapTR variant: segments between junctions plus a connectivity matrix), and the new **path-wise** one.
- The claim: the primitive should be the **path** (a root-to-leaf walk of the directed graph, i.e. one traffic flow), because pieces fragment at junctions and pixels lose structure in the vectorisation. A MapTR-style set predictor learns the paths end to end; a non-learned **Path2Graph** step (discretise, add vertices and edges, merge close vertices) rebuilds the graph; no link head exists.
- The one claim that matters for us: on ground-truth inputs the path and piece recoveries are lossless (1.000 / 1.000) while the pixel-wise recovery already loses 0.21 - 0.23 F1 (Table 10), and with the same decoder and encoder the path-wise model beats the piece-wise one by 0.068 / 0.064 (junction / TOPO F1, nuScenes, my arithmetic from Table 1). The paper scores forks and merges only through Junction TOPO (no fork / merge detection counts) and never states how many paths a scene has.

## 2. Method as built

### Pipeline / architecture (digest)
```
6 cameras 6 x 900 x 1600, resize ratio 0.5
  -> ResNet-50 -> GKT-style 2D-to-BEV  -> F_BEV  200 x 100 cells, 0.3 m grid  (60 m x 30 m)
        |                                       |
        |                                       +-> aux BEV segmentation head phi_seg (CE, training only)
        v
  MapTR-style decoder: randomly initialised learnable path queries, 6 deformable decoder layers
        |    one stage, all paths at once
        v
  per query: class score (focal) + N_v ordered points in (0,1)-normalised BEV coordinates
        |    polyline: N_v = 30 (default); Bezier: N_b control points (3 / 5 / 10 tested)
        v
  Hungarian one-to-one matching to ground-truth paths (training)
  inference: Path2Graph (rule, no gradient) -> directed graph
```
Channel width, number of queries, score threshold, head widths: **not found** in the text (see section 6). Encoder and decoder are shared with the piece-wise MapTR baseline; only the target representation differs.

### Inputs, outputs
- Input: 6 surround cameras; LiDAR-only and vision + LiDAR variants exist (appendix). nuScenes range x in [-15, 15] m, y in [-30, 30] m (30 m x 60 m); Argoverse2 uses 7 cameras at 7 x 2048 x 2048, resize 0.3, height dropped; a 50 m x 100 m range is tested separately (Table 7).
- Output: a set of M ordered paths, each a polyline of N_v points in (0,1) normalised coordinates (or a Bezier curve sampled through a Bernstein matrix), then a directed graph. Direction is the order of the points (root to leaf).
- Ground truth: a directed local lane graph around the ego; how it is cropped and which map is used is not described in the text (the code of the MapTRv2 repository does it, see [../code/lanegap.md](../code/lanegap.md)).

### Graph2Path (labels), Algorithm 1 (digest)
1. Collect roots (in-degree 0) and leaves (out-degree 0) of the ground-truth directed graph.
2. For every (root, leaf) pair run a depth-first search (`findpath`); keep the path if one is found.
3. The label set is all found paths; M = their number.
- Shared segments are not treated: each path is stored in full, so a trunk before a fork appears in every path through it. No pruning, no maximum path length, no minimum length, no de-duplication is described. The number of paths per sample is not reported.
- The released preprocessing differs in how it is done (all simple paths from each root to all leaves, 0.2 m Douglas-Peucker simplification per path), see section 4.

### Path representations (digest)
- **Polyline:** N_p ordered points regressed directly; deformable attention samples BEV features around the points. Default 30 points (Table 4: 10 / 30 / 40).
- **Bezier:** N_b control points b_j; curve point B(t) = sum_{j=0}^{N_b-1} C(N_b-1, j) t^j (1-t)^{N_b-1-j} b_j for t in [0,1]; sampled at parameters t_k, k = 0 .. K-1, as the matrix product B = Gamma x V_Bezier, Gamma in R^{K x N_b} the Bernstein basis matrix with entries C(N_b-1, j) t_k^j (1-t_k)^{N_b-1-j}; "Bezier deformable attention" samples around the K curve points. Tested with 3 / 5 / 10 control points; 5 is best (Table 5). The paper does not say which representation the main numbers use; Table 1 headlines use the polyline (30 points, it is the stated default).

### Losses (digest)
- **Eq. 3 (set loss):** L_bipartite = sum_{i=1}^{N} [ L_Focal(p_hat_{sigma_hat(i)}, c_i) + 1{c_i != empty} L_path(V_hat_{sigma_hat(i)}, V_i) ], sigma_hat the optimal assignment found by the **Hungarian algorithm**, N the number of queries (every query gets a class target, unmatched ones are "empty").
- **L_path** = sum_{j=0}^{N_v-1} L1(v_hat_j, v_j): L1 over the N_v point pairs of a matched path, **in the fixed order of the points** (arc-length equidistant points of the label, same order in the prediction).
- **L_auxseg** = CE(phi_seg(F_BEV), M_BEV): BEV segmentation on the encoder output, **training only**, no inference cost (Table 3: +0.010 junction F1, +0.011 TOPO F1 at 24 epochs).
- **Eq. 5 (total):** L_total = L_cls + L_path + L_auxseg; weights are not written, so presumably 1 each. (The MapTRv2 repository's centerline config uses classification weight 2.0, point L1 5.0, direction cosine 0.005, BEV seg 1.0, but that is MapTRv2 with four map classes, not a LaneGAP config.)
- Assignment rule: Hungarian one-to-one, matching cost not stated (MapTR convention: focal class cost + ordered point L1). No one-to-many branch in the paper.

### Path2Graph (inference), Algorithm 2 - the full algorithm as published (digest)
1. Create an empty directed graph G.
2. For every predicted path: discretise it into an ordered vertex sequence (interval **0.15 m**, Sec. 5.2); add every vertex to G; unless it is the last one, add a directed edge to the next vertex.
3. After all paths are inserted, **merge** overlapping vertices of different paths whose distance is below a threshold (**0.15 m**, "the same as the graph interpolation size", Sec. 5.2); the merged vertex **inherits the predecessors and successors** of both originals.
4. Return G.
- There is **no angle or direction criterion**, no snap, no spur or dead-end removal, no length filter, no junction detection step, no use of the class score inside the merge (it states nothing on a score threshold for keeping paths).
- Not specified: where the merged vertex sits (mean? one of the two?), whether merging is transitive (clusters) or pairwise, the processing order, how self-loops and double edges are handled, the search structure; the interpolation is not said to be by arc length.
- Output: a directed graph whose vertices are 0.15 m apart (about 667 vertices per 100 m of lane, my arithmetic), edges between consecutive vertices; forks and merges are vertices with out-degree or in-degree above 1. Cost: "negligible", 16.5 FPS network only against 15.6 FPS with Path2Graph (about 3.5 ms per frame, my arithmetic from 1000/15.6 - 1000/16.5), while HDMapNet's post-processing costs 17.8 -> 6.8 FPS.

### Metrics (Sec. 4, digest) - needed to read every number below
- **TOPO** (adapted from He and Balakrishnan, WACV 2022, which uses the TOPO idea of Biagioni and Eriksson): interpolate both directed graphs so that connected vertices are 0.15 m apart; two vertices are candidates if closer than **0.45 m**; a **maximal one-to-one matching** gives the pairs P_pair; from every matched pair traverse the directed graph for **less than 7.5 m** on the prediction side and on the ground-truth side to get sub-graphs S_hat and S; match the sub-graph vertices with the same 0.45 m rule; per pair Pre = N_subpair / |S_hat|, Rec = N_subpair / |S|; Precision_TOPO = sum Pre / |V_hat|, Recall_TOPO = sum Rec / |V|, with V_hat, V the interpolated vertex sets of the prediction and the ground truth. F1 is reported but not defined in the text.
- **Junction TOPO:** the same on the N_junction ground-truth junction points (merging and forking points): traverse both graphs for less than 7.5 m from the junction point to get a sub-graph pair, precision and recall per pair; how the pairs are aggregated is not stated.
- Undirected versions of both ignore the predecessor direction (appendix and v1 tables).
- Not stated: whether the directed traversal goes only forward, how junction points are extracted from the ground truth (degree rule), how unmatched ground-truth junctions are scored, tie-breaking of the matching.

### Training recipe (Tables 11 and 12, digest)
- AdamW (beta 0.9 / 0.999, eps 1e-8), lr 6e-4, cosine decay, 500 warm-up iterations, weight decay 0.01, gradient clip 2-norm 35, total batch 32 on 8 RTX 3090, BEV grid 0.3 m. nuScenes 110 epochs for the headline rows (24 epochs for the ablations), Argoverse2 24 epochs. Training time Table 15: HDMapNet 42 h, MapTR 49 h, LaneGAP 58 h (vision), 60 h (vision + LiDAR). Augmentation, number of queries, loss weights: not stated.

### Inference
- Path2Graph above; otherwise: **not found** (score threshold for keeping paths, any NMS or duplicate rule, cap on paths).

## 3. Numbers
All values are author-reported (v3 HTML, digest), nuScenes val 30 m x 60 m camera-only unless stated, same encoder (ResNet-50 + GKT) for all rows. Do not compare with our numbers (100 m x 100 m, frozen camera-radar BEV).

**Table 1 (directed graph; Junction TOPO F1 / TOPO F1; parameters; FPS)** FPS_net / FPS measured on one RTX 3090 + 24-core AMD EPYC 7402; batch size and input size of the timing not given.

| dataset | method | modelling | Junction F1 | TOPO F1 | params | FPS |
|---|---|---|---|---|---|---|
| nuScenes (110 ep) | HDMapNet | pixel | 0.507 | 0.513 | 35.4 M | 6.8 |
| nuScenes | STSU | piece | 0.419 | 0.395 | 35.3 M | 15.1 |
| nuScenes | MapTR variant | piece | 0.496 | 0.465 | 36.0 M | 15.0 |
| nuScenes | LaneGAP | path | **0.564** | **0.529** | 35.9 M | 15.6 (16.5 network only) |
| Argoverse2 (24 ep) | HDMapNet | pixel | 0.533 | 0.535 | 35.4 M | 4.6 |
| Argoverse2 | STSU | piece | 0.477 | 0.449 | 35.3 M | 12.4 |
| Argoverse2 | MapTR variant | piece | 0.522 | 0.500 | 36.0 M | 12.5 |
| Argoverse2 | LaneGAP | path | **0.616** | **0.565** | 35.9 M | 13.0 |

**Table 2 (OpenLane-V2, mIoU, 50 m x 100 m):** HDMapNet 18.3, STSU 31.1, VectorMapNet 25.0, MapTR 35.7, TopoNet 39.0, LaneGAP (dagger) 40.6 (+1.6 over TopoNet). Rows other than LaneGAP are copied from the TopoNet paper; **LaneGAP (dagger) is a variant built on TopoNet's piece-based code base**: only the lane-piece detection is replaced by path-wise representation and the piece-connection modules are removed, other hyper-parameters as TopoNet. It is not the nuScenes model. How mIoU is computed from a graph is not stated beyond rendering predictions with a 0.75 m line width.

**Ablations (24-epoch schedule, nuScenes, Junction / TOPO F1):**
- Table 3, auxiliary BEV segmentation: 0.529 / 0.450 -> 0.539 / 0.461.
- Table 4, polyline points: 10 -> 0.532 / 0.450 (16.3 FPS); 30 -> 0.539 / 0.461 (15.6); 40 -> 0.557 / 0.472 (15.1).
- Table 5, Bezier control points: 3 -> 0.481 / 0.365; 5 -> 0.510 / 0.477; 10 -> 0.501 / 0.388. The text claims an 11.2 % TOPO F1 gain that does not match these cells.
- Table 13, modality (Junction P / R / F1; TOPO P / R / F1; params; FPS): vision 0.554 / 0.524 / 0.539; 0.432 / 0.493 / 0.461; 35.9 M; 15.6. LiDAR 0.608 / 0.594 / 0.600; 0.467 / 0.559 / 0.509; 9.1 M; 8.6. Vision + LiDAR 0.630 / 0.600 / 0.615; 0.504 / 0.561 / 0.531; 39.8 M; 5.5.
- Table 14, schedule: vision 24 ep 0.539 / 0.461, 110 ep 0.564 / 0.529 (P / R at 110 ep: junction 0.591 / 0.539, TOPO 0.547 / 0.511); vision + LiDAR 24 ep 0.615 / 0.531, 110 ep 0.632 / 0.597. Longer training helps TOPO (+0.068) far more than junctions (+0.025), my arithmetic.
- Table 10, recovery from **ground-truth** inputs (Junction P / R / F1; TOPO P / R / F1): pixel 0.898 / 0.678 / 0.773; 0.913 / 0.693 / 0.788. Piece and path 1.000 on all six.
- Table 6, planning with VAD (average L2 / collision): VAD-Tiny 0.78 m / 0.38 %; + MapTR 0.74 / 0.34 %; + LaneGAP 0.70 / 0.29 %. Open-loop L2 only.
- Table 7, large range 50 m x 100 m (Junction / TOPO F1): HDMapNet 0.361 / 0.379, STSU 0.303 / 0.285, MapTR 0.363 / 0.338, LaneGAP 0.392 / 0.388. Table 8, a new (geographically disjoint) split: HDMapNet 0.373 / 0.379, STSU 0.274 / 0.268, MapTR 0.338 / 0.334, LaneGAP 0.400 / 0.398. Table 9, mIoU (nuScenes / Argoverse2): HDMapNet 48.1 / 55.0, STSU 38.2 / 44.7, MapTR 43.6 / 49.3, LaneGAP 49.2 / 55.9. (The schedule of Tables 7 - 9 is not stated in what I read.)
- Table 15, trade-off: HDMapNet pixel 0.507 / 0.513, 6.8 FPS, 42 h; MapTR piece 0.496 / 0.465, 15.0 FPS, 49 h; LaneGAP path 0.564 / 0.529, 15.6 FPS, 58 h; LaneGAP vision + LiDAR 0.632 / 0.597, 5.5 FPS, 60 h.

**Compute.** 35.9 M parameters; 16.5 FPS network only, 15.6 FPS with Path2Graph on an RTX 3090 including the image backbone (about 60.6 ms and 64.1 ms per frame, my arithmetic). Not comparable with our 31 ms chain, which starts from cached frozen BEV features and has no image encoder.

**Caveats.** Range 30 m x 60 m (our +-50 m square is about 5.6 times the area, my arithmetic); camera-only; one-to-one matched metric with 0.45 m tolerance, below our 0.5 m cell; the metric code is not released (section 6); the paper's own text cross-references several tables wrongly (auxiliary branch cited as Table 4 but is Table 3, Bezier as Table 6 but is Table 5, mIoU as Table 10 but is Table 9, both training-settings tables cited as Table 12). The v1 text on arXiv is a different state of the paper: nuScenes only, with a 0.537 / 0.515 row for the model without the auxiliary segmentation and 0.564 / 0.529 with it, 24-epoch vision 0.529 and LiDAR 0.595 (digest of v1).

## 4. Paper versus code
- **No code of the paper's model.** `hustvl/LaneGAP` contains `assets/`, `.gitignore`, `LICENSE`, `README.md` (8 commits, 2 open issues, 155 stars when read). Its README says the LaneGAP code was released together with MapTRv2 in `hustvl/MapTR`. Maintainer replies (LaneGAP issue #4, 2023-09-08): path extraction is a **data preprocessing step** that casts the lane graph into the path representation and lives in the MapTRv2 repository; (issue #5, 2023-09-06): LaneGAP extends MapTR and both were merged into the unified framework MapTRv2.
- What the MapTRv2 branch contains of LaneGAP: a fourth map class `centerline` in the nuScenes map converter (`union_centerline`: graph of lane and lane-connector centrelines, all simple root-to-leaf paths, 0.2 m simplification) and a config `maptrv2_nusc_r50_24ep_w_centerline.py` (20 points per instance, 900 shared queries, results marked WIP). **Path2Graph, the TOPO and Junction-TOPO metric and the Argoverse2 path-wise setting are not released**; three open requests in the MapTR tracker (#100 "Path2Graph" 2023-09-11, #184 2024-08-26, #193 "Realization for LaneGAP metric" 2024-12-20) have no visible reply.
- Differences between the paper text and the released path extraction: all simple paths from each root to all leaves (`networkx.all_simple_paths`) instead of a root-leaf pair loop with DFS (same path set); Douglas-Peucker simplification with 0.2 m tolerance of every path before fixed-number resampling; node identity by coordinates rounded to 3 decimals and explicit pred-last-point to start-point edges, not by a distance threshold; no cap on the number of paths; no minimum length; 20 resampled points per instance in the MapTRv2 config against 30 in the paper. Details in [../code/lanegap.md](../code/lanegap.md).

### Corrections to the level-1 note ([TP04](../papers/TP04_lanegap.md))
1. **Code line:** the note says the landing page only was read and the path-wise code is announced in the MapTR news. Now checked: `hustvl/LaneGAP` has no code (README, LICENSE, assets); licence = Apache License 2.0 (confirmed by the LICENSE file; the copyright line of the licence appendix is not filled); `hustvl/MapTR` branch `maptrv2` licence = MIT, "Copyright (c) 2022 Hust Vision Lab" (confirmed by the LICENSE file; `literature/repos/README.md` still lists MapTR as "licence not stated"). The branch was opened: only Graph2Path exists there.
2. **"Fit to our project: Path2Graph is exactly our tree -> DAG conversion by clustering (0.15 m)" overstates.** The published Path2Graph is five lines: distance-only merge at 0.15 m, no angle, no snap, no spur removal, no junction head, details unspecified, validated only on exact ground-truth paths (Table 10). Our assembly (merge radius 1.5 m, snap 1.0 m, spur removal, junction head, fork relocation) does more and works at ten times the radius. The note's "effort S" holds for Graph2Path labels, not for Path2Graph or the metric (neither is released, section 6).
3. **Results line is incomplete:** add Argoverse2 (LaneGAP 0.616 / 0.565 against MapTR 0.522 / 0.500, 24 epochs), vision + LiDAR at 110 epochs (0.632 / 0.597), the schedule effect (TOPO +0.068, junction +0.025), the FPS hardware (RTX 3090 + 24-core EPYC 7402), and that **LaneGAP (dagger) on OpenLane-V2 is a variant on TopoNet's code base**, not the nuScenes model.
4. **"Training: 110 epochs on nuScenes"** is right for the headline rows; Argoverse2 rows use 24 epochs (Table 12).
5. **"path count per scene not reported"** is confirmed for the paper; our own E46 log gives the count at +-50 m (10.1 GT paths per VAL sample, 14.0 TEST, section 7).
6. Unchanged and confirmed: Hungarian one-to-one, loss terms, 6 deformable layers, 30-point polyline default, Bezier via Bernstein matrix, TOPO thresholds (0.15 m / 0.45 m / 7.5 m), Tables 1, 3 - 5, 10, 13 numbers of the note.

## 5. Transfer to our pipeline

| question | answer |
|---|---|
| what it needs from us | **Graph2Path labels:** only our ground-truth directed piece graph (nodes with in- and out-degree) per frame, offline and cacheable; no sensor data, no frozen-BEV change. **TOPO / Junction TOPO:** predicted and ground-truth directed graphs as vertex + edge lists in the ego frame, interpolated to a spacing far below the tolerance. **Their decoder** would need a trainable BEV encoder or at least a deformable decoder over our cached 200 x 200 x 39 raster (re-implementable with `grid_sample`), plus Hungarian matching: not wanted. |
| what it gives us | Axis A3 / A9. (1) A per-**path** training signal for the hypothesis tree (K12): every root-to-tip chain of our tree is already a LaneGAP path; the paper supplies the label construction, not a loss. (2) A graph-level evaluation, TOPO and Junction TOPO, that scores connectivity along traversals and local junction topology, complementary to F1@1 m, lanes found and J (fork / merge F1 at 6 m). (3) Evidence that decomposing into paths is lossless while heuristic vectorisation of a dense raster is not (Table 10: 0.77 / 0.79 even from perfect GT). |
| conflicts | Hungarian one-to-one matching and a fixed point order in L_path (both forbidden on our training path); a trainable perspective-to-BEV encoder; a rule step (Path2Graph) after the network, used as the *only* graph builder; 0.3 m BEV grid and +-15 x +-30 m range against our frozen 0.5 m, +-50 m raster; camera (+ LiDAR) only; no code for Path2Graph or the metric; shared trunks are predicted again for every path, so a path-set output is **anti-sparse** (our champion has 13.5 vertices per 100 m; Path2Graph's output has about 667 per 100 m by construction). |
| reuse mode | Graph2Path: **re-implement from the description** (about 30 lines, the released converter is MIT and shows the way, [../code/lanegap.md](../code/lanegap.md)). Path-consistency loss: **idea only**, our own formulation (below). Path2Graph: **idea only**, we have a richer assembly. TOPO / Junction TOPO: **re-implement from the paper's description** (no code); the older TOPO of Sat2Graph (MIT label) is an algorithmic reference, not a drop-in. |
| effort | S: Graph2Path labels from our piece graph, path counts per frame, path-level label cache. M: per-path chain loss (padded GT path tensor, pairwise point-to-polyline tensor, soft-min, annealing) and its A/B on two seeds. M: TOPO and Junction TOPO in `evaluation/` with self-tests, because the text leaves aggregation and traversal direction open (section 6). Blocker: no reference implementation to reproduce a published number; a self-made metric cannot be checked against LaneGAP's 0.564 / 0.529. |
| expected gain and risk | Open items it touches: lanes found (0.695 against 0.800 for the control at equal sparsity), junction recall J (0.265 against 0.408), and our own path view E46 (path recall 0.507 VAL / 0.428 TEST, only 8.0 predicted against 10.1 GT paths per sample). A per-path coverage term attacks exactly that headroom; risk: E46 / E51 path **repair** at inference was worth +0.001 and learned link heads stayed within 0.01 of the rules, so a tie inside the noise band (0.010 in S) is the likely outcome on clean frames; a gain is plausible in path recall and in wrong-exit errors at junction mouths, which S barely sees. |
| feeds which experiment / K id | K12 (path term of L) is the home. Proposed sub-variants to register before use: K12a per-path soft-min coverage (GT path -> best chain), K12b chain-to-nearest-GT-path consistency (chain -> best GT path), both rMCL-style relaxed winner-take-all (K38), no Hungarian. TOPO / Junction TOPO: a report next to K47 (J), not a replacement. |

### 5.1 Path-wise supervision as a label and loss construction for our graph
What the paper's labels are: the set of all root-to-leaf paths of the ground-truth DAG (Graph2Path). What changes for a **matching-free point-to-polyline loss**, and what does not:

1. **No change for a loss on the union.** Every edge of a DAG lies on at least one root-to-leaf path, so the union of all paths is the union of all edges. Our current two-sided point-to-polyline distance (predicted vertices to the ground-truth polylines and back) is identical whether the target is the edge set or the path set. Paths add information only if the loss is **per path**.
2. **Per-path, one-sided each way, no matching (my design, nothing of it is in the paper).** Let the predicted tree give chains c (root to each tip, vertices with validity pi), and the label give paths g (cached per frame, padded to a cap).
   - chain -> path: d_c = softmin_g [ mean over vertices p of c of dist(p, polyline g) ], weighted by chain validity. A chain that follows lane A and then drifts to the parallel lane B (or takes the wrong exit of a fork) has all its vertices near the *union* of the labels but near no single *path*; only this term sees it. Distance to the polyline (segment projection), not to the end points, so a chain that is a prefix of a path is not penalised for being short ("better short lane graphs than false ones").
   - path -> chain: e_g = softmin_c [ mean over points q of g of dist(q, chain c) ]: every ground-truth path must be traced end to end by one chain. This is the continuity-of-recall term and the direct counterpart of the E46 path recall (0.507 VAL) that we measured as an evaluation only.
   - The min is a relaxed winner-take-all with gradient only through the selected items (allowed by the project rules; use the rMCL / aMCL temperature schedule of K38, not an argmax), so there is no Hungarian assignment and no fixed point order, unlike LaneGAP's L_path.
   - Direction: add a heading term (cosine between the vertex heading and the tangent of the nearest path segment), otherwise a chain can lock on the oncoming lane 3.5 m away.
   - Shared trunks: with a min over paths nothing needs re-weighting; for any sum over paths weight by 1 / (number of paths through the segment).
3. **Label construction details to copy or avoid.** Roots and leaves include nodes created by the **window crop** (a lane entering from outside is a root), so the path set depends on the crop; a lane leaving and re-entering the window becomes two components. Count paths per frame and cap them (the released code has no cap, `all_simple_paths` grows multiplicatively with forks in series and does not terminate quickly on cycles such as roundabouts); our E46 count at +-50 m is 10.1 (VAL) / 14.0 (TEST) per sample, so a cap of 32 - 48 with a logged overflow rate is enough to start. Resample every path to a fixed number of arc-length-equidistant points only for the *distance tensor*, never as a regression target. The released code simplifies every path with Douglas-Peucker at 0.2 m before resampling; we already have a Douglas-Peucker oracle (14.3 vertices per 100 m), so path labels can be built from the oracle graph for free.
4. **Cost.** Chains C (tens) x paths G (at most 48) x points: a few hundred thousand segment distances per frame, negligible on the GPU next to the ray reads.

### 5.2 Path2Graph versus our DAG assembly
| aspect | Path2Graph (paper) | ours |
|---|---|---|
| input | independent predicted paths, each a 30-point polyline, trunks repeated per path | a hypothesis **tree**: shared prefixes are one object by construction; only convergence (merge) needs joining |
| vertex spacing | 0.15 m after interpolation (metric-driven) | about 7 m (13.5 vertices per 100 m) |
| merge test | Euclidean distance < 0.15 m, vertices of different paths | merge radius 1.5 m, snap 1.0 m |
| direction | none | travel heading available per vertex |
| spurs / dead ends | not handled | spur removal |
| junction handling | implicit (degree > 1 after merging) | junction head + fork relocation |
| evidence it works | lossless on ground-truth paths (Table 10) where duplicates coincide exactly; no experiment on predicted paths separately | S, J, lanes found on VAL |
| gradient | none | none on the assembly; the training path stays differentiable |
Reading: the 0.15 m radius works on ground truth because duplicate segments are identical points; on predicted paths the regression error of a Transformer decoder between two copies of a trunk is not reported, and any copy more than 0.15 m away stays a separate vertex, which the 0.45 m metric tolerance forgives but a sparsity score would count (my reading, not tested by the paper). The one transferable lesson is order of operations: merge by **geometry only after prediction**, and score graph quality after interpolation. Their 0.15 m threshold equals their interpolation step and one third of their match tolerance; our 1.5 m radius is three cells of the 0.5 m raster, and whether that is the right tie to the grid and to the heat-map width (sigma 0.75 m) is what a radius sweep would test. No change to the assembly is suggested by this paper; the only open item it raises is "log root-to-leaf paths per frame; radius sweep" (Literature_Compatibility_Space, T14), now partly answered by E46.

### 5.3 Metric: TOPO / Junction TOPO next to F1@1 m, J and the path view
- Threshold regime: 0.45 m is below our 0.5 m cell; with a frozen 0.5 m BEV and a heat map of sigma 0.75 m we should expect low values at 0.45 m and report a **tolerance curve** (0.45 / 1.0 / 1.5 m, traversal 7.5 m and maybe 15 m), never a single number compared with the paper's 0.529.
- Junction TOPO scores the sub-graph reachable within 7.5 m of a ground-truth junction; our J scores the position of forks and merges within 6 m. A fork found in the right place with a wrong or missing branch passes J and fails Junction TOPO. This is the missing complement and also a check on fork relocation.
- Because the metric needs matching, it is an **evaluation-only** matching; it does not touch the training path.
- Ground truth for Table 10 style self-tests: Graph2Path on the ground truth, Path2Graph-like merge, TOPO against the original must give 1.000 - a free layer-0 test (project rule: self-tests of metric code first).

## 6. Not found / open
- Number of path queries, hidden size, score threshold for keeping paths, loss weights actually used, augmentation, matching cost weights, N_v used in Table 1 beyond "30 default", the representation (polyline or Bezier) behind the headline numbers.
- Path2Graph: position of the merged vertex, transitivity, processing order, self-loop and double-edge handling, search structure, behaviour on predicted (not ground-truth) duplicates; the PDF was not read, so an appendix algorithm figure could hold more (the HTML v3 has none).
- TOPO: F1 definition, direction of the traversal in the directed version, junction extraction rule, aggregation of Junction TOPO, border handling, the matching algorithm; **no implementation is released** (MapTR issue #193 open since 2024-12-20, #100 since 2023-09-11, #184 since 2024-08-26; no maintainer reply visible; LaneGAP issue #4 and #5 do not mention it).
- Ground-truth generation text (nuScenes and Argoverse2): absent from the HTML appendix; the MapTRv2 converter shows nuScenes only. Number of ground-truth paths, vertices or junctions per sample: not reported. Schedule behind Tables 7 - 9. How mIoU (Tables 2, 9) is computed from a graph.
- How SeqGrowGraph obtained its LaneGAP row (landmark / reachability F 53.2 / 47.5 on the default split, from its Table 1; LaneGAP does not report those metrics, the digest says "paper or released checkpoint"): unresolved, no LaneGAP checkpoint was found in what I read.
- Figures (qualitative results, failure cases: distant junctions at the range edge, occlusion, darkness) were not seen; only captions.

## 7. Our experiments that touch it
- **E46 / E47** (loop 4, `Experiment_Table_EN.md`, `STATE.md` line 36): "LaneGAP path view and repair". Path view with `path_eval.py` (criterion: 90 % of the points of a path within 1.5 m of one predicted path; the file is no longer under that name in the tree). Single-frame S2: path recall / precision / F1 VAL 0.492 / 0.597 / 0.539, TEST 0.415 / 0.529 / 0.465; final (A) VAL 0.507 / 0.625 / 0.560, TEST 0.428 / 0.557 / 0.484. Ground-truth paths per sample 10.1 (VAL) and 14.0 (TEST) against 8.0 and 10.6 predicted; interior dead ends per sample 0.40 GT against 0.83 predicted (VAL); pieces per sample 24.0 GT against 11.9 predicted. Verdict in the table: only 0.428 of the TEST paths recovered, path **repair** +0.001, led to the trace network. This is an evaluation and an inference-time repair; **no training loss over paths was run**.
- **E42 / E45 / E46 / E51 / E56:** learned link heads, TopoLogic-style scoring and path repair stayed within 0.01 of the rules and were dropped (growth note in `sources/lit16/growth.md`).
- K12 in `register.md` ("path term of L") has no experiment that uses it as a loss; `sources/proposal_sparse.md` line 76 names "an optional path-consistency loss on the traced chains" as the intended use. Section 5.1 is a concrete version of it.
- Not tried: TOPO / Junction TOPO, a per-path loss, path counts per frame at the loop-16 range with the hypothesis tree, a radius sweep tied to path count.
