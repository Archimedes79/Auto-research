# TP29 - OpenLane-V2: full-text extraction

| field | value |
|---|---|
| paper | OpenLane-V2: A Topology Reasoning Benchmark for Unified 3D HD Mapping. Huijie Wang, Tianyu Li, Yang Li, Li Chen, Chonghao Sima, Zhenbo Liu, Bangjun Wang, Peijin Jia, Yuting Wang, Shengyin Jiang, Feng Wen, Hang Xu, Ping Luo, Junchi Yan, Wei Zhang, Hongyang Li. NeurIPS 2023 Datasets and Benchmarks track. arXiv 2304.10440 (cs.CV): v1 2023-04-20, v2 2023-06-10, v3 2023-10-28 (the text read here). Links opened: arxiv.org/abs/2304.10440, arxiv.org/html/2304.10440v3 (main text and supplementary A - G), github.com/OpenDriveLab/OpenLane-V2 (README, docs/*.md, data/README.md, tutorials/Centerline.ipynb, openlanev2/ source, issue #76, PR #77 and its commit, tags, commit list), the papers TopoMLP (arXiv 2310.06753), TopoLogic (2405.14747), TopoBDA (2412.18951), LaneSegNet (2312.16108 and its README), the SCORE paper (2507.01397) for numbers only |
| axis, tier, K ids | A9 (evaluation, data, metrics), tier 1 core; K133 (OpenLane-V2, "external reference metrics" in the register); no new K id proposed here (a new one for the adapter is suggested in section 5, to be registered first) |
| level-1 note(s) | [TP29](../papers/TP29_openlane-v2-data-and-metrics.md); related: [SN09](../papers/SN09_topobda-the-only-camera-radar.md) (camera-radar rows on subset B), [TP01](../papers/TP01_topologic.md), [TP18](../papers/TP18_score.md) |
| code extraction | [openlane-v2](../code/openlane-v2.md) (metric code only, as asked) |
| sources read | see "paper" row. Paper: v3 HTML only (v1 PDF came back as binary, unreadable; v1 / v2 texts not compared). Code: files of the `master` branch (head commit d54a0dd, 2025-07-02), tags listed on the tags page; the diff of the metric fix (PR #77, commit 5d31f86) through its web page; all read 2026-10-25 |
| reading marks | **digest** for everything that comes from a page (every fetch went through the summarising tool; no raw file was seen, so numbers and control flow may be mis-transcribed). **checked (own arithmetic)** for the cross-checks that I computed from digest numbers (OLS with square root, frame counts, version deltas). **derived** for consequences I reasoned out from the digest (marked as such). Section 5 is my analysis, not paper content |
| extracted by, date | Claude (Sonnet 5.5) literature-extraction agent, 2026-10-25 |

## 1. Problem and claim
- A benchmark, not a model: multi-view images in, a directed lane centre-line graph (centre-lines as 3D polylines, directed adjacency between them) plus small traffic elements (2D boxes in the front view) and their correspondence to centre-lines out. Scored with DET_l, DET_t, TOP_ll, TOP_lt and the combined OLS.
- Gap closed (digest): earlier lane benchmarks score lane detection only; OpenLane-V2 adds the connectivity between lanes and between lanes and traffic elements ("topology reasoning") on 2,000 scenes built on Argoverse 2 (subset A) and nuScenes (subset B).
- The claim that matters for us: a metric pair (Frechet-distance AP for pieces, link-prediction AP for their adjacency) that is the de-facto external yardstick for lane graphs, and whose TOP part changed between devkit versions (TOP_ll 4.1 became 10.9 for the same TopoNet, section 3), so a number is only meaningful with its devkit version.

## 2. Method as built

The "method" here is the data construction and the evaluation procedure. Baselines (STSU, VectorMapNet, MapTR, TopoNet) are only context.

### Data (digest)
| item | subset A (Argoverse 2) | subset B (nuScenes) |
|---|---|---|
| scenes (segments) train / val / test | 700 / 150 / 150 | 700 / 150 / 150 |
| keyframe rate, mean segment length | 2 Hz, 15 s | 2 Hz, 20 s |
| samples train / val / test (TopoBDA paper, Sec. 4.1) | 22,477 / 4,806 / 4,816 | 27,968 / 6,019 / 6,008 |
| cameras, image size | 7, 2048 x 1550 (front view transposed to 1550 x 2048) | 6, 1600 x 900 |
| annotation window | x +-50 m, y +-25 m, ego at the origin | same |
| z of centre-lines | present | set to 0 (the source HD maps have no height) |
| centre-lines per frame (train / val / test) | 26.34 / 26.44 / 26.50 | 24.32 / 24.80 / 23.82 |
| connections per centre-line (train / val / test) | 1.90 / 1.89 / 1.89 | 1.83 / 1.79 / 1.84 |
| traffic elements per frame | 3.70 / 3.69 / 2.80 | 3.58 / 3.76 / 3.25 |
| cities | Miami 35.4 %, Pittsburgh 35.0 %, Washington D.C. 12.6 %, Detroit 11.7 %, Austin 3.1 %, Palo Alto 2.2 % | Boston 55.0 %, Singapore 45.0 % |
| night / rain | 3.0 % / 1.1 % | 11.7 % / 17.4 % |
| radar | none | yes (nuScenes) |
| lane-segment labels (Map Element Bucket, devkit v2.x) | yes (about 240 MB) | none listed in the README |
- Totals (checked, own sums): A 32,099 + B 39,995 = 72,094 samples, matching the "about 72K frames" of the datasheet; per scene 32 (A) and 40 (B) samples, matching 15 s and 20 s at 2 Hz. The paper itself gives no per-subset frame counts; the sample counts come from the TopoBDA text. The subset B val (6,019) and test (6,008) counts equal the official nuScenes val and test sample counts as I remember them (memory), the train count is 162 lower than the official 28,130 (memory); this suggests that subset B uses the official nuScenes scene split, but no OpenLane-V2 page says so (see section 6).
- Annotation range is larger than the usual +-30 m (paper text). Annotation size: about 1.8 M centre-line and 0.25 M traffic-element instances (datasheet D.4; the introduction quotes 2.1 M instance annotations: the two statements are not reconciled).

### Centre-line representation and topology (digest)
- Each lane is one instance: an ordered 3D point list from start to end, the order is the driving direction. 201 points are stored per lane; for evaluation the devkit subsamples every 20th point (point_interval = 20 in the tutorial), i.e. 11 points. The 11 points are not equidistant by construction of the devkit: they are every 20th stored point; whether the 201 stored points are equidistant in arc length is not stated (section 6).
- Centre-lines are regressed from HD-map boundaries (datasheet); lanes with a single predecessor and a single successor are merged into one; lanes are split only at intersections, forks and merges. In graph terms the ground truth is a junction-vertex graph whose edges are whole pieces between branching vertices.
- Topology is a per-frame adjacency matrix in the order of the lane list: entry (i, j) is positive iff the end point of lane i meets the start point of lane j (asymmetric, directed). Most lanes have one predecessor / successor, a complex crossroad up to 7. About 90 % of frames have more than 10 centre-lines, about 10 % more than 40.
- Lane - traffic element correspondence: an undirected bipartite relation; elements link only to centre-lines outside intersections.
- Source of the nuScenes centre-lines (nuScenes map expansion or own annotation): not stated (section 6). Labelling by contractors with several validation stages.
- Per-frame JSON (data/README.md): `annotation.lane_centerline` = list of {id, points float [n, 3], confidence (predictions only)}; `topology_lclc` float [n, n]; `topology_lcte` float [n, k]; ground truth is 0 / 1, predictions 0 - 1. Pose, intrinsics and extrinsics per camera, `meta_data.source` and `source_id` (original dataset and original segment id) are stored per frame.

### Evaluation pipeline (digest of the master-branch code; the order of the steps matters)
```
predictions per frame token (split, segment_id, timestamp):
   lane_centerline [{id, points (n,3), confidence}], traffic_element [...], topology_lclc (n,n), topology_lcte (n,k)
        |  check_results (format only)
        v
1  distance matrix GT x pred per frame:  mask = Chamfer < 3.0 m ;  discrete Frechet (p = 2, raw points, no resampling)
   inside the mask, multiplied by the ego relaxation factor  max(0.5, 1 - 0.005 * d_ego(GT lane))
        v
2  DET_l: for t in {1, 2, 3} m:  per frame greedy matching in descending confidence, nearest GT only,
   TP if distance < t and that GT not yet taken, else FP ;  all frames pooled, sorted by confidence ;
   11-point interpolated AP ;  mean over t
        v
3  matches stored per (t): prediction index -> GT index (NaN if FP)
        v
4  TOP_ll: per t and frame: build a GT-sized prediction matrix from matched x matched cells of topology_lclc,
   fill all other cells with (1 - gt) * (0.5 + eps) ;  per GT vertex (row = outgoing, column = incoming):
   rank the cells with value > 0.5 by value, AP = sum of precision at hits / number of GT neighbours ;
   pooled mean over all vertices, all frames, all t
        v
5  OLS = mean( DET_l, DET_t, sqrt(TOP_ll), sqrt(TOP_lt) )     (scores in [0, 1] in the code, papers print x 100)
```

### Equations (numbered as in the paper v3; digest)
- **E1** OLS = 1/4 [ DET_l + DET_t + f(TOP_ll) + f(TOP_lt) ]. f is "a scale function to emphasize the task of topology reasoning" and is not defined in the paper. The code uses the square root of the 0 - 1 value (checked below).
- **E2 - E3** Frechet distance: a coupling L is a sequence of point pairs with monotonically non-decreasing indices; its norm is the largest distance of a pair; D_Frechet(v, v^) = min over couplings of the norm. Pairs farther apart than t are unmatched.
- **E4** DET_l = 1/|T| sum over t in T of AP_t, T = {1.0, 2.0, 3.0} m, AP = area under the precision-recall curve.
- **E5** D_IoU = 1 - |v_t and v^_t| / |v_t or v^_t|, match threshold 0.75; **E6** DET_t = mean over the 13 attributes of AP_a.
- **E7** TOP = 1/|V| sum over vertices v of [ sum over n^' in N^'(v) of P(n^') * 1(n^' in N(v)) ] / |N(v)|: predicted neighbours of v ranked by confidence, positive if confidence > 0.5, P = precision at that rank, N(v) = true neighbours. Predicted vertices are first matched to GT (Frechet for lanes, IoU for traffic elements), "dummy" vertices are allowed in the text; TOP_ll uses the directed lane graph, TOP_lt the undirected lane - traffic-element graph.
- Worked example in the supplement (Appendix E): vertex 1 with true neighbours {2, 3, 5, 8} and ranked predictions {4, 3, 5, 6, 7} has AP (0 + 1/2 + 2/3) / 4 = 7/24; if the other seven vertices score 0, TOP = 7/192 = 0.036.
- Relaxation (text): thresholds are relaxed for distant lanes, example 1.2 t for a far and 1.1 t for a nearer lane; the exact rule is not in the paper (it is in the code, below).
- Code-level formulas that the paper does not give (digest of `distance.py` / `evaluate.py`):
  - **C1** relaxed distance = D_Frechet(GT lane, pred lane) * max(0.5, 1 - 5e-3 * d_ego), d_ego = smallest norm of the GT lane's points from the origin (the GT list is always the first argument). Equivalent threshold t / max(0.5, 1 - 0.005 d): 1.11 t at 20 m, 1.25 t at 40 m, 2 t at 100 m or more.
  - **C2** Chamfer pre-mask: pairs with Chamfer distance (mean of the two directed mean-nearest-point distances, halved sum) >= 3.0 m are not evaluated (distance stays at the fill value 1024).
  - **C3** 11-point AP: AP = (1/11) sum over r in {0, 0.1, ..., 1} of max precision over ranks with recall >= r (0 if none).
  - **C4** per-vertex AP: sorted by descending confidence, precision at every hit summed, divided by the number of true neighbours; 1 if both neighbour sets are empty, 0 if exactly one is empty.
  - **C5** OLS in the code: mean of DET_l, DET_t, sqrt(TOP_ll), sqrt(TOP_lt).

### Losses, labels, training of the baselines (digest, supplement F)
- No loss belongs to the benchmark. The baselines were trained on both subsets: 8 A100, 24 epochs, about 16 h; resize factor 0.5 and colour jitter; TopoNet with ResNet-50 (C3 - C5), BEV grid 200 x 100, five loss weights 1.0, 2.5, 1.0, 1.5, 0.025; STSU re-based on ResNet-50, Bezier output interpolated to 11 equidistant points; VectorMapNet and MapTR with the range widened to +-50 m x +-25 m, only 2D centre-lines in BEV, MapTR with 11 point queries per lane.
- Matching inside the DETR-style baselines is Hungarian set matching (memory, not seen on a page); the benchmark's own matching is greedy (below), never Hungarian.

### Inference / hidden rules of the metric (digest; consequences marked derived)
1. Matching is per frame, greedy in descending confidence; each prediction looks only at its nearest GT lane (argmin of the relaxed distance); if that GT is already taken the prediction is a false positive even when a second GT lane is within the threshold. A GT lane can be taken once per threshold. Ties in confidence are ordered by NumPy's default sort (unstable).
2. AP is pooled over all frames and sorted globally by confidence: confidences must be comparable across frames. Recall denominator: all GT lanes of all frames.
3. The devkit does no resampling, clipping or capping: prediction points are used as given (any number of points; the format check only needs an array [n, 3]); there is no minimum or maximum number of predictions (check_results has only a limit of 10 authors).
4. TOP in the current code (v2.1): only predictions matched to a GT lane (at the given t) enter the matrix, regardless of their confidence; unmatched predictions are dropped; unmatched GT lanes keep their row and column as vertices and get the fill value. Frames without GT lanes are skipped. The mean is pooled over all (threshold, frame, vertex) triples, so frames with many lanes weigh more.
5. Strict `> 0.5` decides whether a cell is a predicted edge, so a fill of 0.5 + float32 eps counts as a predicted edge with the lowest possible rank.
6. Derived: the discrete Frechet distance always contains the pairs (first, first) and (last, last), so a pair can only match if both end points are within the (relaxed) threshold; the monotone coupling also makes direction matter.
7. Derived: because of rules 1 and 4, spurious extra predictions at the bottom of the global ranking cannot lower AP (the 11-point maximum runs over a superset), and they cannot lower TOP either; they only add chances of matching more GT lanes.
8. Derived from the control flow (to be confirmed by a unit test in the port): for a matched GT lane with no true neighbour, the fill cells of every unmatched GT lane in the same frame count as predicted edges (gt = 0, value 0.5 + eps), so its vertex AP is 0 instead of 1 as soon as one GT lane of the frame is unmatched.

## 3. Numbers

All numbers are author-reported, digest level, on different devkit versions; the version is stated wherever it is known. They are not comparable with each other unless the version column says so, and not with ours.

### Table 3 of the paper (val split, v3 text; DET_l / DET_t / TOP_ll / TOP_lt are x 100)
| method | design | A: OLS | DET_l | DET_t | TOP_ll | TOP_lt | B: OLS | DET_l | DET_t | TOP_ll | TOP_lt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| STSU | instance | 25.4 | 12.7 | 43.0 | 0.5 | 15.1 | 21.2 | 8.2 | 43.9 | 0.0 | 9.4 |
| VectorMapNet | point set | 20.8 | 11.1 | 41.7 | 0.4 | 5.9 | 16.3 | 3.5 | 49.1 | 0.0 | 1.4 |
| MapTR | point set | 20.0 | 8.3 | 43.5 | 0.2 | 5.9 | 21.1 | 8.3 | 54.0 | 0.1 | 3.7 |
| TopoNet | instance | 35.4 | 29.2 | 48.0 | 4.1 | 19.3 | 33.2 | 24.3 | 55.0 | 2.5 | 14.2 |
- Devkit version of this table: not stated in the paper. The TOP values are of the pre-fix kind (TopoNet 4.1 and 2.5 equal the v1.0.0 column of the TopoLogic table below), derived.
- Check of the combination rule (own arithmetic on these numbers, **checked**): TopoNet A (29.2 + 48.0 + 100 sqrt(0.041) + 100 sqrt(0.193)) / 4 = 35.35 (printed 35.4); TopoNet B 33.20 (33.2); STSU A 25.41 (25.4); MapTR A 20.14 (20.0), the small deviations are rounding of the printed TOP values. So f = square root on the 0 - 1 scale, as in the code.
- Text: all methods have low DET_l (centre-lines are not visible, their direction depends on the scene), TOP is weak because it depends on accurate centre-line matches; the qualitative comparison says STSU gives many false-positive centre-lines, MapTR's points are not smooth, VectorMapNet's auto-regressive shapes are better.
- The same baseline name has different numbers in other papers (MapTR DET_l 8.3 here, 17.7 on subset A and 15.2 on subset B in the TopoLogic table; TopoNet DET_l 29.2 / 28.6 / 28.5 / 27.1 in the benchmark paper, TopoLogic table, TopoMLP table, TopoMLP Sec. 4.6): different re-trainings. Quote a baseline only from one source.

### Same models under two devkit versions (TopoLogic paper, Table 1, digest; "v1.0.0" = old TOP, "v2.1.0" = fixed TOP; DET_l and DET_t are one column, i.e. identical)
| subset | method | SD map | DET_l | DET_t | v1.0.0 TOP_ll | TOP_lt | OLS | v2.1.0 TOP_ll | TOP_lt | OLS |
|---|---|---|---|---|---|---|---|---|---|---|
| A | VectorMapNet | no | 11.1 | 41.7 | 0.4 | 5.9 | 20.8 | 2.7 | 9.2 | 24.9 |
| A | MapTR | no | 17.7 | 43.5 | 1.1 | 10.4 | 26.0 | 5.9 | 15.1 | 31.0 |
| A | TopoNet | no | 28.6 | 48.6 | 4.1 | 20.8 | 35.6 | 10.9 | 23.8 | 39.8 |
| A | TopoLogic | no | 29.9 | 47.2 | 18.6 | 21.5 | 41.6 | 23.9 | 25.4 | 44.1 |
| A | SMERF | yes | 33.4 | 48.6 | 7.5 | 23.4 | 39.4 | 15.4 | 25.4 | 42.9 |
| A | TopoLogic | yes | 34.4 | 48.3 | 23.4 | 24.4 | 45.1 | 28.9 | 28.7 | 47.5 |
| B | TopoNet | no | 24.3 | 55.0 | 2.5 | 14.2 | 33.2 | 6.7 | 16.7 | 36.8 |
| B | TopoLogic | no | 25.9 | 54.7 | 15.1 | 15.1 | 39.6 | 21.6 | 17.9 | 42.3 |
- Section 4.1 of that paper says that for centre-lines two versions exist for TOP_ll, TOP_lt and OLS, v1.0.0 and v2.1.0; its Sec. 4.3 quotes TopoNet's v2.1.0 TOP_ll as 10.8 while the table has 10.9 (text and table disagree by 0.1).
- Deltas v2.1.0 minus v1.0.0 (**checked**, own subtraction): TOP_ll +2.3 (VectorMapNet), +4.8 (MapTR), +6.8 (TopoNet), +5.3 (TopoLogic), +7.9 (SMERF), +5.5 (TopoLogic + SD map); subset B +4.2 (TopoNet), +6.5 (TopoLogic); TOP_lt +2.0 to +4.7; OLS +2.4 to +5.0. The fix raises every listed TOP value and the shift is not constant, so no conversion between versions exists.
- This is the level-1 / overview claim "same model TOP_ll 4.1 against 10.9": confirmed, it is TopoNet on subset A, v1.0.0 against v2.1.0 (the level-1 note's v1.0 / v2.0 against v1.1 / v2.1 naming is right). The compatibility study (section k) adds TopoMLP 7.1 against 21.7 and OLS 38.3 against 44.1; the TopoBDA table below has TopoMLP at 21.7 / 44.1 on the fixed metric (digest of columns, see caveat).

### Why the numbers moved (TopoMLP, Sec. 4.6, digest; and the devkit diff)
- TopoMLP states that unmatched lanes or traffic elements got a default confidence of 1.0 in the old metric, so that false positives ranked first, and that pushing each edge confidence to 1.0 or 0.0 around the 0.5 cut (so that true edges rank above those defaults) raised the scores without any model improvement. Their table (subset A, original -> "enhanced" scores): TopoNet TOP_ll 4.0 -> 11.5 (OLS 34.4 -> 38.1), TopoMLP TOP_ll 7.2 -> 19.0 (OLS 38.2 -> 42.2). Their proposal was a correctness factor N_TP / (N_TP + N_FP) on each precision term (their Eq. 8, TOP-dagger: TopoNet 2.0 and 1.0, TopoMLP 4.5 and 1.9 in the two groups of their second table, which of the groups is original and which enhanced is not recoverable from the page). The devkit did not adopt that factor.
- What the devkit changed (PR #77, "OpenLane-V2 Benchmark V2.1", 2023-11-01, digest of the diff): the old topology functions looped over 10 confidence thresholds (the confidences at recall 10 % ... 100 % of the detection step) and kept in the matrix only predictions above the current threshold; every other cell of the GT-sized prediction matrix (cells of unmatched GT lanes and of filtered-out predictions) was filled with `1 - gt`, i.e. a predicted edge with confidence 1.0 wherever the ground truth has no edge and 0 where it has one. The new code has no confidence loop, uses all matched predictions, and fills with `(1 - gt) * (0.5 + eps)`, i.e. the filler edges sit just above the cut and at the bottom of every ranking. Same change in the lane-segment evaluation. Issue #76 (pinned, opened and closed 2023-11-01) only thanks the TopoMLP authors and calls it a "loophole" fix; it names no numbers.
- Reading (derived): after the fix the metric behaves like the old one with "enhanced" scores, which is why honest models gain 2 - 8 TOP points; the 1.0 default of the old version was applied to filler cells of unmatched GT lanes, not literally to unmatched predictions (those were dropped in both versions).

### Later results on the fixed metric (TopoBDA paper, Table 5, digest; its columns are unlabeled in the excerpt: DET_l, DET_t, TOP_ll, TOP_lt, OLS inferred from the text for TopoBDA; the paper says its comparison uses the "V1.1" metric)
| subset | modality | method | DET_l | DET_t | TOP_ll | TOP_lt | OLS |
|---|---|---|---|---|---|---|---|
| A | camera | TopoNet | 28.6 | 48.6 | 10.9 | 23.9 | 39.8 |
| A | camera | TopoMLP | 28.5 | 49.5 | 21.7 | 26.9 | 44.1 |
| A | camera | TopoLogic | 29.9 | 47.2 | 23.9 | 25.4 | 44.1 |
| A | camera | TopoFormer | 34.7 | 48.2 | 24.1 | 29.5 | 46.3 |
| A | camera | TopoMaskV2 | 34.5 | 53.8 | 24.5 | 35.6 | 49.4 |
| A | camera | TopoBDA | 38.9 | 54.3 | 27.6 | 37.3 | 51.7 |
| A | camera + SD | TopoBDA | 42.7 | 52.4 | 34.3 | 41.7 | 54.6 |
| B | camera | TopoNet | 24.3 | 55.0 | 6.7 | 16.7 | 36.0 |
| B | camera | TopoLogic | 25.9 | 54.7 | 21.6 | 17.9 | 42.3 |
| B | camera | TopoFormer | 34.8 | 58.9 | 23.2 | 23.3 | 47.5 |
| B | camera | TopoMaskV2 | 41.6 | 61.1 | 28.7 | 26.1 | 51.8 |
| B | camera | TopoBDA | 45.1 | 61.4 | 34.0 | 27.6 | 54.3 |
| B | camera + lidar | TopoBDA | 57.7 | 62.9 | 45.0 | 35.2 | 61.7 |
- TopoBDA's ablation Table 3 (subset B, ResNet-50, "V1.1m" baseline, centre-line-only score OLS_l): camera DET_l 45.1 / TOP_ll 35.6 / OLS_l 49.9; camera + radar 49.4 / 38.6 / 54.8; camera + lidar 57.5 / 46.0 / 61.6. This is the only camera-radar row on the benchmark known to us (radar adds +4.3 DET_l / +3.0 TOP_ll in that ablation). "V1.1m" and DET_l_ch (a Chamfer-based column?) are not defined in the excerpts; do not mix with Table 5 (camera TOP_ll 35.6 there against 34.0 in Table 5).
- Different papers differ by 0.1 - 0.8 for the same model (TopoNet A TOP_lt 23.8 / 23.9, B OLS 36.8 / 36.0): treat the fixed-metric table as order of magnitude.

### Lane-segment benchmark (devkit v2.x, "Driving Scene Topology", subset A only; digest)
- Definitions (docs/metrics.md and `lanesegment/evaluation`): OLUS = mean of DET_ls (called DET_l in the code), DET_a, DET_t, f(TOP_lsls) and f(TOP_lste) (code keys TOP_ll and TOP_lt, f = square root, five terms). Lane-segment distance = 1/2 [ Frechet(centre-line) + Chamfer(left line) + Chamfer(right line) ] with thresholds {1.0, 2.0, 3.0} m and the same ego relaxation; a Chamfer pre-mask on the centre-line (< 3.0 m); areas (pedestrian crossings and road boundaries, undirected curves) with Chamfer thresholds {0.5, 1.0, 1.5} m; traffic elements IoU 0.75. TOP_lsls and TOP_lste use the same vertex AP and the same fill rule as the centre-line TOP. Submission format (docs): per frame `lane_segment` (id, centerline, left_laneline, right_laneline [n, 3], confidence), `area`, `traffic_element`, `topology_lsls` [n, n], `topology_lste` [n, k]; 10 points per lane-segment line and 20 per area are recommended to align with the ground truth (not mandatory), floats may be float16.
- LaneSegNet paper (arXiv v2, Table 1, ResNet-50, 24 epochs, 10 points per line; evaluation version not stated): LaneSegNet mAP 32.6, AP_ls 32.3, AP_ped 32.9, TOP_lsls 8.1, FPS 14.7; LaneSegNet-tiny 28.5 / 28.2 / 28.7 / 6.8 / 16.2; TopoNet 23.0 / 23.9 / 22.0 / 1.0; MapTR 27.0; MapTRv2 28.5. Its README, "devkit v2.1.0": LaneSegNet mAP 33.5, TOP_lsls 25.4; Map Element Bucket model DET_ls 27.8, DET_a 23.8, DET_t 36.9, TOP_lsls 24.1, TOP_lste 21.3. The README says the fix of a TOP loophole is why TOP_lsls differs from the paper; its prose says the loophole made the value "significantly higher" while its own numbers show the fixed value (25.4) higher than the paper's (8.1): direction unresolved, the numbers are what they are.
- SCORE paper (digest): single-frame LaneSegNet baseline DET_ls 32.3, DET_ped 32.9, TOP_ll 25.4; on a geographically disjoint split LaneSegNet (12 epochs) DET_ls 17.2, TOP_ll 16.2 and SCORE (30* epochs, the asterisk is unexplained in the excerpt) 21.3 / 22.2. Epoch counts differ, so the drop is an upper bound.
- Subset B has no lane-segment labels (README lists no Map Element Bucket for it; the SCORE paper says so too): the nuScenes benchmark is centre-line only.

### Compute and caveats
- Compute: baselines see the training note above; devkit cost not stated; one evaluation over a val split is numpy / `similaritymeasures` work on 11-point curves (cheap).
- Caveats: devkit version (above); unlabeled columns in the TopoBDA excerpt; reproductions of the same baseline differ; TOP depends on DET matching, so TOP is bounded by the matched fraction (section 5); OLS includes traffic elements.

## 4. Paper versus code
- **f(TOP)**: undefined in the paper, square root in the code (checked numerically above).
- **Relaxation**: "1.2 t / 1.1 t" in the paper, the factor max(0.5, 1 - 0.005 d_ego) on the distance in the code.
- **Dummy vertices**: the paper allows dummy vertices for unmatched predictions; the code has none: unmatched predictions are dropped and unmatched GT lanes get filler cells.
- **docs/metrics.md versus code**: the doc (digest) still says the topology metrics are "averaged over different recall levels", which describes the old (x.0) code; the current code has no recall-level loop. It also never defines f, the point count or the relaxation.
- **Points**: the paper says 11 points are used for evaluation; the devkit uses whatever points are in the ground-truth pickle and in the prediction. The ground truth gets its 11 points only if it was preprocessed with `point_interval = 20` (tutorial: "for evaluation"); the default is 1 (all 201 points). The lane-segment task recommends 10 points per line and 20 per area, not 11.
- **Window**: the paper states the +-50 m x +-25 m annotation range; `evaluate` shows no range filter (digest), so predictions outside the window are simply false positives.
- **Traffic-element part**: DET_t is an AP per attribute at IoU 0.75 over 13 attributes (including "unknown").
- **F-score for 3D lane** (OpenLane-style, one-to-one min-cost-flow matching, 100 x samples from -50 to 50, distance 1.5 m, ratio 0.75) is returned next to OLS but is not part of it.
- **Paper tables versus later tables**: see section 3 (versions, re-trainings).

## 5. Transfer to our pipeline
| question | answer |
|---|---|
| what it needs from us | Nothing from the model: this is an evaluation. Needed: our graph per frame in ego metres (x forward, y left, ego at the origin, as in the nuScenes ego frame; to be confirmed for our BEV frame), the nuScenes scene and sample timestamp of each frame, the OpenLane-V2 subset B ground truth of the same frames (annotations of the "OpenLane-V2 info" file, about 7.7 GB, no images needed; CC BY-NC-SA 4.0 plus nuScenes terms; **a download needs the user's OK**) and a port of the metric. No frozen BEV, no SD map, no radar, no lidar, no pose beyond what the ground truth already is. |
| what it gives us | Axis A9: an external, literature-facing number pair for pieces of the graph, DET_l (whole-piece Frechet AP at 1 / 2 / 3 m) and TOP_ll (adjacency AP among matched pieces) on nuScenes. Not OLS: traffic elements are outside our output. |
| conflicts | (1) Whole-curve Frechet matching punishes truncated lanes twice (a false positive and a missed GT lane): our rule "better short lane graphs than false ones" is not rewarded here. (2) The piece cut must follow the ground-truth rule (cut only at forks, merges, intersections and window borders). (3) Window x +-50, y +-25 m against our +-50 m square. (4) The ground truth of OpenLane-V2 differs from our nuScenes map-expansion ground truth, so only the metric, not the data, is shared. (5) Devkit pins (numpy < 1.24, scipy 1.8.0, ortools 9.2.9972, seen in requirements.txt) very probably do not fit Python 3.12 (wheel availability from memory): port instead of installing. (6) No Hungarian, no mm* in the metric (good). (7) Apache-2.0 code, data non-commercial. (8) Evaluation only: nothing here touches the differentiable training path. |
| reuse mode | Re-implement from the description (the metric is about 300 lines of NumPy); Apache-2.0 would allow adaptation with attribution, but only digests were read, so the port follows the description in [openlane-v2](../code/openlane-v2.md) and is verified by the devkit's own dummy evaluation and by unit tests. |
| effort | S for the port, the piece cutter, the resampler and the self-tests (hours; can run on our own ground-truth pieces without any download). M for the real comparison: approval for the 7.7 GB info download, token matching (scene and timestamp), checking the window clipping and the frame offset against the ground truth, split bookkeeping. Blocker: the download approval. |
| expected gain and risk | No model gain; the value is a literature-facing number and a metric that is closer to what a planner consumes (whole lanes and their links) than our pooled point F1. Risks: a low DET_l caused by a different cut or window rule rather than by the model; a TOP_ll that is bounded by DET_l recall; cross-version confusion; temptation to tune on it (do not: select on our VAL). |
| feeds which experiment / K id | K133 stays the benchmark entry. A new technique id for the adapter (piece cutter + 11-point resampler + confidence rule + topology matrix + frame tokens) should be registered before use. Suggested use: a metric-only evaluation of an existing champion output (no new run), then a column in the final report. |

### 5.1 Evaluating our graph with the devkit metric (adapter specification, my design; nothing here is in the paper)
1. **Frames.** Take the frames that exist in OpenLane-V2 subset B (match nuScenes scene and sample timestamp to `meta_data.source_id` and `timestamp`; the exact timestamp convention of subset B is not verified). The devkit asserts that ground-truth and prediction tokens are equal, so build the ground-truth pickle from a `data_dict` that lists exactly the evaluated frames. Ground truth is public for the OpenLane-V2 train and val parts (test annotations are withheld for the servers; the `collect` function skips frames without annotation, which suggests it), so any of our frames that belongs to the official nuScenes train or val scenes can be scored; the published numbers use the 6,019 val frames. To claim a like-for-like number, restrict to frames of the official val scenes that are not in our training set (our TRAIN 8,414 frames may overlap them: check by scene name before anything else).
2. **Ground-truth pickle.** `collect(root, data_dict, name, point_interval=20)` in the centre-line API gives 11 points per ground-truth lane. With the default `point_interval = 1` the ground truth has 201 points and a coarse prediction would be scored against a dense curve (the discrete Frechet distance of a 11-point and a 201-point version of the same curve is already large).
3. **Pieces.** Build the graph from vertices with validity above a threshold (or all, and let the confidence do the work); cut it into maximal directed paths between "branching" vertices: a vertex is a cut vertex unless it has exactly one incoming and one outgoing edge (fork, merge, source, sink, junction). This reproduces the ground-truth rule "single predecessor and successor are merged, split only at forks, merges and intersections". Lane-change or connector-like edges that the ground truth does not contain must not be cut into pieces (check that our graph has none).
4. **Window.** Clip every piece to x in [-50, 50], y in [-25, 25] and drop pieces outside. How the ground truth is clipped at the window (end points exactly on the border or not) is not documented: measure it on the ground truth (fraction of lane end points with |y| = 25 or |x| = 50) before trusting the border pieces; a shift sweep of the whole prediction by -1 ... +1 m is a cheap check for a frame offset between our ego frame and the ground truth.
5. **Resampling.** Resample each piece to 11 points equidistant in arc length from the first to the last vertex in the travel direction (the ground truth uses every 20th of 201 stored points; if those are equidistant the two parametrisations agree for equal curves), z = 0 (subset B has z = 0), float32. 10 points (lane segment) and 20 points (areas) belong to the lane-segment task, not to this one; the "11 or 20" of the compatibility study should read 11 for centre-lines.
6. **Confidence.** One number per piece. Minimum vertex validity is a good first choice (a piece is as reliable as its worst vertex), but it favours short pieces; also report geometric mean and a calibrated variant (a small logistic fit on VAL from min validity, length and distance to ego, evaluation-side only) because DET_l depends only on the ranking of pieces across all frames. Ties are ordered arbitrarily by the devkit, continuous confidences avoid that.
7. **Topology matrix.** `topology_lclc[i, j]` = probability that the end of piece i continues into piece j: our edge probability where the two pieces meet in a cut vertex and our graph has an edge from the last edge of i to the first edge of j; 0 otherwise. If a junction vertex has several incoming and outgoing edges and our graph does not say which in-edge continues into which out-edge, the pairing is undefined: use all pairs (over-predicts at crossings) or a geometric rule, and say which. Only values above 0.5 count as predicted edges, so edge probabilities must be calibrated; ranking among them gives the AP.
8. **Traffic elements.** Empty list and `topology_lcte` of shape (n, 0); report DET_l and TOP_ll separately from the dict, never the OLS (it contains DET_t and TOP_lt, which would be 0).
9. **Call.** `evaluate(ground_truth=<pkl>, predictions=<dict>)` returns `{'OpenLane-V2 Score': {DET_l, DET_t, TOP_ll, TOP_lt, score}, 'F-Score for 3D Lane': {...}}` on a 0 - 1 scale (the dummy evaluation, `predictions=None`, prints 1.0 for all). In our environment: the port, not the devkit (pins).
10. **Sanity layers before any number is quoted** (project layers 0 - 2): the dummy evaluation of the port gives 1.0; ground truth as prediction with confidence 1 gives 1.0; a rigid shift of the ground truth by s metres (confidence 1) gives a Frechet distance of exactly s before the relax factor, so DET_l falls in steps of one third as s crosses 1, 2 and 3 m (softened by the relax factor, which lets far lanes pass at up to 2 t); reversing every piece gives about 0; removing one third of the pieces gives recall-limited DET_l and TOP_ll below the matched fraction; the "oracle topology" (replace the predicted matrix by the GT matrix among matched pieces) is the upper bound of TOP_ll for a given DET matching and separates detection from linking.

### 5.2 What the metric rewards and what it ignores (for our graph)
| aspect | effect |
|---|---|
| graph sparsity / vertices per 100 m | Ignored: polylines are resampled to 11 points; only the maximum deviation of the chord polyline from the true curve matters (about 0.3 m for 7 m vertex spacing on a 20 m radius curve, my estimate). Fewer vertices are neither rewarded nor punished. |
| partial lanes ("short but correct") | Punished: matching is whole-curve and both end points must be within about 1 - 3 m; a piece that stops 10 m early is a false positive and its GT lane is missed (affects DET_l and, through the unmatched row and column, TOP_ll). |
| false lanes | DET_l punishes them through precision (ranked by confidence, so low-confidence ones cost little). TOP_ll ignores them completely (dropped from the matrix). |
| spurious low-confidence extra pieces | Free: appending predictions below all others cannot lower AP and can only add matches (derived). Always report the number of predicted pieces per frame next to the scores. |
| fork / merge accuracy | Only through geometry and adjacency: every piece touching a fork has an end point there, so a fork misplaced by more than the relaxed threshold (about 3 - 4 m) costs the incoming and both outgoing pieces at once (three misses and three false positives); a fork displaced by 6 m, which our J (6 m radius) counts as found, fails here. A correct fork shows up as two correct outgoing adjacency edges among matched pieces. No explicit fork / merge score exists. |
| direction | Rewarded: the Frechet coupling is monotone (a reversed piece does not match) and the adjacency is directed (rows = outgoing, columns = incoming, both scored). |
| per-vertex validity | Used only through the single piece confidence. Interior validity gaps are invisible. |
| edge calibration | Only the 0.5 cut and the ranking above it matter; probabilities below 0.5 are ignored. |
| topology of unmatched pieces | Ignored, but unmatched GT lanes zero their own row and column and, derived, also zero the vertex AP of matched isolated lanes in the same frame. TOP_ll therefore tracks the matched fraction of GT lanes. |
| route / SD map / temporal behaviour / planner use | Not measured. |
| scale | Pooled over frames, so frames with many lanes dominate TOP; confidence must be comparable across frames for AP. |

### 5.3 Warning list for comparisons
1. Always state the devkit: v1.0 / v2.0 ("x.0") or v1.1 / v2.1 ("x.1"). The same model gains 2 - 8 TOP_ll points and 2 - 5 OLS points on the fixed metric; DET_l and DET_t do not change. Papers that do not state the version (TopoMLP's table, the LaneSegNet paper's table, the benchmark paper's Table 3) are of the old kind when their TopoNet TOP_ll is 4.1.
2. v1.1.0 (centre-line only, long-term support) and v2.1.0 appear to give the same centre-line numbers (TopoBDA, which names V1.1, and TopoLogic, which names v2.1.0, list the same TopoNet subset B TOP_ll 6.7); not verified in the code.
3. Never quote OLS for our output; quote DET_l and TOP_ll with the number of predicted pieces per frame and the frames used.
4. The 11-point ground truth requires `point_interval = 20`; a ground truth with 201 points or a prediction with sparse vertices unresampled gives meaningless distances.
5. Subset B has z = 0 and no lane-segment labels; lane-segment numbers (OLUS, DET_ls, TOP_lsls) exist only for subset A (Argoverse 2) and are not available for nuScenes.
6. Re-trainings of the same baseline differ by many points (MapTR DET_l 8.3 / 17.7); use one table per comparison.
7. Splits: subset B follows (probably) the official nuScenes scenes; nuScenes val locations overlap training locations, so val numbers are not geographically disjoint (the SCORE paper's disjoint split lowers LaneSegNet's DET_ls from 32.3 to 17.2 with unequal epochs, TP18). Our own TRAIN may contain scenes of the official val set: check before using val frames.
8. OpenLane-V2 centre-lines come from HD-map boundaries by their own regression; our ground truth comes from the nuScenes map expansion. A score of our output against our ground truth and against theirs measure different things.
9. TOP_ll is bounded by DET matching (derived); a model with low DET_l has a low TOP_ll whatever its linking quality; report the oracle-topology bound.
10. The strict > 0.5 edge cut and the global confidence ranking make calibration part of the score; a monotone change of confidence does not change DET_l ranking but a shift across 0.5 changes TOP.
11. Ties and the unstable sort give small non-determinism; our continuous confidences avoid it.
12. Do not tune on OpenLane-V2 val frames and then quote them; select on our VAL (project rule); the 6,019-frame val set is not our TEST.
13. Licences: ground truth CC BY-NC-SA 4.0 plus the nuScenes terms: research use only, no redistribution of the annotation.

### Corrections to the level-1 note (TP29)
- "loophole: unmatched predictions get confidence 1.0": imprecise. In the old code the filler cells of the GT-sized prediction matrix (cells of unmatched GT lanes and of predictions filtered out by the 10 confidence levels) got `1 - gt`, i.e. confidence 1.0 where the ground truth has no edge; unmatched predictions were dropped in both versions. TopoMLP's wording ("unmatched ... defaulted to 1.0") is the origin of the shortcut.
- Missing in the note: the old topology functions looped over 10 confidence levels and the new ones do not; the new fill is `(1 - gt) * (0.5 + eps)` (the note already has this part right).
- "averaged over rows and columns, thresholds and samples": it is one pooled mean over all (threshold, frame, vertex) triples, not a mean of per-frame means; frames without ground-truth lanes are skipped.
- The sqrt in OLS is confirmed (the paper never defines f); relaxation is now exact: factor max(0.5, 1 - 0.005 d_ego) on the distance.
- DET_t is an AP per attribute at IoU 0.75 over 13 attribute values (including "unknown"), averaged over attributes; the note's "IoU 0.75 over 13 attributes" is right.
- "evaluated with 11": only if the ground truth was preprocessed with `point_interval = 20`; the devkit itself does no resampling and accepts any number of prediction points.
- The "Argoverse-2-based subset A" lane-segment labels are the only lane-segment data; confirmed. Add: subset B has radar; sample counts per split are 22,477 / 4,806 / 4,816 (A) and 27,968 / 6,019 / 6,008 (B), with subset B probably equal to the official nuScenes split.
- The note's "TopoNet baseline (old metric)" numbers are Table 3 of v3 and are right; the fixed-metric TopoNet values are 10.9 / 23.8 / 39.8 (A) and 6.7 / 16.7 / 36.8 (B) (TopoLogic Table 1).
- Licence: code Apache-2.0 is now verified from the LICENSE file; data CC BY-NC-SA 4.0 (README, paper appendix C). The repository URL of the note is correct.
- Not in the note: the metric also exists for lane segments (OLUS, five terms, subset A only); the devkit tags (v1.0.0 2023-07-19, v2.0.0 2023-07-21, v2.0.1 2023-07-27, v1.1.0 and v2.1.0 2023-11-01).

## 6. Not found / open
- Whether the 201 stored points per lane are equidistant in arc length; how ground-truth lanes are clipped at the +-50 / +-25 m window; whether lane-change connectors exist in the ground truth (the paper only says lanes are split at intersections, forks and merges).
- Source of the subset B centre-lines (nuScenes map expansion or own annotation) and the mapping of subset B segments to nuScenes scene names and sample tokens; whether the timestamps in subset B are nuScenes sample timestamps; whether the splits equal the official nuScenes split (the sample counts suggest it; unverified); the frame offset between the OpenLane-V2 ego pose and our ego frame.
- Exact text of the three metric files: only digests were seen, no verbatim lines. In particular the claims in section 2 rules 7 and 8 and all control-flow statements should be confirmed by unit tests of the port, and ideally by running the original devkit once (needs Python 3.8 and the pins; no installs without the user's OK).
- Differences between paper v1, v2 and v3 (the OLS definition and baselines in v1 could not be read: the fetch tool returned the PDF as binary and saved a copy of it by itself under the session's `tool-results` folder in the user profile, 3.2 MB; I did not open or use it, delete it if unwanted).
- The `io`, `dataset`, `visualization` modules, the notebooks beyond preprocessing / dummy evaluation, the leaderboards (CVPR 2023 OpenLane Topology on EvalAI, CVPR 2024 Mapless Driving), and the dataset changes between v1.0 and v2.0 (README only says that APIs and lane-segment / SD-map material differ).
- GitHub API calls returned HTTP 403, so the file tree and commit data come from web pages.
- The TopoBDA excerpt of Table 5 has unlabeled columns; "V1.1m" and DET_l_ch of its ablations are undefined in the excerpts.

## 7. Our experiments that touch it
None. A search of the project outside `literature/` finds OpenLane-V2 only in the register (K133, "external reference metrics"), the experiment table and the compatibility study (section k and adapter row 47 "OpenLane-V2 evaluate.py (x.1), adapt_code, S - M"; adapters G1 - G6 of EV B.8 are there, DET_l "computable after G2"). No E-id has used the data or the metric. The existing diagnostics (point F1, lanes found, J, route success) are pooled-point or landmark metrics, not whole-piece Frechet AP.
