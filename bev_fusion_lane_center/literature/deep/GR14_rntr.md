# GR14 - RNTR (RoadNet Sequence, RoadNetTransformer): full-text extraction

| field | value |
|---|---|
| paper | Translating Images to Road Network: A Sequence-to-Sequence Perspective. Jiachen Lu, Ming Nie, Bozhou Zhang, Reyuan Peng, Xinyue Cai, Hang Xu, Feng Wen, Wei Zhang, Li Zhang. arXiv 2402.08207 (cs.CV). Three versions: v1 13 Feb 2024 = ICCV 2023 conference version (title in the repository README: "... A Non-Autoregressive Sequence-to-Sequence Approach"), v2 31 Aug 2024 = extended version, **v3 20 Aug 2025 = current, the text read here** (the README says TPAMI 2025; the arXiv page lists no journal reference). Numbers of v1 may differ from v3. Links opened: arxiv.org/abs/2402.08207, arxiv.org/html/2402.08207 (v3), github.com/fudan-zvg/RoadNet (now LogosIntelligenceLab/RoadNet) pages, raw files and issues |
| axis, tier, K ids | A9 (evaluation, data, metrics), tier 1 core; K47 (landmark and reachability metrics for forks and merges, our J metric); also the frozen-BEV-encoder precedent of axis A1 |
| level-1 note(s) | [GR14](../papers/GR14_rntr.md), [SN02](../papers/SN02_rntr-frozen-bev-encoder-sequence.md), metric collection [GR28](../papers/GR28_metrics-used-by-growth-papers.md) |
| code extraction | [roadnet](../code/roadnet.md) (MIT; AR-RNTR and LaneGraph2Seq only: **no SAR / NAR code, no checkpoints**) |
| sources read | arXiv abstract page (versions, comments); arXiv HTML v3, whole page (116,813 characters, about 17 separate queries incl. the last 16,813 characters at offset 100000; no appendix beyond the qualitative figures exists); GitHub repository pages (README, release notes, directory listings, commits, issues #3 #9 #10 #11 #12), raw files `configs/rntr_ar_roadseq/lss_ar_rntr_changeloss_test_fp16_torch2.py`, `rntr/nus_reach_metric.py`, `rntr/transforms/roadnet_reach_dist_eval.py`, `rntr/core/centerline/structures/ljc_bz_centerline.py`, `.../pryordered_bz_centerline.py`, `rntr/transforms/centerline_utils.py`, `rntr/transforms/loading.py`, `rntr/ar_rntr.py`, `rntr/ar_rntr_head.py`, `rntr/road_seg.py`, `rntr/data/nuscenes_converter_pon_centerline.py`; all read 2026-10-25. Local files read for section 5: `evaluation/sparse_lanes/graph_eval.py`, `evaluation/sparse_lanes/README.md`, `experiment_execution/plans/Literature_Compatibility_Space_EN.md` (section k), OVERVIEW sections 1, 2b, 5 |
| reading marks | every statement about the paper or the repository is **digest**: all pages came through the summarising fetch tool, which refused to reproduce the metric section verbatim and gave paraphrases; numbers may be mis-transcribed (one internal inconsistency found, see section 3). Statements about our own code (graph_eval.py) are **checked** (file read). The nested-greedy property in section 5.4 is **checked** (own test script, 2000 random trials x 9 tolerances, 0 mismatches). Section 5 is my analysis, not paper content |
| extracted by, date | Claude (Sonnet 5.5) literature-extraction agent, 2026-10-25 |

## 1. Problem and claim
- Predict the road network (a directed acyclic graph of landmarks and Bezier centre-line edges) from six surround cameras (nuScenes, Argoverse) as one **integer sequence** ("RoadNet Sequence") decoded by a transformer. Euclidean content (positions, control points) and non-Euclidean content (category, parent index) share one token stream.
- Three decoders for the same sequence: autoregressive (AR, one token at a time), semi-autoregressive (SAR, a DETR-style key-point set first, then one sub-tree per key point decoded in parallel) and non-autoregressive (NAR, masked-token prediction in 3 iterations). Claim: SAR and NAR beat AR in accuracy and NAR is 47 times faster (Table IV).
- Second claim (v2/v3): on a non-overlapping split the scores collapse (AR landmark F 54.1 -> 36.5); the bottleneck is landmark detection of the BEV encoder, not the sequence head. Remedies: topology-inherited training (TIT, lidar-map teacher) and an SD map as sequence prefix.
- For us the paper matters for two things: its **landmark / reachability metric definitions** (K47) and its **frozen BEV encoder + sequence head** regime. The sequence head itself is out of scope (tokens, argmax, Hungarian key points).

## 2. Method as built

### Pipeline (digest)
```
6 images 128 x 352 --> image backbone (ResNet-50 DeepLabV3+/Cityscapes, or VoVNetV2) --> CPFPN neck (256 ch)
   --> LSS view transformer (depth bins 4..48 m step 1) --> BEV feature map F, 0.5 m cells, x in [-48,48], y in [-32,32] m
       (config grid: 192 x 128 cells, 256 ch; the stride of the map the decoder attends to: not found)
   --> sequence decoder (6 transformer decoder layers, d 256, 8 heads, FFN 1024), cross-attention to F + sine BEV position code
   --> integer tokens (vocabulary 576) --> nodes + Bezier edges (graph)
SAR : key-point decoder (6 layers, DETR-style set, Hungarian) -> <= 34 key points -> parallel-seq decoder (3 layers) over sub-sequences
NAR : SAR weights, fine-tuned as masked-token predictor, 3 mask-predict iterations
```
Config values (digest of the released AR config): `grid_conf` x [-48,48,0.5], y [-32,32,0.5]; `bz_grid_conf` x and y [-55,55,0.5] (220 cells); LSS `d_out` 256; decoder 6 layers, 8 heads, FFN 1024, dropout 0.1; `num_query` 900; `max_center_len` 601; vocabulary `num_center_classes` 576.

### Inputs, outputs
- Input: multi-view images (+ optional SD-map sequence prefix, v2/v3). Output: token sequence -> list of nodes with type, position and quadratic-Bezier middle control point per edge.
- **Graph as DAG** G = (V, E): vertices are landmarks (the text names crossroads, stop-lines, fork points, merge points; four categories), edges are centre-lines with one Bezier middle control point (section III-A).
- **DAG -> directed forest** (III-B): each merge point gets clones of all its parents except the first as children; the original edges are removed, the clones carry category Clone so the direction can be restored. After that every non-root vertex corresponds to exactly one edge (vertex-edge bijection). Order = depth-first search; clone vertices are visited right after their originals; sibling tie-break "random or by coordinates, front-right first" (the digest does not say which one the tables use; the released code sorts by x coordinate, see section 4).
- **Coupled sequence**: six integers per vertex-edge pair: x, y (integer part), category (Ancestor 0 = root, Lineal 1 = first child, Offshoot 2 = further child, Clone 3), parent index (topological index of the parent; implied for Lineal; None for roots; clones share the index of their original), two integers for the Bezier control point (value + 10 to avoid negatives). Claimed lossless, O(|E|) long.
- **Decoupled sequence** (III-C): a vertex sequence (topological order, ends with <EOV>) followed by an edge sequence (per source vertex a group of triples [child index, e_mx, e_my], each group ends with <Split>, sequence ends with <EOE>); <Start> first, <N/A> pads. Table IV marks the **coupled** variant with an asterisk (AR coupled 52.8 / 59.2 against 54.1 / 61.3 for the unmarked AR row), so the main AR row is presumably decoupled; the text never labels the unmarked rows (digest of the caption: "use coupled road network").
- **Embedding ranges (Table I)**: x, y 0-199; category 200-249; parent index 250-349; control points 350-569 (220 values = the 220-cell Bezier grid); noise category 570; EOS 571; Start 572 (printed "57572", typo); N/A 573. Default embedding size 576. Quantisation of the vertex coordinates: the paper text does not give the cell size (digest); the config says 0.5 m cells.

### Equations (numbered as in the paper; digest)
- E1 AR objective: max sum_{i=1..L} w_i log P(y_i | y_<i, F); w_i down-weights the very frequent tokens (Lineal, parent index 0). E13 repeats it as the class-weight ablation.
- E2 key points do not depend on earlier vertices: P(y_i | y_<i, F) = P(y_i | F).
- E3 SAR objective: max sum_{i=1..M} sum_{j=1..L} (log) P(y_{i,j} | y_{1:M,1:j-1}, F, V_kp); M sub-sequences, L positions, V_kp the key points (printed without the log in the digest).
- E4 key-point Hungarian loss: sum_{i=1..M} [ -log p_{sigma(i)}(c_i) + 1{c_i != 0} * || k_i - khat_i ||_1 ]; sigma = Hungarian assignment, c_i class (0 = empty), k_i key-point position.
- E5 NAR objective: max sum_{i,j} (log) P(y_{i,j} | yhat, F, V_kp), yhat = the masked guess. E14: class-weighted version.
- E6 - E8: Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 = 2 P R / (P + R).
- E9 joint objective min over (theta_f, theta_g) of E[ CE(g(f_I(x)), y) ]; E10 stage 1: min over theta_g of E[ CE(g(f_L(x_L)), y) ] with the lidar-map BEV encoder f_L; E11 stage 2: min over theta_f of E[ || f_I(x) - f_L(x_L) ||_1 ]; E12 L_dist = || F_img - F_lidar ||_1.

### Losses and assignment
- AR / SAR sequence loss: class-weighted token cross-entropy. Released config: per-token weight vectors of length 576, all 1.0 except entry 201 (category Lineal) = 0.2 and the connect weight at entry 250 (parent index 0) = 0.2 (digest). Table IX: class weights 1.0 / 0.5 / 0.1 / 0.2 give 60.1-60.2 / 65.5-66.1; the paper does not mark the chosen one.
- SAR key points: Hungarian matching with cost -class probability + L1 (E4). **Hungarian is only in the key-point decoder; AR has none.**
- NAR: masked-token likelihood, mask ratio 90 % during fine-tuning.
- **Sequence noise padding**: noise vertices with random values are appended to the input until a fixed length; their target is the noise category, all other fields are N/A with zero loss. It exists because AR models otherwise stop early. Sequence augmentation (jittering positions) hurt (Table VIII).

### Labels
- From the nuScenes map: only `lane` and `lane_connector` centre-lines (resolution 0.5 m), cropped to +-70 m around the keyframe ego pose, endpoints taken from incoming / outgoing lanes, nodes = line endpoints merged when closer than 2.1 units; no stop-line layer is used by the released converter (digest of `nuscenes_converter_pon_centerline.py` and `centerline_utils.py`). Details in [../code/roadnet.md](../code/roadnet.md) section 4.
- Train / val lists come from `nuscenes_pon_split.py` (the file name says PON; the tool counted about 698 / 148 scenes, unverified). The "new non-overlapping split" has no scene counts and no construction procedure in the text; the code of it was not seen.

### Training recipe (digest)
- BEV encoder pre-trained as a centre-line segmentation network (config name `lss_roadseg_48x32_b4x8_resnet_adam_24e`: x/y half-extents 48 / 32, batch 4 per GPU, 8 GPUs, Adam, 24 epochs, read from the file name only); paper text: loaded and frozen while the sequence decoder trains. VoVNetV2 is not used on the new split because its pre-training used the default nuScenes split.
- AR: 300 epochs, AdamW lr 2e-4, weight decay 0.01, gradient clip 35, batch "2 x 8", linear warm-up over 500 iterations then cosine, mixed precision (config); sequence padded to 6 x 100 (100 vertex-edge groups of 6 integers = 600 tokens, `max_center_len` 601); augmentation flip, rotation (22.5 degrees in the config) and scale 0.95-1.05 of the centre-lines / BEV.
- SAR: same recipe; 6 key-point layers + 3 parallel-seq layers; sub-sequences padded to 6 x 18 (18 groups of 6 integers, inferred); at most 34 key points. NAR: initialised from SAR, 100 extra epochs, mask ratio 90 %, 3 inference iterations.
- TIT (AR only): stage 1 the sequence head on lidar-map BEV features (300 epochs, 20 Hz sweeps, rotation +-45 degrees, scale 0.9-1.1); stage 2 the lidar encoder frozen, the camera encoder trained with L1 to the lidar-map features (24 epochs); stage 3 assemble and fine-tune (100 epochs in IV-B, "a few epochs" in III-K).
- SD-map variant: OpenStreetMap highway nodes and link roads, hand-aligned to the nuScenes map (nuScenes has no GPS), straight edges without control points, cycles broken by duplicating revisited vertices, serialised with the same scheme and placed **before** the RoadNet Sequence as a prompt; the architecture is unchanged.

### Inference
- **AR**: one token per step; code: fixed 600 steps, `topk(k=1)` (greedy argmax), the whole sequence re-embedded every step, no KV cache, no early stop at EOS, no per-position vocabulary mask (digest of `ar_rntr_head.py`); output cut at the first EOS and to a multiple of 6 in `ar_rntr.py`.
- **SAR**: key-point decoder first (all key points in one pass), then the parallel-seq decoder generates the sub-sequences; two factored attentions: intra-seq attention across sub-sequences at the same position j (what makes the sub-trees coordinate; removing it costs 9.7 reachability-F, Table VI), inter-seq attention inside one sub-sequence over earlier positions (causal). Key-point locations plus each sub-sequence's start key point enter as prompt tokens. Key points are vertices with out-degree > 1, in-degree > 1 or in-degree 0; the parent of a key point is duplicated as a Clone to restore the edge.
- **NAR**: start from a fully masked sequence, predict all tokens, re-mask the low-confidence ones, repeat 3 times (token count per re-mask and argmax-versus-sampling not stated).
- No NMS or merge rules are described; tokens are converted to nodes and Bezier edges, coordinates are integers (0.5 m cells in the config).

## 3. Numbers (digest; v3 table numbers)

Protocols: Tables II, III use the PON train / val split of STSU / TPLR (Table II: front camera only); Tables IV, V, VI-IX, XI, XII use the default nuScenes split (the paper calls it overlapping); Table X and the second half of Table V use the new non-overlapping split. Metrics: M = mean (STSU pooled), Detect = detection ratio, C = connectivity (all three of STSU / TPLR, not defined in this paper), L = landmark, R = reachability, P / R / F as E6-E8. Thresholds: sections 5.1 and 5.2.

Table II (nuScenes PON val, front camera):

| method | M-P | M-R | M-F | Detect | C-P | C-R | C-F |
|---|---|---|---|---|---|---|---|
| PINET | 54.1 | 45.6 | 49.5 | 19.2 | - | - | - |
| Poly | 54.7 | 51.2 | 52.9 | 40.5 | 58.4 | 16.3 | 25.5 |
| STSU | 60.7 | 54.7 | 57.5 | 60.6 | 60.5 | 52.2 | 56.0 |
| TPLR | - | - | 58.2 | 60.2 | - | - | 55.3 |
| AR-RNTR | 60.9 | 57.9 | 59.3 | 61.7 | 63.2 | 52.7 | 57.5 |
| SAR-RNTR | 63.5 | 59.9 | 61.6 | 63.5 | 67.1 | 57.2 | 61.7 |
| NAR-RNTR | 62.0 | 59.4 | 60.7 | 62.0 | 66.4 | 56.2 | 60.9 |

Table III (Argoverse; only the AR model is in this table): PINET 47.2 / 15.1 / - ; TR 55.6 / 60.1 / 54.9 ; TPLR 57.1 / 64.2 / 58.1 ; AR-RNTR 57.6 / 64.5 / 58.2 (M-F / Detect / C-F). Margin over TPLR 0.5 / 0.3 / 0.1. SAR and NAR on Argoverse appear only per city in Table XIV (M-F Miami / Pittsburgh: AR 58.2 / 57.2, SAR 60.1 / 60.6, NAR 58.4 / 57.6).

Table IV (multi-camera nuScenes, default split):

| method | L-P | L-R | L-F | R-P | R-R | R-F | FPS (x AR) |
|---|---|---|---|---|---|---|---|
| AR-RNTR, coupled sequence | 60.9 | 46.6 | 52.8 | 73.2 | 49.7 | 59.2 | 0.1 (1x) |
| AR-RNTR | 62.5 | 47.7 | 54.1 | 73.2 | 52.8 | 61.3 | 0.1 (1x) |
| SAR-RNTR | 65.3 | 49.1 | 56.0 | 74.1 | 59.1 | 65.7 | 0.6 (6x) |
| NAR-RNTR | 64.1 | 48.3 | 55.1 | 74.0 | 58.7 | 65.1 | 5.5 (47x) |
| AR-RNTR, coupled, VoVNetV2 | 62.6 | 47.9 | 54.3 | 73.2 | 52.9 | 61.4 | 0.1 (1x) |
| SAR-RNTR, VoVNetV2 | 66.0 | 55.9 | 60.5 | 74.5 | 61.1 | 67.1 | 0.6 (6x) |
| NAR-RNTR, VoVNetV2 | 65.6 | 55.7 | 60.2 | 73.4 | 60.0 | 66.0 | 4.7 (47x) |
| AR-RNTR + TIT | 67.7 | 53.6 | 59.8 | 79.3 | 63.8 | 70.7 | 0.1 (1x) |
| SD-RNTR + TIT | 77.1 | 69.4 | 73.0 | 75.9 | 68.9 | 72.2 | 0.1 (1x) |

(Caption markers as digested: star = coupled road network, dagger = VoVNetV2 backbone pre-trained on extra data; the VoVNetV2 rows are the setting of the ablation tables VI-IX.) Table V (F scores, landmark / reachability): default split AR 54.1 / 61.3, SAR 56.0 / 65.7, NAR 55.1 / 65.1, AR+TIT 59.8 / 70.7, SD+TIT 73.0 / 72.2; **new non-overlapping split** AR 36.5 / 40.9, SAR 36.9 / 43.0, NAR 36.2 / 42.3, AR+TIT 39.1 / 49.8, SD+TIT 54.7 / 62.6. The gap between the splits is 17.6 landmark-F and 20.4 reachability-F for AR; on the new split the three decoders have the same landmark F (about 36.5), which the authors read as a detection ceiling of the BEV features. Table XII (five runs): standard deviations 0.1-0.3 on every column (AR, SAR, NAR); the digest does not say whether the five runs are five trainings.

Ablations that decide transfer (all NAR, VoVNetV2, default split, L-F / R-F):

| table | change | result |
|---|---|---|
| VI | key-point / parallel-seq layers 3/3, 5/3, 6/3, 6/5, 6/6 | 56.1/63.3, 58.2/65.1, 60.2/66.0, 60.5/66.9, 61.3/67.1 (FPS 4.8, 4.8, 4.7, 3.4, 3.0) |
| VI | remove intra-seq attention (6/3) | 55.7 / 56.3 (FPS 5.0): reachability -9.7 |
| VII | mask ratio 50 / 75 / 90 %, 3 iterations | 51.4/51.3, 59.4/62.7, 60.2/66.0 |
| VII | 90 % mask, 1 / 3 / 6 iterations | 59.0/62.7 (8.9 FPS), 60.2/66.0 (4.7), 60.3/66.1 (2.8) |
| VIII | BEV aug only / noise padding only / both / + sequence aug | 58.6/64.3, 57.5/62.7, 60.2/66.0, 59.1/65.2 |
| IX | embedding 576 / 1000 / 2000 (weight 0.2) | 60.2/66.0, 60.1/65.8, 59.7/65.5 |
| XI | AR learning rate 1e-4 / 2e-4 / 3e-4 | L-F 53.0 / 54.1 / 54.0, R-F 60.6 / 61.3 / 60.7 |

Table X (TIT stages, **new split**, ResNet; L-P / L-R / L-F / R-P / R-R / R-F):

| stage | method | L-P | L-R | L-F | R-P | R-R | R-F |
|---|---|---|---|---|---|---|---|
| lidar map | baseline | 47.8 | 38.9 | 42.9 | 60.4 | 24.9 | 35.3 |
| lidar map | + more aug | 52.6 | 41.4 | 46.3 | 70.9 | 25.6 | 37.6 |
| lidar map | + sweep data | 67.2 | 47.2 | 55.4 | 81.1 | 51.0 | 62.6 |
| assembled | freeze BEV | 47.7 | 34.2 | 38.6 | 66.7 | 38.2 | 48.6 |
| assembled | unfreeze BEV | 48.8 | 32.7 | 39.1 | 67.1 | 39.6 | 49.8 |

Arithmetic check by me: every row reproduces F = 2PR/(P+R) to 0.1 **except the freeze row L-F**: 2 x 47.7 x 34.2 / 81.9 = 39.8, the table says 38.6. Either one of the three digested numbers is mis-transcribed or the paper has a typo; the text speaks of a rise of about 0.5 (L-F) and 1.2 (R-F) by unfreezing, which matches the printed F values. Another row, Table IV NAR R-F: 2 x 74.0 x 58.7 / 132.7 = 65.5 against 65.1, within the effect of "mean over thresholds of the per-threshold F" (the released code averages F over thresholds, section 5.1), so not an error.

Robustness (Tables XIII-XV, SAR best or tied in every group): by location (nuScenes) AR M-F Singapore 59.6 / Boston 59.1, SAR 62.0 / 61.1, NAR 60.2 / 61.1; by weather (M-F, Normal / Rainy / Cloudy) AR 59.9 / 58.6 / 59.0, SAR 62.4 / 60.7 / 61.8, NAR 62.0 / 60.3 / 60.2. No night split, no sensor failure.

Compute: AR-RNTR 37.25 M parameters, NAR-RNTR 35.90 M (section IV-D, "computational costs"), SAR not stated; about 6 GB inference memory at batch 2 (NAR); FPS 0.1 / 0.6 / 5.5 (4.7 with VoVNetV2). **GPU type, batch size of the FPS measurement and whether the BEV encoder is included are not stated.** Training wall-clock is not stated; users of the public AR code report 3.6 s per iteration at the start (about 20 days) and about 4 days after upgrading to PyTorch >= 2.2 (issue #3).

Caveats: protocol differences (PON vs default vs new split, 96 x 64 m BEV, nuScenes lane + connector nodes), tolerance sets, and a metric whose released implementation deviates from the text (section 4). Numbers of RNTR are not comparable with our S or J.

## 4. Paper versus code

The public repository (MIT, see [../code/roadnet.md](../code/roadnet.md)) contains the **AR-RNTR** and LaneGraph2Seq code only. A maintainer wrote in issue #3 (a thread about training LaneGraph2Seq) that the original code is confidential and that the released one is a reproduced, slower version (digest; whether this also holds for the RNTR files is not stated), issue #10 asks for SAR / NAR code (open, no answer), issue #7 asks for checkpoints (no answer), issue #12 reports that the loss does not converge and that the inference code has bugs (no answer), nobody in the threads states that the paper numbers were reproduced.

| item | paper (v3) | released code (digest) |
|---|---|---|
| decoders | AR, SAR, NAR | AR only; no SAR / NAR model or config |
| frozen BEV encoder | "loaded and frozen" | class default `freeze_pretrain=True` (requires_grad off for backbone, neck, LSS; no `eval()`, so BN statistics still update), but **both released AR configs set `freeze_pretrain=False`** and load the road-seg checkpoint as `load_from`: the encoder is fine-tuned |
| category names | Ancestor / Lineal / Offshoot / Clone | start / continue / fork / merge (0-3); parent index = "fork from" / "merge with"; confirmed by a maintainer in issue #9 |
| special tokens | noise 570, EOS 571, Start 572, N/A 573 | end 573, start 574, no_known 575 (target mask) |
| landmark thresholds | 0.5 to 5.0 m in 0.5 m steps (10 values) | default list [1,3,5,8,10] in grid cells x 0.5 m = 0.5, 1.5, 2.5, 4.0, 5.0 m (5 values); the configs do not override it |
| reachability thresholds | 0.5 to 2.5 m | [1,2,3,4,5] cells = 0.5, 1.0, 1.5, 2.0, 2.5 m |
| reachability depth | at most 5 edges | chains of 2..5 **nodes** = 1..4 edges (`max_dpt` 5 counts nodes) |
| endpoint matching for reachability | "matched landmarks" | fixed 5 cells = 2.5 m, independent of the reachability threshold |
| matching | nearest GT; one-to-one not stated; Fig. 11 case 3 allows a path predicted twice to count twice | nearest-neighbour, many-to-one, no Hungarian, no greedy |
| landmark set | crossroads, stop-lines, fork, merge points | every node of the sequence graph (depth-1 chains), including degree-(1,1) "continue" nodes at lane / connector boundaries; GT loader is the non-clearing one in train and test; no stop-line layer in the GT converter |
| chain distance | "Chamfer distance" | symmetric mean-of-min distance of 10 points per Bezier segment times a direction penalty tan(-dot)+tan(1)+1 (1.0 for parallel, 2.56 for perpendicular, 4.1 for opposite tangents) |
| headline F | F1 of P and R (E8) | per threshold F from summed counts, then the plain mean over thresholds |
| sequence order | random or coordinate tie-break | deterministic by x coordinate (`orderedDFS=True`) |
| sequence form | coupled (6 integers per group) and decoupled | the AR config uses the coupled 6-integer clause; the decoupled builder `nodesbetween2seq` is in the utilities; which form the unmarked table row uses is not stated (issue #9 asks the same) |
| sequence noise | noise vertices appended | random clauses pad the sequence to max(boxes + 2, 100) boxes in `ar_rntr.py` |

Silent behaviours of the evaluation code that bias the numbers (digest, not run): a frame without a prediction (KeyError) or with any exception is skipped, so missing frames never become false negatives; recall uses prediction-side true positives over (those + ground-truth-side false negatives); unmatched reachability pairs add nothing to FP or FN; chains that dead-end before the depth limit are dropped. See [../code/roadnet.md](../code/roadnet.md) section 5.

### Corrections to the level-1 notes
1. **GR14 / GR28 "landmarks = crossroads, stop lines, fork and merge points"**: true for the paper text, not for the metric code, where every node of the sequence graph is a landmark (also degree-(1,1) lane / connector boundary nodes); the GT converter has no stop-line source. The metric is therefore **not a fork / merge metric**; the project's compatibility study (Literature_Compatibility_Space, section k) assumes "forks, merges, roots" and should say so.
2. **GR14 "SAR ... 6 x 18 tokens"**: it is 18 vertex-edge groups of 6 integers per sub-sequence (AR: 6 x 100 = 600 tokens), at most 34 key points; key points = out-degree > 1, in-degree > 1 or in-degree 0.
3. **GR14 / SN02 "pre-trained BEV encoder frozen"**: the paper text says so, the two released AR configs set `freeze_pretrain=False`. SN02's "unfreezing the encoder barely matters" refers to Table X, which compares frozen and unfrozen **only in the assembled TIT stage** (after distillation to the lidar-map features), not in the default recipe.
4. **SN02 "cell size written '11m', probably a typo for 1 m"**: the config cell size is 0.5 m (`xbound [-48, 48, 0.5]`); 1 m is wrong.
5. **SN02 "BEV channel count not stated"**: LSS `d_out` is 256 in the config. **"Code: not checked"**: checked now, see the code file; **"hardware not read"**: the paper states no hardware at all.
6. **GR28 "reachability: ... matched landmark pairs within 5 edges"**: the code counts 5 nodes (4 edges) and matches the endpoints at a fixed 2.5 m; **"nearest-landmark match"** is many-to-one; the threshold sets in the code are 5 + 5 values, not 0.5 m steps.
7. **GR14 "code ... licence not noted"**: MIT, Copyright (c) 2023 Fudan Zhang Vision Group; only AR (and LaneGraph2Seq) code, no checkpoints; the repository moved to the organisation LogosIntelligenceLab on 2026-10-24 (the fudan-zvg URL still resolves).
8. **GR14 "TIT (distilling camera BEV towards lidar-based features)"**: precise: the teacher is a BEV encoder on a **lidar map** (clean lidar BEV, 20 Hz sweeps), stage 2 freezes it and trains the camera encoder with L1.
9. **GR14 "none" under experiments tried**: our J metric (6 m, 3 m) and the one-to-one greedy matching are the nearest relatives, see section 7.

## 5. Transfer to our pipeline

| question | answer |
|---|---|
| what it needs from us | For the **metrics**: only our predicted and ground-truth graphs in the contract of `evaluation/sparse_lanes` (pieces in metres, ego frame, with successor edges); no model, no mmdet3d. For the **frozen-encoder precedent**: nothing. |
| what it gives us | Axis A9: a published landmark / reachability protocol (tolerance sets 0.5-5 m and 0.5-2.5 m, chain-based reachability) to report next to J and route_f; a topology metric between matched landmark pairs that we do not have; evidence about the seen-to-unseen gap of a frozen BEV encoder (A1) and about location leakage of the default split (A9). |
| conflicts | Tokens, argmax and Hungarian key points (training path); a trainable cross-attention decoder; mmcv 2 / mmdet3d 1.4 / mmengine stack for the code; the released metric is many-to-one, skips frames and has the quirks of section 4; nuScenes non-commercial. The metric itself has no conflict (evaluation only). |
| reuse mode | Metric: **re-implement from the description** (the code is MIT and could be adapted with attribution, but it is bound to mmengine, groups landmarks in a way we do not want and has the bugs listed); sequence decoders: idea only. |
| effort | Fork / merge F1 curve: **S** (hours, extension of `graph_eval`). Directed-chain reachability: **M** (a day or two plus a self-test); blocker: the definition of a path in a sparse graph (section 5.5). TIT-like clean-teacher pre-training: idea only. |
| expected gain and risk | Gain: J split into existence and localisation parts (one curve), a like-for-like statement against published tolerances, a topology metric that is not APLS-sampled. Risk: published numbers are on another GT (nuScenes lane + connector nodes, 96 x 64 m BEV), so only the definitions transfer, not the values; sparse vertices (13.5 per 100 m, about 7 m apart) make the small tolerances almost always zero for forks. |
| feeds which experiment / K id | K47 (exists). Proposed sub-techniques, to be registered before use: tolerance curve of fork / merge / root landmarks with nested greedy matching; directed-chain reachability F with a metre-based path cap. Idea-only candidate for later: clean-teacher pre-training of the graph head (TIT analogue with a GT-rendered map raster instead of a lidar map). No new training run is needed: both metrics re-score stored graphs. |

### 5.1 Landmark metric: precise definition (paper) and what the code does
- **Paper (III-I, IV-B, digest):** each predicted landmark is matched to the nearest ground-truth landmark; TP if that distance is within the threshold, FP otherwise; a ground-truth landmark that is unmatched or beyond the threshold is FN; P, R, F as E6-E8. Thresholds uniformly in [0.5, 5.0] m (III-I lists 0.5 m steps, Fig. 16 plots P and R over them); the text does not say how the per-threshold values become the single number in the tables, whether matching is repeated per threshold, or whether it is one-to-one. The "Mean precision-recall" of Tables II-III uses the same range but is the STSU metric.
- **Which points are landmarks:** the text: crossroads, stop-lines, fork points, merge points (four categories); no extraction rule is given. **Code:** all nodes of the sequence graph (`get_nodechains_dpt(1)`: one single-node chain per node), whatever their degree; GT nodes are the endpoints of nuScenes `lane` and `lane_connector` centre-lines merged within 2.1 units, with the non-clearing loader (degree-(1,1) nodes stay). Roughly: a landmark is a lane or connector boundary, not specifically a fork or a merge.
- **Matching rule in the code:** `node_match` takes, for every predicted point, the nearest ground-truth point (`cdist` argmin) and sets it to "none" beyond the threshold; the same ground-truth node can be hit by several predictions. TP = number of matched predictions; FP = predictions - TP; FN = ground-truth nodes - ground-truth nodes hit at least once. Recall = TP / (TP + FN) with the **prediction-side** TP in the numerator, so duplicates inflate recall (if every ground-truth node is hit, recall is 1 regardless of duplicates). Counts are summed over frames and processes, then P, R, F per threshold (micro average); the reported number is the mean over the threshold list (the metric keys are `LP_<t>`, `LR_<t>`, `LF_<t>` and means `mLP`, `mLR`, `mLF`).
- **Tolerances and why:** 0.5 m = one BEV cell (the vertex coordinates are integer cells of 0.5 m); the upper end 5 m is not justified in the text I could read; for reachability 0.5-2.5 m, again without a reason. Configs and code use 5 values each (units: grid cells x 0.5 m).

### 5.2 Reachability metric: precise definition and what the code does
- **Paper (digest):** two landmarks A, B are reachable if a directed path leads from A to B; only pairs within a maximum of 5 edges count. After landmark matching, a predicted path between the matched predicted landmarks is TP if its Chamfer distance to some ground-truth path A -> B is below the threshold; FN = a ground-truth path with no matched predicted path; FP = a predicted path beyond the threshold of its matched ground-truth path (Fig. 11 case 2); a missed intermediate landmark does not matter if both paths stay within the threshold (case 1); a path predicted twice can count as two TPs (case 3). Thresholds 0.5 to 2.5 m. The Chamfer formula and the path extraction are not given.
- **Code:** (1) chains = all directed paths along child edges with 2 to 5 nodes (BFS over `childs`; chains that dead-end early are dropped; no de-duplication); key = (start node, end node). (2) For every threshold: match nodes with `node_match` at the **fixed** 5 cells (2.5 m); for each ground-truth pair (A, B) take the predicted nodes matched to A and to B and, if that predicted pair has chains, compare chains with `dist_supernode`: symmetric mean-of-min distance over 10 points per Bezier segment, multiplied by the tangent penalty; each predicted chain is assigned to its nearest ground-truth chain, TP if the distance is below the threshold (grid cells), FP otherwise, FN = ground-truth chains not hit; (3) pairs whose predicted endpoints are unmatched add nothing (neither FP nor FN) except that a matched pair with no predicted chain adds all ground-truth chains to FN; counts are micro-summed over frames, F per threshold, mean over thresholds.
- **Net effect:** reachability is a geometric chain-matching score between landmark pairs that are matched at 2.5 m; it is both optimistic (many-to-one, silent undercounting of FP and FN) and strict (direction penalty up to 4x). It is not a graph-theoretic reachability.

### 5.3 Tolerance sets to report
Paper grid 0.5, 1.0, ..., 5.0 m (10 values) and code grid 0.5, 1.5, 2.5, 4.0, 5.0 m are both subsets of the grid {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0} m; the 3 m and 6 m values are our current points. One curve then yields: J6 (unchanged, continuity with loop 16), the paper-style mean over 0.5-5.0 m, the code-style mean over {0.5, 1.5, 2.5, 4, 5} m.

### 5.4 Fork / merge F1 curve over 0.5 - 6 m from our graphs
Our current code (`evaluation/sparse_lanes/graph_eval.py`, checked): `junctions(g)` gives splits (end of pieces with >= 2 successors) and merges (start of pieces with >= 2 predecessors); `_match_points` matches greedily one-to-one by ascending distance up to `tol`; `aggregate` micro-sums matches and counts over frames and computes P, R, F1; `composite` uses J = mean(split6_F1, merge6_F1).
- **Key property (checked):** the greedy one-to-one matching is **nested in the tolerance**: the pairs accepted at tolerance t are exactly the pairs accepted at the largest tolerance whose distance is <= t. So one pass at tol_max = 6 m, keeping the matched distances d_1 <= d_2 <= ..., gives every point of the curve: TP(t) = number of matched distances <= t. I tested this with my own copy of the greedy rule on 2000 random point-set pairs x 9 tolerances: 0 mismatches (continuous coordinates, so no ties). It costs nothing extra over today's J.
- **Recipe:** per frame store the sorted matched distances for splits and for merges; per tolerance t in the grid pool TP(t) over frames (or keep per-frame TP vectors so that `bootstrap.py` can resample scenes), then P(t) = TP(t) / N_pred, R(t) = TP(t) / N_gt, F1(t); report split and merge separately and J(t) = their mean. Add the scalars J6 = J(6 m), mJ(0.5-5) = mean over the 10 paper tolerances, and the gap J(6) - J(2.5), which says how much of the loss is localisation (vertex placement of a fork on a sparse polyline) and how much is existence.
- **Like-for-like landmark set:** RNTR's SAR key points are out-degree > 1, in-degree > 1 and in-degree 0; our contract has pieces between vertices of in / out degree != 1, so our graph nodes are forks, merges, roots and ends by construction. The closest comparable curve is therefore the pooled key-point curve (forks, merges, roots; ends optional) in addition to the separate fork and merge curves. Exclude ground-truth roots / ends cut by the +-50 m border (our ground-truth graphs carry `start_open` / `end_open` keys) and predicted ones within a margin of the border. RNTR's all-node landmark set cannot be reproduced because its extra nodes are nuScenes lane / connector boundaries that our graphs do not contain.
- **Conventions to state with the curve:** one-to-one greedy (stricter than RNTR's many-to-one), micro-aggregation over all VAL frames, every frame scored (RNTR's code skips frames without a prediction), tolerance in metres in the ego frame.

### 5.5 Reachability score from our directed graph
Pieces are our edges and the junction points are our nodes; successor edges give direction. Two variants, both evaluation-only:
- **R-topo (topological, cheap):** match predicted to ground-truth landmarks one-to-one at tolerance t_l (report 2.5 m like RNTR's fixed value and 6 m like J). For all landmark pairs (A, B) with a directed path of at most k pieces (or at most L metres) in the ground truth, ask whether the matched predicted pair is connected by a directed path under the same cap; TP / FP / FN over pairs, F1 per t_l. No geometry, a pure connectivity score for the matched landmarks; the closed-form transitive closure up to k hops is a few matrix products on the node adjacency.
- **R-geo (RNTR-like):** enumerate directed chains of pieces from every node (DFS over successors, cap by path length 40-60 m and at most 4 pieces; **a cap in metres, not in edges**, because our nodes are only branching points, so 4 of our pieces can be 100 m long while 4 RNTR edges are short lane segments); resample each chain every 0.5 m (`resample` / `dense_points` exist); chain distance = symmetric mean nearest-point distance via `cKDTree` (optionally times RNTR's tangent penalty); for each ground-truth chain A -> B and the predicted chains between nodes matched to A and B, TP if the best distance < tau_r, evaluated at tau_r in {0.5, 1.0, 1.5, 2.0, 2.5} m; use a one-to-one assignment of predicted to ground-truth chains so that duplicates do not count twice; count unmatched predicted chains as FP and unmatched ground-truth chains as FN (RNTR's code does not).
- Relation to what we have: `route_f` in `graph_eval` is APLS-like (random source / target pairs within 40 m, path-length ratio), insensitive to small offsets but seeded and sampled; R-geo is deterministic and geometric; the two are complementary. Self-tests to write with it: perfect graph scores 1 at every tolerance; a graph shifted by d scores 0 below d and 1 above (landmark curve step at t = d); a graph with one fork removed loses exactly the fork landmark and the chains through it; duplicated predicted pieces do not raise the score.

### 5.6 How our J (6 m) relates to their landmark metric
| aspect | our J | RNTR landmark F (paper / code) |
|---|---|---|
| point set | forks (piece ends with >= 2 successors) and merges (piece starts with >= 2 predecessors), scored separately, then averaged | all graph nodes (code), i.e. mostly lane / connector boundaries plus forks and merges; one pooled F |
| matching | greedy one-to-one by ascending distance (checked) | nearest ground-truth point, many-to-one (code); paper silent |
| tolerance | 6 m (and 3 m) | 0.5-5 m, averaged over 5 (code) or 10 (paper) values |
| aggregation | micro-sum over frames, then F1; all frames scored | micro-sum per threshold, F per threshold, mean over thresholds; frames without prediction skipped (code) |
| range / data | +-50 m ego square, our GT graph | 96 x 64 m BEV, nuScenes lane + lane_connector, PON default or new split |
Consequence: J(6 m) is the right-hand end of a landmark curve restricted to forks and merges with a stricter matching; it is more lenient than RNTR in tolerance (6 > 5 m) and stricter in matching. A value of J6 cannot be turned into an RNTR landmark F; only the curve at {0.5 ... 5 m} on our graphs, computed with their set of tolerances, is a like-for-like use of the definition, and even then the node sets differ. Never compare our J with their 54-56 (or 36 on the new split).

### 5.7 Does "frozen encoder + head" support our interface assumption?
Partly, with three caveats.
- **For:** the recipe pre-trains the BEV encoder on centre-line segmentation (same objective family as our map-pretrained GaussianCaR) and trains a sequence head on its features; Table X shows that fine-tuning the encoder instead of freezing it changes landmark F by 0.5 and reachability F by 1.2 (new split, after distillation), which the authors call not significant; and the resolution matches ours: 0.5 m cells, 192 x 128 cells, 256 channels against our 200 x 200 cells at 0.5 m with 39 channels.
- **Against / limits:** (1) the frozen-versus-unfrozen comparison exists only in the assembled TIT stage, not for the default recipe; (2) the **released** configs fine-tune the encoder (`freeze_pretrain=False`), and the class default freezes by `requires_grad` only (BN statistics keep updating), so the public code does not demonstrate a truly frozen interface; (3) the decisive effect in the paper is not freezing but the split: AR landmark F falls from 54.1 to 36.5 and reachability from 61.3 to 40.9 when scenes / locations of train and evaluation do not overlap, and SAR / NAR cannot repair it (landmark F identical across decoders); the head is capped by the features. That is exactly our seen-to-unseen gap of the foundation (0.932 on its own training scenes, 0.820 on TEST) and the location-leakage gap of OVERVIEW A9.
- **Useful finding for the interface:** the paper locates the bottleneck in landmark detection of the BEV encoder, and TIT (head trained on clean lidar-map features, then the camera side distilled to them with L1) moves reachability more than landmark F (R-F 40.9 -> 49.8, L-F 36.5 -> 39.1). An analogue for a frozen foundation would pre-train the graph head on a clean, GT-rendered map raster and then train only a small adapter on the cached 39 channels to reproduce that raster's features; this is idea-only here (adapters are in A1 of the overview) and nothing in the paper tests it with a foundation that stays frozen.
- **Conclusion:** supports the assumption that a head trained on a frozen, task-pretrained BEV encoder is a viable regime, but it does not measure the cost of freezing in the default setting and it warns that overlapping splits dominate the number.

## 6. Not found / open
- The reason for the tolerance ranges; how the tables combine thresholds (code suggests mean of per-threshold F; the paper text is silent); whether matching is one-to-one in the original (confidential) code.
- Construction and size of the new non-overlapping split; the PON scene counts (tool count about 698 / 148, unverified).
- Stride and token count of the BEV map the decoder attends to; the GPU and batch size of the FPS numbers; training wall-clock; whether the five runs of Table XII are five trainings.
- Which sequence form (coupled or decoupled) the unmarked table rows use; which class weight was used in the final NAR model.
- SAR / NAR code, checkpoints, the new-split code: not released (issues #7, #10 open). Appendix: none exists beyond two qualitative figures.
- Not read: the v1 (ICCV) text; digest numbers could not be compared with the PDF (PDFs come back binary).

## 7. Our experiments that touch it
- J (split / merge F1 at 6 m) is 20 % of the loop-16 composite S (`composite()` in `evaluation/sparse_lanes/sparse_metrics.py`); at equal sparsity the champion has J 0.265 against 0.408 for the control (OVERVIEW section 1.5); E164 (junction head) reports J 0.140 (Experiment_Table_EN). The 3 m values `split_F1` / `merge_F1` exist in `graph_eval`; `p16_diag` reports fork / merge landmark P / R at 1 / 2 / 4 / 6 m and a one-to-one F1 (Literature_Compatibility_Space, section k). Key-point heads for split / merge reached F1 0.55 / 0.61 in E44 (OVERVIEW A3; tolerance not recorded there).
- Not done: the full 0.5-6 m curve per class, roots in the landmark set, any reachability score, a location-overlap audit of our splits. No experiment used RNTR's sequence decoders (not suitable: tokens and argmax).
