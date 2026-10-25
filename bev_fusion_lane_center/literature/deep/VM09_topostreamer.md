# VM09 - TopoStreamer: full-text extraction

| field | value |
|---|---|
| paper | TopoStreamer: Temporal Lane Segment Topology Reasoning in Autonomous Driving. Yiming Yang, Yueru Luo, Bingkun He, Hongbin Lin, Suzhong Fu, Chao Zheng, Zhipeng Cao, Erlong Li, Chao Yan, Shuguang Cui, Zhen Li (corresponding). CUHK-Shenzhen, Wuhan University, Tencent T Lab. arXiv preprint 2507.00709 (cs.CV), no venue on the page; the repository issue tracker says (2026-08-21) the paper is still under review. v1 2025-07-01, v2 2025-07-20, v3 2025-10-16, v4 2025-10-17, v5 2025-11-12 (current). Links opened: arxiv.org/abs/2507.00709, arxiv.org/html/2507.00709 (v5), .../v3, .../v5 |
| axis, tier, K ids | A5, tier 1 (core), K79 (shared with FASTopoWM) |
| level-1 note(s) | [VM09](../papers/VM09_topostreamer.md), [TP19](../papers/TP19_topostreamer.md), [A15](../papers/A15_topostreamer-streaming-for-lane-segment.md) |
| code extraction | [topostreamer](../code/topostreamer.md) (the repository contains no code: README and four GIFs only) |
| sources read | arXiv abs page (version history); arXiv HTML v5: Sections 1, 3, 4 and Appendix A.1 - A.10 through three summarising fetches, plus one targeted fetch for exact wording; arXiv HTML v3 as an independent cross-check of every table (all numbers agreed with v5); GitHub repository page, REST API (tree, commits, issue #1 and its comments), raw README; base repository LaneSegNet and OpenLane-V2 devkit (see the code file). Read on 2026-10-24. The PDF was not read |
| reading marks | **digest** for everything taken from arXiv and GitHub (every page came through a summarising fetch tool, no page was seen raw; numbers may be mis-transcribed, tables 1, 5, module ablation, 2a / 2b, stream ratio and DN count were read twice from two HTML versions and agree). **computed** = my own arithmetic on digest numbers. **analysis** = my reasoning (Section 5). Nothing is marked "checked" because no page was seen raw |
| extracted by, date | extraction agent (literature sub-agent), 2026-10-24 |

## 1. Problem and claim
Input: surround camera images of consecutive frames (OpenLane-V2 subset A, seven cameras). Output per frame: a set of lane segments, each a centreline plus left and right boundary (ten ordered 3D points each) with class (lane segment / pedestrian crossing) and boundary type (dashed / solid / non-visible), and a lane-to-lane adjacency matrix (connectivity topology). Gap named by the authors: earlier streaming map methods (StreamMapNet, SQD-MapNet) have a positional encoding that does not fit propagated queries, and learn too few attributes across time, while topology is very sensitive to the exact position of each segment. Claim: three additions to a LaneSegNet-style decoder (streaming attribute constraints with identity-followed supervision, dynamic lane-boundary positional encoding, lane-segment denoising incl. topology) give +3.0 mAP in lane-segment perception and +1.7 OLS in centreline perception over the best published methods on OpenLane-V2 subset A. The one claim that matters for us: temporal propagation of queries helps topology scores, but the paper never propagates the adjacency itself.

## 2. Method as built

- **Pipeline / architecture** (digest; channel counts of the decoder are not given in the paper, 256 is the LaneSegNet base value):
```
7 cams 2048x1550 -> ResNet-50 + FPN -> BEVFormer-style PV-to-BEV -> F_bev^t (grid 200 x 100, +-50 m x +-25 m)
 F_bev^(t-1) --Warp(Psi)--+--> GRU --> F~_bev^t (also stored for t+1)          [E1]
 stream memory (t-1): top-K queries Q, centre refs R_C, boundary refs R_B
     --> Q^S_t = MLP(concat(Q_{t-1}, Psi)) + Q_{t-1}                           [E2]
 decoder input, layer 1: N = 200 queries = (initialised, learned) with the lowest-confidence ones
     replaced by K = 66 stream queries (+ their reference points)  + 240 DN queries (training only)
 6 decoder layers (Deformable-DETR style, cross-attention = lane attention on F~_bev, guided by boundary refs):
     self-attn -> Duplicate x M + DBPE(boundary refs) -> lane attention -> MLP merge        [E4]
     refs refined by offsets in every layer
 heads per query: centreline (10 x 3) + offset (left = c + o, right = c - o) | class | boundary type |
     BEV mask (query embedding (x) F~_bev, sigmoid) | adjacency (predecessor / successor MLPs -> pair MLP -> sigmoid)
 extra streaming heads on Q^S_t (before decoding): coordinates, class, boundary type, BEV mask -> L_stream  [E3]
```
- **Which queries are carried and how.** The top-k most confident queries of frame t-1 (selection by confidence, rank-based; the paper does not say whether this is the class score) are taken together with their centre and boundary reference points (stream memory holds past BEV features, queries and reference points, Sections 3.1 and A.4). A learned MLP over the concatenation of the query and the ego transform Psi maps them into the frame-t coordinate system, with a residual (E2). The same Q_{t-1} appears inside the MLP and as the residual (the text notes no further detail). DN queries are never carried. In layer 1 the stream queries and their reference points replace the lowest-confidence queries of the initialised set; the appendix writes "N-k" where K is meant, and K = 66 while "30 %" of 200 is 60 (66/200 = 33 %; the paper is inconsistent, the ablation labels the best ratio "30 %"). Reading (inference): 134 learned queries + 66 stream queries = 200. Stream queries can come from both stream and initialised queries of the previous frame, so a lane can be carried indefinitely.
- **Adjacency / topology between frames.** Nothing in Sections 3.1, 3.2 or A.4 says that adjacency or topology is propagated. The adjacency matrix A is predicted in every frame by a pairwise head on the decoder queries (predecessor and successor projections, concatenated, MLP, sigmoid, threshold 0.5; the head is described explicitly only for the DN queries, A.2; for the main queries it is presumably the LaneSegNet-style relationship head, inference). The streaming constraints (E3) have no topology term: coordinates, classes, boundary type, mask only. So topology is carried only implicitly through the content of the carried queries and is recomputed per frame, for all pairs. Topology enters training in two places: the Hungarian-matched main loss (focal, weight 5.0) and the denoising loss (focal on the DN adjacency, weight 5.0).
- **Inputs, outputs.** Lane segment = centreline L^c = (P, C), left boundary L^l = (P, T), right boundary L^r = (P, T); P = M ordered 3D points (M = 10 per polyline, so 30 points per segment); boundaries are the centreline plus / minus a predicted offset; C class; T boundary type; A adjacency N x N; Psi relative transform between the two frames; M_t BEV semantic mask. Pedestrian crossings are handled as the same object type (class 1). Output set size 200 per frame.
- **Equations** (E1 - E7 = order of appearance as returned by the digest; the paper's own tags were not verified):
  - E1: F~_bev^t = GRU(Warp(F_bev^{t-1}, Psi), F_bev^t)
  - E2: Q_t^S = MLP(Concat(Q_{t-1}, Psi)) + Q_{t-1}
  - E3: L_Stream = L_coord^Stream + L_cls^Stream + L_mask^Stream, with L_coord = L1(L~_t, L_t); L_cls = Focal(C~_t, C_t) + CE(T~_t, T_t); L_mask = CE(M~_t, M_t) + Dice(M~_t, M_t). The targets L_t, C_t, T_t, M_t are ground truth of frame t-1 transformed into frame t (for the lossless variant see Labels). Appendix weights: kappa_1..7 = 1.0 (focal), 0.01 (boundary CE), 1.0 (mask CE), 1.0 (Dice), 0.025 (coord), 1.0 (cls), 3.0 (mask), i.e. L_Stream = 0.025 L_coord + 1.0 L_cls + 3.0 L_mask with the inner weights above. Prediction heads on Q^S_t: centreline = sigmoid(MLP + inverse_sigmoid(R^S_C)) denormalised; boundaries by offset; mask = sigmoid(MLP_mask(Q^S_t) (x) F~_bev^t) (the paper does not define the operator).
  - E4 (DBPE layer): F_pe = {PE(P^B_i)}_{i=1..M}; Q~ = MLP( LA( Duplicate(SA(Q)) + F_pe, F_bev, R_B ) ). PE = sinusoidal encoding of the boundary reference point coordinates (A.6); Duplicate copies each query M times (one per boundary point); LA = lane attention with heads-to-regions sampling guided by R_B; the MLP merges the M copies. M = 8 boundary reference points for the PE (4 left, 4 right, Section 4.2). Reference points are refined by offsets after every layer, so every layer re-encodes the current points.
  - E5 (DN query): F^D_pe = MLP({PE(R^C_i)}); F^D_content = MLP(Concat(Emb(C), Emb(T))); Q^D = MLP(Concat(F^D_pe, F^D_content)); R^C = noised GT centre points.
  - E6 (denoising loss): L_denoise = lambda_1 L_coord^DN + lambda_2 L_cls^DN + lambda_3 L_topo^DN with lambda = 0.025, 1.0, 5.0; L_cls^DN = 1.0 Focal + 0.01 CE; L_topo^DN = Focal(A~, A) with A~ = Sigmoid(MLP_topo(Concat(MLP_pre(Q^D), MLP_suc(Q^D)))).
  - E7 (total): L = alpha_1 L_ls + alpha_2 L_stream + alpha_3 L_denoise with alpha = 1.0, 0.3, 1.0 (Section 4.2; Appendix A.3 prints alpha_1 three times, a typo).
- **Losses.** Main lane-segment loss L_ls = 0.025 L_vec (L1) + 3.0 L_seg (1.0 CE + 1.0 Dice on the BEV mask) + 1.5 L_cls (focal) + 0.01 L_type (boundary CE) + 5.0 L_topo (focal on adjacency). Assignment: Hungarian bipartite matching of the 200 queries to the GT lane segments (the cost terms are not listed in the paper; the base code LaneSegNet uses focal class 1.5, lane L1 0.025, mask CE 3.0, Dice 3.0, see the code file). The adjacency loss is applied to matched queries. Stream loss: no matching of its own, the target follows the identity (see Labels). DN loss: no matching (each DN query is built from one GT segment).
- **Labels / ground truth.** OpenLane-V2 subset A (re-annotated Argoverse 2): lane segments with centreline and both lane lines (arc-length resampled to 10 points in the base code), pedestrian crossings, adjacency lsls, and instance IDs (the paper says OpenLane-V2 provides them). Streaming supervision: earlier methods transform the previous frame's GT by the pose, which loses everything that left the BEV range; TopoStreamer instead tracks the unique IDs of the positive instances that the Hungarian assignment matched to the stream queries and takes the frame-t GT of the same ID ("lossless streaming supervision"); for data sets without IDs the authors suggest mask matching. The README lists `..._gt_tracks.pkl` files produced by a tracking script, which fits this (details not available, no code). What happens to a stream query whose ID is gone at frame t, and how new GT at frame t enters, is not described. Subset B (only centrelines): pseudo lane segments by widening each centreline to a standard 4 m width (A.7).
- **Training recipe.** AdamW, lr 2e-4, cosine annealing, 24 epochs of which the first 12 use single-frame input (stabilises streaming training; the README has two training scripts, stage 1 and stage 2, which fits), batch size 8, NVIDIA V100 GPUs (count and time not stated; the README asks for 4 x A100-40GB or 4 x V100-32GB "for batch size 2"). Each epoch starts by cutting every training sequence randomly into 2 parts (more varied sequences); a batch holds N samples from different scenes read in chronological order; temporal fusion only when the previous sample is from the same scene; inference runs over full sequences. 200 queries, 6 decoder layers, 10 points per polyline, DN 240 queries (groups set per batch), DN noise: box shift factor 0.2 and class-label flip probability 50 %. Gradient through the memory (detach or not): not found, no mention of detach or gradient anywhere in the paper. Hungarian parts: the main loss L_ls (and with it the adjacency loss), and the identity that the stream loss follows. Not Hungarian: DN, the stream targets (identity), the GRU fusion.
- **Inference.** Stream memory from the previous frame; top-K by confidence (K = 66); adjacency edge when sigmoid > 0.5; visualisation confidence threshold 0.3; the paper names no NMS and no cap on the number of output segments (all 200 are scored by the AP metrics). IDs are not needed at inference (they only shape the training targets). Frame 0 has no memory: the 12 single-frame epochs give a single-frame mode (inference), not stated explicitly.

## 3. Numbers

Metrics (digest): OpenLane-V2 subset A (more than 20,000 training and more than 4,800 validation frames; seven cameras at 2048 x 1550). Lane-segment perception: AP over Chamfer and Frechet-style thresholds {1.0, 2.0, 3.0} m for road lines (AP_ls) and pedestrian crossings (AP_ped), mAP = their mean, TOP_lsls = topology mAP on lane segments; Acc_b = new boundary classification accuracy (predictions matched to GT by Frechet distance, left and right boundary types compared, no formula given). Centreline perception: DET_l, TOP_ll, OLS. Devkit version: **not stated in the paper**. Reference points: the base code LaneSegNet states devkit v2.1.0 and that this version fixed a loophole that had inflated TOP_lsls (README of LaneSegNet, digest); the OpenLane-V2 README recommends the x.1 metrics. TopoNet's TOP_ll of 10.9 in Table 5 is the "new devkit" value (the overview notes 4.1 under the old one), which suggests the x.1 devkit for the centreline table (inference). Metric definitions as read in the devkit: see the code file, Section 5.

- **Main results.** Table 1 (lane-segment perception, subset A, ResNet-50, 24 epochs; digest, read twice):

| method | venue | temporal | mAP | AP_ls | AP_ped | TOP_lsls | Acc_b | FPS |
|---|---|---|---|---|---|---|---|---|
| MapTR | ICLR23 | no | 27.0 | 25.9 | 28.1 | - | - | 14.5 |
| MapTRv2 | IJCV24 | no | 28.5 | 26.6 | 30.4 | - | - | 13.6 |
| TopoNet | arXiv23 | no | 23.0 | 23.9 | 22.0 | - | - | 10.5 |
| LaneSegNet | ICLR24 | no | 32.6 | 32.3 | 32.9 | 25.4 | 45.9 | 14.7 |
| TopoLogic | NeurIPS24 | no | 33.2 | 33.0 | 33.4 | 30.8 | - | - |
| Topo2Seq | AAAI25 | no | 33.6 | 33.7 | 33.5 | 26.9 | 48.1 | 14.7 |
| StreamMapNet | WACV24 | yes | 20.3 | 22.1 | 18.6 | 13.2 | 33.2 | 14.1 |
| SQD-MapNet | ECCV24 | yes | 26.0 | 27.1 | 24.9 | 16.6 | 39.4 | 14.1 |
| TopoStreamer | - | yes | 36.6 | 35.0 | 38.1 | 28.5 | 50.0 | 13.6 |

  Note: TopoLogic has a higher TOP_lsls (30.8) than TopoStreamer (28.5) in this table; the page says TopoLogic's dedicated topology measures were not included "for fair comparison". StreamMapNet and SQD-MapNet were re-trained by the authors with 10 points per boundary and the centreline taken as the mean of the two boundaries.

  Centreline perception (the HTML labels it "Figure 5"; subset A, ResNet-50, 24 epochs):

| method | temporal | OLS | DET_l | TOP_ll |
|---|---|---|---|---|
| VectorMapNet | no | 13.8 | 11.1 | 2.7 |
| STSU | no | 14.9 | 12.7 | 2.9 |
| MapTR | no | 21.0 | 17.7 | 5.9 |
| TopoNet | no | 30.8 | 28.6 | 10.9 |
| Topo2D | no | 38.2 | 29.1 | 26.2 |
| TopoMLP | no | 37.4 | 28.3 | 21.7 |
| LaneSegNet | no | 40.7 | 31.1 | 25.3 |
| TopoLogic | no | 39.4 | 29.9 | 23.9 |
| Topo2Seq | no | 42.7 | 33.5 | 27.0 |
| TopoFormer (staged training, pretrained lane detector) | no | 42.1 | 34.7 | 24.7 |
| StreamMapNet | yes | 28.8 | 21.7 | 12.9 |
| SQD-MapNet | yes | 33.9 | 27.2 | 16.4 |
| TopoStreamer | yes | 44.4 | 35.2 | 28.8 |

  Computed (my arithmetic on the rows above): the OLS column equals (DET_l + 100 * sqrt(TOP_ll / 100)) / 2 for five rows (TopoStreamer 44.4, LaneSegNet 40.7, TopoNet 30.8, StreamMapNet 28.8, SQD-MapNet 33.9). So OLS here is the devkit OLS without the two traffic-element terms, with a square root on the topology term. The square root makes TOP_ll gains count less at high values.
  Subset B (OpenLane-V2 B, centrelines only, pseudo lane segments, 24 epochs; appendix A.7): OLS / DET_l / TOP_ll: TopoStreamer 42.6 / 30.9 / 29.4; LaneSegNet 38.7 / 27.5 / 24.9; TopoLogic 36.2 / 25.9 / 21.6; TopoMLP 36.2 / 26.6 / 19.8; TopoNet 25.1 / 24.3 / 6.7; SQD-MapNet 29.1 / 21.9 / 13.3; StreamMapNet 26.7 / 18.9 / 11.9; MapTR - / 15.2 / -; STSU - / 8.2 / -; VectorMapNet - / 3.5 / -.
  Reported gains: versus LaneSegNet +4.0 mAP, +3.1 TOP_lsls, +4.1 Acc_b (computed from the table); versus Topo2Seq +3.0 mAP; versus TopoFormer +2.3 OLS and +4.1 TOP_ll; on subset B +3.9 OLS, +3.4 DET_l, +4.5 TOP_ll over LaneSegNet.

- **Ablations that decide transfer** (digest, read twice; differences are computed by me against the baseline row):

| row (module ablation, "Figure 7") | mAP | AP_ls | AP_ped | TOP_lsls | Acc_b | change vs baseline (mAP / TOP) |
|---|---|---|---|---|---|---|
| baseline LaneSegNet (static PE) | 32.6 | 32.3 | 32.9 | 25.4 | 45.9 | - |
| + DBPE | 33.5 | 32.6 | 34.3 | 26.1 | 47.2 | +0.9 / +0.7 |
| + DBPE + DN | 34.5 | 32.9 | 36.1 | 26.7 | 47.6 | +1.9 / +1.3 |
| + stream + static PE | 33.8 | 32.8 | 34.8 | 26.3 | 47.7 | +1.2 / +0.9 |
| + stream without PE | 33.5 | 32.0 | 34.9 | 27.1 | 48.3 | +0.9 / +1.7 |
| + stream + DBPE | 35.6 | 34.7 | 36.5 | 27.8 | 49.5 | +3.0 / +2.4 |
| + stream + DBPE + DN, no ID tracking | 34.4 | 33.6 | 36.2 | 28.0 | 49.5 | +1.8 / +2.6 |
| + stream + DBPE + DN (full) | 36.6 | 35.0 | 38.1 | 28.5 | 50.0 | +4.0 / +3.1 |

  Reading: (a) query propagation alone (stream + static PE) is worth +1.2 mAP and +0.9 TOP; (b) the positional encoding decides whether propagation pays: with DBPE the stream rows gain 1.8 mAP over static PE and 2.1 over no PE (35.6 vs 33.8 / 33.5), the text says "about 2 points"; (c) removing identity-followed supervision costs 2.2 mAP but only 0.5 TOP and 0.5 Acc_b (full vs no ID tracking); (d) it is not stated whether the earlier "stream" rows use ID tracking; if they do, then adding DN without ID tracking (34.4) is worse than stream + DBPE alone (35.6), i.e. DN and streaming need the identity-followed targets to combine; (e) the largest TOP_lsls values in the table come with the stream rows, but the stream rows hold no topology loss of their own.

  Streaming attribute constraints (Table 2a; base = stream + DBPE with none of the constraints, no DN; check-mark placement inferred from the cumulative row order):

| constraints on | mAP | TOP_lsls | Acc_b |
|---|---|---|---|
| none | 33.8 | 26.1 | 47.8 |
| cls | 34.0 | 26.5 | 49.2 |
| cls + mask | 35.1 | 27.0 | 49.2 |
| cls + mask + coord | 35.6 | 27.8 | 49.5 |

  Lane-segment denoising (Table 2b; same base):

| DN losses | mAP | TOP_lsls | Acc_b |
|---|---|---|---|
| none | 33.8 | 26.1 | 47.8 |
| cls | 35.1 | 27.0 | 49.5 |
| cls + topo | 35.7 | 27.9 | 49.7 |
| cls + topo + coord | 36.3 | 28.0 | 49.8 |

  Reading: the biggest single jump in 2a is the mask term (+1.1 mAP, +0.5 TOP); in 2b the topology term of the DN adds +0.6 mAP and +0.9 TOP, the only explicit topology-specific gain in the paper. The two are sub-additive: constraints alone +1.8 mAP, DN alone +2.5 mAP over the common base, both together +2.8 mAP (36.6 vs 33.8). Because the common base 33.8 / 26.1 / 47.8 is not a row of the module ablation, and 2b's last row (36.3 / 28.0 / 49.8) differs from the full model, the base of 2a / 2b is probably "stream + DBPE without constraints and without DN" (digest text), not verified.

  Stream ratio (appendix Table 2a; share of queries propagated): 10 % mAP 34.8 / TOP_lsls 27.9; 30 % 36.6 / 28.5; 50 % 34.3 / 27.3; 75 % 32.9 / 25.1 (the 75 % row is below the single-frame baseline in TOP, 25.4). DN queries (appendix Table 2b): 120 -> 35.0 / 27.4; 240 -> 36.6 / 28.5; 360 -> 35.1 / 27.7.
- **Compute.** FPS (Table 1): TopoStreamer 13.6, LaneSegNet 14.7, Topo2Seq 14.7, StreamMapNet 14.1, SQD-MapNet 14.1, MapTR 14.5, MapTRv2 13.6, TopoNet 10.5. Computed: 73.5 ms vs 68.0 ms per frame, i.e. about 5.5 ms (8 %) for the memory and the extra heads (denoising is training only). GPU, batch size and input size of the FPS measurement: not found. Parameters: not found. Training time: not found. GPU count: not found in the paper (README: 4 GPUs).
- **Caveats.** Argoverse 2 lane segments (a segment is a lane piece with its own two boundaries, 7 cameras, 2048 x 1550) are not nuScenes lane centre-lines or our graph vertices; the range is +-50 x +-25 m (the base config is 102.4 x 51.2 m, see the code file); the baseline LaneSegNet row in this paper (32.6 mAP) differs from the number in the LaneSegNet README (33.5 mAP, same TOP_lsls 25.4), so the +4.0 against LaneSegNet is relative to the authors' own number; the devkit version is not stated; no temporal-consistency metric was seen (only qualitative Figures 6 and 9: five consecutive frames, moving and stationary ego); no seeds, no variance; the authors re-trained the two streaming baselines themselves. Numbers must not be compared with ours.

## 4. Paper versus code
No code. The repository (YimingYang23/TopoStreamer) holds a README and four GIFs (last commit 2025-07-31); issue #1 asked for the training code, the author answered on 2025-08-18, 2026-05-26 and 2026-08-21 that the paper is under review and the code will be opened on acceptance. The README (digest) says the project is built on LaneSegNet, names a two-stage training (two scripts), a data-processing script and a "tracking" script that produce `data_dict_subset_A_{train,val}_lanesegnet_gt_tracks.pkl` (GT instance tracks), and pins Python 3.8, PyTorch 1.9.0, mmcv-full 1.5.2, mmdet 2.26.0, mmsegmentation 0.29.1, mmdet3d 1.0.0rc6. Differences visible without the code are between the paper and its base code (listed in the code file): boundary-type loss weight 0.01 in the paper vs 0.1 in the LaneSegNet config, BEV range +-50 x +-25 m vs +-51.2 x +-25.6 m, GRU fusion vs the BEVFormer temporal self-attention of the base.

## 5. Transfer to our pipeline

Short answers to the questions of the task (analysis, from the digest above):

1. **How streaming propagation works.** Per frame: keep the 66 most confident of the 200 decoder outputs (queries with content, centre and boundary reference points), update them with an MLP over [query, Psi] plus a residual, put them in place of the least confident learned queries in layer 1, decode as usual. BEV features are warped and fused by a GRU in parallel. Training adds an auxiliary prediction from the updated queries (before decoding) of coordinates, classes, boundary type and mask for frame t, supervised by the GT instance with the same ID.
2. **Topology between frames.** It is recomputed. Each frame a pairwise head over the decoder queries gives A; A is never stored, warped or used as a prior; the streaming loss has no topology term. The topology-specific training signal is the focal loss on matched queries and on DN queries. The curated line "the only temporal model that carries topology" should read "the only temporal lane-segment model with a topology head; topology rides on the carried queries".
3. **Dynamic PE and temporal modules, ablation value.** DBPE alone +0.9 mAP / +0.7 TOP; it is worth about 2 mAP only when queries are propagated (35.6 vs 33.8 / 33.5). Stream alone +1.2 / +0.9. Constraints on top (no DN) +1.8 mAP / +1.7 TOP; ID-followed supervision +2.2 mAP / +0.5 TOP at the end; DN topology term +0.6 / +0.9. Best carried share 30 %; 75 % is worse than no streaming in TOP.
4. **Regime and losses.** Hungarian main loss (200 queries, focal + L1 + mask + dice cost in the base code), adjacency focal on matched pairs, DN without matching, stream targets by tracked IDs of Hungarian positives; 12 single-frame epochs then 12 streaming epochs; sequences cut in two per epoch.
5. **Metrics, numbers, compute.** DET_l and TOP_ll via the OpenLane-V2 centreline evaluation (Frechet 1 / 2 / 3 m, topology at edge confidence 0.5), lane-segment mAP and TOP_lsls via the lane-segment evaluation; devkit version not stated; 44.4 OLS (35.2 / 28.8) against 40.7 (31.1 / 25.3) for LaneSegNet; 13.6 against 14.7 FPS.

| question | answer |
|---|---|
| what it needs from us | Frozen cached BEV 200 x 200 at 0.5 m (39 ch): the BEV warp + GRU of E1 runs on any BEV raster (our analogue is E39 / E41). The query path needs our own per-vertex token (128-d) and a confidence or validity for the keep rule. Needs the relative ego transform Psi between consecutive frames (relative motion, not a global pose; untested under pose error: not found in the paper). Needs a stable GT identity of each vertex over frames (lane token + arc length from the nuScenes map expansion is available offline). No SD map, no image features, no lidar |
| what it gives us | A5 recipe and ablation evidence for: carrying a confident subset, a pose-conditioned token update, a forecast loss on the warped prior, positional encoding recomputed from current positions, and denoising of the prior (incl. an edge loss). A9: OpenLane-V2 DET_l / TOP_ll definitions for an external comparison. Nothing for A1 - A4, A6 - A8 |
| conflicts | Hungarian matching in L_ls and in the definition of the followed ID; fixed 200-query set with a top-k replace rule (hard index choice, gradient only into the selected items); pairwise adjacency over all query pairs recomputed every frame (our edges are explicit); segment-level units, not vertices; perspective-image BEVFormer encoder and camera-only training; Argoverse 2 labels; no code and no licence file (nothing to copy, nothing to inherit). Gradient through memory unknown |
| reuse mode | idea only; re-implement from the description (no code exists) |
| effort | S (hours): (i) ablate the carried share in our time recursion (keep the top p % of previous vertices by validity, p = 10 / 30 / 50 / 100) since 75 % hurt them; (ii) check that the position encoding of a carried vertex token is computed after the ego warp, not stored from the previous frame. M (days): forecast loss on the warped prior against the identity-followed GT of frame t; denoising of the prior (shift vertices by a fraction of the spacing, flip validity with probability 0.5) with an edge-recovery loss. L (weeks, not in the paper): an explicit edge-state memory (warp parent edges and validity with the vertices, update by evidence), because the paper offers no recipe for it |
| expected gain and risk | Small and uncertain. Their temporal part without extra supervision is +1.2 mAP / +0.9 TOP on a weak single-frame baseline (32.6), most of the gain comes from the supervision (constraints, ID following, DN) and the gains are sub-additive. Our single-frame result is strong and plain temporal feature fusion gave +0.008 F1 (E39 / E41). Risk: the forecast loss and the prior denoising need GT identity; the "nearest GT point" way around it is a matching-like step and conflicts with the project rule unless it is expressed through the dense target field; locking in old errors if too much of the prior is kept (their 75 % result) |
| feeds which experiment / K id | K79 (propagate confident queries) exists. Proposed new techniques, to be registered before use (next free ids after K187, check): identity-followed forecast loss on the warped prior (streaming attribute constraints); position encoding recomputed from current positions in every layer / after the warp (DBPE); denoising of the prior with an edge-recovery loss (lane-segment denoising with topology DN). Candidate experiments: carried-share sweep (S) on the loop-16 time recursion; prior denoising with edge loss (M) |

What would transfer to a time recursion on our vertex set with explicit edges and validity:
- Carry only the confident part of the prior and let fresh proposals fill the rest (their optimum 30 %). In our recursion the whole warped set is the prior; the share sweep tests whether a cap helps. Cheap, no training.
- A pose-conditioned update of the carried token (MLP over [token, Psi] with a residual). Identical in spirit to StreamMapNet's update; our recursion already warps vertices and tokens, the MLP over Psi is the part to check.
- A forecast loss on the updated prior before it sees frame-t features (position, validity, and for us the parent edge). It trains the warp and update module directly and needs no matching if each vertex has a GT identity; this is the strongest piece (constraints +1.8 mAP, mask term +1.1 of it).
- Positional encoding recomputed from the current position (after warp and after every refinement). Cheap to verify, was worth about 2 mAP once queries are propagated.
- Prior denoising: noise the geometry and the labels of carried items during training and supervise recovery of position, validity and edges. It uses known identity, not matching, and doubles as exposure-bias training for the recursion.

What depends on a fixed query set or matching and does not transfer:
- The main loss L_ls and its adjacency term (Hungarian on 200 queries) and the "tracked IDs of Hungarian positives" that define the stream targets.
- The top-k / replace-lowest-confidence rule over a fixed set of N queries; our vertex set has variable size and validity per vertex.
- The pairwise 200 x 200 adjacency head recomputed per frame; it is the opposite of an explicit edge memory.
- Whole-segment units: a lane segment of OpenLane-V2 carries its own 3-polyline geometry; our vertices have no instance identity unless we define it.
- BEVFormer image-to-BEV, the camera-only and Argoverse-2 evaluation protocol.

## 6. Not found / open
- Devkit version of the OpenLane-V2 evaluation; GPU, batch and input size of the FPS measurement; parameter count; training time; GPU count.
- Whether gradients flow through the stream memory; what happens to stream queries whose ID disappears; the confidence used for the top-k; the attention mask between DN, stream and learned queries; the Hungarian cost weights of TopoStreamer itself; how reference points are transformed by Psi; how many frames per sequence at inference and the frame rate; whether the earlier stream rows of the ablation use ID tracking; the base of Tables 2a / 2b; the formula of Acc_b; the definition of the mask operator.
- No temporal consistency metric, no seed variance, no robustness to pose error, no failure case study (A.9 names long-tail, curvature, faint markings as limits).
- Inconsistencies in the paper: K = 66 vs "30 %" of 200; "N-k" vs K; alpha_1 printed three times in A.3; a "layer 0" in A.6; "Table" vs "Figure" labels in the HTML.
- Differences from the level-1 notes (VM09 / TP19 / A15): (1) 66 of 200 is 33 %, not exactly 30 %; (2) "carries topology" is too strong, adjacency is recomputed, the streaming loss has no topology term; (3) "Verified: checked" in TP19 does not hold for the equations, which were not seen raw; (4) the repository holds no code at all (A15 says "code: not checked"); (5) TopoLogic has a higher TOP_lsls (30.8) than TopoStreamer in Table 1. Level-1 numbers (module ablation, stream ratio, DN count, results) agree with this extraction.
- Code: none released; not seen; the LaneSegNet base is described in the code file.

## 7. Our experiments that touch it
- E39 / E41 (loop 2, temporal feature fusion, BEV-GRU analogue): +0.008 F1, kept (level-1 notes). STATE.md line 23 (loop 2) records for E41 zero-shot: single frame .892 / .796, plain temporal .893 / route .735 ("topology broken"), gate .893 / .795. The figures are copied as written there, the column meaning was not re-derived. This is the one place where we saw plain temporal fusion damage the topology.
- E42 / E45 / E46 / E51 / E56 learned link heads: within 0.01 of the rules, dropped (level-1 note); streamed or denoised topology: none.
- K79 is listed for E80 (temporal head) and E118 (clean-trained consumers under failure) in the register; their verdicts were not re-read for this extraction.
- Loop 16 time recursion (ego-warped previous vertices + tokens as prior) is part of the champion; no carried-share sweep, no forecast loss, no prior denoising has been run.
