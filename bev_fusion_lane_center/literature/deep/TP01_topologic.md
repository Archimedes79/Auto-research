# TP01 - TopoLogic: full-text extraction

| field | value |
|---|---|
| paper | TopoLogic: An Interpretable Pipeline for Lane Topology Reasoning on Driving Scenes. Yanping Fu, Wenbin Liao, Xinyuan Liu, Hang Xu, Yike Ma, Feng Dai (corresponding), Yucheng Zhang; Institute of Computing Technology, Chinese Academy of Sciences (Xu: Hangzhou Dianzi University). NeurIPS 2024 (venue as given by the repository README and our level-1 note; the arXiv page itself lists no venue and no comments field). arXiv 2405.14747, **v1 only** (submitted 2024-05-23; no later version listed). Links actually opened: arxiv.org/abs/2405.14747, arxiv.org/html/2405.14747 (v1 HTML), github.com/Franpin/TopoLogic (root, commits, directory pages, README summary), raw.githubusercontent.com files `sgnn_decoder.py`, `topologic_head.py`, config `topologic_r50_8x1_24e_olv2_subset_A.py`. |
| axis, tier, K ids | A3 (sparse lane graph, topology and linking), tier 1 core; K30 (geometric link prior: endpoint distance / heading -> link probability; register source "checked"); related K31 (TopoPoint) |
| level-1 note(s) | [TP01](../papers/TP01_topologic.md) (home note), [A17](../papers/A17_topologic-geometric-distance-topology-prior.md) |
| code extraction | none written (decision of the coordinator, 2026-10-25: code extractions only for the most meaningful papers). Repository, licence evidence and the files a later extraction should start with are in section 4. |
| sources read | arXiv abs page (submission history, links); arXiv HTML v1, eight separate queries on 2026-10-25: structure and version; abstract, introduction, related work 2.2; sections 3.1 - 3.3 (equations 1 - 6); 3.4 - 3.6 (equations 7 - 11); verbatim-intent query on Figure 3 text, 3.5, 3.6; sections 4.1 and 4.2; Tables 1 and 2 with section 4.3; Tables 3 - 5 with sections 4.4, 4.5 and conclusion. No appendix or supplement is visible on the HTML v1 page. GitHub pages: repository root and README summary, commit list, directory listings of `projects/`, `projects/topologic/{models,core,utils}`, `models/{dense_heads,modules,detectors}`, `projects/configs`, raw `LICENSE` (HTTP 404). GitHub API calls returned HTTP 403 and were not used. |
| reading marks | Everything about the paper and the code is **digest**: all pages came through the summarising fetch tool (HTML and raw files, no page seen raw), so numbers may be mis-transcribed. Cross-check that passed: for all six TopoLogic / TopoNet rows of Table 1 I recomputed OLS from DET_l, DET_t, TOP_ll, TOP_lt with f = square root on the fractions (section 3), so the Table 1 transcription is internally consistent. Code line numbers are the tool's counts and are good to about +- 3 lines. `relationship_head.py` and the bbox coder were **not read** in this pass (statements about them are carried over from the level-1 note and marked "level-1"). Statements about our own experiments are **checked** (local files read). Section 5 is my analysis; numbers marked "computed" are my arithmetic on constants seen on the pages. |
| extracted by, date | extraction agent (Claude Sonnet 5.5, sub-agent of the literature session), 2026-10-25 |

## 1. Problem and claim
- Predicted: a directed lane-to-lane adjacency matrix (N x N, N = 200 queries) between centre-line instances of OpenLane-V2, from 6 / 7 camera views. Entry (i, j) is positive iff the end point of lane i connects to the start point of lane j.
- Gap named by the paper: existing methods (TopoNet and its successors) put an MLP on independently encoded lane queries, ignore the geometry, and are sensitive to endpoint shifts (connected lanes rarely end exactly at the same point), so they predict too few connections. The authors call this "perception over reasoning".
- Claim: score a pair by a monotone mapping of the distance between the end of i and the start of j (two or three learnable scalars), add a learned query-similarity score, feed the fused matrix to a GNN after each decoder layer. Subset A, v2.1.0 metrics: TOP_ll 23.9 against 10.9 (TopoNet), OLS 44.1 against 39.8; the geometric part alone can be added to already trained models without retraining (TopoNet 10.9 -> 22.3). The authors state the limits themselves: geometry depends heavily on detection accuracy (their Figure 4), and the GNN barely improves lane detection.

## 2. Method as built

### Pipeline / architecture (digest; layer counts from the code config, not from the paper text)
```
7 (A) / 6 (B) images, resized to 1550 x 2048
  -> ResNet-50 (ImageNet) + FPN, 256 ch
  -> BEVFormer encoder (3 layers in the config), BEV grid 100 x 200 (H x W), 256 ch
  -> lane decoder: Deformable-DETR style, 200 queries x 256, 6 layers in the config;
     every layer outputs  points  P_l : 200 x 11 x 3  (x, y, z),  class score,  queries Q_l : 200 x 256
  after EVERY decoder layer l:
     G_dis(l) : 200 x 200  from  end(P_l[i]) vs start(P_l[j])    (no parameters except 2 - 3 scalars)
     G_sim(l) : 200 x 200  from  sigmoid( MLP1(Q_l) MLP2(Q_l)^T )  (two 3-layer MLPs, 256-d)
     G(l)     = lambda1 * G_dis + lambda2 * G_sim               (lambda: 1 x 1, init 1, learnable)
     GNN(Q_l, G(l)) -> queries of layer l+1   (a lane-traffic branch is concatenated in the code)
  separate DETR head: 100 queries for traffic elements (lane-traffic topology TOP_lt; not relevant to us)
```
- **Inputs, outputs:** input camera images; output per sample: up to 200 lanes (11 x 3 points each, class score) and the directed matrices lane-lane (200 x 200) and lane-traffic. Coordinates: ego frame, metres in the OpenLane-V2 config `point_cloud_range = [-51.2, -25.6, -2.3, 51.2, 25.6, 1.7]` (digest), i.e. a BEV of 102.4 m x 51.2 m at about 0.51 m per cell (computed). The lane decoder uses all three coordinates. No padding rule is stated: every query is a candidate lane.

### Equations (numbered as in the paper; digest, each read in a separate query)
- E1 `l_0..l_{n-1} = LaneHead(Q_l^i)`: points from the queries of decoder layer i.
- E2 `d_ij = | l_i^end - l_j^start |`: last point of lane i against first point of lane j. The norm is **not named**, the unit is **not stated**, z is part of the points.
- E3 `D = {d_ij}`, all pairs.
- E4 `f_ours(x) = exp( - x^alpha / (lambda * sigma) )`, x = d_ij; alpha and lambda learnable 1 x 1; sigma the standard deviation of the distance matrix D (section 3.3: of D; section 4.2: "of x", not reconciled). Initial values alpha 0.2, lambda 2 (section 4.2).
- E5 `G_dis = { f_ours(d_ij) }`, values in 0 - 1.
- E6 alternatives of the ablation: (a) `f_gau = exp(-x^2 / 2)`, (b) `f_sig = 2 / (1 + exp(x))`, (c) `f_tan = (exp(-x) - exp(x)) / (exp(-x) + exp(x)) + 1`.
- E7 `Q_emb1, Q_emb2 = MLP1(Q), MLP2(Q)`, Q in R^{N_l x C} (200 x 256); each MLP has three layers (no widths, activations or norms stated).
- E8 `S = matmul(Q_emb1, transpose(Q_emb2))`; E9 `G_sim = sigmoid(S)`. Not symmetric unless MLP1 = MLP2: the matrix is directed, as the labels are.
- E10 `G = lambda1 * G_dis + lambda2 * G_sim`, lambda1, lambda2 learnable scalars, init 1.
- E11 `L = L_det + L_top`.
- GNN: "aggregates features of adjacent lanes with G after every decoder layer"; no update equation in the paper (section 4 of this file gives the code version).

### Code versions of the same quantities (digest of `sgnn_decoder.py`, `topologic_head.py`; line numbers +- 3)
- C1 per decoder layer: `coord = sigmoid(reg_branch(h) + reference)`, then scaled to metres per axis with `pc_range` (x and y always, z when the coordinate dimension is 3; the variable is called `pts_num` but holds the coordinate dimension, pts_dim = 3 in the config).
- C2 `d_ij = sum over c in {x, y, z} of | coord_i[last, c] - coord_j[first, c] |` : **L1 sum**, on the **detached** coordinates (no gradient from the topology term into the points), decoder lines about 89 - 93.
- C3 `topo_ij = exp( - d_ij^P / w ) * (1 - delta_ij)`: P and w are `nn.Parameter` of shape [1], init P = 2, w = 10 (decoder lines about 25 - 28 and 97); diagonal set to zero. `sigma = std(topo)` is computed (line about 95) and **never used**. No normalisation of any kind.
- C4 adjacency handed to the GNN of layer l+1: `A_lclc = lamda_1 * sigmoid(rel_l).detach() + lamda_2 * topo` (decoder lines about 101 - 103); layer 0 receives zeros. The lane-traffic adjacency is `sigmoid(rel).detach()` with no distance term.
- C5 GNN layer for lane-lane (class `LclcSkgGCNLayer`, lines about 319 - 351): `out = X W_self + e * A (X W_fwd) + e * A^T (X W_bwd)`, no bias, no activation, e = `edge_weight`; the FFN (`FFN_SGNN`) concatenates this with the lane-traffic GNN output, applies activation, dropout and a linear 2C -> C, then adds the residual. The config sets `edge_weight = 0.6` (subset A).
- C6 **inference score** (`get_lanes` in `topologic_head.py`, lines about 426 - 476): the distance is recomputed from the last decoder layer's points with the same L1 formula, then `P = 2`, `w = 11.5275`, `lamda_1 = lamda_2 = 1` are **hard-coded**; the learned `decoder.P / w / lamda_1 / lamda_2` appear only in commented-out lines. `score_ij = sigmoid(rel_last)_ij + exp(-d_ij^2 / 11.5275) (1 - delta_ij)`, range 0 - 2, no clipping.

### Losses
- L_det per decoder layer: focal loss on the lane class (gamma 2, alpha 0.25, weight 1.5) + L1 on the 33 regressed coordinates (weight 0.025, all code weights 1.0), after **Hungarian matching** (config: `LaneHungarianAssigner3D` with FocalLossCost 1.5 and LaneL1Cost 0.025; paper text does not restate the matching).
- L_top: only the relation head (G_sim) is supervised. The paper names no type and no weight. In the config the lane-lane relation head uses mmdet FocalLoss (gamma 2, alpha 0.25) with **loss_weight 5**; the lane-traffic head the same. G_dis has no loss: its scalars learn only through the detection loss, through the GNN messages.
- Per-layer supervision: paper and code apply all losses on every decoder layer (earlier layers are logged as d0., d1., ...).
- Normalisation of the topology loss: no `avg_factor` is visible for the lane-lane term in `topologic_head.py` (digest) -> default of the loss, **not found**.

### Labels / ground truth (from code, digest)
- GT adjacency `gt_lane_adj` (N_gt x N_gt, annotation flag `with_lane_adj = True`) is written into a Q x Q zero matrix at the rows / columns of the Hungarian-matched queries: `target[xs, ys] = gt_adj[pos_assigned_gt_inds][:, pos_assigned_gt_inds]` (lines about 266 - 269). **Every other pair is a negative**, including every pair with an unmatched query; no sampling, no positive weight. The target passed to the focal loss is `1 - target` (line about 361): mmdet treats label == num_classes as background, so this is a convention, not a semantic flip (level-1 note; the relation head file was not re-read here).

### Training recipe (paper section 4.2 and config, digest)
AdamW, weight decay 0.01, lr 2e-4 (config: backbone lr x 0.1), cosine annealing, config adds 500 warm-up iterations with ratio 1/3, min lr ratio 1e-3, gradient clipping max norm 35; 24 epochs; 8 x RTX 3090. Batch: paper says 2; the config has `samples_per_gpu = 1` with an auto-scale base batch of 8 (not reconciled). Backbone frozen_stages = 1. Training time: not stated. Augmentation: not stated in the paper.

### Inference
`get_lanes` returns lanes, scores, labels (from the bbox coder, not read) and the lane-lane matrix of C6 and the lane-traffic matrix (sigmoid only). No threshold, NMS or top-k appears in this function; the OpenLane-V2 devkit scores the matrices. No rule step on the topology path beyond the kernel and the sum. The evaluation protocol of TOP_ll in the devkit was not read (TP29 is queued; from memory a rank-based, AP-like score over matrix rows, which would explain why a 0 - 2 score is acceptable: **memory**).

## 3. Numbers

OLS definition (section 4.1, digest): `OLS = 1/4 [ DET_l + DET_t + f(TOP_ll) + f(TOP_lt) ]`, f = square root, applied to the fractions (TOP values printed in percent). Recomputed by me from the table cells: TopoLogic A v2.1.0 = (29.9 + 47.2 + 48.9 + 50.4) / 4 = 44.1; A v1.0.0 41.65 (table 41.6); B v2.1.0 42.35 (table 42.3); A with SD map 47.5; TopoNet A v2.1.0 39.75 (table 39.8). All match, so Table 1 is transcribed consistently. Decomposition of the OLS gain over TopoNet (computed): +4.35 = 1/4 (+1.3 DET_l - 1.4 DET_t + 15.9 for sqrt-TOP_ll + 1.6 for sqrt-TOP_lt); the topology term carries almost all of it, the detection terms are flat.

**Table 1 - centreline, OpenLane-V2** (as printed; validation split per the README, the paper does not say; no seeds or intervals seen)

| subset | method | SDMap | DET_l | DET_t | v1.0.0 TOP_ll | TOP_lt | OLS | v2.1.0 TOP_ll | TOP_lt | OLS |
|---|---|---|---|---|---|---|---|---|---|---|
| A | VectorMapNet | no | 11.1 | 41.7 | 0.4 | 5.9 | 20.8 | 2.7 | 9.2 | 24.9 |
| A | MapTR | no | 17.7 | 43.5 | 1.1 | 10.4 | 26.0 | 5.9 | 15.1 | 31.0 |
| A | TopoNet | no | 28.6 | 48.6 | 4.1 | 20.8 | 35.6 | 10.9 | 23.8 | 39.8 |
| A | **TopoLogic** | no | 29.9 | 47.2 | 18.6 | 21.5 | 41.6 | **23.9** | 25.4 | **44.1** |
| A | SMERF | yes | 33.4 | 48.6 | 7.5 | 23.4 | 39.4 | 15.4 | 25.4 | 42.9 |
| A | **TopoLogic** | yes | 34.4 | 48.3 | 23.4 | 24.4 | 45.1 | 28.9 | 28.7 | 47.5 |
| B | TopoNet | no | 24.3 | 55.0 | 2.5 | 14.2 | 33.2 | 6.7 | 16.7 | 36.8 |
| B | **TopoLogic** | no | 25.9 | 54.7 | 15.1 | 15.1 | 39.6 | **21.6** | 17.9 | 42.3 |

(STSU, VectorMapNet, MapTR rows of subset B exist for v1.0.0 only; the same model scores TOP_ll 4.1 under v1.0.0 and 10.9 under v2.1.0, so the metric version matters more than most method differences.) Section 4.3 of the paper quotes TopoNet's subset-B DET_l as 24.4 and its subset-A TOP_ll as 10.8 against 24.3 and 10.9 in the tables (paper-internal inconsistency).

**Table 2 - lane segment, OpenLane-V2** (subset not stated on the page; TOP_lsls v2.0.0 / v2.1.0 columns from the section 4.3 text): TopoNet mAP 23.0, TOP_lsls 1.0 / -; MapTR 27.0; MapTRv2 28.5; LaneSegNet 32.6 (AP_ls 32.3, AP_ped 32.9), TOP_lsls 8.1 / 25.4; **TopoLogic** 33.2 (33.0, 33.4), TOP_lsls **22.0 / 30.8**.

**Table 3 - mapping function (centreline, subset A, v2.1.0)**

| f | DET_l | DET_t | TOP_ll | TOP_lt | OLS |
|---|---|---|---|---|---|
| tanh (E6c) | 27.6 | 47.2 | 15.1 | 24.8 | 40.9 |
| sigmoid (E6b) | 28.7 | 44.1 | 19.1 | 24.1 | 41.4 |
| Gaussian (E6a) | 28.9 | 46.8 | 21.7 | 23.2 | 42.6 |
| **learned exponent (E4)** | 29.9 | 47.2 | 23.9 | 25.4 | 44.1 |

My observation (computed, not in the paper): the half-value point of each function in units of x is tanh 0.55, sigmoid 1.10, Gaussian 1.18, ours (with the code's constants P = 2, w = 11.5275) 2.83 (section 5.2); the order of TOP_ll follows the order of the half-values. If x is in metres the fixed Gaussian is 2.4 times narrower than the learned one and loses 2.2 TOP_ll. The paper does not state the unit of x for the alternatives, so this is a consistency, not a proof; the paper's own statement is that the learned function has the largest tolerance for endpoint shift (Figure 3).

**Table 4 - topology approach (centreline, subset A, v2.1.0): with and without each part**

| approach | DET_l | DET_t | TOP_ll | TOP_lt | OLS |
|---|---|---|---|---|---|
| MLP (pair classifier, TopoNet style) | 27.8 | 46.8 | 10.8 | 23.9 | 39.1 |
| similarity only (E7 - E9) | 28.1 | 46.4 | 12.9 | 23.7 | 39.8 |
| GeoDist only (E2 - E5) | 28.6 | 44.1 | 20.1 | 23.1 | 41.4 |
| **both (E10)** | 29.9 | 47.2 | 23.9 | 25.4 | 44.1 |

Reading: geometry alone is worth +9.3 TOP_ll over the MLP, the similarity term adds +3.8 on top of it, similarity alone adds +2.1 over the MLP. DET_l moves between 27.8 and 29.9 across arms although the topology term has no direct path into the points (single runs, no seeds seen).

**Table 5 - GeoDist as post-processing on trained models (no retraining)**

| model | setting | metric | before | after |
|---|---|---|---|---|
| TopoNet | centreline | TOP_ll | 10.9 | 22.3 |
| SMERF | centreline + SDMap | TOP_ll | 15.4 | 26.2 |
| LaneSegNet | lane segment | TOP_lsls | 25.4 | 29.6 |

How the post-hoc kernel was parameterised is not stated; the released inference code uses fixed constants (section 4), which suggests fixed constants here too (my inference).

**Compute:** parameters, FPS and training time are **not stated**. The geometric term itself is an elementwise operation on a 200 x 200 matrix (40,000 pairs); the code builds (B, 200, 200, 11, 3) repeated tensors although only the first and last point are used (computed: about 5 MB per sample and tensor in fp32), so it can be made nearly free.

**Caveats:** OpenLane-V2 devkit versions change TOP_ll by a factor of two to three; subset A and B differ in cameras (7 / 6) and map source (Argoverse 2 / nuScenes); numbers are validation, single run, no confidence intervals seen; none is comparable with our F1 / S / J.

## 4. Paper versus code

Repository: https://github.com/Franpin/TopoLogic (owner Franpin; the URL on the level-1 note is correct). Read: branch `main`, 22 commits, latest listed `c7b37f7` (all recent commits titled "update readme", dated May 2025 on the commit page; the code release is announced in the README for 2024-10-06). 133 stars, 16 forks, 4 open issues. Stack per README: Linux, Python 3.8, CUDA 11.1, PyTorch 1.9.1, torchvision 0.10.1, `requirements.txt`; the model files use mmcv / mmdet / mmdet3d registries (config `type=` dicts, `custom_imports`, mmdet losses and assigners). One checkpoint is listed in the README (ResNet-50, 24 epochs, subset A, OLS 44.1); none for subset B.

**Licence:** none found. The repository root lists `figs/`, `openlanev2/`, `projects/`, `tools/`, `README.md`, `requirements.txt`; no LICENSE file; `raw.githubusercontent.com/Franpin/TopoLogic/main/LICENSE` returns HTTP 404; the README summary states no licence (digest). The GitHub API was not reachable (403), so a licence field in the repository metadata was not checked. Consequence under our rules: **read only**, describe and re-implement from the description, copy no code. The formulas themselves are in the paper and need no code.

**Where a later code extraction should start** (files and functions; I did not read further):
- `projects/topologic/models/modules/sgnn_decoder.py`: class `TopoLogicSGNNDecoder` (parameters `w, P, lamda_1, lamda_2`; the distance / kernel / adjacency block at lines about 75 - 108), `FFN_SGNN`, `LclcSkgGCNLayer`.
- `projects/topologic/models/dense_heads/topologic_head.py`: `get_lanes` (inference score with the hard-coded constants), `loss_single` (GT adjacency from the assignment, `1 - target`, per-layer losses).
- `projects/topologic/models/dense_heads/relationship_head.py`: the relation head (`loss_rel`, the pair scoring) - not read in this pass.
- `projects/topologic/core/lane/{assigners,coders,match_costs}` and the bbox coder `LanePseudoCoder` (unit of the points returned to `get_lanes`: normalised or metres, see below); `openlanev2/` (the evaluation devkit copy, TOP_ll).
- `projects/configs/topologic_r50_8x1_24e_olv2_subset_A.py` (loss weights, `edge_weight`, `pc_range`) and the subset-B config.

**Differences paper versus code (digest):**

| item | paper text | released code / config |
|---|---|---|
| distance norm | `\|.\|`, not named, 3-D points | L1 sum over x, y, z, on detached coordinates in metres (decoder) |
| mapping | `exp(-x^alpha / (lambda sigma))`, alpha 0.2, lambda 2, sigma = std(D) | `exp(-d^P / w)`, P init 2, w init 10; no alpha, no lambda; `sigma` computed and unused (in the decoder and in the head) |
| what is used at inference | learned parameters implied | hard-coded P = 2, w = 11.5275, both lambda = 1; learned values only in comments |
| fusion | `G = l1 G_dis + l2 G_sim`, both in 0 - 1 | training: `l1 * sigmoid(rel).detach() + l2 * topo` into the GNN; inference: plain sum, range 0 - 2 |
| diagonal | not mentioned | zeroed |
| topology loss | G_sim only, type and weight not named | mmdet FocalLoss (gamma 2, alpha 0.25), weight 5, on the relation head; all pairs not matched-and-adjacent are negatives |
| matching | not restated | Hungarian assigner (focal cost 1.5, L1 cost 0.025) and the GT adjacency indexed by it |
| GNN | no equation | forward and backward messages with A and A^T, `edge_weight` 0.6, lane-traffic branch concatenated, linear 2C -> C, residual |
| batch | 2 on 8 GPUs | `samples_per_gpu = 1`, auto-scale base batch 8 |
| data version | v1.0.0 and v2.1.0 metrics reported | README tells to install the OpenLane-V2 v1.0.0 data; the config names neither version |

**Open code question that touches our numbers:** in the decoder the distance is on coordinates scaled to **metres**; in `get_lanes` the points are taken from the last layer's `all_lanes_preds` and no denormalisation is visible before the distance, while `bbox_coder.decode` runs after it (digest). Whether the inference-time distance is in metres or in normalised units, and therefore whether the constant 11.5275 means "half value at 2.8 m" at inference, is not settled by the pages I read. The training-time kernel (init w = 10, learned) is in metres with certainty (digest of the scaling lines), and 11.5275 is close to that init, which supports the metre reading. A later code extraction must read the head's `forward` and the coder to close this.

### Corrections to the level-1 note (TP01) and to A17
- TP01 "GNN edge weight 0.5": the subset-A config sets `edge_weight = 0.6`; the 0.5 may be a class default (not checked).
- TP01 says the topology loss weight is not named: the paper does not name it, the config does: focal loss, weight 5, on the relation head.
- TP01 "Hungarian matching (from memory)": now seen in the config (`LaneHungarianAssigner3D`), and the label matrix is built through it; digest level.
- TP01 "mmcv layers (from memory)": the config uses mmcv / mmdet registries throughout; confirmed at digest level.
- TP01 "batch 2 on 8 RTX 3090" is the paper's statement; the released config has 1 sample per GPU (not reconciled; add to the code differences).
- TP01 reports only v2.1.0 numbers for subset A and B; v1.0.0 numbers (A: TOP_ll 18.6, OLS 41.6; B: 15.1, 39.6) are in the same Table 1 and show how strongly the devkit version matters.
- TP01 "half value at 2.8 m": correct as arithmetic on the hard-coded constants, but with two qualifiers: the distance is an **L1 sum over x, y, z** (the Euclidean half-value is 2.0 - 2.8 m for xy offsets, down to 1.6 m for equal offsets in all three axes; computed), and the unit at inference is not confirmed (above).
- A17 "differentiable: yes" and "compat_conflicts: none": the coordinates are detached in the code (no gradient from the topology term into the points) and the labels use Hungarian assignment, so the conflict list is not empty for our rules; the repository now opened (A17 says "not opened") has no licence file.
- A17 "geometric part can be added to a trained model without retraining" is right (Table 5), with the note that the released inference code uses fixed constants rather than re-fitted ones.

## 5. Transfer to our pipeline

| question | answer |
|---|---|
| what it needs from us | Only builder outputs in the ego BEV frame: vertex or tip positions (metres, 2-D), headings, and for the optional semantic term the vertex tokens. No BEV features, no images, no lidar, no SD map, no global pose. Works on the frozen cached 200 x 200 (0.5 m, 39 ch) pipeline because it acts on head outputs. |
| what it gives us | Stage A3, link and merge decisions: a smooth score for "tip i belongs to vertex / edge j" that can replace the hard merge radius (1.5 m) and snap (1.0 m) of the DAG assembly by a calibrated, differentiable weight; a per-link probability for the planner and for the route head; the same kernel can serve as the association weight of the time recursion (the Gaussian of the vertex-level log-odds design in [VM12](VM12_l2g-map.md), section 5.2). |
| conflicts | (1) Hungarian matching is used for the labels in the code: we do not need it, our link labels come from geometry (point-to-polyline distance to GT lanes). (2) The code detaches the coordinates: a literal port would block the gradient into the positions, which our rule "gradient reaches every module" does not allow; whether the link loss should move positions is a design choice (stop-gradient on one side is the safe default). (3) The summed score is in 0 - 2, not a probability; our validities must be calibrated. (4) Endpoint pairs only: it assumes lanes are cut at junctions. (5) No licence: no code is copied; the formula is in the paper. (6) mmcv: nothing is needed, the core is about 6 lines of PyTorch, the GNN two matrix products, the relation head an MLP dot product. |
| reuse mode | Re-implement from the description (idea level); nothing to port. |
| effort | S (hours): kernel with learnable bandwidth fitted on cached builder outputs of TRAIN and tested as an assembly weight. M (days): integration as the soft merge weight inside training, with polyline distance and heading term. Blocker: none technical; the open item is a reliable measure of link quality beyond S (J at 6 m is coarse). |
| expected gain and risk | Quality: a tie inside the noise band is the evidence-based prediction (E42 / E45 and E133, section 7). Value that remains: one hand-set rule less on the training path, a calibrated per-link score, a single bandwidth shared by assembly and time association. Risk: a bandwidth that is too wide merges neighbouring lanes (released kernel gives 0.35 at 3.5 m lateral offset, computed), a bandwidth that is too narrow drops true links when tip error is larger than the average vertex error. |
| feeds which experiment / K id | K30 stays the id of the geometric link prior. The polyline-distance, anisotropic and refit variants below are new forms: register a K id (proposal: variant of K30) before an experiment uses them. |

### 5.1 The geometric-distance topology in one paragraph, for us
A pair score `k(d) = c * exp(-d^P / w)` of the distance between the end of one piece and the start of the next, with P and w learned scalars. All the effect is the width: the paper's own Table 3 shows +2.2 TOP_ll for a learned wide kernel over a fixed Gaussian, and the ranking tanh < sigmoid < Gaussian < learned follows the half-value distance (computed, section 3). The semantic term adds +3.8 TOP_ll on top of the geometry (Table 4), which in OpenLane-V2 is where the endpoints are noisy by metres. It is the kernel width, not the architecture, that has to be re-derived for sub-metre vertices.

### 5.2 Refitting the bandwidth at 0.5 m vertex error
**What the released constants say (computed).** Half value at `d_1/2 = sqrt(w ln 2)`: w = 11.5275 gives 2.83; the trained-in-metres init w = 10 gives 2.63. Values of the released kernel `exp(-d^2 / 11.5275)`: d = 0.5 m 0.98, 1 m 0.92, 2 m 0.71, 3 m 0.46, 3.5 m 0.35, 5 m 0.11, 8 m 0.004. Equivalent Gaussian standard deviation `sqrt(w/2)` = 2.4 m. For L1 distance over x, y, z the Euclidean half-value is 2.0 - 2.8 m for planar offsets (computed from the L1 / L2 ratio, the z extent of the road is small).

**Where our numbers differ.** Vertex error 0.43 - 0.54 m (project figure from the compatibility study; definition not repeated there, so the derivation below states its assumption); lane spacing about 3.5 m, so the released kernel gives 0.35 to the neighbouring lane at equal heading; vertices of the champion sit 7.4 m apart (100 / 13.5, computed), so a tip and "the nearest existing vertex of the same lane" differ along the lane by up to 3.7 m even when the lane is perfect: a vertex-to-vertex distance is the wrong feature on a sparse graph, the distance to the polyline is the right one.

**Derivation (my own, with assumptions).** Let the two positions have independent isotropic Gaussian errors with per-axis standard deviation s each; the difference has per-axis variance 2 s^2. If the competing candidates are spread uniformly in the plane, the posterior that a pair at distance d is a true link is proportional to `exp(-d^2 / (4 s^2))`, i.e. a Gaussian kernel with **w = 4 s^2** (the same holds for the 1-D lateral distance to a polyline). If the 0.5 m figure is the mean radial error, s = 0.5 / 1.2533 = 0.40 m (Rayleigh mean), so w = 0.64 m^2 and the half value is 0.67 m; if it is the RMS radial error, s = 0.35 m, w = 0.5 m^2 and half value 0.59 m (computed). Three reference bandwidths result:

| setting | w (m^2) | half value (m) | k(0.5) | k(1.0) | k(1.5) | k(3.5) |
|---|---|---|---|---|---|---|
| error-derived (w = 4 s^2, s = 0.40 m) | 0.64 | 0.67 | 0.68 | 0.21 | 0.03 | 5e-9 |
| matched to the present merge radius 1.5 m | 3.25 | 1.50 | 0.93 | 0.74 | 0.50 | 0.02 |
| released (TopoLogic) | 11.53 | 2.83 | 0.98 | 0.92 | 0.82 | 0.35 |

(all computed.) The error-derived value is a lower bound: uniform negatives are optimistic at junction mouths and next to parallel lanes, and tip errors may exceed the average vertex error (the end of a ray is extrapolated; not measured, measure it).

**Fit procedure (proposal, nothing run).**
1. Build labelled candidate pairs from cached builder outputs on TRAIN: for each tip, the candidate attach points (vertices and nearest points on polylines within 8 m); label = the tip's GT lane continues into / merges with the GT lane of the candidate (decided from GT polylines by point-to-polyline distance, no matching).
2. Fit `p = c * exp(-d^P / w)` (c, w, optionally P) by binary cross-entropy; start with P = 2 as in the released code; use d_lateral and d_along separately if the pooled fit is poor (kernel `exp(-d_lat^2 / w_lat - d_lon^2 / w_lon)`).
3. Add the heading term of E133 (`exp(kappa (cos dtheta - 1))`): TopoLogic gets direction for free from the pairing end(i) -> start(j) (the opposite lane has its start at the other end), a vertex-to-vertex distance does not.
4. Report the fitted half value next to the three reference rows, the reliability diagram on VAL, and S, Q, J, vertices per 100 m for the builder run with each setting.

### 5.3 Endpoint-based versus vertex-based use
- **TopoLogic is endpoint-based on pieces:** only `end(i)` against `start(j)` of whole lane instances, because OpenLane-V2 centre-lines are cut at junctions and the label is defined by endpoint coincidence. The cost is O(N^2) on 200 instances.
- **Our graph has shared vertices by construction**, so an endpoint pair exists only where a decision is made: a growing tip against existing vertices (continuation, merge), a branch start against an interior point of an existing edge (fork), and the clustering of paths into the DAG (merge radius 1.5 m, snap 1.0 m). Because a branch may start at the interior of an edge, the pair must be allowed to be (endpoint, polyline): `d(end_i, polyline_j)` with the projection parameter u giving the location of the new vertex; this is also the shape of the fork relocation.
- **Vertex-based use as a dense kernel:** a (tips x vertices) matrix of `k(d, heading)` used as an attention bias or soft merge weights; with a few dozen tips and a few hundred vertices the cost is negligible next to the 31 ms chain. It needs the heading term and the polyline distance (section 5.2); a plain vertex-to-vertex Euclidean kernel inherits the sparse-graph stagger and the neighbouring-lane leakage.

### 5.4 What TopoLogic could still add after E42 - E56 and E133
Honest summary: for the link decision itself, probably nothing in S or Q. Our labels are geometric as in OpenLane-V2, so geometry decides (the literature predicts this, section 7), the learned heads were within 0.01 of the rules, and the TopoLogic-form prior without any learning landed in the same band (E133). What is left:
1. **Differentiable assembly.** Replace the hard radii (merge 1.5 m, snap 1.0 m: rule steps) by the calibrated kernel and give the bandwidth a gradient. This is about the standing rule, not about S; the prediction is a tie.
2. **The unrefit bandwidth was the tested one.** E133 used the published width (`exp(-d^2 / 11.5)` times a heading term), 1.9 times wider (half value) than the present merge radius and about 4 times wider than the error-derived value (computed). A refit is therefore untested; the observed J drop (0.119 against 0.149 and 0.160 for the control seeds) and the 1.7 extra vertices per 100 m are consistent with a too-wide kernel but a single seed cannot say so.
3. **Calibration.** The paper's sum is a score in 0 - 2; ours would be a probability per link that the planner and the time recursion can use, and that can be checked with a reliability diagram (open item 4 of the overview).
4. **A shared kernel.** The same fitted `exp(-d^2 / w)` can be the association weight for the vertex-level evidence memory ([VM12](VM12_l2g-map.md) section 5.2) and for the merge weights, with one bandwidth to calibrate instead of two.
5. **The semantic half** (+3.8 TOP_ll where endpoints are 3 m noisy) only helps where geometry is ambiguous and the tokens carry information the geometry lacks (junction mouths, stop lines). The learned pair heads of E42 / E46 are the experiment of this kind and tied the rules; I would not repeat it before the refit kernel is in.

### 5.5 Proposed experiment (not run; for the loop owner's card)
Arms on the loop-16 builder, same trained weights where possible: (0) hard radii, control; (1) kernel at the published bandwidth (= E133 arm); (2) kernel with w fitted on TRAIN by BCE (section 5.2); (3) (2) plus heading term fitted; (4) (3) with separate lateral and longitudinal widths and polyline distance. Report S and Q, F1 at 0.5 / 1 / 1.5 m, J, vertices per 100 m, ms per frame, reliability of the link probability; 2 seeds, paired scene bootstrap, noise band 0.010 in S. Prediction to write down: |dS| < 0.010 for all arms, J within +- 0.03, no cost in ms; a gain only in calibration. Kill criterion: arm 2 below arm 0 by more than the band on S or Q.

## 6. Not found / open
- Unit of the inference-time distance (normalised or metres) and therefore the meaning of the hard-coded w = 11.5275 at inference (section 4); how the post-hoc experiment of Table 5 parameterised the kernel.
- Learned final values of P, w, lambda1, lambda2 in the released checkpoint (the inference constants suggest P = 2, w = 11.5275, lambda about 1, but this is an inference from the commented lines).
- Widths and activations of the two 3-layer MLPs, the GNN equation and layer count (code only), topology loss normalisation (`avg_factor` for the lane-lane term), augmentation, training time, parameters, FPS.
- How the SD map enters the "TopoLogic + SDMap" rows (TOP_ll 23.9 -> 28.9; the method sections describe no SD branch): relevant to our route head (A6), not found on the HTML page.
- Number of positive entries in the adjacency labels and the class imbalance (not discussed); no seeds or intervals for any result.
- README digest lists OLS 44.1 beside the subset-B v2.1.0 table where the paper has 42.3 (page labelling ambiguous; unresolved).
- `relationship_head.py`, the bbox coder, the subset-B config and the OpenLane-V2 TOP_ll implementation were not read.
- The PDF and the NeurIPS camera-ready / supplement were not read; the HTML v1 has no appendix. If the camera-ready differs from arXiv v1 it was not seen.

## 7. Our experiments that touch it
- **E42 / E45** (K30): learned link head and TopoLogic-style link scoring for splits and merges: within 0.01 of the rule links (verdict D in the experiment table; dropped according to the level-1 note). **E46 / E47** (LaneGAP path view and repair; repair +0.001) and **E51 / E56** (node-conditioned dense head +0.003 F1; direction-consistency acceptance for connectors, connectors lose 0.026 split F1) point the same way.
- The lever that did work was endpoint supervision, not link scoring: **E44 / E53** split / merge F1 at 6 m 0.34 / 0.38 -> 0.55 / 0.61 (key points) -> 0.71 / 0.64 (Bezier connectors), at a route F cost of 0.025.
- **E133** (loop 16, run E123.B1.r0.00, baseline arm B1; K30): D7 with the geometric link prior in TopoLogic form (`exp(-d^2 / 11.5)` times a heading term, nothing learned) instead of the learned link head: S 0.697, F1@1 0.820, F1@0.5 0.657, J 0.119, 14.1 vertices per 100 m, false length 0.026, VAL subset of 250 frames, 1 u. The control D7 (E130, seed 0 / E134, seed 1): S 0.701 / 0.701, J 0.149 / 0.160, 12.4 / 12.3 vertices per 100 m. S is inside the noise band 0.010; J and vertex count are worse for the prior, single seed, no bootstrap run. The bandwidth was the published one, not refit.
- Planning documents that cite it: `experiment_execution/plans/Literature_Compatibility_Space_EN.md` (adapter row 7 "bandwidth re-fitted", technique row 30, recommendation "do not stack TP01, TP13, TP14" because geometry priors replace each other, CoPo with the fixed mapping -0.5 and UniTopo +0.5 in the literature).
- Not tried: bandwidth refit to our vertex error, polyline / anisotropic distance, the kernel as a differentiable assembly weight with a gradient into the bandwidth, calibration of the link probability.
