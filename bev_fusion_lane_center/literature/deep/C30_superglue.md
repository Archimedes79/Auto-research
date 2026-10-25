# C30 - SuperGlue: Sinkhorn optimal transport with a dustbin: full-text extraction

| field | value |
|---|---|
| paper | SuperGlue: Learning Feature Matching with Graph Neural Networks. P.-E. Sarlin, D. DeTone, T. Malisiewicz, A. Rabinovich. CVPR 2020 (oral). arXiv 1911.11763 (v1 2019-11-26, v2 2020-03-28). Links actually opened: https://arxiv.org/abs/1911.11763, https://arxiv.org/html/1911.11763v2, https://ar5iv.labs.arxiv.org/html/1911.11763 |
| axis, tier, K ids | A6 (assignment to a map graph), tier 1 (core). No K id of its own yet. Neighbours in [../register.md](../register.md): K145 (PolyWorld: Sinkhorn in training, exact assignment at inference), K75 (MAT, hard road assignment), K17 (our route head), K74 (map as noise), K99 (OrienterNet). A new K id is proposed in section 5 (register first). |
| level-1 note(s) | [C30](../papers/C30_superglue-sinkhorn-optimal-transport-with.md) (only note of this paper). Related notes: [C31 LightGlue](../papers/C31_lightglue-matchability-scores-instead-of.md), [C32 weighted Procrustes](../papers/C32_differentiable-weighted-procrustes.md), [C01](../papers/C01_online-navigation-refinement-mat.md) / [PL01](../papers/PL01_online-navigation-refinement-sd-roads.md) (MAT), [PL46 OrienterNet](../papers/PL46_orienternet.md) |
| code extraction | [../code/supergluepretrainednetwork.md](../code/supergluepretrainednetwork.md). The repository named in the queue (magicleap/SuperGluePretrainedNetwork) exists and is the right one (inference code and weights only; training code is not released). LICENCE: noncommercial research only, READ ONLY for us. |
| sources read | arXiv abstract page (version history); arXiv HTML v2: Sections 3, 3.1, 3.2, 3.3, 4, 5, Tables 1-4, Appendix C and E (summarised); ar5iv: Sections 3.4, 5.1, 5.4, 6, Table 1 and 4 captions, layer-count statements. Repository pages and raw files (see the code extraction). Local project notes (checked). All 2026-10-24. |
| reading marks | EVERY web page of this file was read through the summarising fetch tool: all paper facts are **digest** (numbers may be mis-transcribed; the summariser also flagged the Table 1 header as garbled in one fetch and gave a consistent column mapping in a second fetch). Facts about the project come from local files (**checked**). Statements tagged [derived] are my own algebra from the equations, [judgement] are my design opinions, [memory] were not seen on any page. Not read: Tables 5-7 (HPatches, YFCC100M, Aachen), Appendix A/B/D details, Figures. |
| extracted by, date | Claude (sub-agent of the literature session), 2026-10-24 |

## 1. Problem and claim
**Mark: digest.** Input: two sets of local features, each keypoint with position, detection confidence and a descriptor (M points in image A, N in image B). Output: a partial soft assignment matrix P in [0,1]^(M x N) in which every point has at most one partner (E1), unmatched points get an explicit mass in a "dustbin", and a hard match list is read from it at inference. The paper's gaps: hand-made matching rules (nearest neighbour, mutual check, ratio test) and separate outlier filters know nothing about context; SuperGlue learns matching and rejection jointly. The one claim that matters for us: **a single learnable bin score plus Sinkhorn gives a differentiable assignment with an explicit "no match" output, trained by a plain negative log-likelihood on ground-truth matched and unmatched points** (E10), with no Hungarian algorithm on the training path. The paper itself calls Sinkhorn "a differentiable version of the Hungarian algorithm". The attention network in front of it is what the authors credit for most of the accuracy (Table 4), but the output layer (E7-E10) is the part that transfers to a map assignment.

## 2. Method as built
**Mark: digest** (paper equations and hyper-parameters); the arithmetic marked [derived] is mine.

### 2.1 Pipeline

```
 image A -> detector+descriptor (SuperPoint, frozen) -> p_i = (x, y, c), d_i in R^D (D = 256)   i = 1..M
 image B -> same                                     -> p_j, d_j                                   j = 1..N
        |
        x0_i = d_i + MLP_enc(p_i)                                         (E2)  keypoint encoder
        |
        L alternating message-passing layers on ONE complete graph with two edge types
          odd layers: self edges (inside an image), even layers: cross edges (between images)
          x_{l+1}_i = x_l_i + MLP([x_l_i || m_{E->i}])                    (E3)  residual update
          m_{E->i} = sum_j alpha_ij v_j , alpha = softmax_j(q_i . k_j)    (E4)  4 heads, own weights per layer
        |
        f_i = W x_L_i + b                                                 (E6)  matching descriptors
        |
        S_ij = <f_i^A, f_j^B>                                             (E7)  M x N score matrix
        |
        S_bar = S plus one dustbin row and column, all filled with ONE learnable scalar z   (E8)  (M+1) x (N+1)
        |
        Sinkhorn on exp(S_bar) with marginals a, b, T iterations          (E9)
        |
        P_bar ((M+1) x (N+1)); P = P_bar[1:M, 1:N]  (dustbins dropped)
        |
        inference only: row/column max, mutual check, threshold 0.2 -> match list, -1 = unmatched
```

- **Inputs, outputs.** Keypoint positions are normalised by the image size before the encoder (paper appendix E: largest image dimension; the code centres them and scales by 0.7 times the larger side, see the code extraction). Keypoint counts: training 512 (homography), 400 (indoor, fixed), 1024 (outdoor, fixed), SIFT variant 1024; testing up to 512 / 1024 (indoor) / 2048 (outdoor) SuperPoint points (NMS radius 4 px for homography, 3 px outdoor). Output of the network: the matrix P_bar (log domain in the code); the hard match list is a post-step.
- **Sizes.** D = 256. Keypoint encoder: MLP with 5 layers, output widths 32, 64, 128, 256, D, BatchNorm + ReLU, about 0.1 M parameters (appendix C). Message MLP: 2 layers, widths 2D then D, BatchNorm + ReLU. 4 heads. About 0.66 M parameters per layer, 12 M in total (appendix C). **Layer count:** Section 4 says L = 9 layers of alternating self / cross attention, Appendix C says "18 layers"; the released code builds 9 (self, cross) pairs = 18 single-type layers, and 18 x 0.66 M + 0.1 M = 12 M agrees with that [derived]. Read "L = 9" as nine self/cross pairs. The ablation row "Smaller (3 layers)" is not specified further (pairs or single layers: not found).
- **Runtime.** About 69 ms (15 FPS) per indoor pair on a GTX 1080 (Section 4); appendix C: 14.5 FPS at 512 keypoints and 11.5 FPS at 1024 per image (Figure 11, not seen). LightGlue's paper reports 70.0 ms for SuperGlue (see [C31](../papers/C31_lightglue-matchability-scores-instead-of.md), digest).

### 2.2 Equations (numbering of the paper)
- **E1** partial assignment: P 1_N <= 1_M and P^T 1_M <= 1_N, P in [0,1]^(M x N). Each point has at most one partner; the inequality is what allows "unmatched".
- **E2** x0_i = d_i + MLP_enc(p_i), with p_i = (x, y, c).
- **E3** residual update x_{l+1}_i = x_l_i + MLP([x_l_i || m_{E->i}]); E = E_self for odd l, E_cross for even l, starting at l = 1; image B is updated in the same way.
- **E4** m_{E->i} = sum_{j:(i,j) in E} alpha_ij v_j, alpha_ij = softmax_j(q_i^T k_j). (The text of the equation shows no scale factor; the code scales, see section 4.)
- **E5** q_i = W1 x_i^Q + b1; [k_j; v_j] = [W2; W3] x_j^S + [b2; b3]; Q and S are the query and source image, i.e. (Q,S) in {A,B}^2: self layers use Q = S, cross layers use the other image as source. Projections are per layer, shared over the keypoints of both images; 4 heads.
- **E6** f_i^A = W x_L_i^A + b (same for B).
- **E7** S_ij = <f_i^A, f_j^B>.
- **E8** S_bar[i, N+1] = S_bar[M+1, j] = S_bar[M+1, N+1] = z, with z a single learnable scalar.
- **E9** a = [1_M ; N], b = [1_N ; M]; P_bar 1_{N+1} = a and P_bar^T 1_{M+1} = b. P_bar is the entropy-regularised optimal-transport plan for scores S_bar and marginals a, b; computed by iteratively normalising exp(S_bar) along rows and columns for T = 100 iterations (Section 4), which the authors liken to alternating row and column softmax. Afterwards the dustbin row and column are removed: P = P_bar[1:M, 1:N].
- **E10** Loss = - sum_{(i,j) in Mset} log P_bar[i,j] - sum_{i in I} log P_bar[i,N+1] - sum_{j in J} log P_bar[M+1,j], where Mset are the ground-truth matches and I, J the ground-truth unmatched points of A and B.

**[derived] What the marginals mean.** Real row i sums to 1 over N+1 columns, so P_bar[i,N+1] = 1 - sum_j P[i,j] is the probability that point i is unmatched. The dustbin row has total mass N and the dustbin column has total mass M so that, in the worst case, every point of the other image can fall into the bin. The dustbin-to-dustbin entry equals the total matched mass K = sum_ij P_ij (dustbin row: N = (N - K) + corner). All dustbin entries share one score z, so a point is "unmatched" when its best real score is clearly below z; the abstain decision is relative to this one global number, not conditioned on the point or on the frame.

**[derived] Log-domain Sinkhorn** (as in the released code, the paper does not mention the log domain): with potentials u (rows) and v (columns), initialised to zero, repeat T times: u = log a - logsumexp_j(S_bar + v); v = log b - logsumexp_i(S_bar + u); at the end log P_bar = S_bar + u + v. In this form the regularisation weight is fixed to 1: the sharpness of P_bar is set by the scale of the descriptors f (and, in the code, by a 1/sqrt(D) factor on the scores). The released code normalises the marginals by M+N (probabilities then sum to 1) and multiplies the plan back by M+N at the end. Because the loop updates the rows first and the columns last, the column marginals are exact after the last iteration and the row marginals only approximately (the residual shrinks with T); for a row-wise categorical output with few iterations one would end on the row update.

### 2.3 What "differentiable" means here
- **Forward.** P_bar is a smooth function of S_bar (only exp, sums, divisions or logsumexp). The loss E10 reads log P_bar at index pairs given by the ground truth, never at indices chosen by the model; there is **no argmax, no Hungarian algorithm and no matching step on the training path**.
- **Backward.** Gradients flow through the unrolled iterations into the bin score z, the matching descriptors f, the position encoder (so into the keypoint coordinates as inputs) and all message-passing layers. The training code is not released; the released inference code consists of plain tensor operations without a custom backward, so ordinary autograd through the T iterations is the natural reading [judgement; implicit differentiation is not mentioned on any page I saw]. With T = 100 the unrolled graph is deep but cheap for small matrices.
- **End-to-end option.** Section 5.4: back-propagating through the descriptor network of the detector raises pose AUC@20 from 51.84 to 53.38 (indoor). The detector's keypoint selection (NMS, thresholding) stays non-differentiable and the detector is frozen in the main experiments.
- **Non-differentiable at inference only:** the row / column maximum, the mutual-nearest-neighbour check, the confidence threshold (0.2) and the "-1 = unmatched" marking. The paper itself says the optimal-transport formulation enforces reciprocity softly, similar to the mutual check, and describes no explicit mutual check; the released code does apply one (section 4). The paper also notes that matches can be used without thresholding, passing their confidences to a later weighted estimator (relevant for us: we keep P_bar soft).
- **Non-differentiable but outside the model:** the ground-truth construction (below) and the keypoint detection.

### 2.4 Labels (ground truth)
- Matches: keypoint pairs whose reprojection error is the minimum along both the row and the column (mutual minimum) and below 3 px (homography), 5 px (indoor), 3 px (outdoor). Indoor and outdoor use depth maps and camera poses for the reprojection; the homography case uses the known synthetic homography.
- Unmatched: keypoints whose minimum reprojection error exceeds 15 px (indoor) or 5 px (outdoor) (threshold for the homography case: not found); keypoints in between are **ambiguous and ignored in the loss** (they appear in neither sum of E10).
- No weights between the three sums of E10; the normalisation (per pair, per keypoint) is not stated (not found).

### 2.5 Training recipe
- Optimiser Adam, constant learning rate 1e-4 for the first 200 k (homography) / 100 k (indoor) / 50 k (outdoor) iterations, then exponential decay with factors 0.999998 / 0.999992 / 0.999992 until iteration 900 k. Batch 32 (homography), 64 (indoor), 16 (outdoor) image pairs with SuperPoint features; 24 with SIFT.
- Data: homography model on synthetic homographies and photometric distortions of Oxford and Paris distractor images, generated on the fly; indoor model on ScanNet (about 230 M pairs selected by overlap 0.4-0.8); outdoor model on MegaDepth (overlap 0.1-0.7), initialised from the homography weights. SuperPoint descriptor network frozen unless stated.
- Hardware and total time: not found. Augmentation beyond the synthetic homographies: not found.

### 2.6 Inference
Test-time confidence threshold 0.2 (paper and code), T = 100 iterations (paper; the code's scripts default to 20, see section 4), keypoint caps as in 2.1. In the released code the match list is: row-wise and column-wise maximum of the real block of P_bar, kept only if both directions agree (mutual) and the probability exceeds the threshold; everything else is -1.

## 3. Numbers
**Mark: digest** (transcribed by the summariser; the README of the repository reproduces the ScanNet and YFCC rows within 0.05, which supports the transcription).

**Table 1, homography estimation (synthetic homographies, SuperPoint features).** AUC of the homography with a RANSAC estimator and with plain DLT, match precision P, recall R:

| matcher | AUC RANSAC | AUC DLT | P | R |
|---|---|---|---|---|
| NN | 39.47 | 0.00 | 21.7 | 65.4 |
| NN + mutual | 42.45 | 0.24 | 43.8 | 56.5 |
| NN + PointCN | 43.02 | 45.40 | 76.2 | 64.2 |
| NN + OANet | 44.55 | 52.29 | 82.8 | 64.7 |
| SuperGlue | 53.67 | 65.85 | 90.7 | 98.3 |

Reading: SuperGlue recovers almost all possible matches (R 98.3) while rejecting most outliers (P 90.7); with it, plain DLT beats RANSAC. The outlier filters PointCN and OANet raise precision but not recall.

**Table 2, indoor pose estimation, ScanNet** (pose AUC @5 / @10 / @20 degrees, match precision P, matching score MS), selected rows:

| features | matcher | @5 | @10 | @20 | P | MS |
|---|---|---|---|---|---|---|
| SIFT | NN + ratio test | 05.83 | 13.06 | 22.47 | 40.3 | 01.0 |
| SIFT | SuperGlue | 06.71 | 15.70 | 28.67 | 74.2 | 09.8 |
| SuperPoint | NN + mutual | 09.43 | 21.53 | 36.40 | 50.4 | 18.8 |
| SuperPoint | NN + OANet | 11.76 | 26.90 | 43.85 | 74.0 | 25.7 |
| SuperPoint | SuperGlue | 16.16 | 33.81 | 51.84 | 84.4 | 31.5 |

**Table 3, outdoor pose estimation, PhotoTourism**, selected rows (same columns): SuperPoint + NN + mutual 09.80 / 18.99 / 30.88, P 22.5, MS 04.9; SuperPoint + OANet 21.03 / 34.08 / 46.88, P 52.4, MS 08.4; SuperPoint + SuperGlue 34.18 / 50.32 / 64.16, P 84.9, MS 11.1; SIFT + SuperGlue 23.68 / 36.44 / 49.44, P 74.1, MS 07.2.

**Table 4, ablation (indoor, SuperPoint)** (AUC@20, precision, matching score):

| variant | AUC@20 | P | MS |
|---|---|---|---|
| NN + mutual (baseline) | 36.40 | 50.4 | 18.8 |
| no graph network (encoder + linear + optimal matching layer) | 38.56 | 66.0 | 17.2 |
| no cross-attention (all layers self-attention, same parameter count) | 42.57 | 74.0 | 25.3 |
| no positional encoding | 47.12 | 75.8 | 26.6 |
| smaller (3 layers) | 46.93 | 79.9 | 30.0 |
| full | 51.84 | 84.4 | 31.5 |

What decides transfer: (1) The optimal-matching layer alone (row 2) lifts precision by 15.6 points over NN + mutual (66.0 against 50.4) but lowers the matching score (17.2 against 18.8): the dustbin buys precision by rejecting. Most of the recall comes from the graph network. (2) Cross-attention and the position encoder are both needed. (3) End-to-end training into the descriptor network: 51.84 -> 53.38 AUC@20.

**Compute.** 12 M parameters, about 69 ms per pair (GTX 1080), 14.5 FPS at 512 and 11.5 FPS at 1024 keypoints. No CPU numbers, no memory numbers (not found).

**Caveats.** Image matching and relative pose, not map assignment; SuperPoint / SIFT inputs; pose AUC thresholds in degrees; numbers are not comparable with ours or with other papers. The paper shows no experiment with one-to-many structure and no ablation of T or of the value of z (not found).

## 4. Paper versus code
**Mark: digest** (repository pages; details in the code extraction). Released: inference code, indoor and outdoor SuperPoint+SuperGlue weights; the authors state that training code, SIFT and homography models will not be released.
- Layers: code `['self','cross'] * 9` = 18 single-type layers; paper Section 4 says L = 9, Appendix C says 18.
- Sinkhorn iterations: class default 100 (as in the paper); the command-line scripts (`match_pairs.py`, `demo_superglue.py`) default to 20.
- Code uses the log domain, normalises the marginals by M+N and rescales; the paper text shows neither.
- Scores are divided by sqrt(256) = 16 in the code (E7 shows no scale); attention logits are scaled by the per-head width (E4 shows none).
- `bin_score` is a scalar parameter initialised to 1.0 (the paper gives no initial value).
- The code applies an explicit mutual check plus the 0.2 threshold; the paper text describes the soft reciprocity of the transport only.
- Matching scores of mutual pairs below the threshold are not zeroed in the output (the index is -1 but the score stays).
- No padding masks and the marginals use the tensor sizes: padded keypoints would take part in attention and in the transport. A batch needs equal keypoint counts.
- Position normalisation: centre and scale by 0.7 x larger image side (code) against "largest image dimension" (paper appendix E).

## 5. Transfer to our pipeline
**Mark: [judgement] on top of digest facts; nothing here was run.**

| question | answer |
|---|---|
| what it needs from us | Only tokens: vertex tokens (128-d, from the builder / time recursion, with validity) and SD-segment tokens (pooled from the SD raster tokens of the route head, or from a segment encoder). No image features, no lidar, no frozen-BEV change. Labels: for each predicted vertex the true SD road id (or "none") and for each SD segment whether a perceived lane exists in range; both can be generated from the nuScenes map expansion and the simulated ADASIS-like SD map (the level-1 note [C30](../papers/C30_superglue-sinkhorn-optimal-transport-with.md) already states that the unmatched labels come from the map expansion). Map-error augmentation (shift, rotation, dropped roads, ghost roads, a whole wrong map) to create the unmatched cases. |
| what it gives us | Stage A6 (and a second use in A5): (a) an explicit "no SD road / do not trust the map" probability per vertex; (b) a soft per-vertex distribution over SD segments, from which a soft route probability follows; (c) a per-segment "not seen" mass; (d) soft correspondences with outlier mass for a closed-form registration step ([C32](../papers/C32_differentiable-weighted-procrustes.md)). |
| conflicts | (1) One-to-one marginals do not fit many perceived lanes per SD road (section 5.3). (2) The 9-pair GNN (12 M parameters) is far larger than our route head (0.69 M); we would take only E7-E10, not the attention stack (our route head already cross-attends vertices to SD tokens). (3) T = 100 sequential iterations: use few iterations and fixed shapes (CUDA-graph friendly, K186). (4) A single global bin score cannot express frame-level map trust. (5) The repository is under a noncommercial-research licence whose terms give derivative ownership to the licensor: read only, re-implement from the equations. (6) Hard mutual check and threshold are inference rules, not wanted on our path. No Hungarian matching, no argmax on the training path, no mm-stack, no perspective image, no lidar. |
| reuse mode | Re-implement from the description (E7-E10 are about 30 lines of PyTorch; nothing is copied). Idea level for the GNN. |
| effort | S for the row-softmax-with-bin head (variant V1) and for temporal association; M for the capacitated variant V2 with labels, augmentation and calibration; L for the registration chain (Procrustes + re-run). Blocker: labels for "road absent from the SD map" and "wrong map" exist only as synthetic corruptions. |
| expected gain and risk | Targets open item 2 of [../OVERVIEW.md](../OVERVIEW.md) (route attribute with road-level fallback and abstention) and the "map trust / abstain output (dustbin column)" item of PROGRAM.md. Gain not measured. Risks: calibration of the abstain mass (open item 4), distribution shift from synthetic to real map errors, label ambiguity at junction mouths and under registration error, and the possibility that a plain 3-way softmax head (on-route / off-route / no-map) does as well; that head is the control. |
| feeds which experiment / K id | New K id to register before use (next free after K187; name proposal: "dustbin assignment of vertices to SD segments: row softmax with learnable bin, capacitated log-Sinkhorn variant"). Related: K145, K75, K17, K74, K99, K185. No E id assigned. |

### 5.1 Assignment between perceived vertices and SD segments: construction
[judgement] Rows i = 1..M are the perceived vertices (or, with K185, the sample points every <= 4 m along edges), tokens t_i from the existing route head after its cross-attention to the SD tokens; columns j = 1..N are SD segments (a road cut into pieces of 20-50 m, or the raster cells on SD roads), tokens g_j with a route flag r_j. Score: S_ij = <W_f t_i, W_g g_j> / sqrt(d), optionally plus a geometric term such as -dist(vertex i, polyline j)^2 / (2 sigma^2) under the current registration estimate (the same idea as the endpoint-distance prior K30). Augment with a bin column z_i and a bin row z'_j and a corner entry. Two differences from the paper that cost nothing: the bin scores can be **per item** (z_i = linear(t_i), as LightGlue's matchability does, see [C31](../papers/C31_lightglue-matchability-scores-instead-of.md)) instead of one scalar, which lets the abstain decision depend on context; and the row masses can be the soft validities a_i = v_i of the builder instead of 1, which handles our padded vertex sets without masks (padded rows get mass 0).

```
 vertex tokens t_i (validity v_i)          SD segment tokens g_j (route flag r_j)
          \                                   /
        route head cross-attention (exists, E104b) -> contextualised tokens
          S_ij = <W_f t_i, W_g g_j>/sqrt(d) (+ geometry)
          S_bar = [[S, z_i], [z'_j, z0]]
          V1: row softmax over [segments, bin]     V2: capacitated log-Sinkhorn, rows last
          p_on_i = sum_j P_ij r_j   p_off_i = sum_j P_ij (1 - r_j)   p_abstain_i = P_i,bin     (sum = 1)
```

### 5.2 The dustbin as "no match / abstain"
[judgement] Two different bins carry two different meanings, both with a label source:
- **Bin column (vertex unmatched):** the map does not explain this vertex (road missing in the SD map, a private road, wrong or shifted map, registration failure). Train with the third sum of E10 on vertices whose true SD road is absent or whose map was replaced. Its mass is the per-vertex map-trust signal; a frame score can be the validity-weighted mean of it, or a separate head on pooled tokens. Calibration can be checked on the unmatched / matched split; because E10 is a proper scoring rule the in-distribution calibration is plausible, but nothing in the paper tests it.
- **Bin row (SD segment unmatched):** no perceived lane found for this segment (out of range, occluded, sensor failure, missed lane). This reports the opposite doubt: perception incomplete rather than map wrong. In the segment-capacity variant V2 it is part of the transport; in V1 it needs its own coverage head (e.g. noisy-or of the P_ij over vertices, BCE against "a GT lane exists within range").
- A false perceived vertex (no GT lane) should probably be **ignored** in the assignment loss, like ambiguous keypoints in the paper, so that the bin does not mix "map wrong" with "perception wrong". Decision open.
- At inference the abstain flag is a threshold on a calibrated mass (post-step after the soft path, like the existing route threshold); the planner can also take (p_on, p_abstain) directly.

### 5.3 What breaks with one-to-many structure (many perceived lanes per SD road) and the fixes
[derived + judgement] E9 gives every real column exactly mass 1. A road segment with k perceived vertices (several lanes, several vertices per lane along a 30 m segment; at 13.5 vertices per 100 m of lane and 3 lanes this is about 12, my estimate) can therefore pass at most mass 1 to its k vertices in total.
1. **Infeasible ground truth.** E10 asks P_bar[i,j] -> 1 for all k vertices; the best attainable per-vertex value is about 1/k, the rest is forced into the bin column. The abstain mass is then biased upward by roughly 1 - 1/k, growing with lane count and segment length, unrelated to map trust, and p_on = sum_j P_ij r_j is shrunk by the same factor. The gradient keeps fighting an unsatisfiable constraint. Worst case among the failure modes.
2. **Bin capacities are wrong.** The bin column is sized M, the bin row N, assuming one-to-one.
3. **Junction mouths.** A vertex near a split can belong to two SD segments; row mass 1 splits (acceptable as probability), but its "unmatched" label is ill-defined: ignore these as ambiguous.
4. **Registration error.** Under a 0-3 m pose error the nearest segment in the noisy frame is a wrong label for close parallel roads: labels must use the true road id from the simulation, not nearest-in-the-shifted-frame.
5. **Padding / varying M.** The reference code has no masks (see section 4); use validity as row mass or masks.
6. **One global z.** Cannot express frame-level distrust except through the descriptors.

Fixes, from cheap to more faithful:
- **V1, row softmax with a bin column (no column constraint).** This is exactly the first half-step of Sinkhorn (u update with v = 0) [derived]. Many-to-one is allowed, the bin keeps its meaning. Loses the competition between vertices for a segment; lane-level exclusivity then has to come from the graph itself. This is the "dustbin + softmax" of the level-1 note.
- **V2, capacitated transport.** Replace the column marginals by capacities c_j (expected vertex count of segment j: lanes x length / vertex spacing, from the SD lane count if the SD record has one, else predicted by a head on g_j, with slack so the constraint acts as an upper bound) and size the bin masses as in the paper: a = [v_i ; sum_j c_j], b = [c_j ; sum_i v_i] (both total sum_i v_i + sum_j c_j). The capacity stops a single mapped road from absorbing every parallel unmapped road next to it. Risk: a wrong c_j forces true vertices into the bin.
- **Iteration count as a knob.** After 0.5 iteration the plan is V1; every further iteration adds column competition; T -> infinity with unit capacity is the paper's setting and the worst case here. Ablate T in {row only, 1, 3, 10, 100}.
- **Coarser rows.** Pool the vertices of a lane (an edge chain) to one lane token: the multiplicity per segment drops to the lane count, which is small and known from the SD record; a per-lane route attribute follows. Lane-level exclusivity needs lane slots (segment x lateral order) as columns, only possible if the SD map carries a lane count (not checked in our simulated map).
- **Lane-level attribute is not an assignment output alone.** The assignment gives road-level evidence at vertices beyond a junction mouth; the lane-level attribute for the vertices before the fork must travel backwards along the perceived edges (a lane is on route if a successor is, e.g. noisy-or over children), which is the path-aware part of MAT ([PL01](../papers/PL01_online-navigation-refinement-sd-roads.md)) and of our edge-wise asking (K185).

**Where the one-to-one marginals do fit [judgement]:** (a) **temporal association** of vertices from frame t-1 (ego-warped) to frame t: a vertex has at most one successor, births and deaths go to the bin: SuperGlue as designed, relevant to A5 (fusion of the detections themselves); (b) **junction nodes**: perceived forks / merges against SD nodes (few, near one-to-one), a good correspondence set for registration; (c) lane slots if the SD lane count exists.

### 5.4 Registration (no pose given)
[judgement] The bin is what makes correspondences usable for pose: outliers go to the bin, and the remaining soft correspondences can weight a closed-form 2-D Procrustes / point-to-polyline fit ([C32](../papers/C32_differentiable-weighted-procrustes.md); SE(2) closed form to be derived by us). Alternating "assign, estimate (dx, dy, dtheta), re-run the score" with few iterations is the structure of classical ICP with a learned, outlier-aware assignment. OrienterNet ([PL46](../papers/PL46_orienternet.md), K99) solves the same step with a pose volume instead; the two can be compared at map errors of 1-3 m. SuperGlue's own position encoder is not equivariant to the unknown offset, so the score must be trained with pose noise (the encoder sees positions in the frame it is given).

### 5.5 Cost [judgement, not measured]
With the vertex counts of our graphs (order of 100 rows) and tens of SD segments, S has a few thousand entries; 3-20 log-domain iterations are about 2 x T tiny logsumexp kernels, well under a millisecond with a captured CUDA graph (K186). It has to be measured on an idle GPU at batch 1 like every module (project rule), with fixed shapes padded by validity. If the columns are raster cells (N in the thousands) the matrix grows to the order of 10^5-10^6 entries; still cheap for T <= 20, but then V1 is enough.

### 5.6 Proposed experiment shape (not run, no E id)
Arms: control A = current route head (E104b / K185 along edges); control B = same head with a third output "no map" (3-way softmax, strong simple baseline); V1 (row softmax + bin over segments); V2 (capacitated, T in {1, 3, 10}); each with per-item vs scalar bin; at least 2 route-head seeds (spread up to 0.03 on clean frames, 0.05-0.09 under failure according to the project rules), noise band 0.010, paired scene bootstrap. Evaluate on clean maps (route success must not drop), on shifted maps 0-3 m, with 10-30 % roads removed, with ghost roads, and with a whole wrong map (abstain precision / recall of the bin, reliability diagram of p_abstain). Report ms per frame. The decision rule: adopt only if abstention quality improves clearly with route success on clean frames inside the noise band.

## 6. Not found / open
- Training hardware and total time; loss normalisation (per pair or per point); augmentation beyond synthetic homographies; the homography-case threshold for "unmatched".
- Initial value and learned value of z; any ablation of T (the arXiv text gives 100 only); any analysis of the dustbin mass as a confidence or of its calibration.
- Whether training differentiates through the 100 unrolled iterations or implicitly (training code unreleased).
- Tables 5-7 (HPatches, YFCC100M, Aachen Day-Night), Figures 4, 7, 11, 16 and Appendix D attention analysis were not read.
- Behaviour with one-to-many structure, with thousands of points (memory), with moving sensors: nothing in the paper.
- For our use: whether the simulated SD map carries a lane count per link; how many SD segments and vertices a typical frame has (counts here are estimates).

## 7. Our experiments that touch it
None used SuperGlue or a dustbin (level-1 note C30: tried before = none). Neighbours so far: E104b route head by cross-attention (route success 0.853-0.820, kept) and E103 correlation volume (dropped), both from the notes [C01](../papers/C01_online-navigation-refinement-mat.md) / [PL01](../papers/PL01_online-navigation-refinement-sd-roads.md); K185 route head asked along the edges (0.798 -> 0.888, control 0.885, validation only). Nothing with an explicit abstain output for the map has been run; it is listed as a tree-search item in HANDOVER.md / PROGRAM.md.
