# VM12 - L2G-Map: full-text extraction

| field | value |
|---|---|
| paper | L2G-Map: Local-to-Global Mapping via Hierarchical Diffusion Refinement and Elliptical Bayesian Fusion. Siyu Li, Xinying Hong, Fei Teng, Kang Zeng, Hao Shi, Beiping Hou, Zhiyong Li, Kailun Yang. arXiv 2503.02578 (cs.CV; cross-lists cs.RO, eess.IV); no venue stated. v1 submitted 2025-03-04, v2 (this text) revised 2026-09-08. **Warning: v1 under the same arXiv id is a different paper (TS-CGNet, Hong et al., repository krabs-H/TS-CGNet); only v2 contains L2G-Map and the log-odds fusion.** |
| axis, tier, K ids | A5 (temporal fusion of detections), tier 1 core; K24 (log-odds fusion of evidence over time, register source "from memory"); also touches K06 |
| level-1 note(s) | [VM12](../papers/VM12_l2g-map.md) (its line "Tried in our experiments: none" is out of date, see section 7) |
| code extraction | [l2g-map](../code/l2g-map.md): the repository named in the paper does not exist (HTTP 404); no code of this paper is public |
| sources read | arxiv.org/abs/2503.02578 (v2 abstract page); arxiv.org/html/2503.02578v2 (HTML, seven separate queries: structure, method and equations, setup, all tables, prose of III-C / IV-B, IV-B to IV-F and conclusion, a literal re-read of Eq. 7 - 12); arxiv.org/html/2503.02578v1 (this is TS-CGNet, used only to learn the v1/v2 situation); arxiv.org/pdf/2503.02578 (binary, unreadable by the tool); GitHub pages and API listings of lynn-yu, krabs-H/TS-CGNet; all read 2026-10-24. Local files read for section 5 and 7: `training/models/p16_time.py`, `p16_builder.py` (warp_prior, add_prior, fuse_prior), `p16_ref.py`, `register.md`, `experiment_execution/plans/Experiment_Table_EN.md` |
| reading marks | every statement about the paper is **digest**: all pages came through the summarising fetch tool (no page was seen raw), so numbers may be mis-transcribed; where I could cross-check arithmetic it is said so. Statements about our own code and experiments are **checked** (files read). Classical occupancy-grid theory is **memory**. Section 5 is my analysis, not the paper's content |
| extracted by, date | extraction agent (Claude, sub-agent of the literature session), 2026-10-24 |

## 1. Problem and claim
- Offline construction of a **global BEV semantic map** (divider, pedestrian crossing, boundary) from a camera-only sequence: a per-frame local map is refined, then all refined local maps are accumulated into a global raster. Closed loop: the global map is fed back as a prior into online mappers.
- Two parts: **HPDR** (hierarchical prior diffusion refinement, learned): a conditional diffusion U-Net that refines the coarse local map using the previous frame (map and BEV features, ego-aligned) and an OSM centre-line raster. **ESBF** (elliptical space Bayesian fusion, **hand-specified**): per-cell log-odds accumulation with class priors and a weight that falls with the elliptical distance of a cell from the ego vehicle.
- The claim that matters for us: log-odds accumulation with a per-class base-rate prior and a distance reliability weight beats direct projection and probabilistic-mean fusion on global IoU, without any learning in the fusion. Nothing in the paper is about vertices, graphs, lanes with topology, or pose error.

## 2. Method as built

### Pipeline (digest)
```
multi-view images (real-world test: one 256 x 1240 panorama)
  -> image encoder -> view transformer (MLP or IPM, depends on the baseline) -> F_bev
  -> decoder -> coarse local map Map  (3 class channels: divider, pedestrian crossing, boundary)
  -> HPDR: DDIM diffusion U-Net, latent z_t + F_f (previous map + previous BEV features, aligned by Tr)
           + OSM centre-line raster by cross-attention at every level -> refined local map Map_r (probabilities)
  -> align to the global frame with the vehicle pose, bilinear footprint f(P)
  -> ESBF: L <- L + c * f(P) * (logit p_obs - logit p_prior) per cell and class
  -> Map_g = sigmoid(L_0 + sum_t dL^w_t)            (global raster)
  -> closed loop: Map_g as prior for HDMapNet (deep features fused into the BEV feature)
                  and StreamMapNet (guides instance decoding)
```
Tensor shapes: only the panorama size (256 x 1240) is stated. Local extents are 60 x 30, 120 x 60, 240 x 60 (Table I); global grid size and resolution: **not found**.

### Inputs, outputs
- Input to ESBF: refined local probability maps Map_r (per class), the vehicle pose per frame. Output: a global probability raster per class. Output is a **raster**; the vector results (Table IX) come from StreamMapNet, which consumes the raster as a prior. There is no vector output of L2G-Map itself.
- Pose source: the text only says local maps are aligned "according to the vehicle pose". For nuScenes / Argoverse it does not say whether ground-truth ego poses are used (presumably yes; not stated). For the real-world run, LiDAR and GPS give the pose alignment.

### Equations (numbered as in the paper; digest, checked twice against two separate queries)
- E1 sparse temporal fusion (HPDR): F_m = ReLU(phi_1(Map^{t-1'} + Map^t)), Map^{t-1'} = Tr(Map^{t-1}); Tr = alignment of the previous frame to the current one by the ego-motion matrix, phi_1 a 3 x 3 convolution.
- E2 dense temporal fusion: F_e = phi_2(Concat(Tr(F_bev^{t-1}), F_bev^t)), phi_2 a 1 x 1 convolution + ReLU.
- E3 F_f = phi_3(F_m + F_e). E4 latent injection z'_t = z_t + F_f.
- E5 centre-line encoder: F_o = ReLU(BN(DCV([ReLU(DCV(Map_o))] x 2))), dilated convolutions on the binary OSM centre-line map. E6 F'_m = SpatialTrans(F_o, F_m) + F_m (cross-attention at every hierarchy level).
- **E7** L_t = L_{t-1} + dL_t, with L = log(p / (1 - p)).
- **E8** dL_t = log(p_obs^t / (1 - p_obs^t)) - log(p_prior / (1 - p_prior)). p_obs: probability of the refined local map at that cell and class; p_prior: static class prior, set from empirical statistics of the observation sequence: **0.10 divider, 0.02 pedestrian crossing, 0.09 boundary**.
- **E9** elliptical normalised distance d_n(l_x, l_y) = sqrt((l_x / r_x)^2 + (l_y / r_y)^2); r_x lateral and r_y longitudinal perception range (the text mis-prints r_y as "y_x"). (l_x, l_y): position of the cell relative to the ego vehicle (implied; units not stated). Values of r_x, r_y: **not found**.
- **E10** confidence c = 1 / (1 + d_n^2): Cauchy form, 1.0 at the vehicle, 0.5 on the boundary of the perception ellipse (d_n = 1), heavy tail.
- **E11** dL^w_t(x) = dL_t(x) (x) c(l_x, l_y) (x) f(P); the operator (x) is not defined (element-wise product); f(P) = bilinear interpolation weight that spreads a local pixel over the 4 neighbours of a non-integer global grid position.
- **E12** Map_g = sigma(L_0 + sum_{t=1..T} dL^w_t(x)); sigma and L_0 are not defined in the text (sigmoid; L_0 = log-odds of the prior, presumably).
- E13 focal loss (1/N) sum_i BCE(y_i, y^_i) (1 - p_t_i)^gamma, N = number of classes; p_t and gamma not defined or given.
- E14 Global IoU = TP_all / (TP_all + FP_all + FN_all) with all map elements merged into one foreground class.

Reading of E7 - E12 (memory, classical): this is the standard inverse-sensor-model update of occupancy-grid mapping. For conditionally independent observations the posterior odds equal prior odds times the product of observation odds divided by prior odds; E8 is that likelihood ratio, E11 scales it by a reliability exponent (a tempered likelihood, c and f(P) between 0 and 1), E12 is the sigmoid of the sum.

### Losses, labels, training (digest)
- Only HPDR is trained. Loss: focal (E13). No loss on the fusion. Labels: nuScenes / Argoverse map layers rasterised (details not given).
- Two stages: pre-train 20 epochs on ground-truth maps with 50 % random masking (simulates a coarse prior), then fine-tune 10 epochs on coarse maps + BEV features of the local mapper. Adam, StepLR, one RTX A6000. Learning rate, batch size, StepLR parameters, noise schedule: **not found** (the v1 text, a different paper, mentions lr 1e-4 and batch 8; do not transfer).
- Denoiser: lightweight U-Net of 27 modules (8 up/down-sampling, 15 ResBlock / SpatialTransformer with skips, central 4 plain ResBlocks). Sampling: DDIM, best at 3 steps.
- Matching: none in the fusion. No Hungarian anywhere on the fusion path.

### Inference / hidden rules
- No clamping of L, no decay or forgetting, no epsilon on the probabilities, no cap on T are described (**not found**). Every frame contributes with equal right except for the distance weight; the world is assumed static.
- Frames are accumulated offline: all frames of a sequence (and, by construction, future frames) can contribute to the same cell. This is not a causal online filter.
- Robustness: only HPDR vs P-MapNet local maps under "FrameLoss" (not defined); the abstract number 3.27 % is the medium level (easy 3.10, hard 3.10).

## 3. Numbers (digest; table numbers as printed on the v2 HTML page)

Table I, local maps, per-class IoU and mIoU (* = copied from the cited paper). The long-range part is nuScenes only.

| method | OSM | nuS 60x30 Div / Ped / Bound / mIoU | Argo 60x30 Div / Ped / Bound / mIoU |
|---|---|---|---|
| LSS* | - | 39.5 / 15.5 / 40.7 / 31.90 | 31.4 / 5.5 / 26.2 / 21.03 |
| HDMapNet* | - | 42.1 / 21.1 / 42.8 / 35.33 | 57.3 / 28.1 / 47.3 / 44.23 |
| DiffMap* | - | 42.1 / 23.9 / 42.2 / 36.07 | - |
| P-MapNet | yes | 44.3 / 23.3 / 43.8 / 37.13 | 53.5 / 30.1 / 47.3 / 43.63 |
| GenMapping | yes | - | 59.3 / 37.0 / 48.4 / 48.23 |
| L2G-Map (HPDR) | yes | 45.9 / 25.8 / 45.4 / 39.03 | 60.4 / 37.0 / 49.4 / 48.93 |

| method | nuS 120x60 Div / Ped / Bound / mIoU | nuS 240x60 Div / Ped / Bound / mIoU |
|---|---|---|
| HDMapNet* | 39.20 / 23.00 / 39.10 / 33.77 | 31.90 / 17.00 / 31.40 / 26.77 |
| P-MapNet | 45.50 / 30.90 / 46.20 / 40.87 | 49.00 / 40.90 / 46.60 / 45.50 |
| L2G-Map (HPDR) | 47.30 / 33.20 / 47.30 / 42.60 | 51.60 / 43.90 / 49.60 / 48.37 |

Table II, **global fusion** (nuScenes, per city; cells are mIoU / Global IoU). This is the table that matters for the log-odds question; "Direct Projection" = direct accumulation of instances after confidence filtering, "Probabilistic Mean" = probabilistic additive (mean) fusion (no formula given).

| fusion | local model | singapore-onenorth | singapore-hollandvillage | singapore-queenstown | boston-seaport |
|---|---|---|---|---|---|
| Direct Projection | P-MapNet | 22.77 / 26.60 | 32.88 / 39.82 | 19.41 / 25.90 | 27.34 / 34.64 |
| Direct Projection | HPDR | 23.41 / 26.63 | 33.88 / 39.48 | 20.24 / 25.55 | 27.01 / 34.37 |
| Probabilistic Mean | P-MapNet | 16.60 / 22.48 | 30.18 / 41.27 | 12.99 / 20.06 | 24.00 / 30.91 |
| Probabilistic Mean | HPDR | 17.92 / 23.70 | 32.19 / 42.91 | 14.03 / 21.52 | 25.24 / 32.09 |
| ESBF | P-MapNet | 22.32 / 27.22 | 39.15 / 46.57 | 17.84 / 24.11 | 31.58 / 38.10 |
| ESBF | HPDR | 25.25 / 28.83 | 39.04 / 47.08 | 20.87 / 27.11 | 32.68 / 39.25 |

Four-city means, **computed by me from the cells above** (mIoU / Global IoU): HPDR: ESBF 29.46 / 35.57, Direct 26.14 / 31.51, Prob. Mean 22.35 / 30.05. P-MapNet: ESBF 27.72 / 34.00, Direct 25.60 / 31.74, Prob. Mean 20.94 / 28.68. Consequences: ESBF minus Prob. Mean with HPDR is +7.11 mIoU / +5.52 Global IoU, matching the paper's text (7.11 and 5.51), so the HPDR cells are consistent; ESBF minus Direct Projection is only +3.3 mIoU / +4.1 Global IoU (HPDR) and +2.1 / +2.3 (P-MapNet). The probabilistic mean is **worse than direct projection** in mIoU in all eight columns and in both four-city means (it blurs); in Global IoU it is better only in hollandvillage. ESBF does not win everywhere: with P-MapNet, queenstown is lower than direct projection in both metrics (17.84 / 24.11 against 19.41 / 25.90) and onenorth in mIoU (22.32 against 22.77). (The fetch tool's own summary of the P-MapNet means did not match my recomputation; I trust the cells.) The global scores (about 30 mIoU) are well below the local ones (42.6 mIoU at 120 x 60), so accumulation does not by itself give a better map than one good frame; the paper does not discuss this (different GT and extent, my reading).

Ablations that decide transfer:
- Table III (HPDR): diffusion only 41.33 mIoU, + sparse temporal features 42.30, + dense temporal features 42.60.
- Table IV (ESBF, Global IoU / mIoU; its last row equals the four-city HPDR ESBF mean computed above, so the rows are presumably averages over the cities): Bayesian only 35.06 / 28.02; + bilinear 35.13 / 28.14; + distance confidence 35.57 / 29.46. The bilinear footprint is worth 0.07 / 0.12, the distance weight 0.44 / 1.32.
- Table VII (initial prior): uniform 50 % prior 27.90 / 20.50, heuristic class prior 35.06 / 28.02, i.e. 7.2 Global IoU and 7.5 mIoU. The explanation in the paper (a uniform prior over-estimates sparse classes) is not obviously consistent with E8, where p_prior = 0.5 removes the subtraction and pushes every cell below 0.5 negative; the threshold and L_0 handling are not stated, so treat the gain as an empirical finding with an unclear mechanism.
- Table V (prior learning): add 42.27, concat 41.53, cross-attention 42.60 (the text calls it Table VI). Table VI (sampling): DDIM 3 steps 42.60, DDIM 1 step 42.57, DDPM 1000 steps 42.57.

Closed loop (Table IX): HDMapNet 33.77 mIoU, + ProbMap 48.47, + L2G-Map 52.03 (+18.26); StreamMapNet 63.40 mAP, + PMap 81.45, + L2G-Map 83.44 (table difference 20.04, text says 20.00). Which split the global map was built from and whether it contained the evaluation frames is **not stated**; if the offline map was built from the same scenes that are evaluated, these are prior-map gains, not online gains.

Compute: Table VIII, building the global map for the 6019 nuScenes validation frames: 15 min with P-MapNet local maps (local mIoU 40.87, Global IoU 24.11) against 18 min with HPDR (42.60, 27.11); the two Global IoU values equal the queenstown ESBF cells in Table II, so Table VIII is a single-city figure (my cross-check). Refinement of a 256 x 1240 panorama: about 0.13 s per frame (real-world run, qualitative only). Parameters, FPS, ESBF cost alone: **not found**.

Caveats: no pose-error or noisy-pose experiment of any kind; no clamp/decay ablation; no comparison of ESBF against a learned fusion; metrics differ between local (per-class IoU) and global (mIoU and class-agnostic Global IoU); numbers of different papers are not comparable with ours.

## 4. Paper versus code
No code. The announced repository `github.com/lynn-yu/L2G-Map` returns HTTP 404 on the page, on the GitHub API and for a raw README; it is not among the six public repositories of the account (Bi-Mapper, DTCLMapper, GenMapping, HierDAMap, HGeo-Topomap, NRSeg). See [../code/l2g-map.md](../code/l2g-map.md).

## 5. Transfer to our pipeline

| question | answer |
|---|---|
| what it needs from us | Not a global pose: the vertex-level version needs only the **relative ego motion between consecutive frames**, which our time recursion already uses (`warp_prior`, tensors Rp, Rc, tp, tc). No SD map, no image features, no lidar. Per frame: vertex positions, travel headings, validity logits, edge logits; optionally the dense heat map. Works on the frozen cached BEV because it acts on head outputs. |
| what it gives us | Axis A5 (and A8 / A9): a validity memory with Bayesian semantics, closing open item 3 of the overview; a fused per-vertex and per-edge probability that is meant to be calibrated (open item 4); the same recursion can carry the route probability. |
| conflicts | (1) L2G assumes a global pose and fixed cells; our vertices move and live in the ego frame. (2) Offline and acausal; we need a causal filter. (3) No clamp, no decay, independent-evidence assumption: all three are wrong for consecutive frames of one frozen foundation. (4) Our current implementation (`fuse_prior`) associates by a hard nearest-neighbour gate (argmin, 1.5 m, 30 deg): a rule step on the training path whose gradient reaches only the current evidence. (5) Nothing to copy: no code exists. |
| reuse mode | Idea only / re-implement from the description (the formulas are 6 lines). |
| effort | S: log-odds on the dense heat map in the ego frame (option R below). M: vertex and edge evidence with soft polyline association, parameters trained on cached builder outputs. L: end-to-end through the builder over several frames. Blocker for M / L: a reliable calibration measure and a failure protocol to show a gain (clean frames are a tie, see section 7). |
| expected gain and risk | Clean frames: expect a tie in S (E173 against E175 and E192 were ties); gain expected in stability (flicker) and under failure (night, cameras off) and in calibration; risk: over-confidence from correlated evidence, wrong association at parallel lanes and at junction mouths, sparse vertices (see 5.2). |
| feeds which experiment / K id | K24 stays the id; this extraction upgrades its formula source from "from memory" to checked-via-digest (E7 - E12). Proposed new technique ids (to be registered before use): soft polyline association with dustbin for evidence transfer; ego-distance Cauchy reliability weight; leaky soft-clamped log-odds state; field-evidence read-out at warped vertices. |

### 5.1 What L2G actually contributes (what is worth keeping)
Ranked by the evidence in the ablations: (a) **log-odds sum against probability mean**: +7.1 mIoU / +5.5 Global IoU with HPDR maps (Table II), and the mean is worse than no fusion; (b) **base-rate prior subtraction** (E8, heuristic against uniform prior): 7.2 Global IoU (Table VII); (c) **distance reliability weight** c = 1 / (1 + d_n^2): +0.4 Global IoU / +1.3 mIoU (Table IV); (d) bilinear footprint: +0.07 / +0.12, only a raster artefact. In our code `fuse_prior` adds two logits without any prior subtraction, which corresponds to L2G's weak "uniform prior" variant; whether the base rate of a candidate node (not of a cell) is near 0.5 and the subtraction therefore does not matter has to be measured (fraction of valid nodes among candidates on TRAIN) before deciding. Not a defect claim.

### 5.2 What it takes to run the same evidence fusion on vertices (my design; nothing here is in the paper)
State after frame t-1, in the ego frame of t-1: vertices j with position x_j, travel heading theta_j, log-odds L_j; edges (child to parent pointer) with log-odds L^e_j; tokens h_j.

1. **Warp** (exists): x'_j = R_c^T (R_p x_j + t_p - t_c), headings rotated alike. Pose enters only as the relative motion of one step; because the positions of frame t come from the detections of frame t, pose error never accumulates in positions, only the soft association loses some precision per hop. This is the structural difference to L2G, whose map errors pile up in the global frame.
2. **Association without identity, by geometry, to the polyline and not to the vertex.** The graph is sparse (champion 13.5 vertices per 100 m, so about 7 m between vertices), the two graphs sample the same lane at different arc positions, and the present gate of 1.5 m therefore pairs only a fraction of the nodes (my back-of-envelope estimate: roughly 3 m of 7 m, below half). Use point-to-segment projection: for a current vertex i and a warped previous edge k = (a, b): u_ik = clamp(<x_i - x'_a, x'_b - x'_a> / |x'_b - x'_a|^2, 0, 1), d_ik the distance to the projected point, weight w_ik = exp(-d_ik^2 / (2 sigma^2)) * exp(kappa (cos(dtheta_ik) - 1)). The heading term is needed: opposite lanes lie about 3.5 m apart with reversed heading. Add a **dustbin** weight w_0 (as in SuperGlue, see [C30](../papers/C30_superglue-sinkhorn-optimal-transport-with.md)) and normalise: w^_ik = w_ik / (w_0 + sum_k' w_ik'). Interpolated prior log-odds of the hit: L_ik = (1 - u_ik) L'_a + u_ik L'_b. Mass that goes to the dustbin is **birth**: the vertex starts from the prior L_0. Column normalisation (or a few Sinkhorn steps) stops two dense current vertices from both claiming the same previous segment (double counting). No argmax, no Hungarian; the outcome is a soft matrix, as the project rules demand.
3. **Evidence model (E8 for vertices).** Let the builder emit a validity logit l_i that is trained to be a calibrated single-frame posterior (BCE against geometric labels, already the practice) and let l_0 = logit of the base rate of valid nodes. Evidence: dL_i = l_i - l_0.
4. **Update (E7, E10, plus what L2G lacks).**
   L_i = L_0 + lambda * (Ltilde_i - L_0) + c_i * dL_i,   Ltilde_i = L_0 + sum_k w^_ik (L_ik - L_0),
   c_i = 1 / (1 + (x_i / r_x)^2 + (y_i / r_y)^2) with r_x, r_y near the BEV half-extent of 50 m (this makes a vertex at the BEV border count half), then a soft clamp L_i <- L_max tanh(L_i / L_max). lambda < 1 is forgetting towards the prior (L2G has none); L_max bounds the state so that train and test states have the same range regardless of sequence length. With lambda = 1, c = 1, no clamp this is exactly L2G E7 - E8.
5. **Negative evidence for previous-only vertices (the part cells get for free).** A raster accumulates evidence for every cell, also "nothing here". A vertex set only sees evidence at current vertices, so a previous vertex that is not re-detected gets no update. Fix: read the **current dense heat map** (our dense head) at the warped position x'_j with `grid_sample`, p_obs = H_t(x'_j), and apply E8 with a visibility weight; this is L2G's cell update evaluated at a moving point. It is what makes **death** possible without a hard rule: a vertex in view that the current frame does not support loses log-odds frame by frame; a vertex that left the +-50 m square simply drops out of the state; no deletion on the training path, the usual top-k slot cap (gradient through selected items is allowed) and the inference threshold are the only discrete steps.
6. **Edges and forks.** Keep one log-odds per parent edge. Prior for an edge: the same soft association applied to both endpoints, weight = product of the endpoint weights (matrix form A L^e A^T on the sparse edge list), evidence from the current edge head, same update. A fork or merge probability is a smooth function of its edge probabilities (for example the probability that at least two out-edges are valid, a noisy-OR style expression), so forks are fused through the edges without a fork-specific rule.
7. **Differentiability.** Every operation above is smooth in positions, logits and the scalar parameters (sigma, kappa, w_0, lambda, L_max, r_x, r_y, l_0; parametrise them through softplus / logit to keep them in range): warp, kernel weights, projection parameter u (clamp has a subgradient), softmax-style normalisation, `grid_sample`, sigmoid. The only discrete operations are the existing top-k and the final threshold. Two training regimes: (A) **fusion-only**: cache the single-frame builder outputs of whole sequences (frozen BEV, builder frozen or detached) and train just the fusion scalars by BCE on the fused validity through K = 3 - 4 unrolled frames; minutes of GPU time, no builder in the loop; (B) joint training with truncated backpropagation through the recursion, with the prior state no longer detached, so that the head learns to emit a **likelihood ratio** instead of a posterior. Keep the pose noise (sigma_t 0.15 m, 0.5 deg, 10 % cold-start frames) already used in `p16_time.py`, and train with random cold starts so the first frames of a scene are in distribution.
8. **Calibration and double counting.** Consecutive frames of one frozen foundation share errors (the same occlusion, the same blind far field), so the independence assumption of E7 over-counts. Worked example from E8 with the 0.10 divider prior: three consecutive uninformative observations p_obs = 0.5 each add +2.2 and move the cell from 0.10 to 0.99 (L_0 = -2.2, sum of increments 6.6). Remedies: lambda < 1, a learned temperature a on the accumulated part (our `fuse_prior` already uses a = 0.9), L_max, and a reliability diagram of the fused validity per vertex (expected calibration error) as a standard output of the experiment.
9. **Option R (cheapest test, effort S).** Apply L2G unchanged to the **dense heat map** in the ego frame: L_t = lambda * warp(L_{t-1}) + c * (logit H_t - logit p_0), warp by `grid_sample` (the bilinear footprint f(P) is exactly what the resampling does), then feed sigmoid(L) or L as extra channels to the builder. This is pixel-based fusion of evidence, to be compared with the feature-level fusion that gave +0.008 F1 (E39 / E41) and with the vertex-level versions above; p_0 is the fraction of cells within the heat-map radius of a centre-line, measured on TRAIN.

### 5.3 Proposed experiment (not run; to be put on a card by the loop owner)
Arms on the same sequences: (0) single frame; (1) present K24, hard gate, one step (E173); (2) L2G-literal on vertices: hand-set sigma, lambda = 1, c from E10, no clamp; (3) the same with lambda, L_max, a fitted (regime A); (4) option R. Report S and Q, F1@1, vertices per 100 m, ms per frame, **and** a flicker / stability number, reliability (ECE) of the fused validity, and the loop-17 failure protocols (night, blur, cameras off) where a memory should pay; two seeds, paired scene bootstrap, noise band 0.010 in S. A prediction I would write down: tie on clean VAL in S, a gain in stability and under failure only if arms 3 or 4 beat arm 2.

## 6. Not found / open
- Global grid size and resolution, r_x and r_y, the exact meaning and units of (l_x, l_y), the epsilon used when taking logits, L_0, the decision threshold for Map_g, the definition of "FrameLoss" and of the easy / medium / hard levels, the global ground truth used in Table II, the split used for the closed loop, the pose source for nuScenes and Argoverse: none is stated in the HTML text I could query.
- No clamping, decay, forgetting, nor an ablation of any of them; no pose-error test; no comparison to a learned fusion; no code.
- The PDF could not be read (binary), no appendix or supplement was visible; the project page, if any, was not found.
- Text errors noted: the second perception range is written "y_x"; the prior-learning ablation is Table V but cited as Table VI; the fusion is called "Bayesian optimization" in the text although it is recursive log-odds updating; the vector gain is 20.04 in the table and 20.00 in the text.
- Related probabilistic accumulation paper surfaced by a search, **not read**: "Mapping like a Skeptic: Probabilistic BEV Projection for Online HD Mapping" (arXiv 2508.21689, BMVC 2025), which accumulates by confidence; it may be a second precedent for A5.

## 7. Our experiments that touch it
The level-1 note says "none"; that is out of date. A one-step, hard-gated vertex version of K24 was built and run in loop 16 (BB28):
- E173 (E123.D7.r3.t1, `training/models/p16_time.py` + `p16_builder.py: add_prior, fuse_prior`): previous-frame nodes (the own single-frame prediction of the previous frame, no gradient, top 64, validity above 0.3) warped by the ego motion become extra seed candidates (validity times 0.9); where a current node lies within 1.5 m and 30 deg of a previous node, logits are fused as a * logit(0.9 * pi_prev) + b * logit(pi_now), a = 0.9, b = 1.0, no prior subtraction, unmatched nodes unchanged; training with pose noise (0.15 m, 0.5 deg) and 10 % frames without a prior. Result S 0.725 (VAL subset of 250 frames, 8 u) against the length control E175 S 0.724: a tie inside the noise band.
- E176 (junction head + the same time recursion): S 0.723; E192 (VAL250, idle GPU) reports the time recursion as a tie in S. The recursion costs no gradient to the memory side and carries no state beyond one step.
- Not tried: recursive accumulation of a log-odds state, prior subtraction, ego-distance weighting, forgetting / clamp, soft polyline association, negative evidence from the dense head, edge-level evidence, evaluation under failure or on stability, calibration of the fused validity.
