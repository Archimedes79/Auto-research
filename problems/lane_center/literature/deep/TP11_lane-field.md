# TP11 - Lane Field: full-text extraction

| field | value |
|---|---|
| paper | Learning Lane-Structured Conditioning Fields for Online HD Mapping. Andrea Bertogalli, Anthony Dell'Eva, Marcello Ceresini, Federico Pirazzoli, Lorenzo Cipelli, Filippo D'Addeo, Nicola Marinello, Massimo Bertozzi (University of Parma, University of Bologna, VisLab / Ambarella; the project page lists the affiliations, the order of two names differs between arXiv and the project page). arXiv 2610.25284 [cs.CV], v1 submitted 2026-10-18, the only version at reading time; the arXiv page has no comment line. Project page and repository README say "accepted at WACV 2027". Links opened: arxiv.org/abs/2610.25284, arxiv.org/html/2610.25284 (main text and supplement A - J), andberto.github.io/LaneField/, github.com/andberto and github.com/andberto/LaneField, raw README |
| axis, tier, K ids | A2 (dense lane-centre field), tier 1 core; K169 (direction-tensor field as conditioning, register status "partly"; this extraction upgrades its source to digest of the full text) |
| level-1 note(s) | [TP11](../papers/TP11_lane-field.md) (home note), [A34](../papers/A34_lane-field-dense-lane-structure.md); corrections in the list below |
| code extraction | none: no code is released. The only repository found, `andberto/LaneField`, holds the project-page website (README + `dist/` web build); the project page says "Code coming soon". See section 4. No file in `../code/` |
| sources read | arXiv abs page; arXiv HTML v1, 102,299 characters, read in 14 separate queries (structure; Sec. 3.1 / 3.2; 3.3; 3.4 / 3.5; 4.1 - 4.3 with Table 1; 4.4 - 4.6 with Tables 2 - 4; supplement C / I / J with Tables 2, 5, 6, 7; supplement A, B, D - G and failure cases; placement and gate details; gradient-flow prose; 4.6 setup and related work; the last 2,299 characters, which are only the arXiv site footer); project page; GitHub pages (api.github.com returned HTTP 403, so `dist/` was not listed); one web search for a repository (nothing found). All read 2026-10-25. Local files read for section 5 and 7: `training/train/dense_head.py`, `training/data/prep/make_gt_arrays.py`, `training/data/p16_data.py`, `experiment_execution/plans/Experiment_Table_EN.md`, `Literature_Compatibility_Space_EN.md` |
| reading marks | every statement about the paper is **digest**: all pages came through the summarising fetch tool (no page was seen raw), so numbers may be mis-transcribed. Cross-checks that I did by arithmetic are marked "my check". Equations 1 - 8 and 16 - 18 were returned identically by three independent queries. Statements about our own code are **checked** (files read). Section 5 is my analysis, not the paper's content; derivations in it are my own mathematics (classical, **memory**-level, re-derived from the paper's formulas) |
| extracted by, date | extraction agent (Claude Sonnet 5.5, sub-agent of the literature session), 2026-10-25 |

## Corrections to the level-1 notes

1. **Gradient-flow ablation (TP11 Conflicts line; compatibility study row 8 "36.1 vs 34.1 mAP with / without gradient into the encoder").** 34.1 is the configuration "stop-grad In and Out" (both cuts at once). Cutting only the encoder-to-field-head path ("Stop-grad In") gives 34.8, cutting only the head-to-decoder path ("Stop-grad Out") gives 33.3, fully attached 36.1, baseline without any field 32.1 (supplement Table 7). Every cut variant therefore still beats the baseline by +1.2 to +2.7 mAP. **In all variants the host loss still trains the whole encoder; no variant freezes it.** The study does not show what a frozen encoder does, and the claim "the reported gain needs the field gradient to reach the BEV encoder" is too strong. All numbers are single runs, no seeds.
2. A34 says the stop-gradient variants lie at 34.1 - 34.8: they lie at **33.3 - 34.8**.
3. TP11 "uses width annotations (we have no lane width ...)": the width only sets the spread sigma = f * mean width (f = 0.5, with a lower clamp sigma_min whose value is not given); the SD-map variant uses **a fixed sigma** and works (36.8 mAP). Width is not a blocker for us; a fixed sigma is enough.
4. TP11 / A34 "5-channel field": the field R has five channels (M_xx, M_xy, M_yy, proximity P, confidence C), but the **head outputs a different five**: M_xx, M_xy, M_yy, proximity mean and a Laplace log-scale of the proximity. The confidence C is computed from the predicted tensor and is not a network output; the uncertainty (log-scale) exists **only for the proximity**, not for the tensor channels.
5. TP11 "guidance is switched off at intersections": the confidence C equals |cos(theta)| for two equally weighted lanes at angle theta (my derivation, 5.2), so the gate closes only for near-perpendicular crossings; at forks and merges with small angles it stays open. Also, "sigma_i = 0.5 w_i" is Eq. 4 with f = 0.5 plus the clamp of the supplement (Eq. 14).
6. TP11 "Matching: not described": the field itself needs no matching (analytic targets). The three host models keep their own set-matching losses (host details not read here).
7. TP11 loss line is incomplete: the proximity NLL is weighted by a lane-proximity weight (w_min 0.1, tau 0.2, alpha 20) and the gradient loss compares signed finite differences; see section 2.
8. TP11 / A34 "no repository link": a GitHub repository `andberto/LaneField` exists but contains only the project-page site. A34 "compat_conflicts: none" and "effort S/M" are too optimistic: see section 5 (sign-free tensor, ribbon-shaped proximity, no deformable decoder, no ablation against concatenation).
9. TP11 header "authors not seen": see the table above. Venue WACV 2027 comes from the project page and README only.

## 1. Problem and claim
- Online HD-map construction (OpenLane-V2 lane segments and centre-lines, also Argoverse 2 and nuScenes map elements) from surround cameras with a trained BEV encoder and a DETR-like vector decoder. Priors are either dense rasters (easy to inject, weak geometry) or sparse vectors (geometry kept, special encoders needed).
- The paper proposes the **Lane Field**: a dense, BEV-aligned field that stores, per cell, the local lane structure (orientation tensor, proximity, direction confidence). It is built in closed form from lane centre-line polylines (ground truth, or SD-map polylines) and **predicted** by an auxiliary head from the BEV features.
- It is used twice, as a plug-in: (a) SPADE-like modulation of the BEV features, (b) stretching the tangential part of the learned deformable-attention sampling offsets along the local lane axis, gated by confidence, predicted uncertainty and a per-head scalar.
- The claim that matters for us: a supervised, analytic, unoriented dense field adds +2.0 to +4.0 mAP on top of LaneSegNet (OpenLane-V2 subset A), and the field built from SD roads alone gives +4.7. Nothing about vertices, graphs, time, sensor failure or frozen encoders.

## 2. Method as built

### Pipeline (digest)
```
7 surround cameras -> ResNet-50 -> BEVFormer encoder -> F_BEV  (100 x 200 x C, +-50 m x +-25 m, so 0.5 m cells - my arithmetic; C not stated)
   |
   +-> field head (DeepLabV3-like, +4.8 M parameters, layers / channels / dilations not given)
   |      outputs: Mxx, Mxy, Myy, proximity mean mu, Laplace log-scale s_p   (5 maps, H x W)
   |      C = Delta / tr computed from the predicted tensor (not supervised, not a network output)
   |
   +-> (a) modulation: residual SPADE-like block (+2.4 M):  field -> shared latent -> two conv branches -> gamma, beta (H x W x C)
   |          F_out = gamma * F_in + beta  -> modulated F_BEV  (block "can be stacked"; placement inside the encoder not stated)
   |
   +-> (b) guidance (about 0 parameters + one scalar theta_h per attention head):
              at the reference point r of a query, bilinear read of M, C, mu, s_p
              -> principal axis t of M (unoriented) -> tangential component of every learned offset stretched by up to 4x
   |
   v
DETR-like decoder of the host (LaneSegNet lane segments; MapTRv2 / GeMap / TopoPoint as plug-in hosts) -> vector instances
```
The framework is stated to be agnostic to the sensing modality and the BEV encoder. Queries are **not** initialised or conditioned by the field; the paper describes only the offset rescaling (digest; "not described" otherwise).

### Inputs, outputs
- Input of the field head: F_BEV only. Output: the five maps above on the BEV grid. Field in the BEV frame of the ego vehicle; lanes are polylines C = {C_k = (x_k, y_k)} with left / right boundary offsets giving a width w_k per point (how the offsets become a width: not given).
- SD-map variant (Sec. 4.5): the head is removed; the same analytic construction is run on SD road polylines with a fixed sigma (value not given), and the result feeds both modulation and guidance. Whether it is rasterised on the BEV grid is implied, not stated.

### Equations (numbering as printed; the main text has 1 - 12, the supplement continues; I did not see Eq. 15 and 19)
- (1) segment i with endpoints A_i = C_k, B_i = C_{k+1}: length l_i = |B_i - A_i|, unit tangent t_i = (B_i - A_i) / l_i, mean width w_i = (w_k + w_{k+1}) / 2.
- (2) segment contribution: I_i(p) = l_i * integral over t in [0,1] of exp(-|p - A_i - t (B_i - A_i)|^2 / (2 sigma_i^2)) dt. This is the **line integral of a Gaussian along the segment** (arc-length units).
- (3) closed form: I_i(p) = (sigma_i sqrt(2 pi) / 2) * exp(-d_perp^2 / (2 sigma_i^2)) * [erf(u1) - erf(u0)], with u0 = -(p - A_i)^T (B_i - A_i) / (sqrt(2) sigma_i l_i), u1 = u0 + l_i / (sqrt(2) sigma_i); d_perp = distance from p to the infinite line through the segment (supplement Eq. 13: the projection parameter is deliberately **not clamped** to [0,1]; the erf terms take care of the ends).
- (4) spread: sigma_i = f * w_i, f = 0.5; supplement Eq. 14: sigma_i = max(f * w_i, sigma_min), sigma_min value not given. Kernel value at the lane edge (distance w/2) is exp(-1/(8 f^2)); supplement Table 1: f = 0.25 / 0.50 / 0.75 / 1.00 gives 13 / 61 / 80 / 88 percent (my check: exp(-2) = 0.135, exp(-0.5) = 0.607, 0.80, 0.88, all match).
- (5) direction tensor: M(p) = sum_i I_i(p) t_i t_i^T = [[Mxx, Mxy], [Mxy, Myy]]. **Sign-free**: t and -t give the same matrix.
- (6) proximity: P(p) = sum_i I_i(p). Additive over all segments.
- (7) confidence: eigenvalues lambda_{1,2} = (tr +- Delta) / 2, tr = Mxx + Myy, Delta = sqrt((Mxx - Myy)^2 + 4 Mxy^2); C = (lambda1 - lambda2) / (lambda1 + lambda2) = Delta / tr.
- (8) the field R(p) = [Mxx, Mxy, Myy, P, C], H x W x 5, same grid as F_BEV. Main text: Mxx, Myy, P, C in [0, 1], Mxy in [-1/2, 1/2]; directional channels and confidence soft-masked by proximity.
- Targets (supplement C, Eqs. 16 - 18), tau_P = 1.0, k = 20, tau_m = 0.3, epsilon not given: P_gt = P / (P + tau_P); M_gt = M / (tr M + epsilon) * sigmoid(k (P_gt - tau_m)); C_gt = C * sigmoid(k (P_gt - tau_m)). The trace normalisation makes tr(M_gt) equal to the mask, so P carries the amplitude and M only the shape.
- (9) per-pixel Laplace beta-NLL of the proximity: l_i = (log(2 b_i) + |mu_hat_i - mu_i| / b_i) * stopgrad(b_i)^beta * w_prox_i, b = exp(s_p), variance 2 b^2; (10) w_prox = w_min + (1 - w_min) * sigmoid(alpha (mu - tau)).
- (11) L = w_R L_R + w_G L_G + w_L L_L with 5.0 / 2.0 / 1.0. L_R (Eq. 20, 21): mean Smooth-L1 over the three tensor channels. L_G (Eqs. 22 - 26): mean absolute difference of the **signed** finite differences (x and y) of the tensor channels between prediction and target. L_L (Eqs. 27, 28): the NLL of Eq. 9. Constants beta 0.5, w_min 0.1, tau 0.2 (note: differs from tau_m = 0.3 of the target mask), alpha 20 (supplement Table 2: "selected empirically on validation").
- (12) guided sampling: s_new = r + [1 + g_C(r) g_U(r) g_H(h) (Delta_par - 1)] s_par t + s_perp n. r reference point; t the unit vector along the principal axis of M(r) with an arbitrary sign; n its normal; s_par, s_perp the learned offset in that frame (the sign of t cancels because s_par is signed in the same frame). Gates (supplement F, Table 3): g_C = sigmoid(alpha_C (C - tau_C)), alpha_C = 10, tau_C = 0.3; g_U = sigmoid(alpha_U ((mu_hat - std) - tau_U)), std = sqrt(2) exp(s_p), alpha_U = 10, tau_U = 0.3 (a **lower-confidence-bound** gate: open only if mean minus one standard deviation of the proximity exceeds 0.3); g_H = sigmoid(alpha_H theta_h), alpha_H = 2, theta_h learnable per attention head; Delta_par = 4.0 (maximum tangential stretch).

### Losses and assignment
Field head: Eqs. 9 - 11 above, no matching of any kind (dense per-pixel targets from analytic ground truth). The weight of the field loss against the host loss, and any ramp-in, are not stated. The host decoders keep their own losses. The valid-pixel sets Omega are "valid BEV locations", not defined further.

### Labels / ground truth (how the field is derived from the lane graph)
- Per **lane centre-line polyline** (with width), not per graph node: every polyline segment contributes through Eq. 3, all contributions add. The graph structure (successor / predecessor links) is not used; consequently a trunk shared by two lane polylines would be counted twice if both polylines carry it (the paper does not discuss de-duplication).
- Several lanes in one cell, forks, merges, crossings: **not discussed explicitly**. The additive sums in Eqs. 5 and 6 handle them implicitly: P counts overlapping support, M mixes the axes, C drops when the axes disagree. Intersections are named only through the effect on C and on the gate (supplement F, Fig. 7: C is near zero in the ambiguous region, guidance suppressed, sampling stays near the learned offsets).
- Properties I derived from the formulas (my check, classical math): (i) the sum over the segments of a polyline equals the line integral over the whole polyline, so the field does not change if collinear vertices are added or removed; (ii) for a long straight segment the peak of I is sigma * sqrt(2 pi) in the coordinate units, so with sigma = 1.75 (lane width 3.5) the peak is about 4.4 and P_gt about 0.81 if the unit is metres (about 0.90 if cells of 0.5 m; the unit is not stated); (iii) with sigma = 0.5 w, two neighbouring lanes 3.5 m apart sit exactly at the limit 2 sigma where a sum of two equal Gaussians stops being bimodal, so P is a flat-topped road ribbon, not a thin ridge per lane.

### Training recipe (digest)
- LaneSegNet host: OpenLane-V2 subset A, seven cameras, ResNet-50, BEVFormer encoder, BEV 100 x 200 over +-50 m x +-25 m, 24 epochs, global batch size 8, AdamW, the original LaneSegNet learning-rate schedule, 8 NVIDIA RTX 4000 Ada GPUs. The main text does not say whether the field head is trained jointly with the host or in a separate stage; the "fully attached" gradient configuration implies joint training. Training time, learning rate of the head, warm-up: not stated.
- Argoverse 2 and nuScenes (Sec. 4.6): MapTRv2 6 epochs on AV2, GeMap 12 epochs on AV2, MapTRv2 24 epochs on nuScenes; batch size 16; BEV range +-30 m x +-15 m; classes boundary, divider, centre-line, pedestrian crossing as fragmented polylines. Backbone and cameras not stated there. GeMap was not run on nuScenes.

### Inference
No NMS or post-processing of the field itself. Guidance at inference reads the predicted field at the query reference point; modulation uses the predicted field. The paper shows maps of the last decoder layer only; whether guidance is used in every decoder layer is not stated.

## 3. Numbers (digest; table numbers as printed; a dagger in the paper marks a baseline re-trained from the official code)

Table 1(a), OpenLane-V2 subset A, lane-segment perception, ResNet-50, 24 epochs:

| method | mAP | AP_ls | AP_ped | TOP_lsls |
|---|---|---|---|---|
| StreamMapNet (temporal) | 20.3 | 22.1 | 18.6 | 13.2 |
| TopoNet | 23.0 | 23.9 | 22.0 | - |
| SQD-MapNet (temporal) | 26.0 | 27.1 | 24.9 | 16.6 |
| MapTR | 27.0 | 25.9 | 28.1 | - |
| MapTRv2 | 28.5 | 26.6 | 30.4 | - |
| LaneSegNet (re-trained) | 32.1 | 32.2 | 31.9 | 25.5 |
| TopoLogic | 33.2 | 33.0 | 33.4 | 30.8 |
| Topo2Seq | 33.6 | 33.7 | 33.5 | 26.9 |
| LaneSegNet + Lane Field | 36.1 (+4.0) | 35.8 (+3.6) | 36.4 (+4.5) | 28.4 (+2.9) |

Table 1(b), centre-line perception (DET_l, TOP_ll, OLS_l): MapTR 17.7 / 5.9 / 21.0; TopoNet 28.6 / 10.9 / 30.8; TopoMLP 28.3 / 21.6 / 37.4; TopoLogic 29.9 / 23.9 / 39.4; LaneSegNet (re-trained) 31.1 / 25.0 / 40.6; TopoFormer (pre-trained lane detector) 34.7 / 24.1 / 41.9; TopoPoint (re-trained) 31.1 / 28.5 / 42.2; Topo2Seq 33.5 / 27.0 / 42.7; **LaneSegNet + Lane Field 35.2 / 28.0 / 44.1 (+4.1 / +3.0 / +3.5)**; **TopoPoint + Lane Field 32.8 / 30.0 / 43.8 (+1.7 / +1.5 / +1.6)**. The paper does not say whether the other rows are copied. My check: OLS_l = (DET_l + 100 sqrt(TOP_ll / 100)) / 2 reproduces 30.8, 40.6, 42.2 and 44.1; the TopoNet TOP_ll of 10.9 is the old-protocol value (the compatibility study records 4.1 versus 10.9 for the same model), so Table 1(b) mixes devkit versions. Devkit version: not named.

Ablations that decide transfer (LaneSegNet host, OpenLane-V2 subset A; mAP / AP_ls / AP_ped / TOP_lsls):

| Table | configuration | mAP | AP_ls | AP_ped | TOP_lsls |
|---|---|---|---|---|---|
| 2 | baseline | 32.1 | 32.2 | 31.9 | 25.5 |
| 2 | + guidance only | 33.2 | 33.4 | 32.9 | 26.8 |
| 2 | + modulation only | 34.9 | 35.7 | 34.0 | 27.9 |
| 2 | both | 36.1 | 35.8 | 36.4 | 28.4 |
| 3 | field built from SD roads (head removed, fixed sigma) | 36.8 | 36.6 | 37.0 | 29.7 |
| suppl. 7 | stop-grad In and Out (AugMapNet-like setting) | 34.1 | 34.2 | 34.0 | 26.7 |
| suppl. 7 | stop-grad In (encoder -> head cut) | 34.8 | 34.4 | 35.2 | 27.0 |
| suppl. 7 | stop-grad Out (head -> decoder cut) | 33.3 | 32.3 | 34.3 | 27.2 |
| suppl. 7 | fully attached | 36.1 | 35.8 | 36.4 | 28.4 |

Reading (mine): modulation is worth about +2.8, guidance +1.1, the two add up (+3.9 against +4.0). Detaching in any way keeps a gain of +1.2 to +2.7 mAP; "stop-grad Out" gives no lane-segment AP gain (32.3 against 32.2) and only a pedestrian-crossing and topology gain; the orderings (In and Out 34.1 below In 34.8) are non-monotone, so with single runs and differences of 0.7 I would not read an order into them. mAP is the mean of AP_ls and AP_ped in every row I checked. **Missing ablations** (not found in what I read): field head with supervision but no modulation and no guidance (separates auxiliary dense supervision from conditioning); modulation against plain concatenation or the add-back of AugMapNet; any single gate (g_C, g_U, g_H) on or off; sigma factor f by metric (only the weight at the lane edge and qualitative figures); supervised against unsupervised field; the effect on forks or junctions.

Table 4, other datasets / hosts (mAP and class APs; units not stated): AV2 MapTRv2 62.0 -> 64.2 (divider 66.3 -> 69.5, ped 54.8 -> 56.4, boundary 64.6 -> 67.1, centre-line 62.0 -> 63.6); AV2 GeMap 70.3 -> 71.2; nuScenes MapTRv2 55.8 -> 57.7 (divider 55.9 -> 59.2, ped 52.5 -> 52.9, boundary 60.1 -> 62.8, centre-line 54.9 -> 55.8). My check: the mAP equals the mean of the class columns in all rows. The text says the largest gains are on boundaries and dividers.

Attention analysis (supplement F Table 4 and project page; 1,000 validation samples, last decoder layer): mean angular error of the sampling pattern to the local lane axis 35.3 deg (original Lane Attention) -> 11.4 deg; anisotropy of the sampling pattern 13 % -> 62 %; covered area 33.2 m^2 -> 78.2 m^2. The examples shown come from a head whose gate "converges close to one", so they show the maximum stretch, not the typical behaviour (authors' admission).

Compute (supplement Tables 5, 6; single RTX 4000 Ada; batch size and input size not stated): parameters LaneSegNet 45.4 M, +4.8 M head, +2.4 M SPADE, attention guidance about 0, total 52.6 M; MapTRv2 40.6 M -> 47.8 M; TopoPoint 53.6 M -> 60.8 M. FPS / TFLOPs: LaneSegNet 14.6 / 1.3 -> 13.9 / 1.5; MapTRv2 33.2 / 1.2 -> 29.9 / 1.4; TopoPoint 13.8 / 1.4 -> 13.0 / 1.6. In ms (my arithmetic, 1000 / FPS): about +3.4, +3.3, +4.5 ms on hosts of 68.5, 30.1, 72.5 ms. Table 5 prints identical values for "+SPADE", "+Attn." and "+All", so the two mechanisms cannot be costed separately from it.

Caveats: single runs, no seeds, no confidence intervals; baselines re-trained by the authors; OpenLane-V2 numbers are protocol-dependent and are not comparable with ours (different data, metric, range, frame rate); all gains are with a **trained** camera encoder and decoder.

Limitations stated by the authors: inherently lane-centric (pedestrian crossings are not encoded; a missed-crossing failure case is shown); guidance only for deformable attention; topology improves only indirectly; SD priors are sparse and lack width and lane count; rare roundabouts fail for baseline and method alike; the model can predict lane support in unannotated areas (driveways, parking-lot entrances) but with high variance, which the uncertainty gate uses to switch guidance off.

## 4. Paper versus code
**No code.** The project page (andberto.github.io/LaneField) states "Code coming soon" under "Code & models" and offers a contact form for code requests. GitHub `andberto/LaneField` (listed as updated 2026-10-16, 7 commits on `main`, 0 stars, no description, no licence file visible) holds only `README.md` (title, abstract, the sentence "accepted at WACV 2027", two figure previews) and a `dist/` folder of web assets; no model, training or data code, no weights. `api.github.com` returned HTTP 403, so the content of `dist/` was not listed (the figure paths in the README suggest web-build assets only). None of the other 14 repositories of the account has a related name. A web search found nothing else. Consequence: no file in `../code/`, the licence is unknown, and everything in section 2 comes from the paper text; the unspecified details (head architecture, sigma_min, epsilon, placement of the modulation block, loss weighting against the host) cannot be checked. Re-read this section when the code appears.

## 5. Transfer to our pipeline

| question | answer |
|---|---|
| what it needs from us | Frozen cached BEV 200 x 200 at 0.5 m (39 ch): yes, the field head would be a small head on the cached features, as our dense head already is (U-Net, base 48 channels, `dense_head.py`). Ground truth: our centre-line graph polylines plus a **fixed sigma**; no lane width, no ego pose, no SD map, no image features, no lidar. For the SD-map variant: SD roads as polylines (we have them, simulated ADASIS-like). Not needed and not available: a deformable-attention decoder, a trainable BEV encoder. |
| what it gives us | Axis A2 (dense field) targets and extra channels: an order-independent, sign-free axis tensor; an **additive** proximity (counts overlapping lane support, where our heat map takes the maximum); an analytic anisotropy confidence; a Laplace-scale uncertainty on the heat map. For A3: heading / fan-width hints for the verified-ray builder. For A8: a lower-confidence-bound gate, i.e. a principled "better short than false" switch. |
| conflicts | (1) Sign-free: the tensor drops the **travel direction** that our tips, rays and planner need. (2) Sigma = half lane width gives a road-ribbon, not a thin ridge; our heat map (sigma 0.75 m) is the better localiser. (3) The guidance mechanism lives inside deformable attention (no host here). (4) The measured gains come with a trained encoder; no frozen-encoder test. (5) The confidence C does not flag Y-forks and merges (small angles), which dominate our junction metric J. (6) No code, licence unknown. No Hungarian matching, no argmax on the field path; C and the principal axis are smooth except at the isotropic point (Delta = 0), which the gate g_C closes. |
| reuse mode | Re-implement from the description (the closed form is a few lines; no code exists). Idea only for the sampling guidance. |
| effort | S: new target arrays (tensor, additive density, optional Laplace channel) and three to five extra outputs of the dense head; precompute is cheap (200 x 200 cells times a few hundred segments per frame, closed form). M: using the new channels inside the ray builder (fan width, heading, gate) and the evaluation at junctions. Blocker: clean-frame F1 of the dense head is near saturation (E31 / E33: even a perfect heat map + direction field gives route F only 0.838), so a gain has to show at J, direction error at junction cells or under failure, not in plain F1. |
| expected gain and risk | Clean VAL S: expect a tie inside the 0.010 band (extra channels re-describe what the 42-channel input already carries: heat + direction concatenated to the 39 features, the AugMapNet-style route of E94). Possible gain: junction cells (order-independent label), J, calibration of the heat map. Risks: correlated evidence (same features, same head), the sign loss, extra channels that only add noise to the builder. |
| feeds which experiment / K id | K169 (registered: direction-tensor field as conditioning). Candidates to register before use: (a) **additive line-integral density with closed-form rendering of a vertex graph into a field** (differentiable in the vertex coordinates; 5.4); (b) **lower-confidence-bound gate from a Laplace scale on the heat map** (g_U). |

### 5.1 How the Lane Field relates to our dense head
Our dense head (`training/train/dense_head.py`, `make_gt_arrays.py`, checked): heat map target y = exp(-d^2 / (2 * 0.75^2)) with d the exact distance to the **nearest** centre-line, capped at 4 m (union semantics); focal-weighted soft BCE; direction target = unit travel vector (dx, dy) drawn along the polylines with `cv2.line` three cells wide, loss 1 - cos on cells with |gt| > 0.5. Side by side:

| aspect | our head | Lane Field |
|---|---|---|
| presence | max-type Gaussian of nearest distance, sigma 0.75 m | sum of line integrals, sigma = 0.5 * lane width (about 1.75 m), saturated target P / (P + 1) |
| direction | signed unit vector (2 ch), travel direction | unsigned tensor (3 ch, M_xx, M_xy, M_yy), trace-normalised and masked |
| overlap of several lanes | the drawn label is whichever lane was rasterised last (`cv2.line` overwrites) | order-independent sum |
| ambiguity measure | none | C = Delta / tr, analytic |
| uncertainty | none in the head (the builder has a hazard along the rays) | Laplace log-scale of the proximity, beta-NLL |
| supervision of the tensor | cosine loss | Smooth-L1 (5.0) + signed-gradient L1 (2.0) + NLL (1.0) |

The tensor is the classical second-moment (structure-tensor) form of an orientation field: M = (tr / 2) * (I + [[cos 2a, sin 2a], [sin 2a, -cos 2a]]) for one axis at angle a, so it is a doubled-angle vector plus an amplitude (my identity).

### 5.2 Could a field replace the heat map at junctions? Short answer: no, but it can fix two weak points
Derived from Eqs. 5 - 7 (my mathematics): for two lanes with weights I1, I2 at angle theta, C^2 = (I1^2 + I2^2 + 2 I1 I2 cos 2 theta) / (I1 + I2)^2; for equal weights **C = |cos theta|**. Consequences:
- Crossings near 90 degrees give C near 0 (flagged), but a Y-fork with 20 degrees gives C = 0.94 and one with 40 degrees gives 0.77: **forks and merges are not flagged**. The gate g_C is half open at C = 0.3 (theta about 72.5 degrees). So C is a crossing detector, not a fork / merge detector; our junction head and J (fork / merge F1 at 6 m) need more than this.
- Two opposite lanes of one road (reversed travel) have identical tensors: the field cannot say which way a lane runs. A tip needs a sign; the field gives only the axis. Keep the signed channel.
- A rank-2 tensor holds one axis plus an anisotropy: with three or more lane directions it cannot recover the branches; with exactly two equal-weight branches it gives the bisector and the opening angle but not which branch is which.
- What it does fix: (i) the direction label at a crossing no longer depends on drawing order; (ii) the additive proximity carries a **lane-multiplicity cue** (overlap of two branches near a fork root, two lanes of a crossing) that a max-type heat map hides; (iii) C gives a smooth, free ambiguity cue at crossings.
- Proposed use (my design, not in the paper): keep the thin heat map and the signed direction, and **add** (a) the tensor (2 free channels after trace normalisation) and the additive density as auxiliary targets with a sigma of 0.75 m or one fixed larger value, (b) a Laplace log-scale on the heat map, (c) an order-independent signed label: the density-weighted sum of signed unit tangents, whose norm relative to the density tells where the sign is unreliable (crossings, opposite lanes). Cost: three to five more output channels of a head that is already part of the 31 ms chain; the closed-form labels are computed offline.

### 5.3 The field as candidate / sampling guide for the verified-ray builder
The paper's guidance is a gated anisotropic stretch of the sampling pattern along the unsigned axis (Eq. 12). Our counterpart is the teardrop of 12 rays from each tip (position + heading). Ideas, ranked by how much they add over what the rays already read (heat map + direction from the 42 channels):
1. **Fan width from C and the lower-confidence bound.** Narrow the teardrop when C is high and g_U is open, widen it where C is low; the same product of gates as Eq. 12, applied to the angular spread instead of the tangential offsets. The sign comes from the tip heading, the field supplies the axis. Cheap, differentiable if the fan is a smooth function of the gates.
2. **Heading refinement** of a tip to the local principal axis (angle = 0.5 * atan2(2 Mxy, Mxx - Myy), sign taken from the tip), blended by g_C; the atan2 is singular at the isotropic point, so only through the gate.
3. **g_U as an extra input to the survival / hazard of a ray** (validity of "a lane is here at all"); our hazard already encodes this, so expect a small effect and some double counting; the new thing is a calibrated lower bound from a trained scale.
4. Not useful: stretching sample points tangentially by up to 4x (our rays are already long and narrow, and the length is chosen by verification).
Honest limit: all of these read the same dense head, hence the same features as the heat map; they are not independent evidence. The expected benefit is at junction mouths and parallel-lane clutter, not on clean straight lanes.

### 5.4 Differentiable rendering of a vertex graph into a field (idea built on the paper's closed form, not in the paper)
Eq. 3 is smooth in the segment endpoints, so the field of a **predicted** graph can be rendered on the 200 x 200 grid with gradients to the vertex coordinates (and, with edge weights, to edge validity). Uses: a matching-free consistency term between the builder output and the dense head's field, and an analysis-by-synthesis score for hypotheses of the tree (how much of the rendered field is explained by the dense field). Against the ground truth we already have the point-to-polyline distance both ways, so the new value is only as a self-consistency / hypothesis check and as one more route to the open item "dense-then-sparse with gradient through the conversion" (overview section 3, item 6). Cost estimate (mine): 40,000 cells times a few hundred edges, local windows make it cheaper; to be timed before any claim. Untested; the paper only renders ground truth.

### 5.5 What needs a trainable encoder and what does not transfer
- The gain split by gradient path (36.1 attached against 33.3 - 34.8 detached) is measured with a trained encoder; the part that comes from the field loss shaping the encoder (auxiliary dense supervision) is absent in our frozen setting. Our dense head already is such a supervised dense head and its output is concatenated to the features (the AugMapNet-like route; "stop-grad In and Out" in the paper is described as that setting and gives 34.1, +2.0 over the baseline), so the paper gives no evidence that FiLM modulation beats our concatenation: the comparison is missing.
- Our situation corresponds most closely to "stop-grad In" (the field head reads frozen features; the downstream loss may reach the head): 34.8, +2.7 over the baseline and -1.3 below fully attached. The comparison is loose: in the paper the host loss still trains the encoder in that variant.
- SPADE modulation of the cached 39 channels by a field predicted from the same channels is a self-conditioning step (a second stage); it adds capacity, not information, so expect at most a refinement effect; test it only against a matched-capacity second stage.
- Deformable-attention guidance has no host in our chain; only the geometric idea (5.3) transfers.
- ResNet-50 / BEVFormer / camera-specific parts: not applicable.

### 5.6 Proposed experiment (not run; to be put on a card by the loop owner)
Arms on the same frozen features and builder: (0) current dense head, 42-channel input; (1) dense head with the extra targets (tensor, additive density, Laplace scale) as auxiliary outputs only, builder input unchanged (control for extra supervision); (2) arm 1 with the extra channels also fed to the builder; (3) arm 2 with fan width and heading refinement from C and g_U (5.3). Report S and Q, F1@1, J, lanes found, vertices per 100 m, direction error at junction cells, ms per frame (idle GPU, batch 1), two seeds, paired scene bootstrap, noise band 0.010 in S; add the loop-17 failure protocols for the uncertainty channel. A prediction I would write down: tie in S on clean VAL; any gain in J or junction-cell direction error; arm 3 is the only one that can matter for the builder.

## 6. Not found / open
- Head architecture (layers, channels, dilations), input normalisation, C of F_BEV; sigma_min; epsilon in Eqs. 17, 18; units of the coordinates in Eq. 2 (metres or cells) and hence the absolute scale of P; the fixed sigma of the SD variant; how the boundary offsets become a width; de-duplication of shared trunks; weight of the field loss against the host loss; optimiser and learning rate of the head; where in the encoder the modulation sits, how many blocks, kernel sizes, hidden width; whether guidance acts in every decoder layer; devkit version for OpenLane-V2; seeds and variances; training time.
- Ablations absent from the text I read: supervision-only (no conditioning), concatenation / AugMapNet add-back against SPADE, each gate, sigma factor by metric, fork / junction subsets, frozen encoder, temporal use, sensor-failure or noise tests, camera-radar use.
- The authors' own statements on code: only "coming soon". The supplement was read in HTML only; figures were not seen.
- Table 5 prints the same speed for the three variants; the cost of each mechanism alone is unknown.

## 7. Our experiments that touch it
None for the tensor form. Related: E31 / E33 (dense heat map + direction field to a graph with the tracer: F1 0.879 / 0.851 VAL / TEST; even a perfect field gives route F 0.838), E32 (sparse DETR-style decoder 0.058 F1 below the grid approach), E37 (dense head at full maturity, F1 0.885 VAL), E44 / E53 (learned split / merge key points, junction work), E94 (trace network with fan sampling on the 42-channel input: heat + direction concatenated to the 39 features, F1 0.883 / 0.851). Nobody has tried an unsigned tensor target, an additive density, a Laplace scale on the heat map, or a closed-form render of the predicted graph. Do not repeat the plain "concatenate heat + direction" route as if it were new: it is the AugMapNet-like baseline here.
