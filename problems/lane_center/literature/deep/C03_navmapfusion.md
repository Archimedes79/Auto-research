# C03 - NavMapFusion: full-text extraction

| field | value |
|---|---|
| paper | NavMapFusion: Diffusion-based Fusion of Navigation Maps for Online Vectorized HD Map Construction. T. Monninger, Z. Zhang, S. Staab, S. Ding. WACV 2026. arXiv 2512.03317 (cs.CV; only version v1, submitted 2025-12-03). Links opened: https://arxiv.org/abs/2512.03317 , https://arxiv.org/html/2512.03317 (and /v1), https://github.com/tmonnin/navmapfusion |
| axis, tier, K ids | A6 (SD prior inside perception; also touches A8), tier 1 (core), K74 |
| level-1 note(s) | [C03](../papers/C03_navmapfusion-sd-map-cross-attention.md) (the only note of this paper; SMERF, the SD-map encoder it reuses, is [PL43](../papers/PL43_smerf.md)) |
| code extraction | [navmapfusion](../code/navmapfusion.md). The official repository contains **no code** (README, LICENSE, one image); the host model's code (tmonnin/mapdiffusion) and one third-party reproduction exist, both described there |
| sources read | (1) arXiv abstract page (v1 only, no appendix, no supplement, 36 references; the HTML has no appendix or limitations heading). (2) arXiv HTML v1, whole paper, in six passes with different questions (method, setup, tables, robustness text, introduction/conclusion, headings). (3) MapDiffusion paper arXiv 2507.21423 (HTML, host model). (4) SMERF arXiv 2311.04079 (HTML, SD-map encoder). (5) GitHub: tmonnin/navmapfusion (page, README, LICENSE, API metadata, issues), tmonnin/mapdiffusion (README, config, mapper, head, transformer, coef.py, vector_eval.py), seonghyun713/NavMapFusion (third-party reproduction: README, config, mapper, SD encoder, transformer, head, dataset, OSM script). (6) Local: STATE.md lines 70-99, register.md rows K17, K74, K75, K98. All read 2026-10-24 |
| reading marks | **digest** for everything that comes from a web page: the only reader available is a summarising fetch tool, so no page was seen verbatim. Tables 1-5 were fetched twice with different prompts and agreed cell for cell; the relative gains and the class means were re-computed by hand and match. **checked** only for Section 7 and the data-builder remarks in Section 5 (local project files read directly). Statements marked "inference" are mine |
| extracted by, date | extraction agent (Claude), 2026-10-24 |

## 1. Problem and claim
Predict a local vectorised HD map (lane dividers, road boundaries, pedestrian crossings; polylines in the ego frame) from six camera images and a coarse OpenStreetMap road-polyline map (the "navigation map", SD map). The gap: earlier SD-prior methods inject the map into the BEV encoder or into the query initialisation; this paper conditions the **decoder** of a diffusion map model on the SD map. The claim that matters: because the model is a diffusion decoder, a disagreement between prior and sensors is treated as noise that the denoiser removes, so consistent regions are reinforced and outdated map segments are suppressed (the introduction says the model can "attenuate or amplify individual elements in a probabilistic manner"); with train-time perturbations of the SD map it stays useful when the map is shifted, thinned out or wrong. Headline: mAP 22.4 -> 27.2 (+21.4 % relative) at 100 m x 50 m, gains growing with range (+0.0 % at 60 m, +57 % at 150 m).

What the paper does **not** contain (digest): any explicit trust, gate, confidence or abstain output for the map; any measurement of whether the network recognises a wrong map; any formal link between the discrepancy and the diffusion noise level (see 2.6); Argoverse 2 or any dataset other than nuScenes; parameter counts; an appendix.

## 2. Method as built

### 2.1 Pipeline and architecture
```
 6 cameras --f_BEV (StreamMapNet encoder: ResNet-50 + BEVFormer-style, GRU)--> BEV grid B (H x W x C)
 OSM road polylines P_road --f_SD (SMERF map encoder)--> S = M tokens (one per polyline)

 random queries X_T  (100 polylines x 20 points x 2, normalised to [0,1])
   |  k = 5 DDIM steps; only the decoder g is iterated, f_BEV and f_SD run once
   v
 decoder g, L = 6 layers; each layer, with add-and-norm after every block:
     self-attention -> MCA (queries X_t, keys/values S) -> SCA (deformable attention over B) -> FFN
   + MLP head: 20-point polyline (the clean map element) and class score per query
```
- **Order matters (Tab. 5):** Map Cross-Attention (MCA) before Spatial Cross-Attention (SCA) gives +1.1 % relative mAP. The authors' reading: MCA supplies the coarse prior, SCA then updates it with sensor evidence (a Bayesian-style argument without derivation).
- **SD-map encoder f_SD:** one embedding per road polyline. Final model: SMERF's transformer encoder; an MLP variant (4 layers) is only in the ablation. SMERF details (digest of the SMERF paper): each polyline resampled to 11 points, coordinates normalised to the BEV range and mapped to (0, 2 pi), sinusoidal embedding (d = 32, temperature 1000), optional road-type one-hot, a linear projection, then 6 self-attention layers (4 heads best in SMERF's ablation); output M x H. NavMapFusion itself does not state the point count, token dimension, maximum number of polylines or whether road type is used.
- **BEV dropout:** the BEV grid B is zeroed with probability d_BEV = 0.30 during training, to keep the model from over-relying on the sensor features; in the decoder this silences the SCA branch so the map is the only evidence. Effect: +2.3 % relative mAP (Tab. 5), AP_ped 28.1 -> 32.1, AP_div 22.8 -> 20.7.
- **Host model:** MapDiffusion (same first author, IROS 2025) = StreamMapNet encoder + diffusion decoder without the temporal query propagation. Numbers of the host (nuScenes, disjoint split, 60 m x 30 m, thresholds 0.5/1/1.5): mAP 35.6 against 33.8 for StreamMapNet; 8.0 FPS with five steps, 12.8 FPS with one step.

### 2.2 Inputs, outputs
- Inputs: six monocular camera images per frame; SD map = OSM road-level polylines only (no lane count, no route), taken from P-MapNet's pre-processed OSM data, **manually aligned to the nuScenes localisation** (i.e. the ego pose is given; registration is not part of their problem).
- Output: up to 100 polylines of 20 points (2-D, normalised to the perception range, ego at the origin), each with a class score for ped_crossing / divider / boundary; post-processing keeps queries above the score threshold. Perception range 100 m x 50 m in the main result (60 x 30 up to 150 x 75 in Tab. 3). The numbers 100 queries x 20 points and the [0,1] normalisation are from the host model's code and config (digest), not from the paper text.

### 2.3 Equations (numbering as in the paper; symbols explained)
- E1: M_HD = m(U, M_SD). U: six-camera image sequence, M_SD: SD map, M_HD: local HD map.
- E2 (forward process on the **ground-truth vector map** X_0 = M_HD): X_t = sqrt(abar_t) X_0 + sqrt(1 - abar_t) eps, eps ~ N(0, I), alpha_t = 1 - beta_t, abar_t = prod_{s<=t} alpha_s.
- E3 (training loss as written): L(theta) = E || eps - eps_theta(X_t, t) ||^2.
- E4 (reverse step): X_{t-1} = 1/sqrt(alpha_t) * ( X_t - (1 - alpha_t)/sqrt(1 - abar_t) * eps_theta(X_t, t) ) + sigma_t z, z ~ N(0, I).
- E5 (sensor conditioning, deformable attention as in StreamMapNet): SCA(z_q, v_q, B) = sum_{k=1..K} A_qk W B[v_q + dv_qk]; attention weights A_qk and offsets dv_qk are predicted from the query.
- E6 (map conditioning): MCA(Q, K, V) = softmax(Q K^T / sqrt(d)) V with Q = X_t W_Q (noisy queries), K = S W_K, V = S W_V (SD-map tokens). Plain multi-head attention; the paper mentions no positional term on the keys, no gate, no mask handling.
- E7: M_HD = g( f_BEV(U), M_SD ).
- Not in the paper (digest; from the host code, see [navmapfusion](../code/navmapfusion.md)): the beta schedule is cosine, abar(t) = cos^2( ((t/N + 0.008)/1.008) * pi/2 ), total T = 1000; the DDIM sampling update with stochasticity eta, sigma = eta * sqrt( (1 - abar/abar_next) (1 - abar_next) / (1 - abar) ) and x_next = sqrt(abar_next) x0_hat + sqrt(1 - abar_next - sigma^2) eps_hat + sigma * noise; eps_hat is derived from the predicted x0 by eps_hat = (sqrt(1/abar) x_t - x0_hat) / sqrt(1/abar - 1).

### 2.4 Losses, labels, assignment
- The paper shows only E3 and states that the training setup and the losses follow MapDiffusion. The MapDiffusion paper names a line loss between the denoiser output and the clean map x_0 over all diffusion steps plus a class loss (digest); no weights in either paper.
- **Paper versus code on the target:** E3 is the generic DDPM noise-prediction loss. The host code predicts the clean polylines x_0 directly (the eps used in sampling is derived from x_0), and trains with a DETR-style set loss: Hungarian assignment of the 100 predictions to the ground truth (focal cost 5.0 + L1 cost 50.0, beta 0.01, with point-order permutation), focal classification loss weight 5.0 (gamma 2, alpha 0.25), L1 line loss weight 50.0, and auxiliary losses on all 6 decoder layers (digest of config and head). So the host loss **uses Hungarian matching on the output side**; the map side has no matching.
- Labels: nuScenes map elements as vectors (ped crossing, divider, boundary), 20 points each. Ground truth is padded to 100 elements with random polylines (N(0.5, 0.25), clipped to [0,1]) before noising (host code, digest).
- **SD-map side labels:** none. The SD map has no loss of its own; nothing is supervised about its reliability (see 2.7).

### 2.5 Training recipe
- 24 epochs, batch size 1 per GPU, 8 NVIDIA V100; AdamW, learning rate 2e-4, cosine annealing (host config: weight decay 1e-2, gradient clip 35, 1000 warm-up iterations, backbone learning-rate multiplier 0.1, 3,480 iterations per epoch = 27,846 // 8, 83,520 in total; digest). BEV grid 100 x 50 cells (the paper digest gives this for the 100 m x 50 m range; the host config uses the same 100 x 50 cells for 60 m x 30 m, so the cell size at the other ranges is unclear); decoder 6 layers; d_BEV = 0.30.
- Diffusion training (host code, digest): one timestep t ~ U{1..1000} per batch (not per sample), noise drawn from N(0.5, 0.25) clipped to [0,1] (not a standard normal), noisy polylines clipped to [0,1].
- **Train-time perturbations of the SD map (Section 4.7), the part that matters for us:**

| family | how the test-time fault is defined | training choice | what the paper shows (mAP, 100 m x 50 m) |
|---|---|---|---|
| outdated geometry (element dropout) | each SD polyline removed independently with probability d_SD | train d_SD in {0.1, 0.2, 0.3}; 0.2 best on an intact map, 0.3 most stable at high drop rates, **0.1 best over the whole range** | without dropout training the model falls below the no-SD-map reference for d_SD > 0.7 (figure only, no values) |
| misalignment | raw OSM against the manually aligned OSM | aligned-trained vs unaligned-trained | 26.6 (aligned/aligned), 26.0 (aligned-trained, unaligned test), 26.8, 26.5: marginal effect, in the authors' words |
| location inaccuracy | Gaussian noise, sigma in metres, on the SD map (per polyline or one global shift is **not stated**) | sigma = 1 m chosen as compromise | 26.6 at sigma 0; noticeable drop beyond sigma 1 m for the untrained model; the sigma = 1 m model keeps its peak up to test sigma 3 m (figure only) |
| localisation error (wrong map) | with probability p the SD map is replaced by (a) the map of a random location in the map area, or (b) the localised map of a random other nuScenes sample | p = 0.1 for both | untrained: 26.6 -> 20.1 (a) and 26.6 -> 17.8 (b) at p = 1. Trained with p = 0.1: (a) 25.9 at p = 0 and 24.4 at p = 1; (b) 24.7 at p = 0 and 24.0 at p = 1 |

- No flag, input channel or loss tells the network that a sample carries a wrong map (digest of 4.7.4).
- **Inference-side caveat (inference from the numbers):** the "clean" reference of all robustness studies is 26.6, which is the model **without** BEV dropout (Tab. 3, d = 0.0); the final model with BEV dropout (27.2) is not swept. BEV dropout pushes the decoder towards relying on the map, so the final model's behaviour under a wrong map is unknown and plausibly worse than the swept model.

### 2.6 "Discrepancy is diffusion noise": what is real
1. The forward process (E2) noises only the **ground-truth map vectors** (queries); neither the SD map nor the BEV features are noised, and the SD map is not part of X_t. It enters only as keys/values of a cross-attention.
2. The noise level is the diffusion timestep t: uniform random per batch in training, a fixed 5-step DDIM schedule at test. It is **not tied to any measured discrepancy** between map and sensors; the paper has no sentence assigning a t to a discrepancy and no analysis of early versus late steps (digest: not discussed).
3. Where t enters the network (host code, digest): sinusoidal embedding (128) -> MLP (512); concatenated with the query encoding for the reference-point offset, and used as a per-layer FiLM scale/shift on the query before each decoder layer (out = out * (1 + scale) + shift). The MCA block has no t-dependent gate in the third-party reproduction; the official NavMapFusion layer is unknown.
4. Practical mechanism for "suppress outdated segments": the sampling loop keeps only queries whose score exceeds tau = 0.5 after each step and replaces the others by fresh random polylines (a rule step), and the network has learned, from the corrupted-map training, to give low scores to elements the BEV does not support. Both are indirect; the paper reports no per-element evidence that the suppression happens.
5. Conclusion: the idea is a conceptual framing of a conditional diffusion decoder with a robust training recipe. What carries over is the recipe (2.5), the MCA-before-SCA ordering, and the sweep protocol, not a noise-level mechanism.

### 2.7 How "map trust" is represented and learned
- **No explicit trust, gate, confidence or abstain signal for the map** exists in the paper (digest answer to a direct question; the third-party reproduction has none either). The MCA output is added through the ordinary residual and norm; there is no learned scalar, no zero-initialised gate, no flag input.
- Trust is therefore **implicit and amortised**: the decoder compares MCA and SCA evidence and has learned, from p = 0.1 wrong-map samples, 10 % element dropout and 1 m noise, to hedge. The only per-query confidence is the class score.
- Evidence that this works: with p = 0.1 training, mAP under 100 % wrong maps (24.0 / 24.4) stays above the 22.4 of the map-free MapDiffusion. Weak evidence: the two models differ in capacity and training, and attention weights or any map-detection accuracy are not reported.
- Related but unused: the host paper turns the spread over n = 10 sampled maps into an uncertainty (variance +31 % in occluded areas); NavMapFusion mentions no uncertainty result.
- BEV dropout is the opposite of a trust mechanism: it teaches the model to live on the map alone.

### 2.8 Inference
DDIM with k = 5 steps (k = 1 and 3 in Tab. 4), eta = 0.5, score threshold tau = 0.5; start from random polylines (N(0.5, 0.25) clipped); after every non-final step drop queries below tau, refill to 100 with fresh random polylines; last step outputs the x_0 prediction; final post-processing score threshold from the config. Batch size 1 and the random draws are hard-coded in the host code (not seeded, so outputs are stochastic). The BEV and SD encoders are computed once per frame. The "tau" and re-randomisation steps are rule steps on the inference path.

## 3. Numbers
All values author-reported, nuScenes, StreamMapNet's geospatially disjoint split ("newsplit"), mAP over ped / divider / boundary, Chamfer thresholds as stated; digest, each table fetched twice.

**Table 1 (100 m x 50 m, thresholds {1.0, 1.5, 2.0}; * = taken from StreamMapNet, others reproduced by the authors)**

| method | AP_ped | AP_div | AP_bound | mAP |
|---|---|---|---|---|
| VectorMapNet* | 12.0 | 8.1 | 6.3 | 8.8 |
| MapTR* | 8.3 | 16.0 | 20.0 | 14.8 |
| SQD-MapNet | 24.8 | 18.8 | 23.7 | 22.4 |
| StreamMapNet | 24.8 | 18.4 | 25.6 | 22.9 |
| StreamMapNet-MCA (StreamMapNet + SD encoder + MCA, no diffusion) | 28.4 | 22.2 | 27.8 | 26.1 |
| MapDiffusion (no SD map) | 23.4 | 21.6 | 22.2 | 22.4 |
| NavMapFusion | 32.1 | 20.7 | 28.9 | 27.2 |

StreamMapNet-MCA is +14.0 % over StreamMapNet; NavMapFusion is +21.4 % over MapDiffusion. A deterministic MCA on StreamMapNet already gets 26.1, so most of the gain comes from the SD map and its cross-attention, and the diffusion decoder accounts for at most 1.1 mAP (27.2 vs 26.1) at 100 m; whether StreamMapNet-MCA uses the same SD encoder and BEV dropout is not stated, so this is an upper bound. (inference from the table)

**Table 2 (long range and other prior-map methods)** thresholds and split differ per block: dagger = 120 m, original split, {1.0, 1.5, 2.0}; asterisk = 120 m, original split, {0.5, 1.0, 1.5}; double dagger = 100 m, disjoint split, {1.0, 1.5, 2.0}, mAP over ped and divider only. NavMapFusion here is trained without dropout.

| method | AP_ped | AP_div | AP_bound | mAP |
|---|---|---|---|---|
| ScalableMap (dagger) | 44.8 | 49.0 | 43.1 | 45.6 |
| NavMapFusion (dagger) | 57.5 | 60.2 | 57.3 | 58.4 |
| MapTR-SDMap (asterisk) | 22.0 | 27.2 | 19.5 | 22.9 |
| P-MapNet (asterisk) | 14.6 | 19.4 | 38.7 | 24.2 |
| NavMapFusion (asterisk) | 34.4 | 37.2 | 25.7 | 32.4 |
| MapEX, GT-boundary prior (double dagger) | 22.0 | 17.2 | - | 19.6 |
| M3TR, GT-boundary prior (double dagger) | 24.7 | 18.9 | - | 21.7 |
| NavMapFusion (double dagger) | 28.1 | 22.8 | - | 25.5 |

**Table 3 (effect of the map by range; without MCA = MapDiffusion; d_BEV = 0.0; double dagger thresholds {0.5, 1.0, 1.5}, asterisk {1.0, 1.5, 2.0})**

| range | map | AP_ped | AP_div | AP_bound | mAP | gain |
|---|---|---|---|---|---|---|
| 60 x 30 (double dagger) | no | 32.9 | 31.4 | 42.4 | 35.6 | |
| | yes | 31.8 | 30.7 | 44.4 | 35.6 | +0.0 % |
| 80 x 40 (double dagger) | no | 18.9 | 19.5 | 24.3 | 20.9 | |
| | yes | 20.2 | 19.8 | 25.7 | 21.9 | +4.8 % |
| 100 x 50 (asterisk) | no | 23.4 | 21.6 | 22.2 | 22.4 | |
| | yes | 28.1 | 22.8 | 29.0 | 26.6 | +18.8 % |
| 120 x 60 (asterisk) | no | 17.2 | 7.66 | 10.4 | 11.8 | |
| | yes | 22.4 | 14.0 | 17.7 | 18.0 | +52.5 % |
| 150 x 75 (asterisk) | no | 11.4 | 4.2 | 2.9 | 6.1 | |
| | yes | 13.0 | 8.2 | 7.7 | 9.6 | +57.4 % |

**Table 4 (inference settings, 100 x 50, d = 0.0)**

| steps k | eta | tau | FPS (A6000) | mAP |
|---|---|---|---|---|
| 1 | 0.5 | 0.5 | 14.7 | 26.4 |
| 3 | 0.5 | 0.5 | 10.3 | 26.6 |
| 5 | 0.5 | 0.5 | 8.1 | 26.6 |
| 5 | 0.1 | 0.5 | 8.1 | 26.5 |
| 5 | 0.9 | 0.5 | 8.1 | 26.6 |
| 5 | 0.5 | 0.1 | 8.1 | 26.1 |
| 5 | 0.5 | 0.9 | 8.1 | 26.6 |

One step costs 0.2 mAP and nearly doubles the speed (AP_ped 27.7, div 22.6, bound 28.9 at k = 1).

**Table 5 (incremental ablation, 100 x 50)**

| row | AP_ped | AP_div | AP_bound | mAP | step |
|---|---|---|---|---|---|
| baseline (MapDiffusion) | 23.4 | 21.6 | 22.2 | 22.4 | |
| + MLP f_SD | 26.3 | 20.7 | 28.3 | 25.1 | +12.1 % |
| + SMERF f_SD | 31.5 | 18.9 | 28.6 | 26.3 | +4.8 % |
| + MCA before SCA | 28.1 | 22.8 | 29.0 | 26.6 | +1.1 % |
| + dropout on B | 32.1 | 20.7 | 28.9 | 27.2 | +2.3 % |

Most of the gain comes from having any SD-map encoder and cross-attention (+12.1 % with an MLP encoder); the SMERF encoder, the ordering and BEV dropout each add 1-5 %, and the per-class picture moves (AP_div falls from 22.8 to 20.7 with BEV dropout).

**Robustness (Section 4.7, mAP; values from the text, curves only in Figs. 6-7):** see the table in 2.5. Reference points: no map 22.4; clean map 26.6; wrong map (random place) 20.1; wrong map (another frame) 17.8; with p = 0.1 training 25.9 / 24.4 and 24.7 / 24.0. A wrong map without such training is therefore **worse than no map** by 2.3 (random place) and 4.6 (other frame) mAP.

**Compute:** FPS on an NVIDIA A6000 at 100 x 50: 14.7 / 10.3 / 8.1 for k = 1 / 3 / 5. Parameters, FLOPs, memory and training time: not stated. Training hardware: 8 V100, 24 epochs.

**Caveats:** (1) thresholds, split and range differ between rows (see markers); do not compare across tables or with our F1 / route success. (2) Metric is vector mAP of map elements; no route attribute, no lane graph, no topology. (3) Manual alignment: the ego pose is given. (4) Only nuScenes; the OSM prior comes from P-MapNet's pre-processing. (5) Single run per configuration, no seeds or intervals reported (digest: none seen). (6) Several robustness numbers exist only as curves.

## 4. Paper versus code
No official code. The README of tmonnin/navmapfusion lists four configs (streammapnet, streammapnet_mca, mapdiffusion, navmapfusion) and the results of Table 1, and says the code will be released soon (state at the reading date: one commit of 2025-12-02, README, LICENSE, one image; issue #2 of 2026-03-05 asks for the release, no reply). Differences visible in the available related code:
- Eq. 3 predicts eps; the host code predicts x_0 and uses a Hungarian set loss with auxiliary losses.
- A third-party reproduction (not by the authors; its README text is copied from the official one, including the claim of being the official implementation) runs at 60 m x 30 m, 110 epochs on 2 GPUs, with SD dropout and coordinate noise switched off (rates 0.0 in its config) and **no wrong-map substitution at all**; BEV dropout 0.30 is implemented as one random draw per batch that zeroes the whole BEV tensor. It is therefore not a check of the paper's robustness recipe. Details in [navmapfusion](../code/navmapfusion.md).
- The paper says nothing about the number of SD tokens, point count or road types; the reproduction uses 11 points, 256-d tokens, no road types (13 types are cached by its loader).

## 5. Transfer to our pipeline
| question | answer |
|---|---|
| what it needs from us | SD map as random-access tokens in the ego frame (we have the SD raster tokens of the route head); a data loader that can substitute another frame's map, a random place's map, a thinned map (our builder already does all three). No image features, no lidar. Their BEV encoder is trainable and camera-only; ours is the frozen 39-channel cache, which only changes the SCA side. Ego pose is given in the paper; in our task registration is part of the problem |
| what it gives us | A6 / A8: (1) a four-family corruption recipe for the SD input with concrete ranges (element dropout 0.1-0.3, shift sigma 1 m, wrong map p = 0.1 from two sources) and an evaluation sweep; (2) evidence that a wrong map without such training is worse than no map; (3) the ordering "prior first, evidence second"; (4) a large-range argument: the SD map helps perception more the farther the range (+0.0 % at 60 m, +18.8 % at 100 m x 50 m, +57 % at 150 m). No trust signal |
| conflicts | Diffusion decoder: 5 iterations, 8.1 FPS (123 ms) on an A6000 against our chain of 31 ms; random-query start, stochastic output, score filter and re-randomisation are rule steps on the inference path; Hungarian matching on the output side in the host loss; Python 3.8 / torch 1.9 / mmcv 1.6 / mmdet3d stack; GPL-3.0 and no released code; trainable BEV encoder; given pose; the output is vector mAP, not a route attribute |
| reuse mode | idea only (training recipe and sweep protocol), re-implemented from description in our own data builder; nothing to port |
| effort | S for one extra sweep (a joint grid over shift x dropped fraction x wrong-map share, as in the paper's four figures) with the existing data builder; M for an explicit map-support output (proposals below), blocker: per-vertex labels for partly wrong maps |
| expected gain and risk | Small for the recipe, because our map-fault training is already a superset of the paper's four families (data builder `p8_dataset.py` + `p8_lib.draw_sd` + `sd_lib.degrade_sd`, read 2026-10-24): other-sample wrong map (row offset 397 inside the same part, share 0.05-0.1), no map (0.05-0.1), rigid pose error sigma in {0,1,2,3} m with yaw 2 sigma deg, side-road dropout 0.3 in 30 % of the draws, shape-point tolerance and lane-count errors, per-road lateral bias (sigma 1-3 m, share 0.5), plus abstention targets. Two real differences: (a) the paper drops any polyline, we never drop the route polyline (`degrade_sd` keeps it); the case "the SD road under our own route is missing or stubbed" is untested and is the one that hurts a lane-level attribute most; (b) the paper reports each family on its own curve with the model trained on a different rate per curve, we train one mixed model, so a family-by-family curve (route success against shift sigma, dropped fraction, wrong-map share) is a cheap addition. Warning: do not copy BEV dropout in the sense of relying on the map more; it works against abstention (K98's source dropout 0.3 already covers the useful variant). Risk: the paper's metric is geometry mAP, so its rates (10 %, sigma 1 m, p 0.1) say nothing about the best rates for route success |
| feeds which experiment / K id | K74 (registered: SD cross-attention, prior as noise, BEV dropout) and K98 (gate and source dropout) cover the recipe. Proposed new techniques, to be registered before use: **K-new-a** "route-polyline dropout / stub in map-fault training" (new fault family, small change in `degrade_sd`), **K-new-b** "explicit per-vertex map-support output and fault-level head" (own idea, below) |

**Proposals for an output that says "the SD map does not match what I see" (own ideas, none tested; the paper gives no recipe):**
1. *Keep the label-free agreement gate as the reference* (perceived graph against SD road geometry; AUROC 0.99 in E83) and use any learned signal only if it beats it on the same wrong-map and failure cells. A learned trust head cost accuracy in E78.
2. *Per-vertex map-support logit trained with the NavMapFusion fault families.* Because every corrupted sample is generated by us, its truth is known by construction: label = 1 where an SD road within the registration tolerance and with compatible heading explains the vertex, 0 for wrong-map samples and for vertices whose SD road was removed by element dropout. This extends E87's abstention target (route flag forced to zero on a wrong map) from "whole map" to "this part of the map", and it separates the three cases the route head must tell apart: map right (use it), road missing in the map (perceived lane stays, route attribute unknown), map wrong (abstain, road-level fallback impossible).
3. *Corruption-level regression head.* The only place where a "noise level" is an input in this model is the timestep FiLM. A literal version of "discrepancy = noise" for us: sample a fault level (shift in metres, dropped fraction, wrong-map flag) in training, supervise a small head to regress it from the pair (vertex tokens, SD tokens), and use its output as the trust scalar that scales the SD contribution (FiLM on the SD branch). Cheap (a few k parameters), differentiable, and it yields a number to threshold for abstention.
4. *Counterfactual pass with a null map.* The route head is small (0.69 M): run it with the map and with a trained "no map" input (p_none samples) and use the disagreement as one feature of the trust estimate; disagreement alone cannot separate informative from contradictory maps, so it is only a feature.
5. *Dustbin assignment* (from SuperGlue, [C30](../papers/C30_superglue-sinkhorn-optimal-transport-with.md), not from this paper): a vertex-to-SD-road assignment with a dustbin row and column; the mass in the dustbin on the perception side is "no SD road explains this vertex", on the map side "no perceived lane explains this SD road". Fully differentiable and gives the road-level fallback as a by-product.
6. *Time recursion:* the same corruption recipe applies to the ego-warped previous vertex set (stale or drifted prior, odometry error); a prior-discrepancy gate there is the A5 analogue of the map-trust gate.

## 6. Not found / open
- Any explicit map-trust output; per-element evidence that the network detects a wrong map (attention weights, AUROC); calibration. Not in the paper.
- Whether location noise is a rigid shift of the whole map or independent jitter (the third-party reproduction applies independent per-coordinate noise, which deforms the roads instead of shifting them); the exact BEV-dropout form (per batch or per sample).
- Robustness of the final model with BEV dropout; any seeds or confidence intervals; parameter count; training time; the grid size used at 100 m x 50 m; the number of val samples of the disjoint split (the host config shows 27,846 training samples).
- Number of SD polylines per frame, token dimension, use of road type, the point count of the OSM resampling.
- The loss weights as used by NavMapFusion (assumed equal to the host config: focal 5, L1 50, Hungarian).
- Official code and checkpoints (not released; no Hugging Face release, issue #1 open since 2025-12-04).
- Everything is digest-level; no PDF or TeX source was read. A page-level read of Section 3.4, 4.3 and 4.7 and of Figs. 6-7 would settle the open points above.

## 7. Our experiments that touch it
Source: `experiment_execution/STATE.md` lines 74-98, read 2026-10-24 (**checked**); numbers are the file's, not re-run.
- E78 (loop 7): SD-conditioned pixel head. A variant with a learned trust head (S0 0.744 at 1500 steps) was worse than the variant without it (0.768); final head has no trust head and uses a label-free agreement score; with a 2 m registration error or no map the route success falls (0.660 and 0.257 for the trust-head variant on TEST).
- E83 (loop 8): segment-level assignment network; route success 0.881 / 0.879 / 0.868 / 0.839 at map error 0 / 1 / 2 / 3 m; the network does not abstain with a wrong map; label-free agreement gate AUROC 0.99.
- E87 (loop 9): training against map faults (wrong map p 0.1, no map p 0.1, per-road bias p 0.5 with sigma 1-3 m, abstention target 0.3): per-road bias 2 m 0.663 -> 0.847; flagged route length on a wrong map 60.7 -> 4.2 m, with no map 60.1 -> 0 m, clean cells unchanged. This is the NavMapFusion recipe (wrong map p = 0.1 drawn from another sample of the same part, shift sigma 1-3 m, side-road dropout 0.3) plus abstention, already done (data builder read directly: `temp/archive/misc/p8_dataset.py` lines 125-135, `training/data/p8_lib.py::draw_sd`, `training/data/sd_lib.py::degrade_sd`). Not run: dropping or stubbing the route polyline itself (`degrade_sd` never drops it); a family-by-family sweep with one model trained per rate (we train one mixed model). The level-1 note C03 lists element-dropout and wrong-map training samples as not recorded as run; that entry is outdated, both are in the builder (suggest correcting its tried_before field).
- E104 (loop 10/11): cross-attention from vertex tokens to SD-map tokens with a gated fusion (K17, K98); wrong map flags 4.6 m of route; route success 0.853 / 0.843 / 0.832 / 0.820 after selecting the threshold on VAL.
- K98 (UMPE: presence gate, source dropout 0.3) was already the model for the gate and dropout in E87 / E104.
