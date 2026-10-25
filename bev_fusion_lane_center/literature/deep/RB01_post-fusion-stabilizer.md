# RB01 - Post Fusion Stabilizer (PFS): full-text extraction

| field | value |
|---|---|
| paper | Post Fusion Bird's Eye View Feature Stabilization for Robust Multimodal 3D Detection (short name: Post Fusion Stabilizer, PFS). Trung Tien Dong, Dev Thakkar, Arman Sargolzaei, Xiaomin Lin (all University of South Florida, College of Engineering). arXiv comment "8 pages, IROS 2026"; arXiv 2603.05623 (cs.CV, cs.AI); v1 2026-03-05, v2 2026-09-03 (v2 read). Funding line: DEVCOM Army Research Laboratory CRADA and the NVIDIA Academic Grant Program. Links opened: https://arxiv.org/abs/2603.05623 , https://arxiv.org/html/2603.05623 , https://arxiv.org/html/2603.05623v2 , https://www.alphaxiv.org/overview/2603.05623 (empty overview, no code link) |
| axis, tier, K ids | A8 (robustness, sensor failure), tier 1 (core, row RB01 in `../papers.csv`); no K id exists yet (proposal in section 5) |
| level-1 note(s) | [RB01](../papers/RB01_post-fusion-stabilizer-identity-initialised.md) (the only note of this paper) |
| code extraction | no repository (section 4) |
| sources read | (1) arXiv abstract page v2. (2) arXiv HTML v2, whole paper (no appendix, no supplement; 29 references), in about ten passes with different questions: structure, Block 1, Block 2, Block 3, curriculum, setup and Table I, Table II (twice), Table III, Table IV, Table V and Section IV-B, discussion, introduction, references, a second verification pass on the curriculum and on the corruption definitions, and one pass on the data flow of Eq. (8). (3) alphaXiv overview page (no content), two web searches for code (nothing). (4) Local: level-1 note, `../papers.csv` row RB01, `../../evaluation/PROTOCOL.md` line 22, `../../HANDOVER.md`, `../OVERVIEW.md` section 1.4. All read 2026-10-25 |
| reading marks | per section: 1 digest; 2.1-2.8 digest (equations E1-E8 returned twice, consistent); 3.1 digest (cross-checked against Table IV); 3.2 digest; 3.3 digest; 3.4 digest; 3.5 digest plus marked inference; 3.6 digest plus my arithmetic; 4 digest (search and abstract pages through a summarising tool); 5 inference on top of **checked** local files (PROTOCOL.md, HANDOVER.md, OVERVIEW.md); 6 digest; 7 **checked** (HANDOVER.md, experiment folder READMEs); 8-10 my proposals. No section is marked memory: nothing is written from memory. In detail: **digest** for every statement about the paper: the only reader available is a summarising fetch tool, so no page was seen verbatim. Equations of Blocks 1-3 were returned in two independent passes and agree. Table II was refused as a whole in the first pass and obtained row by row in the second; its 20 cells per host that also appear in Table IV (rows A0 and A3, five conditions, mAP and NDS) agree with Table IV, which is the only cross-check available. Tables III, IV, V were fetched once each (Table V partly twice, text and table agree except where noted). **checked** only for section 7 and the local facts in section 5 (project files read directly). Statements marked "inference" or "my arithmetic" are mine and are not in the paper |
| extracted by, date | extraction agent (Claude), 2026-10-25 |

## 1. Problem and claim
Camera + LiDAR BEV fusion detectors (hosts: BEVFusion, UniBEV) lose accuracy under sensor corruption and weather. Published remedies change the fusion architecture or retrain a specialised model. PFS leaves the host (encoders, fusion, head) frozen and inserts a small trainable module on the fused BEV tensor, between the fusion stage and the unchanged detection head. The module has three blocks that all start near identity: a global shift normalisation, a per-cell reliability map, and a gated expert residual. It is trained in three stages on clean data plus synthetic corruptions.

The claim that matters: a 3.3 M parameter module, frozen host, lifts mAP under several corruptions (largest: Low Light, Rain Lens, 8-beam LiDAR, Fog) at a latency cost of 8.1 % on BEVFusion, and keeps clean mAP. Small or no gain for 6-camera dropout, occlusion and miscalibration on BEVFusion. All numbers are detection mAP / NDS on nuScenes; nothing is shown for maps, lanes or radar.

## 2. Method as built

### 2.1 Pipeline and architecture
```
 host (frozen):  6 cameras + LiDAR -> encoders -> BEV fusion -> F_fused   B x C x H x W
                                                       |             (C, H, W: symbolic in the paper;
                                                       |              C = 256 is consistent with the 99 K count, inference)
                  F_lidar = raw LiDAR BEV feature of the host (second input of Block 2; tensor not named)
                                                       v
 PFS (trainable, about 3.3 M parameters in total)
   Block 1  shift normalisation   F_fused               -> F_shift   B x C x H x W   about 99 K
   Block 2  reliability map       F_shift ++ F_lidar    -> R         B x 1 x H x W   about 664 K
                                  F_clean = R * F_shift                (soft mask, all channels)
   Block 3  experts + gate        [F_clean ; R]         -> dF, G     B x C x H x W   about 2.5 M
                                  F_out = F_shift + G * dF           <- note: base is F_shift, not F_clean
                                                       v
 host detection head (frozen, unchanged)  -> boxes     (host loss: focal + L1)
```
- **Input / output:** the module maps B x C x H x W to the same shape; the head sees only F_out (Figure 2 caption). Block 2 needs a second tensor, the host's LiDAR BEV feature; if it is not available the formula duplicates F_shift as both halves of the input (Eq. 2, second case). The paper does not say which case was used for BEVFusion or UniBEV, and does not evaluate the fallback separately.
- **Parameter sum:** 99 K + 664 K + 2.5 M = about 3.26 M (my arithmetic), stated total 3.3 M; Table V adds 3.3 M to both hosts (40.8 -> 44.1, 72.7 -> 76.0).
- **Which tensor is tapped, C/H/W:** not stated. The only grid size printed is 108 x 108, used for the Block 2 target (2.4).
- **Data flow caveat (from Eq. 8, inference from the equations; the paper says nothing explicit):** F_clean = R * F_shift enters the experts and the gate but not the output. The paper states that the correction bypasses the suppression of Block 2 ("effectively bypassing Block 2's suppression to restore signal from the normalized stream"). So in the full model the reliability map reaches the head only through the experts' input and the gate, never as a direct mask on the output. What the head sees in the configuration "A2 = Blocks 1 + 2 only" (Table IV, Stage 2) is not stated; F_shift would give Block 2 no task gradient at all, F_clean is the only reading under which Block 2 can change the A2 numbers (inference).

### 2.2 Block 1: shift normalisation (Section III-A)
Idea: low light and similar camera shifts cause a global drift of the BEV feature statistics; a conditional affine correction gated by a learnable scalar removes it.
- g = spatial average pool of F_fused (a vector over channels); (gamma, beta) = MLP(g), a two-layer MLP printed as Linear 256 -> 128 -> 512 (activation not stated; 512 = 2 x 256 would be gamma and beta per channel if C = 256, inference). Parameter check, my arithmetic: 256*128+128 + 128*512+512 = 98,944, which matches "about 99 K".
- GN = Group Normalisation of F_fused (group count, GN affine, epsilon not stated).
- E1: `F_shift = sigmoid(a) * ( gamma (.) GN(F_fused) + beta ) + ( 1 - sigmoid(a) ) * F_fused`, a = learnable scalar, initialised to -5.0 (sigmoid = 0.0067 at start). (.) = channel-wise product.

### 2.3 Block 2: spatial reliability estimation (Section III-B)
Idea: a per-cell map of how much of the original sensor data survived, learned from the ratio of point counts.
- E2: `R = sigmoid( ConvNet( F_shift ++ F_lidar ) )` if the host provides F_lidar; `R = sigmoid( ConvNet( F_shift ++ F_shift ) )` if not. ++ = channel concatenation. R is in [0,1], shape B x 1 x H x W.
- ConvNet: three 3 x 3 convolutions with stride 2, then one transposed convolution back to H x W. Channel widths, normalisation, intermediate activations and the upsampling factor (three stride-2 layers imply a factor 8) are not stated.
- Filtering (printed between Eq. 2 and Eq. 3, unnumbered): `F_clean = R (.) F_shift`, a soft multiplication, no threshold.
- Target and loss (Eqs. 3-5):
  - E3: `L_total = L_det + lambda_rel * L_rel`
  - E4: `L_rel = 1_corr * BCE(R, T) + alpha_anchor * 1_clean * BCE(R, 1.0)` (the second term is the "Anchor Loss")
  - E5: `T_ij = clamp( D_corr_ij / D_clean_ij , 0, 1 )`, with T_ij = 1 where D_clean_ij = 0
  - D_ij = point count of BEV cell (i,j) after voxelisation onto a 108 x 108 grid (the sensor is not named in that sentence; context says LiDAR). D_corr from the corrupted point cloud, D_clean from the clean one of the same frame.
  - 1_corr / 1_clean = indicators for corrupted / clean samples; the anchor pulls R to 1 on clean samples and prevents a drift to a uniform low value. Stage 2 uses lambda_rel = 1.0, alpha_anchor = 0.2. The text says the anchor matters for the roughly 70 % of samples without LiDAR corruption.
  - How R (B x 1 x H x W) is compared with a target on a 108 x 108 grid (resampling) is not stated. At inference no clean reference is needed: R comes from the features alone.
- Consequence (inference, see 3.5): T is defined from LiDAR point counts only. A camera-only corruption leaves the LiDAR counts unchanged, so T = 1 for it; the map is trained to say "reliable" under camera failure.

### 2.4 Block 3: expert correction and inpainting (Section III-C)
- Inputs: F_clean and R (R is called the hole map). Two experts, a semantic E_s and a geometric E_g, both on the concatenation [F_clean ; R]. Their architecture (layers, kernels, dilation, widths) is not stated; the only remark is that they use the standard Kaiming initialisation. Weights w_s, w_g are scalars; how they are obtained (learned parameters or predicted) is not stated.
- E6: `dF = w_s * E_s([F_clean ; R]) + w_g * E_g([F_clean ; R])`
- E7: `G = sigmoid( Conv([F_clean ; R]) )` (kernel not stated)
- E8: `F_out = F_shift + G (.) dF`
- The paper says the gate should open only where R is near 0. Nothing forces that: G also sees F_clean, and no loss ties G to R.

### 2.5 Identity initialisation: how exactly
| block | initial setting (paper) | value at start | exact identity? |
|---|---|---|---|
| 1 | scalar a = -5.0; MLP, GN and gamma/beta init not stated | sigmoid(a) = 0.0067 (paper); output = 0.9933 F + 0.0067 (gamma*GN(F)+beta) | no, near identity: the branch is down-weighted, not removed |
| 2 | bias of the last layer of ConvNet = +4.0 | R about 0.98 (sigmoid(4), my arithmetic) | no; only matters for F_clean, which does not reach the output directly |
| 3 | bias of the gate conv = -4.0; experts Kaiming, last layer not zeroed | G about 0.018 (sigmoid(-4)) | no: 1.8 % of a random expert output is added at step 0 |
The paper states in the Section III introduction that the whole module is mathematically an identity transform at initialisation; with the values above that is an approximation, not an identity. Exact identity would need a zero-initialised last layer or a zero gate (our option, not the paper's). Training moves the clean score anyway: see Table IV, rows A1/A2 (BEVFusion clean mAP 68.7 / 68.9 against 69.2), so identity at step 0 does not bound the change after training; the anchor, the 49 % clean data and Block 3 bring it back to 69.5.

### 2.6 Losses
- L_det = the host's own detection losses (focal and L1), back-propagated through the frozen head into PFS in every stage. No feature-regression loss to clean features, no distillation, no auxiliary heads.
- Stage 2 adds lambda_rel * L_rel (E3-E4, weights above). Stage 1 and Stage 3: detection loss only is stated (whether L_rel stays on in Stage 3 is not stated).
- No assignment / matching in the module; the host's own target assignment is used and is not described.

### 2.7 Curriculum and corruption recipe (Section III-D, IV-A1, IV-A2, Table I)
Host frozen in all stages. AdamW (named for Stage 1; later stages not stated), cosine annealing (Stage 1), batch size 4, gradient clipping max norm 5.0 (all stages except Stage 1), 14 epochs in total (6 + 4 + 4).

| stage | trained | epochs | lr | corruption pool |
|---|---|---|---|---|
| 1 shift normalisation | Block 1 | 6 | 2e-4 | a limited pool of global corruptions: Rain Lens, Low Light, Beam Reduction, Miscalibration |
| 2 reliability calibration | Blocks 1 + 2 jointly | 4 | 1e-4 | full eight-corruption suite |
| 3 expert correction | Block 3 (Blocks 1, 2 frozen) | 4 | 2e-4 | not stated |

- **Sampling:** camera-side and LiDAR-side corruptions are drawn independently with p = 0.30 each, giving 49 % clean, 21 % camera only, 21 % LiDAR only, 9 % both (consistent: 0.7*0.7, 0.3*0.7, 0.7*0.3, 0.3*0.3). Which corruption type is drawn within a side, and which severity (Table I defines L1-L3), is **not stated**; the text says Level 3 only for evaluation.
- **Generation:** the paper names no library and releases no code; corruptions are applied to raw inputs (all six images / the point cloud; implied, not explicit) with a fixed per-sample seed so every method sees the same corrupted samples.

Table I (severity levels, as printed):

| side | corruption | L1 | L2 | L3 | definition (paraphrase) |
|---|---|---|---|---|---|
| camera | Rain Lens (haze alpha / drops) | 0.15 / 80 | 0.45 / 200 | 0.70 / 350 | global haze, quadratic-falloff droplets, vertical smears (droplet sub-parameters not stated) |
| camera | Low Light (gamma / sigma) | 1.8 / 0.018 | 3.2 / 0.045 | 5.5 / 0.10 | power-law darkening I_dark = I^gamma plus shot noise scaled by sqrt(I_dark + 1e-4) |
| camera | Occlusion (area) | 15 % | 35 % | 60 % | opaque, centre-biased masks over the horizon |
| camera | Dropout (views N) | 1 | 3 | 6 | N camera views zeroed (which views: not stated) |
| LiDAR | Beam Reduction (step k) | 2 (16-ch) | 4 (8-ch) | 8 (4-ch) | keep every k-th elevation bin |
| LiDAR | Range Drop (P_max @ r_max) | 0.70 @ 50 m | 0.85 @ 45 m | 0.95 @ 40 m | piecewise-linear drop probability reaching P_max at r_max |
| LiDAR | Sector Drop (azimuth) | 30 deg | 60 deg | 90 deg | remove points in an azimuth sector |
| LiDAR | Miscalibration (rot, trans) | 0.23 deg, 0.04 m | 0.85 deg, 0.08 m | 2.0 deg, 0.20 m | rigid extrinsic perturbation (which extrinsic: not stated) |

Evaluation: Table II at Level 3 for all cases, same eight corruptions as in training (the paper does not explicitly confirm that the evaluation instances differ from training instances). "Camera-only" tests corrupt all six images with clean LiDAR; "LiDAR-only" tests corrupt the points with clean images. Metrics mAP and NDS of nuScenes (centre distances 0.5, 1, 2, 4 m), official train / val split.

### 2.8 Training budget, parameters, inference cost
- Budget: 14 epochs, batch 4, lr 1e-4 to 2e-4. Number of iterations, hardware and wall-clock time: **not stated** (the only GPU named is the RTX 4080 Super used for latency). Training set size for PFS is not stated.
- Parameters: 99 K + 664 K + 2.5 M = 3.3 M; no per-expert breakdown.
- Inference: one forward pass, soft masks, no thresholds, no clean reference, no temporal state, no test-time adaptation (future work only).
- Cost (Table V, section IV-B4, RTX 4080 Super, single-sample inference per the digest): BEVFusion 70.9 -> 76.6 ms (14.1 -> 13.1 FPS), printed overhead 8.1 % (from the printed latencies 5.7/70.9 = 8.0 %, my arithmetic); UniBEV 417.0 -> 436.0 ms (2.4 -> 2.3 FPS), printed 4.5 %. Same 3.3 M parameters added in both hosts but 5.7 ms against 19 ms (my arithmetic), so the absolute cost depends on the host's BEV tensor and cannot be carried over to another grid. Memory, batch size > 1, input size: **not stated**.

## 3. Numbers
All values author-reported, nuScenes val, detection mAP (NDS where given), digest. The paper's own table compares methods on one protocol; nothing here is comparable with other papers or with our numbers.

### 3.1 Table II - sensor corruptions and failures, level 3 (mAP / NDS; (r) = marked with a dagger in the paper: reproduced by the authors from open code)
Columns: Clean; LiDAR: Beam Reduction (8 beams), Range Dropout (95 % @ 40 m), Sector Dropout (90 deg), Miscalibration (2.0 deg, 0.2 m); camera: Rain Lens (350 drops), Low Light (gamma 5.5), Occlusion (60 % area), Dropout ("6 drops").

| row | Clean | Beam Red. | Range Drop | Sector Drop | Miscalib. | Rain Lens | Low Light | Occlusion | Dropout (6) |
|---|---|---|---|---|---|---|---|---|---|
| BEVFusion (r) | 69.2 / 59.7 | 12.1 / 24.8 | 52.6 / 50.0 | 54.7 / 52.4 | 46.8 / 44.1 | 60.0 / 55.0 | 58.8 / 54.4 | 64.6 / 57.3 | 62.6 / 56.3 |
| BEVFusion + PFS | 69.5 / 71.2 | 17.5 / 27.1 | 53.4 / 60.6 | 56.9 / 63.2 | 47.5 / 55.5 | 63.6 / 68.3 | 63.2 / 64.9 | 65.3 / 69.0 | 63.8 / 68.7 |
| UniBEV (r) | 65.3 / 57.2 | 25.6 / 29.9 | 53.2 / 50.0 | 56.5 / 52.2 | 48.0 / 45.1 | 56.5 / 52.8 | 55.4 / 52.3 | 61.5 / 55.5 | 56.7 / 53.1 |
| UniBEV + PFS | 65.3 / 57.2 | 33.8 / 35.5 | 53.4 / 50.1 | 60.7 / 66.5 | 52.9 / 59.4 | 58.1 / 62.8 | 58.9 / 60.9 | 64.7 / 63.6 | 58.4 / 64.0 |
| CMT (r) | 70.1 / 72.9 | 32.6 / 45.7 | 58.5 / 65.2 | 60.7 / 67.0 | 50.1 / 58.0 | 58.9 / 66.6 | 55.4 / 64.3 | 64.1 / 70.2 | 62.0 / 68.6 |
| DeepInteraction (r) | 68.9 / 58.3 | 7.74 / 22.6 | 49.9 / 47.84 | 53.6 / 50.2 | 45.2 / 42.0 | 54.4 / 49.1 | 58.8 / 51.6 | 59.5 / 52.7 | 53.7 / 47.3 |
| SparseFusion (r) | 70.5 / 72.5 | 5.2 / 23.3 | 68.2 / 71.2 | 32.4 / 50.8 | 47.2 / 56.1 | 60.4 / 66.9 | 55.9 / 64.4 | 64.4 / 70.7 | 55.4 / 63.8 |
| MoME (r) | 72.1 / 74.4 | 23.1 / 41.4 | 60.0 / 66.8 | 61.2 / 67,6 (comma as printed) | 52.6 / 60.2 | 50.7 / 62.7 | 44.6 / 59.5 | 64.1 / 69.7 | 63.8 / 69.8 |

Gains of PFS in mAP points (my arithmetic from the table) and the share of the host's own drop that is recovered (gain / (host clean - host corrupted)):

| case | BEVFusion gain | share of drop | UniBEV gain | share of drop |
|---|---|---|---|---|
| Rain Lens | +3.6 | 39 % | +1.6 | 18 % |
| Low Light | +4.4 | 42 % | +3.5 | 35 % |
| Occlusion | +0.7 | 15 % | +3.2 | 84 % |
| Dropout (6 cameras) | +1.2 | 18 % | +1.7 | 20 % |
| Beam Reduction | +5.4 | 9 % | +8.2 | 21 % |
| Range Dropout | +0.8 | 5 % | +0.2 | 2 % |
| Sector Dropout | +2.2 | 15 % | +4.2 | 48 % |
| Miscalibration | +0.7 | 3 % | +4.9 | 28 % |
| Clean | +0.3 | - | 0.0 | - |

NDS caveat (my arithmetic from the table): the NDS gains of PFS are large and not tied to the mAP gains. BEVFusion: +10.5 to +13.3 NDS in every column except Beam Reduction (+2.3), including clean (59.7 -> 71.2, +11.5, for +0.3 mAP) and Dropout (56.3 -> 68.7, +12.4, for +1.2 mAP). UniBEV: +5.6 to +14.3 in seven columns, but 0.0 on clean and +0.1 on Range Dropout. The reproduced BEVFusion and DeepInteraction NDS values are about 13-15 points below CMT, SparseFusion and MoME in the same table, although their mAP is in the same range. The paper does not explain this. Use the mAP columns; treat NDS as unexplained (a protocol difference between the host reproduction and the PFS run is possible, not stated).

### 3.2 Table III - nuScenes-C weather (mAP, as printed; the table has no mAP label, the text calls them mAP; severity not stated; modality C = camera, L = LiDAR)
| method | mod. | Fog | Snow | Sunlight |
|---|---|---|---|---|
| PGD / FCOS3D / DETR3D / BEVFormer | C | 12.8 / 13.5 / 27.9 / 32.8 | 2.3 / 2.0 / 5.1 / 5.7 | 22.8 / 17.2 / 41.6 / 41.7 |
| PointPillars / SSN / CenterPoint | L | 24.5 / 41.6 / 43.8 | 27.6 / 46.4 / 55.9 | 23.7 / 40.3 / 54.2 |
| FUTR3D | LC | 53.2 | 52.7 | 57.7 |
| Transfusion | LC | 53.7 | 63.3 | 55.1 |
| BEVFusion | LC | 54.1 | 62.8 | 64.4 |
| DeepInteraction | LC | 54.8 | 62.4 | 64.9 |
| CMT | LC | 66.3 | 62.6 | 63.6 |
| MoME | LC | 67.9 | 63.5 | 65.2 |
| BEVFusion + PFS | LC | 67.9 | 65.5 | 65.6 |
| UniBEV + PFS | LC | 65.7 | 63.6 | 61.3 |
There is no UniBEV baseline row in Table III. The text gives +25.5 % (Fog), +4.3 % (Snow), +1.9 % (Sunlight) for BEVFusion: these are relative gains (13.8/54.1, 2.7/62.8, 1.2/64.4; my arithmetic), not a contradiction with the table. The abstract's "+1.2 %" and "+4.4 %" are absolute mAP points. Fog, Snow and Sunlight come from nuScenes-C, a different corruption generator from the eight-corruption suite of Table I; the paper does not list them as training corruptions.

### 3.3 Table IV - block ablation (cumulative; mAP / NDS)
| host | config | Clean | Rain Lens | Low Light | Beam Red. | Miscalib. |
|---|---|---|---|---|---|---|
| BEVFusion | A0 base | 69.2 / 59.7 | 60.0 / 55.0 | 58.8 / 54.4 | 12.1 / 24.8 | 46.8 / 44.1 |
| | A1 + Block 1 | 68.7 / 60.0 | 61.6 / 56.0 | 61.4 / 56.7 | 11.9 / 25.5 | 46.8 / 44.6 |
| | A2 + Blocks 1,2 | 68.9 / 60.0 | 62.3 / 56.5 | 61.8 / 57.0 | 14.9 / 25.0 | 47.0 / 44.6 |
| | A3 + all | 69.5 / 71.2 | 63.6 / 68.3 | 63.2 / 64.9 | 17.5 / 27.1 | 47.5 / 55.5 |
| UniBEV | A0 base | 65.3 / 57.2 | 56.5 / 52.8 | 55.4 / 52.3 | 25.6 / 29.9 | 48.0 / 45.1 |
| | A1 + Block 1 | 65.3 / 57.2 | 56.9 / 53.2 | 57.6 / 53.4 | 26.4 / 30.0 | 48.2 / 45.1 |
| | A2 + Blocks 1,2 | 65.2 / 57.2 | 57.1 / 56.5 | 57.6 / 56.8 | 26.5 / 30.0 | 48.2 / 45.1 |
| | A3 + all | 65.3 / 57.2 | 58.1 / 62.8 | 58.9 / 60.9 | 33.8 / 35.5 | 52.9 / 59.4 |
The table has no camera Dropout or Occlusion column, no single-block-alone rows, no "Block 3 without Block 2", no identity-initialisation off, no plain fine-tune of an unconstrained module of equal size, and no seeds. Increments in mAP points (my arithmetic, BEVFusion): Low Light +2.6 (Block 1), +0.4 (Block 2), +1.4 (Block 3); Rain Lens +1.6, +0.7, +1.3; Beam Reduction -0.2, +3.0, +2.6; Miscalibration 0.0, +0.2, +0.5; Clean -0.5, +0.2, +0.6. UniBEV: Low Light +2.2, 0.0, +1.3; Rain Lens +0.4, +0.2, +1.0; Beam Reduction +0.8, +0.1, +7.3; Miscalibration +0.2, 0.0, +4.7. The text: Block 1 corrects global drift, Block 2 helps most under LiDAR beam loss on BEVFusion (12.1 -> 14.9, +23.1 % relative), Block 3 gives the largest gains and recovers clean accuracy; UniBEV shows the strongest synergy for beam loss (+32.0 % relative) and miscalibration (NDS 45.1 -> 59.4).

### 3.4 Table V - real-world physical test, no fine-tuning, no recalibration (200 synchronised frames)
Platform: lab vehicle, one forward RGB camera, roof 32-beam LiDAR, 10 Hz, nearest-timestamp matching with 50 ms tolerance; clear daytime against low-light evening urban driving. How the missing five cameras of the hosts are handled: not stated.

| method | mAP day | mAP night | params (M) | latency (ms) | FPS | overhead |
|---|---|---|---|---|---|---|
| BEVFusion | 32.87 | 24.08 | 40.8 | 70.9 | 14.1 | - |
| + PFS | 35.33 | 29.20 | 44.1 | 76.6 | 13.1 | 8.1 % |
| UniBEV | 33.87 | 24.75 | 72.7 | 417.0 | 2.4 | - |
| + PFS | 36.65 | 28.83 | 76.0 | 436.0 | 2.3 | 4.5 % |
Text-versus-table mismatch (digest, both versions of the numbers seen): for UniBEV the text gives 34.17 / 25.63 (baseline) and 36.85 / 30.41 (PFS), gains +2.68 / +4.78, while Table V gives 33.87 / 24.75 and 36.65 / 28.83, gains +2.78 / +4.08 (my arithmetic). BEVFusion text and table agree (+2.46 day, +5.12 night).

### 3.5 What the paper says about the weak camera-dropout gain, and what the design suggests
- **Paper:** the 6-camera dropout gain for BEVFusion (62.6 -> 63.8) is described as matching MoME and CMT (Table II: MoME 63.8, CMT 62.0 on the same column); occlusion (+0.7) and BEVFusion miscalibration (+0.7) are small; the prose gives **no explanation** and has no limitations section. The only explanation offered anywhere is that LiDAR-side performance is limited by the host's geometry extraction.
- **Design facts (digest):** the reliability map is supervised with LiDAR point-count ratios only (E5); a camera-only corruption does not change the points, so T = 1 and the anchor pulls R to 1 on the 70 % of samples without LiDAR corruption. The map is therefore never trained to flag a camera failure; the paper reports no accuracy of R for any case. Future work lists a per-channel reliability and temporal context (a recurrent or transformer aggregator for transient dropouts) and test-time adaptation using the Anchor Loss as the label-free signal.
- **Inference (mine):** (1) with all six cameras zeroed the host runs on its other branch and the gap to clean is the smallest of the camera cases (6.6 mAP for BEVFusion), so the headroom is small and the recovered share is 18 %; (2) nothing in the module can create camera information that was never encoded; (3) the camera-side gains that exist (Low Light, Rain Lens) come mostly from Block 1 and Block 3 (increments in 3.3), i.e. from a global statistic correction and a learned residual, not from the reliability map; (4) Block 2 pays off where its target is defined, LiDAR density (BEVFusion beam loss +3.0 in the A1 -> A2 step). The paper does not test any of these readings.

### 3.6 Caveats
- One run per cell, no seeds, intervals or significance statement found in the digest; differences below about 1 mAP point cannot be separated from noise on this evidence.
- Training and Table II evaluation use the same eight-corruption suite (Table I); the held-out evidence is nuScenes-C (Table III, other generator, no severity stated, no UniBEV baseline) and the 200-frame physical test (one camera, one vehicle, no pre-registration of the frame choice). No held-out corruption type inside Table II.
- Severity mix of the training corruptions and the sampling of types are not stated, so the recipe cannot be rebuilt exactly.
- Baselines marked with a dagger were reproduced by the authors; NDS anomalies in 3.1; MoME sector NDS printed with a comma; the printed Range-Dropout NDS of both baseline hosts is 50.0 (may be a coincidence or a typo).
- Reference [12] is titled as a general multimodal model paper but is cited as TransFusion; a cross-reference to the corruption details points at Table III (digest). Editing slips like these lower trust in unverified cells.

## 4. Paper versus code
No code. Search (arXiv abstract and HTML pages, alphaXiv overview, two web searches) found no repository, no project page and no code-availability statement; the paper cites MMDetection3D (the baselines were built on it). Licence: not applicable; the paper's HTML carries the arXiv perpetual non-exclusive licence (seen in a search result title). Stack implied: PyTorch inside MMDetection3D, nuScenes. No `../code/` file was written. If the authors release code later, check the licence file before reading: without a licence file or under GPL-3 / AGPL it is read-only.

## 5. Transfer to our pipeline
| question | answer |
|---|---|
| what it needs from us | the frozen cached fused map (200 x 200 x 39) as F_fused; per-frame pairs of clean and failure features for the training scenes (our failure series gives clean/failure halves and a matched clean reference; coverage on the training split must be checked, see 6); gradients through the frozen head(s): the dense head first, the chain later (a route loss into the builder gave nothing in E214, so a dense-head or graph-loss that already trains is the cheapest source); for Block 2 a per-cell damage target that does not use LiDAR |
| what it gives us | stage A8 (sensor failure): a repair of the 39-channel map before all downstream heads, one module for all of them; a per-cell damage raster R that could feed the existing gates (idea only) |
| conflicts | (1) the reliability target is LiDAR point counts: no LiDAR in the cache (the radar equivalent exists only for radar loss); (2) Block 2 expects the host's LiDAR branch tensor as a second input: we only have the fused map, i.e. the paper's duplicate-input fallback, which the paper never evaluates separately; (3) detection losses and the host's target assignment (we use our own dense / graph losses, no matching); (4) hosts have a strong redundant sensor (LiDAR) behind a lost camera, our radar adds little to lane geometry, so the "fall back on the other branch" regime does not exist for us; (5) identity is approximate in the paper (2.5); (6) no conflict with the differentiable-path rule: the module is a pixel-based raster transform of convolutions, pooling, GroupNorm and sigmoids (no argmax, no matching, no rule step) |
| reuse mode | re-implement from description (no code); keep Block 1 and the gate trick, replace the Block 2 target and make the identity exact |
| effort | M (days). Blocker: paired failure caches over enough training scenes and a Block 3 design (the paper gives none, we choose it) |
| expected gain and risk | gain is plausible for global shifts (dark, fog, blur) through Block 1, as in the paper's low-light and rain cases; small for lost cameras and extrinsic error (the paper's own camera-dropout and miscalibration cases); the loop-17 head-side result already is "fine-tuning on failure frames helps night / blur, nothing for lost cameras / extrinsics" (level-1 note), so the open question is whether a feature-side module in front of frozen heads beats that at equal data and budget |
| feeds which experiment / K id | proposed new technique, to be registered in `../register.md` before use (no K id exists): identity-gated feature repair between the frozen BEV and the heads. Variants worth separate rows: (a) exact-identity initialisation (zero-initialised last layer of the correction and a zero gate), (b) R trained on a paired-feature damage target (below), (c) optional feature-consistency loss to the paired clean features. Related, not read for this file: [RB02](../papers/RB02_modalpatch-learned-history-based-feature.md), [RB06](../papers/RB06_grace-bev-trust-gate-failsafe.md), [RB26](../papers/RB26_ncgr-noise-conditional-gated-rectification.md); row 2609.35264 in `../papers.csv` (zero-initialised per-channel residual repair, same recipe, separable inputs) |

### 5.1 What transfers
- **Placement and the frozen-head contract:** module on the raster, same shape in and out, head untouched, trained with the head's own loss. This is exactly our interface.
- **Block 1 as built** (global average pool, small MLP to per-channel gamma and beta, GroupNorm branch, learnable scalar gate): cheap, no dependence on LiDAR; with C = 39 the MLP is tiny. Expected to target night, fog and blur.
- **Gated residual correction (Eqs. 6-8):** F_out = F_shift + G * dF; with a spatial gate G the module can act only where needed.
- **Independent per-side corruption sampling with p = 0.30** (49 / 21 / 21 / 9 % mix) and a clean-anchor term: keeps clean quality; our failure series has the same two sides (mostly camera-side: dark, blur, fog, cam_front_off, cam_front3_off and the extrinsic error extr3, for which PROTOCOL.md does not say which sensor is perturbed; radar side: radar_off).
- **Staged curriculum** (global shift first, then reliability, then correction with the earlier blocks frozen): easy to reproduce on cached features, each stage is a short run.
- **Reliability target from a paired damage ratio** (Eq. 5): the idea carries over if the ratio is computed on something we have. Two candidates (mine): radar return count per cell from the nuScenes radar files (clean against radar-dropped frames; covers radar_off only), and a feature-space damage ratio from the paired caches, e.g. clamp(norm of failure feature / norm of clean feature, 0, 1) or one minus cosine similarity per cell (covers every failure type for which a paired clean cache exists). Both are untested.

### 5.2 What does not transfer
- The LiDAR point-count target and the LiDAR branch input of Block 2; the 108 x 108 grid.
- The evaluation: detection mAP / NDS, nuScenes-C weather, the physical test; no number of this paper says anything about thin structures such as lane centre lines at 0.5 m cells, far range, or graph output.
- The regime of the camera-dropout case: a host that keeps a strong second sensor.
- The claim of an exact identity at start; the latency numbers (host-specific tensors, other GPU); Block 3 as a recipe (architecture unspecified).
- Anything about extrinsic error: the paper's miscalibration gains are small on BEVFusion and larger on UniBEV, and the module has no geometric operator.

## 6. Not found / open
Not stated in the paper (digest): C, H, W of the tapped tensor and the layer that is tapped; what F_lidar is; MLP activation, GroupNorm groups; ConvNet widths and upsampling factor; expert architecture, number of layers, kernels, dilation, widths; how w_s, w_g are computed; gate kernel; whether Block 2 sees F_lidar or the duplicated fallback in the experiments; what the head receives in the A2 configuration; training pools of Stage 3; severities and types actually sampled in training; which cameras are dropped and whether all six are meant by "6"; which extrinsic is perturbed; optimiser, scheduler and clipping of Stages 2-3; training set size, iterations, hardware, wall-clock time; memory; batch size of the latency test; seeds; severity of nuScenes-C; a plain fine-tune or an unconstrained-module control; held-out corruption types; real-world handling of the five missing cameras; code.

Open for us: (1) do the paired failure caches (dark, blur, fog, cam_front_off, cam_front3_off, radar_off, extr3) exist for the training scenes, in how many folds, and can combined camera + radar faults be generated (the paper trains on 9 % both)? (2) is a radar-count target enough for radar_off, and does the paired-feature damage map track real damage (checked against the per-cell loss in the head output)? (3) does the repair help a head that was already fine-tuned on failure frames (E197 / E214 / E220), or only the clean-trained head? (4) cost at 200 x 200 x 39: one 3 x 3 convolution with 64 input and 64 output channels at this size is about 1.5 GMAC (my arithmetic), so Block 3 experts must be narrow or run at half resolution to stay within a few ms; measure ms per frame on the idle GPU at batch 1.

## 7. Our experiments that touch it
No run reproduces PFS. Nearest work, all failure-side remedies on frozen heads (verdicts from `../../HANDOVER.md` and the level-1 note; the experiment folders only point to the loop-17 report and `experiment_execution/loop17_recovery`):
- E193: failure series with the champion and control ([E193](../../experiments/E193_champion-d7-control-failure-series-plus/README.md)); the series definition is in [PROTOCOL.md](../../evaluation/PROTOCOL.md) (line 22): dark, blur, cam_front3_off, fog, radar_off, extr3, cam_front_off; per scene first half clean, second half one failure type.
- E195 hold / EMA memory on the input, E199 learned gated memory (12.5 k parameters), E207 learned gate behind the map-conditioned builder: input-side or gate-side remedies; numbers in the loop-17 report, not re-read here.
- E197 / E198: failure-trained dense head and failure-frame fine-tune of the builder; level-1 verdict: helps night / blur, nothing for lost cameras / extrinsics.
- E209: the map benefit transfers to failure types the builder never saw (night / blur held out): the holdout design to copy.
- E214 / E220: joint fine-tune with failure frames: the route loss into the builder gives nothing; failure-frame fine-tune of route head and builder lifts failure-half route success, mostly from the head, and only a little in the best chain; E222: no planner-level effect.

## 8. Testing recipe for us (proposal; the card with prediction and decision rule is written before the run)
Layers: 0 self-test of the module (exact identity at start: head output bit-equal or within 1e-6; shapes; runs on batch 1); 1 reproduce a stored head score on clean frames through the identity module; 2 sanity runs (null, shuffled pairing, clean-only training must not drift); 3 DEV / VAL250 proxy on the dense head; 4 VAL protocol with statistics on the chain.
- **Perturbation matrix** (our series, grouped by the mechanism PFS addresses): global drift: dark, blur, fog; spatial hole: cam_front_off, cam_front3_off; modality loss: radar_off; geometry: extr3; plus clean. Training mix as in the paper (independent camera-side and radar-side faults, p = 0.30, clean anchor), once combined faults exist. Hold out one type per group at training time (copy E209: night and blur held out), evaluate on all. If severities can be cached, add an L1-L3 sweep for dark and cam_front3_off.
- **Controls (equal data, equal steps, equal parameter count, 2 seeds each):** (C0) identity module (null); (C1) same architecture, random initialisation, no gates, no reliability head, plain task-loss fine-tune of the module (the "plain fine-tune" control); (C2) head-side fine-tune on the same failure frames (existing E197 / E214 recipe); (C3) cumulative ablation Block 1 / +2 / +3 as in Table IV, and Block 3 alone; (C4) Block 2 with a shuffled target (damage map of another frame) to test that R carries information; (C5) matched clean reference on the same frames (head on the paired clean features: a sanity level, not a bound).
- **What to report:** Q and S on VAL (clean half and failure half separately, seen and held-out types separately), paired scene bootstrap, noise band 0.010 in S; F1@1 for the dense head; R diagnostics (mean R on clean frames close to 1; rank correlation of R with the per-cell damage on failure frames); ms per frame at batch 1 on the idle GPU with cached features, parameters of the module, clean regression as its own number.
- **Decision-rule skeleton:** KEEP if the failure-half Q gain over C2 (and over C1) exceeds the noise band with the bootstrap interval excluding 0, the clean half stays within the band of C0, and the added ms is below the budget set on the card; TIE (take the cheaper variant) inside the band; INCONCLUSIVE means add a seed. The numeric prediction goes on the card.
- **Order:** Block 1 alone first (hours, targets dark / fog / blur and needs no new target), then Block 2 target and Block 3 only if Block 1 leaves a gap on the spatial-hole group.

## 9. Novelty and relevance for our problem
The novelty is the combination of a frozen host, a near-identity module on the fused BEV tensor, a staged curriculum with a clean anchor, and a supervised reliability map; the parts are standard (affine normalisation, soft masks, gated residuals) and the authors hedge the novelty claim themselves. For us it is the closest published match to the interface (frozen fused raster in, same-shaped raster out, head untouched) and a ready recipe for how to mix clean and corrupted data and how to stage the training; the evidence is detection on nuScenes with a LiDAR-backed host, single runs, the same corruption suite in training and test, and a reliability map that the paper's own design leaves blind to camera failures. Its value for us is therefore the recipe and the controls we add (exact identity, a plain-fine-tune control, held-out failure types, a radar- or feature-based damage target), not its numbers.

## 10. Corrections and additions for the level-1 note (note not edited)
1. "formulation ... exact loss form not stated": now known: L_total = L_det + 1.0 * L_rel in Stage 2, L_rel = BCE(R, T) on corrupted samples + 0.2 * BCE(R, 1) on clean samples, T = clamped ratio of corrupted to clean LiDAR point counts per cell on a 108 x 108 grid; detection loss is the host's focal + L1; Stages 1 and 3 use detection loss only (E3-E5).
2. "input: fused BEV feature map only": Block 2 also takes the host's raw LiDAR BEV feature (duplicate fallback otherwise), and the reliability map is a LiDAR-density map; it is not supervised for camera failures. Add this to compat_conflicts.
3. "training: ... nuScenes-C fog, snow, sunlight": those are evaluation conditions (Table III), not part of the eight-corruption training suite listed in Table I (camera: rain lens, low light, occlusion, dropout; LiDAR: beam reduction, range dropout, sector dropout, miscalibration).
4. "paper text and Table III disagree on snow / sunlight gains": they do not; the text gives relative gains (+4.3 %, +1.9 %), the table gives +2.7 and +1.2 points. The genuine text/table mismatch is the UniBEV physical-test row (34.17 / 25.63 / 36.85 / 30.41 in the text against 33.87 / 24.75 / 36.65 / 28.83 in Table V).
5. compute: add UniBEV 72.7 -> 76.0 M parameters; GPU RTX 4080 Super (single-sample); printed overhead 8.1 % and 4.5 %; the absolute added time differs by host (5.7 ms against 19 ms).
6. "identity-initialised": only approximately (sigmoid gates at 0.0067 and 0.018, reliability bias +4); Block 1 and 2 lower BEVFusion clean mAP before Block 3 restores it (Table IV: 68.7, 68.9, 69.5 against 69.2).
7. "6-camera dropout gain is small": still true (+1.2 BEVFusion, +1.7 UniBEV), but the paper offers no reason; the design reasons are in 3.5. The NDS columns of the BEVFusion rows are unexplained; quote mAP.
8. "code: none found on the page": confirmed on the abstract page, HTML, alphaXiv and two searches.
9. Suggested wording for the RB01 statement in `../papers.csv` (the parent session decides): "Near-identity module between a frozen fused BEV map and an unchanged head: global shift normalisation, a LiDAR-density reliability map, gated expert residual, three-stage curriculum with an independent p = 0.30 corruption mix; largest gains for low light, rain lens, beam loss and fog, small for 6-camera dropout. Closest interface match to ours, but its reliability map is trained on LiDAR point counts only, identity is approximate, no seeds, no plain-fine-tune control, no code. Use: re-implement Block 1 and the gated residual on cached rasters with an exact-identity start and a paired-feature damage target."
