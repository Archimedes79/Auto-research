# VM01 - StreamMapNet: full-text extraction

| field | value |
|---|---|
| paper | Streaming Mapping Network for Vectorized Online HD Map Construction; Tianyuan Yuan, Yicheng Liu, Yue Wang, Yilun Wang, Hang Zhao; WACV 2024 (pp. 7356-7365 per the repository citation); arXiv 2308.12570. Links opened: https://arxiv.org/html/2308.12570, https://ar5iv.labs.arxiv.org/html/2308.12570, https://arxiv.org/abs/2308.12570 (abstract page: v1 2023-08-24, v2 2023-08-27 = current), repository pages and raw files listed in the code extraction |
| axis, tier, K ids | A5 (also A9 for the splits); tier 1 core; K45 (temporal map propagation, memory); new technique ids proposed in section 5 (not yet in the register) |
| level-1 note(s) | [VM01](../papers/VM01_streammapnet.md) (home), [SN18](../papers/SN18_streammapnet-bev-level-gru-fusion.md), [A04](../papers/A04_streammapnet-bev-warp-gru-and.md), [RB35](../papers/RB35_temporal-redundancy-streammapnet-and-a.md); split context in [VM46](../papers/VM46_geographically-disjoint-splits.md) |
| code extraction | [streammapnet](../code/streammapnet.md) (GPL-3.0: read only) |
| sources read | arXiv HTML of v2 (method 3.1-3.5, experiments 4.1-4.6, conclusion, appendix A / B; all tables 1-6), ar5iv copy for a second read of section 3.4, abstract page; repository: README, file tree, config files `nusc_newsplit_480_60x30_24e.py` and `av2_newsplit_608_60x30_30e.py`, model / head / sampler / converter / evaluation files (list in the code extraction); Roddick and Cipolla split file (mono-semantic-maps, `src/data/nuscenes/splits.py`), 2026-10-24. The arXiv PDF came back as binary and was not used. No supplementary material exists on the pages seen |
| reading marks | every section below is **digest**: all pages were read through the summarising fetch tool (it returned structured walkthroughs, not the raw text); equations and numbers are as the tool transcribed them. Items that I derived by arithmetic or by combining two digests are marked "(derived)". Nothing in this file is **memory** unless stated |
| extracted by, date | extraction sub-agent (Claude), 2026-10-24 |

## 1. Problem and claim
Input: a stream of synchronised six-camera frames (nuScenes) or seven ring cameras (Argoverse 2, AV2) with ego poses; output per frame: a set of vectorised map instances (pedestrian crossing, lane divider, road boundary), each a class and a polyline of N_p = 20 ordered points in the ego frame. Prior single-frame detectors (VectorMapNet, MapTR) degrade at a larger range (100 x 50 m against 60 x 30 m) and flicker under occlusion. The paper adds (1) multi-point attention (all 20 previous-layer points are the deformable-attention reference points, so cost stays O(N_p)), (2) streaming temporal fusion with two memories: top-k query propagation with transformed reference points, and an ego-warped BEV feature fused by a GRU, and (3) new non-overlapping train / val splits for nuScenes and AV2. The one claim that matters for us: a recurrent memory of the previous frame's own outputs (queries + warped geometry) and of the previous BEV state gives large gains on the geographically cleaner splits (nuScenes new split 60 x 30 m: 33.9 mAP against 20.9 for MapTR, Table 2), at 13.2 - 14.2 FPS.

## 2. Method as built

### Pipeline / architecture (digest; sizes from the nuScenes 60 x 30 m config and the paper)
```
6 images 480x800 (AV2: 608x608)
  -> ResNet-50 (caffe style, DCN in stages 3-4, lr x0.1) + FPN (256 ch, 3 levels)
  -> 1-layer BEVFormer encoder (spatial cross-attention, 8 points, 3 levels)   -> BEV 256 x 50 x 100  (60 x 30 m, 0.6 m cells, derived)
  -> [memory 1] warp previous fused BEV by relative ego pose, ConvGRU(h = warped, x = current) + LayerNorm -> fused BEV (stored, detached)
  -> 1x1 conv 256 -> 512
  -> decoder, 6 layers, Nq = 100 queries x 512 d
        layer 0 : 100 learned queries (reference polylines from a linear map of the query embedding)
        [memory 2] after layer 0: keep top Nq - k = 67 new queries by layer-0 class logit, concatenate k = 33 propagated queries
                   (their reference polylines = previous polylines mapped by the relative pose)
        layers 1-5 : self-attention over all 100, multi-point deformable cross-attention (20 points x 1 level, 8 heads), FFN 1024
        per layer : class head (3 logits) and regression head (40 = 20 x 2 absolute coordinates, sigmoid, normalised to the ROI)
  -> per frame 100 instances; training: Hungarian matching at every layer; inference: scores + polylines
```
- **Inputs / outputs.** Per frame: images, camera intrinsics and extrinsics, ego-to-global translation and rotation, scene name. Output coordinates are normalised to the ROI and denormalised by x * (roi + eps) + origin with origin = -roi / 2 (ego frame; the first ROI entry is 60 m or 100 m, the second 30 m or 50 m). The set size is fixed (100); nothing in the output marks a padded or invalid element except the class score.
- **Memory state per sequence** (digest): query memory (33 x 512), reference-point memory (33 x 20 x 2, normalised ROI coordinates), target memory (33 x 20 x 2, training only), BEV memory (256 x 50 x 100 fused feature). All are stored detached.

### Equations (numbering as in the paper)
- E1 - E4 (original deformable attention): O = Offset_Embed(Q), W = Weight_Embed(Q), Q_i = sum_j W_j * DA(Q, R + O_j, F_BEV), R_next = sigmoid(sigmoid^-1(R) + Reg(Q)).
- E5 - E8 (multi-point attention): the sampling is summed over the N_p reference points P_i^j and N_off offsets each; P_next = sigmoid(Reg(Q)) (absolute coordinates, not residual). The ablation (Table 5 row c) says direct prediction is better than the relative one for the single-frame model.
- E9 query propagation: Q_t = phi(concat(Q_{t-1}, flatten(T))) + Q_{t-1}; T = 4 x 4 transform between the two frame coordinate systems, phi = MLP.
- E10 reference points: P_t = T * homogeneous(P_{t-1}), xy part only.
- E11 - E12 transformation loss: P_hat = Reg(Q_t); L_trans = sum over the N_p points of SmoothL1(P_hat^j, P_t^j).
- E13 - E14 BEV fusion: F~_{t-1} = Warp(F_{t-1}, T); F_t = LayerNorm(GRU(F~_{t-1}, F_t)) (GRU idea taken from Neural Map Prior).
- E15 line loss: L_line = min over the permutation group Gamma (MapTR) of (1/N_p) sum_j SmoothL1(p_hat_j, p_gamma(j)).
- E16 matching cost: L_match = lambda_1 * L_line + lambda_2 * L_focal. E17 total: L_train = lambda_1 * L_line + lambda_2 * L_focal + lambda_3 * L_trans.

### Losses
- Main: focal loss on classes (weight 5.0 in the config; alpha 0.25, gamma 2) and smooth-L1 line loss (weight 50.0, beta 0.01, divided by the number of points), both normalised by the number of positives, applied at each of the 6 decoder layers with equal weight (deep supervision, each layer with its own Hungarian matching).
- Assignment: Hungarian (scipy `linear_sum_assignment` on CPU) per image and layer; cost = 5 * focal cost + 50 * smooth-L1 line cost, the latter minimised over the permutations of the GT polyline (for N_p = 20, 2(N_p - 1) = 38 permutations per GT line, derived from the code). Background queries get no regression loss; background class weight 0 in the normaliser.
- Transformation loss: weight printed in the paper as "lambda_1 = 5.0" in the third slot of the list "50.0, 5.0, 5.0" (typo for lambda_3, digest). In the configs it is `trans_loss_weight = 0.1`, applied on top of the regression loss class whose own `loss_weight` is 50.0, which gives an effective factor 5.0 relative to the per-point-normalised smooth-L1 (derived; this reconciles the 5.0 of the paper and the 0.1 of the config - both level-1 notes were right at different levels). It is divided by the number of non-background propagated queries and added once (not per decoder layer).
- Target of the transformation loss: **see section 4**. The paper text (digest of E10 - E12 on two pages) reads as if the target were the previous frame's predicted polyline mapped by T; the released code uses the matched ground-truth polyline of the previous frame (stored per query slot, all-zero for queries that matched nothing, those rows get weight 0).

### Labels / ground truth
Vector map from the nuScenes map expansion (AV2: log map JSON): classes `ped_crossing` (polygon contour turned into a closed polyline after merging nearby crossings whose orientations agree within about 8 degrees), `divider` (lane dividers + road dividers as polylines), `boundary` (contour of the union of road segments and lanes, inset 0.2 m from the ROI edge so that the clipping edge is not a false boundary; the layer `drivable_area` is deliberately not used). GT polylines are resampled to N_p = 20 points evenly by arc length, normalised to the ROI, and expanded into 2(N_p - 1) point-order permutations (closed rings: all cyclic shifts of both directions; open lines: the line and its reverse, padded with a sentinel value 1e5). Evaluation GT uses the geometry simplified with tolerance 0.2 m and resampled to 200 points inside the evaluator. Details in the code extraction.

### Training recipe
- AdamW, lr 5e-4 (batch 32: 8 GPUs x 4), weight decay 0.01, backbone lr multiplier 0.1, linear warm-up 500 iterations from 1/3, cosine decay to a ratio of 3e-3, gradient clipping at norm 35, SyncBN. 24 epochs on nuScenes (870 iterations per epoch = 27,846 // 32; 20,880 iterations), 30 epochs on AV2 (750 per epoch, 22,500 iterations). Hardware: 8 x GTX 3090 (the page text reads "88 GPUs" and "batch size 3232", a rendering artefact). Training time: not found.
- Hyper-parameters fixed for all settings "without tuning": Nq = 100, N_off = 1, N_p = 20, k = 33 (config: int(Nq / 3)), loss weights 50 / 5 / 5.
- **Streaming schedule.** Memory is detached: no gradient reaches the previous frame (the store call clones and detaches). Sequence splitting: a custom infinite batch sampler gives every batch slot its own stream. For the first num_iters_to_seq iterations (4 epochs = 3,480 on nuScenes; AV2 config: total_iters // 6 = 3,750 = 5 epochs, which differs from the paper's "4 epochs") it yields single random frames (memory resets on nearly every step because the scene changes) = single-frame warm-up, taken from SOLOFusion. After that each scene (group) is cut at a random point into seq_split_num = 2 chunks, chunks are visited in shuffled order, and a slot advances one frame per iteration through its chunk; a new chunk (new scene) resets the slot. The paper says "randomly divide into 2 splits at the start of each epoch"; the code re-randomises each time a group has been consumed once (digest), nothing is keyed on the epoch counter.
- The propagated state is the model's own previous output (queries and polylines), so the recurrence is trained on-policy; only the transformation loss reaches back to ground truth.
- Augmentation: photometric distortion and grid mask on images; no geometric augmentation of the map or of the poses.

### Inference
- Frames of a scene are processed in order, batch 1 per GPU, memory slot reset when the scene name changes. After each frame: rank all 100 final-layer queries by the maximum class logit, keep the top 33 (queries, polylines), store them. No score threshold on the training path; evaluation uses all outputs ranked by score (mAP). The multi-GPU test sampler hands whole scenes to ranks round-robin and forbids shuffling.
- Hidden rule steps: (a) rank-based top-k with a fixed k, no confidence threshold, no age or decay; (b) reference points are clipped to [0, 1] of the ROI after the pose transform (points that left the window are squashed onto the border instead of dropped); (c) first-frame handling is a branch (all 100 new queries, zero prior); (d) no NMS or duplicate removal between propagated and new queries: suppression relies on self-attention trained by the one-to-one Hungarian assignment.

## 3. Numbers
(Digest of the arXiv HTML tables; protocols differ from ours and between rows; do not compare with our F1 / S.)

**Table 1, Argoverse 2 new split (R50, FPS as reported)**

| range | method | image | epochs | AP_ped | AP_div | AP_bound | mAP | FPS |
|---|---|---|---|---|---|---|---|---|
| 60x30 | VectorMapNet | 384x384 | 120 | 35.6 | 34.9 | 37.8 | 36.1 | 5.5 |
| 60x30 | MapTR | 608x608 | 30 | 48.1 | 50.4 | 55.0 | 51.1 | 18.0 |
| 60x30 | StreamMapNet | 608x608 | 30 | 56.9 | 55.9 | 61.4 | 58.1 | 14.2 |
| 100x50 | VectorMapNet | 384x384 | 120 | 32.4 | 20.6 | 24.3 | 25.7 | 5.5 |
| 100x50 | MapTR | 608x608 | 30 | 46.3 | 36.3 | 38.0 | 40.2 | 18.0 |
| 100x50 | StreamMapNet | 608x608 | 30 | 60.5 | 44.4 | 48.6 | 51.2 | 14.2 |

**Table 2, nuScenes new split**

| range | method | image | epochs | AP_ped | AP_div | AP_bound | mAP | FPS |
|---|---|---|---|---|---|---|---|---|
| 60x30 | VectorMapNet | 256x480 | 120 | 15.8 | 17.0 | 21.2 | 18.0 | 3.8 |
| 60x30 | MapTR | 480x800 | 24 | 6.4 | 20.7 | 35.5 | 20.9 | 16.0 |
| 60x30 | StreamMapNet | 480x800 | 24 | 29.6 | 30.1 | 41.9 | 33.9 | 13.2 |
| 100x50 | VectorMapNet | 256x480 | 120 | 12.0 | 8.1 | 6.3 | 8.8 | 3.8 |
| 100x50 | MapTR | 480x800 | 24 | 8.3 | 16.0 | 20.0 | 14.8 | 16.0 |
| 100x50 | StreamMapNet | 480x800 | 24 | 24.8 | 19.6 | 24.7 | 23.0 | 13.2 |

**Table 3, AV2 original split, 50 m:** VectorMapNet 30.2, MapTR 47.5, StreamMapNet 57.7 mAP. **Table 4, nuScenes original split, 30 m:** VectorMapNet 40.9 (110 epochs), MapTR 48.7, BeMapNet 59.8 (512x896, numbers taken from its paper because the code was not public), StreamMapNet 62.9 (24 epochs).

**Table 5, ablation (AV2 new split, 50 m range, mAP; no FPS or parameter columns)**

| row | variant | mAP |
|---|---|---|
| a | single-frame baseline with relative prediction | 33.7 |
| b | a, minus multi-point attention | no value (does not converge well) |
| c | + direct prediction | 41.7 |
| d | + query propagation without transformation loss | 42.8 |
| e | + transformation loss | 43.7 |
| f | + BEV fusion | 46.1 |
| g | + image size 608x608 (= full model) | 51.2 |

Reading for transfer: the temporal parts add 1.1 (query propagation) + 0.9 (transformation loss) + 2.4 (BEV fusion) = 4.4 mAP over the strong single-frame row c (derived); the bigger single step is the image size. Single runs, no seeds, no noise band. The paper has no ablation of k, of the memory length, or of the detach.

**Metrics.** AP with chamfer matching at {0.5, 1.0, 1.5} m (30 m range) and {1.0, 1.5, 2.0} m (50 m range), mean over thresholds and the three classes; ranges 60 x 30 m and 100 x 50 m, i.e. 30 / 50 m in front and behind, 15 / 25 m to each side. VectorMapNet and MapTR were re-run from their official code with only range and split changed.

**Table 6, overlap of train and val areas (km^2; area of the union of 30 m radius discs around ego positions)**

| dataset | split | train | val | overlap | share of val |
|---|---|---|---|---|---|
| AV2 | original | 3.46 | 1.04 | 0.56 | 54 % |
| AV2 | new | 3.58 | 0.54 | 0.00 | 0 % |
| nuScenes | original | 2.00 | 0.92 | 0.79 | 85 % |
| nuScenes | new | 1.64 | 0.54 | 0.06 | 11 % |

(Section 4.1 says "more than 84 %" for nuScenes and appendix A "more than 85 %": the paper is not consistent with itself.)

**Compute.** 13.2 FPS (nuScenes, R50, 480 x 800) and 14.2 FPS (AV2, 608 x 608) = 75.8 and 70.4 ms per frame (derived). GPU used for the FPS figure: not found (the benchmark script times only the model forward at batch 1 after 5 warm-up iterations with synchronisation, excluding data loading but including host-to-device copy, default 2,000 samples). Parameters, memory and a latency breakdown: not found.

**Caveats.** Instance-level polyline AP, not a graph metric; no seeds; the released checkpoints give slightly different numbers than the paper (README, nuScenes new split 60 x 30 m: 34.1 mAP with 32.2 / 29.3 / 40.8; 100 x 50 m: 22.4; AV2 60 x 30 m: 58.3, 100 x 50 m: 51.6; old split 63.4 with a 30-epoch config while Table 4 says 24 epochs).

## 4. Paper versus code
Digest-level comparison (code read as walkthroughs, paper as HTML). Items marked (!) should be checked in the PDF before anybody relies on them.
1. (!) **Target of the transformation loss.** Paper (E10 - E12): P_t is the previous points mapped by T, read by the digest as the previous predicted polyline ("not ground truth"). Code: `propagate` maps the stored matched GT polyline of the previous frame (target memory filled from the Hungarian assignment of the previous frame; zero rows for unmatched queries, weight 0). So the loss needs labels and inherits the matching. Level-1 note VM01 follows the paper reading.
2. **Weight.** Paper: third weight printed as 5.0 (typo in the symbol). Config: 0.1 times the regression loss weight 50 (derived effective 5.0). Normalisation by the number of foreground propagated queries is in the code only.
3. **Pose input.** Paper: flatten(T), a 4 x 4 matrix (16 numbers). Code: the upper 3 x 4 block of the previous-to-current ego transform (12 numbers, translation in raw metres, no normalisation), repeated for every query; MLP c_dim = 12.
4. **Query update network.** Two linear layers (524 -> 1024 -> 512 with LayerNorm and ReLU in between) plus residual; both weight matrices are zero-initialised, so the update starts as the identity (code only; about 1.06 M parameters, derived).
5. **Confidence for the top-k.** Paper: "highest confidence". Code: maximum class logit over the 3 classes (raw logit, same ranking as the sigmoid).
6. **Where the carried queries enter.** Paper and code agree: after decoder layer 0 (`prop_add_stage = 1`); the 67 kept new queries are those with the highest layer-0 maximum logit (code).
7. **GRU.** Paper: GRU + LayerNorm. Code (digest): ConvGRU with three 1 x 1 convolutions without bias on the concatenation, sigmoid gates, **no tanh on the candidate**, output (1 - z) * h + z * q, LayerNorm over channels. h = warped previous fused BEV, x = current BEV. First frame: h is a detached copy of the current BEV.
8. **Warp.** Paper: Warp(F, T). Code: the BEV cell centres (z = 0) are mapped from the current ego frame into the previous one with float64 matrices and `grid_sample` (bilinear, zero padding, align_corners False); the full 3-D pose is used, not SE(2).
9. **Splits.** Paper: new 700 / 150 splits for both datasets. Code: nuScenes lists have 698 train + 148 val scene names (hand-tallied by the tool) and a 50-name `CALIBRATION_SCENES` list; the AV2 id files hold about 749 / 149 log ids (val count from the tool, train from the byte size 27,713 / 37 = 749, derived). Not reconciled.
10. **Warm-up length.** nuScenes config: 4 epochs. AV2 config: total_iters // 6 = 5 epochs.
11. **AV2 frame rate.** Paper: 2 Hz. Code: the info files hold every 10 Hz lidar timestamp; the 2 Hz comes from `interval = 5` as a plain stride over the flat sample list; the stored `prev` links are not updated after striding (digest), the scene name alone drives the memory reset.
12. **Hidden steps not in the text:** clipping of transformed reference points and targets to [0, 1]; zero-padded prior for first frames; y-axis flip of the reference points before the deformable attention (image-row convention); next-layer reference points are detached predictions.

## 5. Transfer to our pipeline
| question | answer |
|---|---|
| what it needs from us | Memory part: the relative ego motion between consecutive frames (the code builds it from global poses of both frames, but only prev-to-current is used, so odometry is enough; the "no pose given" problem of the SD-map registration is separate). Cached BEV: the warp + GRU runs on any raster, but the recurrent state must be our head's own hidden map: the frozen 39-channel cache is the same at every step and cannot be a recurrent state, so the paper's BEV-GRU reduces to a two-input fusion head-side (= our E39 / E41). Query part: a persistent per-element token (our 128-d vertex token), a rank / validity score, per-element labels for the transformation loss. No SD map, no image features, no lidar. |
| what it gives us | A5: (a) a pose-conditioned, zero-initialised residual token update; (b) transformed previous geometry as the prior of the next frame; (c) a label-supervised "carry" loss; (d) the streaming training schedule (single-frame warm-up, two chunks per scene, per-slot memory, detach) and per-scene reset at test; (e) A9: the new nuScenes / AV2 splits and the overlap measure of appendix A. Does not give: validity accumulation, edge or fork carry, pose-error robustness, sensor-failure behaviour. |
| conflicts | Hungarian matching in the main loss (6 assignments per frame) and the carry-loss labels are produced by that matching; fixed query count with concatenation into a DETR decoder and joint self-attention; polyline instances without edges or fork / merge states; hard top-k by logit; mmcv / mmdet / mmdet3d stack and a compiled deformable-attention op (BEV encoder not needed by us); **GPL-3.0 code: read only**; trainable BEV encoder feeds the GRU; the clip-to-ROI rule. |
| reuse mode | Re-implement from description (idea only; GPL). Candidates in order: overlap audit of our splits; token-update MLP; transformed-vertex prior; streaming sampler; head-side warp + GRU (already tried). |
| effort | Overlap audit S (hours; ego poses only). Token update + vertex warp inside the existing time recursion S - M. Chunked streaming sampler with per-slot memory and detach M (days; touches the trainer). A DETR-style query memory L and not recommended. |
| expected gain and risk | In the paper the temporal parts add 4.4 mAP over a strong single-frame row (Table 5, single run, instance AP). Our own analogue of the BEV part gave +0.008 F1 (E41), inside the 0.010 noise band of S; do not expect more from the feature side. The real use is consistency and recovery from a lost frame or sensor, which the paper never tests (no pose-noise test, no failure test, no stability metric, no k or memory-length ablation). Risks: a prior made of the model's own output can confirm its own errors (no decay, no evidence update); the trained carry loss uses GT of the previous frame, so on a frame after an error the prior is trained to be better than it will be at test (exposure gap partly closed by on-policy propagation); pose error shifts the prior by the pose error. |
| feeds which experiment / K id | K45 already names the family. To register before use (proposals, not yet in `../register.md`): (1) pose-conditioned zero-init residual token update on the vertex set with a matching-free carry loss; (2) location-overlap audit of splits (30 m disc union area as in appendix A, plus the 5 m criterion of VM46); (3) chunked per-slot streaming sampler with single-frame warm-up. |

### 5.1 The new splits: how they are built and what that means for ours
- **nuScenes:** the "new split" is not computed in the repository. `tools/data_converter/nusc_split.py` holds three hard-coded lists of scene names (698 train, 148 val, 50 calibration; counts hand-tallied by the tool) with no function, no comment and no provenance; the converter flag `--newsplit` swaps them in for the official lists (digest) and writes `nuscenes_map_infos_{train,val}_newsplit.pkl`; four failed scenes are skipped. The first and last entries of all three lists are identical to the file `splits.py` of Roddick and Cipolla (mono-semantic-maps), and the paper's section 4.1 says the nuScenes split is theirs (digest). So StreamMapNet did not create a new nuScenes split; it adopted an existing scene-level re-split and measured it: 11 % of the val area still overlaps (Table 6), so it is *not* fully location-disjoint. Their reason for the re-split (page not readable: PDF binary, ar5iv digest unclear) is not confirmed here.
- **AV2:** train and val log ids are in `av2_train_split.txt` and `av2_val_split.txt`; with `--newsplit` the converter reads them, without it the id lists are replaced by directory listings (the digest says "overwritten", unclear whether on disk). The paper (appendix A) says all AV2 logs were merged and re-divided so that objects, weather and cities are balanced and overlap is zero. The algorithm that produced the id lists is not in the repository.
- **How reproducible:** the splits themselves are reproducible (lists shipped); their construction is not (no script, no seed, no criterion beyond the prose). The overlap measure is reproducible from ego positions alone (union of 30 m radius discs per split, area of the intersection against the val area; no map needed). Lilja et al. (VM46) give an independent 5 m criterion and report 2.1 % overlap for this nuScenes split (level-1 note, not re-read).
- **For us:** our TRAIN / VAL / TEST (212 / 22 / 21 scenes from the first three blobs) are scene-disjoint but not location-disjoint. Cheap audit (S): compute the disc-union overlap of VAL and TEST against TRAIN from nuScenes ego poses, report it next to the S numbers, and add a VAL subset that is filtered by a distance threshold to TRAIN (e.g. 5 m and 30 m) as a robustness row. Ranking-preserving drops are typical (VM46: MapTR 48.7 to 20.9 on the clean split), but the effect on a frozen map-pretrained foundation is unknown and not checkable here. A full region split of 255 scenes would leave very few scenes per fold (see VM46 note).

### 5.2 Which part of the query memory can act on a vertex set with validity (our time recursion)
Can transfer:
- **Warp of the previous geometry** into the new ego frame (their reference-point transform). For our vertices in ego metres it is a plain SE(2) transform of positions; parent edges survive because they are index-based. Difference to make: vertices that leave the +-50 m window get validity 0 (or are dropped), not clipped onto the border.
- **Token update** MLP([token, relative pose]) with residual and zero initialisation (identity at the start). For 128-d tokens, a hidden width of 256 and a 3-number pose (dx, dy, dyaw, or cos / sin) this is about 0.07 M parameters (derived), against about 1.06 M in the original. Use a normalised pose; the original feeds raw metres.
- **Carry selection by rank** works because memory is detached (selection needs no gradient). For a graph, carry by branch or with parent closure, or carry the whole set with validity as a soft weight, so that isolated orphan vertices are not created by an independent top-k.
- **Decode-back loss** (transformation loss): decode the updated token to a position with the same head that decodes fresh vertices and compare with the previous-frame target moved by T. Needs a per-vertex target. Matching-free options: (i) if the builder assigns a GT point to every vertex by construction (teacher-forced growth), store it with the vertex and move it by T; (ii) my proposal, not in the paper: penalise the distance of a warped valid prior vertex to the current dense GT centre-line (distance transform of the current GT), with a validity target from the visibility of the GT point. Both avoid Hungarian.
- **First frame / reset:** a zero-validity prior is the natural "no prior"; no separate branch is needed.
- **Training schedule:** single-frame warm-up (about one sixth of the iterations), then chunks of consecutive frames with a per-slot memory, detached, on-policy (the prior is the model's own previous output). Gradient still reaches every module of the current-frame path; only the credit assignment across time is cut (to be stated in the experiment card, it is a deviation from "gradient reaches every module" only along the time axis).
- **BEV side:** warp + ConvGRU on the head's hidden dense map instead of the frozen cache (the cache-level variant is E41).

Cannot transfer:
- The **fixed set of 100 query slots** with the carried ones concatenated to fresh ones and a joint self-attention: our builder has no query decoder; there is no slot to give up.
- The **Hungarian main loss** (and with it the labels of the carry loss and the one-to-one pressure that suppresses duplicates between carried and fresh queries). Without a matching, duplicate suppression between the prior and the new detections must come from somewhere else: use the prior as context (cross-attention) for the builder instead of as extra output instances, or add a validity update rule (log-odds, K24) so that a vertex seen again increases the same vertex instead of creating a second one.
- **Polyline-as-instance** semantics (20 ordered points, 38 permutations) and the absence of edges, forks and merges.
- **Confidence only as a rank:** no accumulation of evidence over frames, no decay; our open item 3 (validity memory) gets nothing from the paper.
- **Pose trust:** the memory trusts the transform completely; there is no test with pose noise, which matters for us as soon as the pose is estimated.

## 6. Not found / open
- Paper text of the target of E12 in the PDF (predicted versus GT) and the exact printed third weight; the digest is ambiguous (section 4, item 1).
- GPU of the FPS figure, parameter count, training time, memory footprint; any k ablation, memory-length study, effect of detaching; pose-noise and sensor-failure experiments; stability metrics (RB35 mentions a later benchmark that measures stability, not read here).
- How the AV2 id files were produced; why 700 / 150 in the text but about 749 / 149 ids; why Table 4 says 24 epochs while the old-split config is a 30-epoch file.
- Raw source of `StreamTensorMemory`, `MapTransformerDecoder_new` and `ConvGRU` was read only as walkthroughs; exact activation choices (no tanh in the candidate) should be re-checked if the GRU is ever ported.
- Table 5 row b value (the text only says it converges poorly).

## 7. Our experiments that touch it
(From the level-1 notes, not re-verified here.) E39 / E41: feature-level temporal fusion on the cached BEV, +0.001 / +0.008 F1, kept (analogue of the BEV-GRU). E31 / E32: matched sparse decoder, dropped (the dense heat map + tracer beat it by +0.058 F1). System B input memory (ego-warped previous input). Loop-16 time recursion on the vertex set (ego-warped vertices and tokens as prior, one step) and loop-17 hold / gate memory (E193 - E212). Set / token propagation with a decode-back (transformation) loss and the location-overlap audit: not done.
