# VM04 - MapTracker: full-text extraction

| field | value |
|---|---|
| paper | MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping. Jiacheng Chen, Yuefan Wu, Jiaqi Tan, Hang Ma, Yasutaka Furukawa (Simon Fraser University; Furukawa also Wayve). ECCV 2024 (oral). arXiv 2403.15951 (cs.CV), v1 2024-03-23, v2 2024-10-12 (this text). Links opened: arxiv.org/abs/2403.15951, arxiv.org/html/2403.15951 (v2 HTML, nine separate queries), map-tracker.github.io, github.com/woodfrog/maptracker (README, docs, code files, see the code extraction), arxiv.org/html/2503.07168 (HisTrackMap, only for the G-mAP definition) |
| axis, tier, K ids | A5 (temporal fusion of detections), tier 1 core; K45 (temporal map propagation and memory, register status "checked"). Proposed new K ids in section 5 (not yet registered) |
| level-1 note(s) | [VM04](../papers/VM04_maptracker.md) (home note), [A05](../papers/A05_maptracker-raster-vector-memory-with.md), [VM45](../papers/VM45_c-map-and-g-map.md) (C-mAP / G-mAP). Related: [VM15](../papers/VM15_histrackmap.md) (G-mAP), [VM44](../papers/VM44_mas.md), [VM01](../papers/VM01_streammapnet.md), [VM12](../papers/VM12_l2g-map.md) |
| code extraction | [maptracker](../code/maptracker.md) (GPL-3.0 plus a non-commercial notice: read only) |
| sources read | arXiv v2 HTML: abstract, 1 Introduction, 3 MapTracker (3.1 - 3.4), 4 benchmark (4.1, 4.2), 5 Experiments (Tables 1 - 5, speed, implementation details), 6 Conclusion/limitations, appendix 0.A (transformation loss, memory fusion), 0.B (consistent GT, track extraction Algorithm 1, C-mAP Algorithm 2), 0.C (online merging), 0.D (Tables 6 - 8); project page; GitHub README, docs (installation, getting_started, data_preparation), config `maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py`, and the code files named in the code extraction. All read 2026-10-24 |
| reading marks | **every statement about the paper and about the code is digest**: all pages came through the summarising fetch tool, no page was seen raw, so numbers may be mis-transcribed (the tool refused near-verbatim transcription, so equations are given in symbolic form reconstructed from its description, not copied). Statements about our own files (section 5, 7) are **checked** where a file is named, **memory** where marked. Section 5 is my analysis, not the paper's content |
| extracted by, date | extraction agent (Claude, sub-agent of the literature session), 2026-10-24 |

## 1. Problem and claim
- Predict a vector HD map (pedestrian crossings, lane dividers, road boundaries; polylines of 20 points in the ego BEV frame) online from six surround cameras, frame by frame, and make the sequence of local maps temporally consistent (an element keeps its identity and shape while the vehicle moves). Not a lane-centre graph; no topology, no forks or merges.
- Gap: query-based mappers (MapTR family, StreamMapNet) are accurate per frame but flicker (elements appear, vanish, split, merge between frames), and standard mAP does not see it. Two contributions beyond the model: a benchmark with "consistent" ground truth (elements get global ids over a scene) and a consistency-aware mAP, C-mAP.
- Claim that matters: treat mapping as tracking. Keep two memories (raster BEV latents and per-element vector latents) over the last 20 frames, propagate tracked elements with a pose-conditioned MLP, and fuse four past memories chosen by travelled distance (about 1, 5, 10, 15 m). Result (digest, Table 1): nuScenes consistent GT 76.1 mAP / 69.1 C-mAP against StreamMapNet 70.4 / 56.4, at 11.5 FPS.
- What the paper itself says it does not do: merges and splits of road elements (stated limitation).

## 2. Method as built

### Pipeline (digest; shapes from text, config and code)
```
6 images 480 x 800 (nuScenes), ResNet-50 (Caffe weights, DCNv2 in stages 3-4) + FPN (3 levels, 256 ch)
  |
  v                              BEV memory buffer: last 20 frames, each M_BEV 50 x 100 x 256
BEV module (BEVFormer-style, trainable, 2 encoder layers)           ^ written every frame
  1. warp M_BEV(t-1) by ego motion P (affine + bilinear grid_sample) -> initial BEV queries;
     cells outside the old map keep a learnable per-cell embedding   ("MaskBlend")
  2. deformable self-attention
  3. spatial cross-attention to image features (BEVFormer), FFN
  4. memory fusion: 4 past M_BEV chosen by ego distance (1/5/10/15 m), warped to t,
     concatenated with the current one, 3 x 3 conv + residual block(s), added back
  (2-4 run once per encoder layer, 2 layers = "repeated twice")
  -> M_BEV(t) 50 x 100 x 256 -> seg head (linear, 2 x 2 up-projection) -> 3 x (100 x 200) masks
  |
  v                              VEC memory buffer: per tracked element a ring of 512-d latents
VEC module (query decoder, 6 layers)                                  ^ written every frame
  tracked latents (from t-1) -> PropMLP(latent, pose of P(t,t-1)) ; + 100 new learnable queries
  per layer: instance self-attention -> BEV cross-attention (StreamMapNet multi-point attention)
             and memory cross-attention (<= 4 strided past latents of the same element, aligned by PropMLP,
             relative-frame sin/cos encoding) -> FFN
  -> class score (1 FC) and polyline of 20 points (3-layer MLP) per query
  -> positives (score threshold) = output Y(t) with persistent ids; their latents go to the VEC memory
```
- BEV memory covers 15 m left/right and 30 m front/back (60 x 30 m ROI, 100 x 50 cells = 0.6 m cells), embedding 256. Vector latents are 512-d. The code runs the memory cross-attention in parallel with the BEV cross-attention and adds the two (digest of `MapTransformerLayer`), the paper text lists the three operations in sequence.

### Inputs, outputs
- Input per frame: 6 images, camera parameters, ego pose (4 x 4 per frame; the motion P^t_{t-m} between frames t and t-m is derived from the poses, exact poses are used at test time). nuScenes 2 Hz keyframes (480 x 800); Argoverse2 7 cameras (608 x 608), annotated at 10 Hz and subsampled by 4.
- Output: positive elements, each a class (3 classes) and 20 ordered points in ego metres within the ROI, plus a scene-wide persistent id (tracked elements keep the id, newborn ones get a new id); optionally BEV semantic masks. No padding rule: the number of elements varies.

### Equations (numbers as in the paper; symbolic form reconstructed from the digest, **not** copied)
- E1 (BEV loss): L_BEV = lambda1 * L_focal(S, S_hat) + lambda2 * L_dice(S, S_hat); per-pixel focal, per-class Dice; lambda1 = 10, lambda2 = 1.
- E2 (assignment of new elements): omega_new(t) = argmin over bipartite matchings Omega(t) of L_match(new GT elements, new predictions); a "hierarchical" cost like MapTR (focal classification term + permutation-invariant line term). The paper does not name the solver; the code uses scipy `linear_sum_assignment`.
- E3 (inductive assignment): omega(t) = omega_track(t) U omega_new(t), with omega_track(0) empty and omega_track(t) = omega(t-1) for t > 0: tracked queries inherit last frame's assignment and are not re-matched.
- E4 (tracking loss): L_track = lambda3 * L_focal(class, omega(t)) + lambda4 * L_line(polylines, omega(t)); lambda3 = 5, lambda4 = 50.
- E5 (total): L = L_BEV + L_track + lambda5 * L_trans, lambda5 = 0.1.
- Transformation loss (appendix 0.A.1, unnumbered): the vector heads are applied to the propagated latents of t-1 (after PropMLP); the targets are the t-1 GT elements moved to frame t with the ego motion; focal + line terms, as in E4. The appendix writes the focal term with omega(t-1) and the line term with omega(t): probably a notation slip.

### Losses (what needs matching)
| term | weight | normalisation | matching needed? |
|---|---|---|---|
| BEV focal / Dice | 10 / 1 | per pixel / per class | no |
| class focal (gamma 2, alpha 0.25 in config) | 5 | sync across GPUs, positives + negatives*bg weight (bg weight 0 in the head default) | yes for new queries (E2); inherited for tracked ones |
| line L1 (beta 0.01 in config) | 50 | by number of positives | same assignment |
| transformation (forward and backward in code) | 0.1 | positives for regression, tracks for class | no (uses the inherited assignment and the GT moved by the pose) |
- Matching cost in config: focal cost 5 + lines L1 cost 50 (permute=True: the point order of a polyline is searched). Hungarian therefore appears only for the "new" part of E3; the inherited pairs are forced inside the same cost matrix (code, digest: inherited pair cost -1, its row and column infinite elsewhere; a tracked query whose GT disappeared is a false-positive query, its row infinite, and it is supervised as background).
- Offline, building the ground-truth tracks also uses Hungarian matching (on mask IoU, see below).

### Labels / ground truth (the "consistent GT")
Question: how are instances matched over time?
1. Per-frame geometry: start from the MapTR code base; replace its pedestrian-crossing processing by StreamMapNet's merging with orientation checks (orthogonal crossings are not merged) plus extra geometric constraints; for Argoverse2 fix the divider graph tracing (whole dividers were lost) and always close crossing polygons. Reason: the public GT variants produce elements that split, merge or vanish between frames only because of the cleaning rules (the paper's own check: MapTR's GT evaluated on the consistent benchmark, Table 8 below, crossings only 80.6 C-AP on nuScenes).
2. Frame-to-frame matching: move the older frame's elements into the newer frame with the ego motion (clipped to the ROI, not dropped), rasterise both sets into instance masks (crossings as filled polygons, others as polylines; thickness 3 pixels on a 200 x 100 canvas in the config; a mask with fewer than 20 pixels is dilated with a 7 x 7 ellipse), compute mask IoU per class, solve the optimal one-to-one assignment (code: `linear_sum_assignment` on -IoU) and keep a pair only if IoU > 0.01 (code; the paper gives no cutoff). Adjacent frames only (look-back 1 for GT).
3. Chain the pairwise matches into tracks: frame 0 gets fresh ids per class; later a matched element inherits the id of its predecessor, an unmatched one gets a new id. The output is `instance_ids` per scene and frame.
- Forks, merges, crossings: the assignment is strictly one-to-one. A split gives one branch the old id and the other a new id; a merge ends one of the two ids; the paper says the ground truth "does not represent these properly either" (example: a U-shaped boundary that splits into two lines). An element that is absent from one frame and returns later gets a new id (follows from matching adjacent frames only: no look-back, no gap bridging for GT). How a crossing that touches a boundary is treated: not described.
- Track extraction for baselines without ids (Algorithm 1): keep predictions with score > tau = 0.4, move earlier elements to the current frame, match mask IoU, look back N frames (main tables N = 1; N = 3, 5 in appendix Tables 6 - 7), unmatched elements get new ids. Code (digest): same Hungarian-on-IoU, cutoff 0.001, then a greedy, order-dependent id propagation in which nearer frames win and an id already used in the frame is not accepted twice.

### Training recipe (Section 3.4, 5, config; digest)
- Three stages: (1) BEV pre-training with L_BEV only (image backbone + BEV encoder), (2) vector-decoder warm-up with everything else frozen and a large batch, vector memory switched on after 500 iterations, (3) joint training of all parameters. Epochs nuScenes 18 / 6 / 48 (72 total), Argoverse2 12 / 3 / 20 (35). Batch 16 / 48 / 16.
- AdamW, lr 5e-4, weight decay 0.01, cosine to 1.5e-6; code config stage 3: lr multipliers 0.1 (image backbone) and 0.5 (BEV transformer, positional encoding, seg decoder), gradient clipping 35, linear warm-up 500 iterations, SyncBN, 83,904 iterations at 2 samples per GPU x 8 GPUs, 27,968 samples per epoch.
- Clips: each training sample is the current frame plus 4 previous frames drawn at random without replacement from the 10 previous ones and sorted in time (`multi_frame=5`, `sampling_span=10`). The image backbone is frozen (no gradient) for the history frames. In the code the training memory bank holds exactly the 4 history frames (`mem_len = history_steps = 4`): **the distance-strided selection is applied only at test time** (buffer 20); in training the "stride" is the random spacing of the clip. (my reading of config and code, digest)
- Pose robustness: Gaussian noise is added to the inter-frame transform before it enters PropMLP and the propagation of points. The paper gives no sigma. Code: Euler angles std 0.08 rad (comment: about 5 degrees, 15 degrees at 3 sigma) and translation std 0.25 m, z noise zeroed; a commit of 2025-05-05 "Fix typo in pose noising" corrected the Euler axis order (zyx -> zxy), so checkpoints may have been trained with the old order (unknown). Whether the targets of the transformation loss are moved with the noisy or the true matrix is not clear from what I read. There is no experiment on pose error in the paper.
- Track-query robustness: tracked queries below score 0.4 are dropped between frames during training; unmatched queries above 0.4 (and, if `track_fp_aug` is on, a random 5 percent of the rest; off in the stage-3 config) are inserted as false-positive tracks so the decoder learns to kill them.
- Hardware: 8 x RTX A5000, about 3 days.

### Inference (what is a hidden rule step)
- Thresholds (paper and mapper code agree): first frame 0.4 on the 100 new queries; later frames 0.5 for propagated queries and 0.6 for new ones. Tracked elements that fall below their threshold are dropped for good (not stored for propagation, their id is not reused); an element that reappears later is a newborn with a new id and a fresh memory. Newborn ids are consecutive integers.
- Positive output = class score above threshold; the paper describes no NMS and no merge step in the model (none seen in the code files read either).
- Memory selection at test time (code, digest): the bank is a ring buffer of up to 20 frames (BEV and vector). Distance is the straight-line xy distance between the global ego translation of each stored frame and the current frame (displacement, not path length). For the targets in reverse order (15, 10, 5, 1 m): if some not-yet-covered entry lies at distance >= r, take the nearest such entry and mark it and all farther ones as covered; otherwise take the farthest not-covered entry that is closer than r; if neither exists fall back to the four nearest entries. Fewer than four stored: all. For a stationary vehicle all distances are near zero, so the "otherwise" branch picks four distinct entries ordered by their tiny distances (paper: silent).
- Vector memory per element: only entries that exist for that element are fused (shorter history, fewer keys, padding mask); an element with no stored entry gets no memory update.
- Online merging into a global map (appendix 0.C, for visualisation and for G-mAP of other papers): merge by global id; crossings by convex hull of all points, dividers and boundaries by interpolation of stored and new points; described as simple and imperfect.

## 3. Numbers

### Main results (author-reported; digest; protocol: 60 x 30 m ROI, AP by Chamfer at 0.5/1.0/1.5 m, three classes)
Table 1 (nuScenes, original split; G.T. = ground truth used for train and test; AP_p crossing, AP_d divider, AP_b boundary):
| G.T. | method | epochs | AP_p | AP_d | AP_b | mAP | C-mAP |
|---|---|---|---|---|---|---|---|
| StreamMapNet-GT | StreamMapNet | 110 | 68.0 | 71.2 | 68.0 | 69.1 | - |
| StreamMapNet-GT | MapTracker | 72 (multi-frame) | 77.3 | 72.4 | 74.2 | 74.7 | - |
| consistent | MapTRv2 | 110 | 69.6 | 68.5 | 70.3 | 69.5 | 50.5 |
| consistent | StreamMapNet | 110 | 70.0 | 72.9 | 68.3 | 70.4 | 56.4 |
| consistent | MapTracker | 72 | 80.0 | 74.1 | 74.1 | 76.1 | 69.1 |

Table 2 (Argoverse2, consistent GT): MapTRv2 70.9 mAP / 56.1 C-mAP; StreamMapNet 70.3 / 57.5; MapTracker 76.9 / 68.3.
Table 3 (geographically non-overlapping splits, consistent GT, mAP / C-mAP): nuScenes 60 x 30 m StreamMapNet 33.5 / 22.2, MapTracker 40.3 / 32.5; nuScenes 100 x 50 m 23.0 / 14.6 against 36.2 / 27.5; AV2 60 x 30 m 64.4 / 54.4 against 71.3 / 63.2; AV2 100 x 50 m 54.6 / 41.3 against 64.8 / 55.7. (The 100 x 50 m evaluation uses Chamfer thresholds 1.0/1.5/2.0 m in the C-mAP code.)

### Ablations that decide transfer
Table 4 (nuScenes, consistent GT; mAP / C-mAP):
| variant | mAP | C-mAP |
|---|---|---|
| MapTRv2-style detection baseline | 69.9 | 56.1 |
| StreamMapNet (conditional detection, memory embedding) | 70.4 | 56.4 |
| tracking, single memory (no fusion) | 70.8 | 62.4 |
| tracking + fusion of the 4 newest memories | 74.9 | 68.1 |
| tracking + fusion + distance stride | 76.1 | 69.1 |
- Reading (my arithmetic): tracking-style propagation without fusion gives +6.0 C-mAP at +0.4 mAP over StreamMapNet; fusing several memories gives +4.1 mAP / +5.7 C-mAP; the stride adds only +1.2 mAP / +1.0 C-mAP. No seeds or intervals are reported.
Table 5 (buffer 20; nuScenes, 60 x 30 m; mAP / C-mAP): 4 newest (buffer 4) 74.9 / 68.1; strides {0,0,0,0} 75.0 / 68.2; {1,2,3,4} 75.8 / 68.8; {1,3,5,7} 75.9 / 68.8; {1,5,10,15} 76.1 / 69.1; {5,10,15,20} 73.8 / 51.5. The collapse of C-mAP when the 1 m memory is missing shows that the nearest memory is essential. No ablation of MaskBlend, of pose noise, or of the 20-frame buffer length.
Appendix Table 6 (nuScenes, C-mAP with different id extraction; C-mAP-bar = upper bound without consistency check): MapTRv2 50.5 / 58.9 / 60.0 at look-back 1 / 3 / 5 (bar 64.9); StreamMapNet 56.4 / 60.5 / 61.0 (bar 65.9); MapTracker own ids 69.1, IoU-extracted ids 69.7 / 70.4 / 70.7 (bar 72.5). Table 7 (AV2): MapTracker own ids 68.3, look-back 5: 71.0 (bar 72.8). Table 8: MapTR's own GT evaluated as a prediction on the consistent benchmark: nuScenes C-mAP 93.5 (crossings 80.6, dividers and boundaries 100), AV2 87.1 (dividers 65.7).
- Reading (my arithmetic): the gap to the upper bound is 3.4 for MapTracker, 9.5 for StreamMapNet, 14.4 for MapTRv2. For the baselines C-mAP rises by about 10 points just by a longer look-back of the post-hoc id extraction, and MapTracker's own ids are worse than the post-hoc IoU ids at every look-back: C-mAP mixes model consistency with the quality of the association rule.

### Compute
- 11.5 FPS (RTX 6000, batch 1, nuScenes, full camera pipeline) against StreamMapNet 14.2 and MapTRv2 12.5; about 19 percent slower than StreamMapNet, attributed to un-batched for-loops in the vector memory fusion. The conclusion says about 10 FPS is short of real time. Training: 8 x A5000, about 3 days. Parameters and GPU memory: **not found**. Checkpoint repository on HuggingFace (cccjc/maptracker): 14 folders, 9.42 GB in total (stages 1 - 3, splits and ranges).
- Test-time id extraction for baselines is expensive: at look-back 5 about 4 times the MapTRv2 runtime (paper).

### Caveats
- Numbers are on MapTracker's own regenerated ground truth (consistent GT) and are not comparable with the original MapTR/StreamMapNet benchmarks; "original split" overlaps geographically between train and val (Table 3 shows the drop on the disjoint split). C-mAP applies only to positive predictions, so it cannot reach mAP even with perfect ids (C-mAP-bar is the practical ceiling). Range 60 x 30 m, camera only, element classes are not lane centre-lines. No pose-error, missing-frame or sensor-failure experiment.

## 4. Paper versus code
See [../code/maptracker.md](../code/maptracker.md), section 3. Short list (digest): (a) code adds the vector-memory attention in parallel to the BEV attention; (b) test-time thresholds are set by the mapper (0.4 / 0.6 / 0.5) although the head file carries other defaults; (c) strided selection is test-only, training memory is the random clip; (d) MaskBlend is a hard mask (a cell counts as valid when the channel sum of the warped feature is positive); (e) noise sigmas exist (0.08 rad, 0.25 m); (f) GT IoU cutoff 0.01, pred-track cutoff 0.001, thin-mask dilation; (g) C-mAP code enforces only the "different prediction id claims an already claimed GT id" rule, not the prose rule "any unmatched ancestor invalidates the match" (see the subsection "C-mAP as implemented" below).

### C-mAP as implemented (question: matching, thresholds, identity)
- Per frame and class: predictions sorted by score; Chamfer distance to the nearest GT (polylines resampled to 200 points); true positive if the distance is within the threshold and that GT is not yet covered in the frame (greedy), else false positive. Thresholds {0.5, 1.0, 1.5} m for the 60 m ROI, {1.0, 1.5, 2.0} m for the 100 m ROI (code); AP by area under the precision-recall curve; C-mAP = mean over 3 classes and the thresholds.
- Identity enters through a per-scene, per-class record: the first prediction id that claims a GT global id is stored. If a later frame matches a prediction with a different id to the same GT id, that prediction is turned into a false positive (never the reverse). So the metric as coded penalises id switches. It does not penalise a GT element that is simply missed in some frames and then found again under the same id, and a missed first appearance does not invalidate later matches (code, digest). The paper's prose (Section 4.2) says a match is removed if any ancestor of the same track was not matched; Algorithm 2 in the appendix (record B_rec, as in the code) is the id-switch rule. Quirk: unmatched predictions all map to GT id -1, which is an ordinary key, so two different unmatched ids also trigger the rule.
- AP is pooled over all frames of all scenes (not per scene); the record resets per scene. Baselines get ids from Algorithm 1 (above); MapTracker uses its own ids.
- **G-mAP is not in the MapTracker paper or repository** (neither the README nor the paper mentions it). It was introduced by HisTrackMap (Yang et al., arXiv 2503.07168, digest): GT vectors of N_seq frames are moved to a global frame and rasterised; per-frame predictions are merged with a Merge function; polygons scored by rasterised IoU AP, polylines by Chamfer AP with GT points from farthest-point sampling; distance thresholds {0.25, 0.5, 0.75, 1.0} m; a prediction may match several shorter GT instances; a false positive counts only if valid within 2 m. HisTrackMap reports G-mAP for MapTracker 49.4 against its own 50.2 (nuScenes, 72 epochs). Identity enters only through the merge, not through a record of ids. The VM45 note attributes the merge function to MapTracker: it is the appendix 0.C merge used for the baselines' global maps.

## 5. Transfer to our pipeline
(my analysis; the only checked facts are the project files named)

| question | answer |
|---|---|
| what it needs from us | The memory idea needs only ego poses per frame (available: nuScenes ego pose; "no pose given" is a separate problem) and a ring buffer of past states. As a model, MapTracker needs a trainable camera-to-BEV encoder (BEVFormer, ResNet-50, 3-stage training), which our frozen cached 39-channel 200 x 200 / 0.5 m foundation excludes. Its BEV grid (100 x 50 cells of 0.6 m, ROI 60 x 30 m, 256 ch) is smaller than ours (+-50 m). No lidar, no SD map. |
| what it gives us | Axis A5 and A9. (1) A recipe for fusing several past memories chosen by ego displacement with a mandatory nearest memory. (2) A pose-conditioned propagation MLP with a matching-free "transformation" loss (forward and backward). (3) Pose-noise augmentation of the warp (sigma 0.08 rad, 0.25 m, code). (4) A learned fill for cells without history. (5) The idea of a consistency-aware metric and a GT sanity check (evaluate the GT as a prediction: Table 8). (6) Evidence that the id-based C-mAP depends on the association rule (Table 6). |
| conflicts | Hungarian matching for new queries and for GT tracks (offline); per-element identity inheritance (a query is an element, our unit is a vertex with parent edges and forks/merges, which MapTracker explicitly does not handle); hard thresholds 0.4 / 0.5 / 0.6 for birth and death (rule step on the inference path; ours is a soft validity); class argmax for the label of tracked queries; trainable BEV encoder and perspective features; mm-stack (Python 3.8, PyTorch 1.9, mmcv-full 1.6, mmdet 2.28, mmdet3d 1.0.0rc6); **GPL-3.0 plus a non-commercial notice: read only**. The distance rule uses poses (inputs), not perception outputs, so it does not violate "no rule step on the training path". |
| reuse mode | Re-implement from the description (idea only; no code is copied). |
| effort | S: distance-strided selection of cached frames + warp + fusion conv on the cached BEV or on the dense head output (hours to days). M: vertex-level memory with attention over K selected past vertex sets, relative-pose encoding, propagation loss without matching. M: identity-free consistency metrics in `evaluation/sparse_lanes/`. L (not planned): tracker with persistent ids. |
| expected gain and risk | Gain claimed by the paper comes mostly from fusing several memories (+4.1 mAP / +5.7 C-mAP over one memory) and from tracking-style propagation, the stride adds about +1 (Table 4/5). Our own evidence is weaker: a single warped previous frame gave +0.008 F1 and a third less flicker (E41), memory as extra channels +0.000 - 0.005 (E80), the time recursion on the graph is a tie in S (E192). So expect a small gain in S and a larger one in flicker, failure phases and fork persistence (open item 3 of the overview). Risks: (a) nuScenes keyframes are 2 Hz, so at 5 - 10 m/s consecutive frames are already 2.5 - 5 m apart and the 1/5/10/15 m rule degenerates to roughly the last 3 - 4 keyframes (my estimate from speed; MapTracker runs on the same 2 Hz frames, which may explain its small stride gain); (b) older memories reach 1.5 - 3 s back and misalign when the odometry is off (E85: needs within 1 degree per sqrt(s)); (c) GPU time: fusing 4 warped cached frames of 39 channels is about 4.5 G multiply-adds for a 3 x 3 conv from 195 to 64 channels on 200 x 200 (my arithmetic), vertex-level attention is negligible. |
| feeds which experiment / K id | Proposed, to be registered before use: **K-A** distance-strided multi-memory selection with a mandatory nearest memory (poses only); **K-B** matching-free propagation (transformation) loss with pose-conditioned token update and pose-noise augmentation; **K-C** identity-free temporal-consistency metrics (below). K45 already covers MapTracker as "temporal map propagation, memory". |

### 5a. What would transfer to a memory on a lane graph with explicit edges and validity
Design sketch (re-implementation from the idea):
1. **Memory content.** Per past frame: vertex positions (ego metres), 128-d tokens, validity logits, local parent-edge list (and fork/merge flags), route probability if wanted, and the frame's ego pose. Ring buffer of about 20 frames (cheap: a few hundred vertices each).
2. **Selection (K-A).** Pick up to four past frames by ego displacement, nearest first, with the MapTracker rule (nearest entry beyond each target, covered-mask, fallback to the newest four). Re-tune targets in terms of the real frame spacing (2 Hz keyframes). Always include the newest frame. For a stationary ego, fall back to time lags because displacement carries no information (the paper is silent).
3. **Warp and fusion.** Move vertices by the relative pose (exact). Do not use persistent element ids: instead each current vertex (seeds from the warped prior plus new candidates) attends to memory vertices by location (soft attention with a distance kernel in the current frame, key encoding with relative position, heading difference and the time/displacement index of the memory), so association is soft and many-to-one at forks and merges. Validity: fuse the K selected validity logits as evidence (log-odds sum with learned weights, K24) rather than only copying; edge prior: an edge (u, v) receives the kernel-weighted vote of memory edges whose both end points lie within tolerance (soft AND). This keeps the whole path differentiable with no matching and no argmax; the selection of frames is a function of poses only.
4. **Learned fill (MaskBlend analogue).** A learned "unseen" token (and a coverage mask channel instead of MapTracker's channel-sum test) for the region ahead that no memory covers; MapTracker's code decides validity by the sign of the channel sum (digest), a coverage mask from the warp itself is cleaner for a 39-channel raster.
5. **Matching-free propagation loss (K-B).** MapTracker's transformation loss decodes the propagated latent and compares it with the previous GT moved by the pose. For a graph, the target of a warped prior vertex needs no identity: the nearest point on the current GT centre-line graph (distance field or projection) and a validity target "a GT centre-line lies within tau of the warped position". At a fork both branches lie within tau of the mouth vertex and the nearest-point target is still well defined, so forks and merges cost nothing extra. Add the backward (cycle) term and train the token update with a pose input (positional encoding of quaternion + translation, residual) as MapTracker's PropMLP. Noise on the warp in training: start from sigma 0.08 rad / 0.25 m and compare against our synthetic pose-jump and map-offset failure sets (they exist; the paper has no pose-error test).
6. **What does not transfer.** Persistent ids and per-instance memory slots, learned new-element queries with Hungarian, hard birth/death thresholds, per-instance for-loops (the authors blame them for the slowdown: batch with padded key masks), the 72-epoch three-stage schedule.

### 5b. A temporal-consistency metric for graph outputs without instance identity
Why not C-mAP as is: it needs prediction ids (our vertices have none), and as implemented it counts id switches only. We already have an identity-free measure: `stability` (point F1 at 1 m between the ego-warped previous graph and the current graph in the commonly visible region, with the same quantity on GT as reference; E41 table column, old `temp/archive/misc/stability.py`, restoration pending per `evaluation/sparse_lanes/README.md`). Its weakness: a constant wrong graph is perfectly stable. Proposal (all use GT poses and GT geometry for pairing, never predicted ids; report with paired scene bootstrap, the GT-versus-GT value as reference):
| metric | definition | what it catches |
|---|---|---|
| C-F1 (consistency-aware point F1) | A predicted point counts as a true positive at frame t only if it matches GT within tau and the GT point it matches was, whenever visible in the previous N frames (via poses), also matched by some prediction in each of them (AND over N = 1, 3, 5). GT identity comes from our piece graph / pose, not from the prediction. | the prose definition of C-mAP (missed ancestor invalidates), which the code does not implement; flicker in and out |
| transition rate | For every GT point visible in frames t-1 and t, d_t = 1 if a prediction lies within tau. Report loss rate P(d_t = 0 given d_{t-1} = 1) and gain rate per 100 m of GT length, always next to recall (compare at equal recall). | flicker without ids; separates "lost" from "late" |
| displacement consistency | For GT-matched predictions in both frames: distance between the warped previous point and the current point on the same GT lane; fraction above 0.5 m (or the lateral jump to a neighbouring lane). | lane jumps, id-switch analogue |
| fork / merge persistence | For each GT junction landmark visible during an approach: fraction of frames detected (our J at 6 m and a tolerance curve 0.5 - 6 m), number of on/off switches per approach. | the graph-specific failure the literature does not measure |
| route flip rate | Warp the route probability of frame t-1 to t at GT points; fraction of points where the 0.5 decision flips while the GT label does not; mean absolute change; abstention flips. | what hurts the planner |
| window union (G-mAP style) | Accumulate predicted points of N_seq frames in the world frame, cluster at 1 m; ratio of union length to GT union length (bloat/ghost lanes) and F1 of the union. | drift and duplicate lanes |
Caveats: pairing uses GT poses, so the metric is independent of pose estimation; stability must always be read together with accuracy (S/Q); and the C-mAP lesson (Table 6) applies: do not let a post-hoc association rule decide the score.

## 6. Not found / open
- Parameters, GPU memory, ms per component (only 11.5 FPS for the whole pipeline).
- Pose-noise sigmas (found in code only), any test with pose error, missing frames or sensor failure; effect of MaskBlend, of the 20-frame buffer, of the pose noise (no ablation).
- Whether the transformation-loss targets use the noisy or the true matrix; whether history BEV features carry gradient into earlier frames (digest suggests the BEV transformer does, the image backbone does not).
- Behaviour with a stationary or very slow vehicle (not discussed); the exact rule of `prepare_temporal_propagation` (head file defaults differ from the mapper's values).
- Licence text of the repository's custom LICENSE (the fetch tool would not reproduce it); licence of the checkpoints on HuggingFace.
- Seeds and intervals for every table.

## 7. Our experiments that touch it
From `experiment_execution/plans/Experiment_Table_EN.md` (checked, rows read; verdicts are the table's):
- E41: temporal feature fusion with a warped previous frame against query propagation: fusion +0.008 F1 and one third less flicker, query propagation 0 (the MapTracker idea at its smallest: one memory).
- E20: causal temporal fusion of distance fields (CRN) 0.675 against 0.670. E80: warped previous output as extra input channels +0.000 - 0.005. E79: ego-warped average of the lane output +0.042 lane F1 in a failure phase. E105 / E112: input memory with hold, spatial hold: large gains under night and blur.
- E85: graph tracks with keep evidence and bridging, +0.110 failure-phase lane F1, needs odometry within 1 degree per sqrt(s). E61: tracking of the map alignment and lane-id assignment keeps route success at 0.92 where per-frame registration falls from 0.845 to 0.608. E106: memory of the route attribute on vertices, no gain.
- E176 / E192: time recursion in the loop-16 builder (previous nodes warped as seeds, log-odds validity fusion) is a tie in S; loop 17 hold / gate memory is E193 - E212 (not re-read here).
- Distance-strided multi-memory, MaskBlend, pose-noise training of the warp, the transformation loss and the identity-free consistency metrics of 5b: **untested**.
