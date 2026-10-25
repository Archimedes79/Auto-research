# PL43 - SMERF: full-text extraction

| field | value |
|---|---|
| paper | Augmenting Lane Perception and Topology Understanding with Standard Definition Navigation Maps (SMERF = SD Map Encoder Representations from transFormers). Katie Z. Luo (Cornell), Xinshuo Weng, Yan Wang, Shuang Wu, Jie Li (NVIDIA), Kilian Q. Weinberger (Cornell), Yue Wang, Marco Pavone. arXiv 2311.04079, only v1 exists (7 Nov 2023). Venue: ICRA 2024 according to the level-1 notes and to a web-search summary of the Cornell project page; **neither the arXiv abstract page nor the NVIDIA research page, both opened, lists a venue** (both show the preprint only), and the Cornell page itself was not readable (certificate error). Code https://github.com/NVlabs/SMERF (correct, see the code file) |
| axis, tier, K ids | A6 (SD-map prior inside perception), tier 1 core; no K id of its own in the register (the idea family is K17 for our route head, K74 / K77 for later SD-map papers); a K id is proposed in section 5 |
| level-1 note(s) | [PL43](../papers/PL43_smerf.md) (home note), [C09](../papers/C09_smerf-sd-map-transformer-encoder.md), [TP17](../papers/TP17_smart-and-smerf.md) (SMART and SMERF) |
| code extraction | [smerf](../code/smerf.md) |
| sources read | all 2026-10-24, all through the summarising fetch tool unless stated: arxiv.org/html/2311.04079 (v1; abstract, Sec. III-A..C, Sec. IV text, Tables I - VII, Figs. 1 - 4, several targeted queries); arxiv.org/abs/2311.04079 (versions); research.nvidia.com publication page (venue); github.com/NVlabs/SMERF (README, commits, root listing), api.github.com tree of the repository (file paths and sizes), raw files: `LICENSE`, `mmdetection3d/projects/toponet_openlanev2/configs/toponet_smerf.py`, `projects/openlanev2/configs/baseline_large_ptsrep_smerf.py`, `.../baseline/models/modules/map_graph_encoder.py`, `map_embedding.py`, `spatial_cross_attention.py`, `bevformer_constructer.py`, `encoder.py`, `custom_base_transformer_layer.py`, `.../datasets/pipelines/transforms.py`, `.../toponet/models/detectors/toponet_map_graph.py`, `openlanev2/sd_maps/*.py`, `openlanev2/centerline/evaluation/evaluate.py`, `data/OpenLane-V2/process_disjoint_split.py`; context: arxiv.org/html/2409.02415 (survey, only for what it says about SMERF), local notes PL44, C03, TP17. `arxiv.org/pdf/2311.04079` came back as binary and was not usable (the fetch tool cached a binary copy in its own tool-results folder; I did not open it and did not download anything myself) |
| reading marks | Sec. 1 - 4: **digest** (summarising tool; the paper has no line numbers and the tool paraphrases). The seven results tables were cross-checked by arithmetic (OLS formula, section 3.1): 15 of 16 distinct rows reproduce, one does not (marked). Sec. 5: own reasoning on digest facts; numbers computed by me are marked "computed". Nothing is **checked** against the PDF, nothing is from **memory** except where a line says so |
| extracted by, date | literature extraction sub-agent (Claude), 2026-10-24 |

## 1. Problem and claim

Input: surround cameras (7 in the config, Argoverse 2 ring cameras) plus an SD map built from OpenStreetMap (OSM) around the ego pose; output on OpenLane-V2 subset A: lane centre-lines, traffic elements, lane-to-lane topology and lane-to-traffic-element topology. Gap as the paper frames it: HD maps are costly, SD maps are cheap and global, but nobody had studied them as a prior for vectorised lane-topology prediction. Claim: a small transformer encoder over SD-map polylines, fused into a BEV lane-topology model by cross-attention and trained end-to-end without extra labels, lifts two hosts (a BEVFormer + deformable-DETR baseline and TopoNet) on every metric; the gain is largest at far range and at intersections. The SD map is a **geometry prior for lanes, not a route**: all roads are given with a road class and none is marked as the planned route. Pose noise, wrong maps and missing maps are not studied.

## 2. Method as built

### 2.1 Pipeline [digest]

```
OSM (osmnx, network_type all) crop around ego pose -> polylines, class per polyline (8 classes)
   -> ego frame (translate to ego, rotate by -yaw)                       [offline, per frame, pickle]
   -> resample every polyline to N = 11 evenly spaced points (x, y)
   -> normalise (x, y) to (0, 2 pi) w.r.t. the BEV range, sinusoidal embedding (16 ch per coordinate, T = 1000)
   -> flatten 11 points: 11 x 32 = 352 ch, concat one-hot class (8)  -> token input 360 ch, one token per polyline
   -> Linear(360 -> 256) -> 6 x TransformerEncoderLayer (d 256, 4 heads, ffn 256 in the config)  -> S in R^(M x 256)
BEVFormer encoder, 3 layers, BEV grid 100 x 200 (0.512 m cells), 256 ch, 7 cameras
   per layer: temporal self-attn -> LN -> spatial cross-attn to image features (deformable) -> LN
              -> MAP CROSS-ATTENTION (4 heads, queries = BEV cells, keys / values = S) -> LN -> FFN -> LN
   -> BEV features -> host lane head (TopoNet: 200 queries, SGNN decoder; baseline: 100 queries deformable DETR)
```

- **Inputs, outputs.** M polylines per frame (M is not stated anywhere; in the code the maps are padded to the largest M in the batch and a key-padding mask is built from the counts). Each polyline is an ordered list of 11 (x, y) points in ego metres plus one class id. Outputs are the host's: 11-point 3D centre-lines (200 queries for TopoNet), traffic-element boxes, lane-lane and lane-traffic-element relation matrices. Evaluation window 50 m ahead and behind, 25 m to each side.
- **SD map attributes that reach the network (code, digest).** Geometry of an OSM edge and **one** class out of eight: road (OSM `road`), pedestrian group (pedestrian, footway, path, steps; the dataset class table also maps cross_walk and side_walk here), truck_road (motorway, trunk and their links), highway (primary, secondary, tertiary and links), residential (residential, living_street), service, bus_way (busway, bus_guideway), other (every other tag, e.g. cycleway, track, unclassified). The paper's Table IV groups them differently ("Road" = highway + residential, "Ped-Way" = pedestrian ways and crossings, "Others" = bus_way, truck_road, service and similar). **Not kept:** node ids and adjacency (so "road-level topology" is implicit in the point geometry), lane count, width, one-way flag, name, speed limit, any route or destination information. A polyline with several OSM tags is exploded into several polylines with one class each (code comment: unresolved TODO).
- **Frame.** Points are transformed into the ego frame with the 3-DoF pose G_p (2-D position + heading) which the paper calls the GPS pose; in the code it is the dataset's own per-frame pose (`frame.get_pose()`), converted to WGS84 for the OSM query. No noise is added anywhere in the pipeline.

### 2.2 Equations [digest; the paper's equations are unnumbered, numbering is mine]

- **E1 embedding (paper).** `E(p, 2j) = sin(p / T^(2j/d))`, `E(p, 2j+1) = cos(p / T^(2j/d))`, T = 1000, d = 32.
- **E2 normalisation (paper text and config).** `p' = (p - offset) / range * 2 pi` per coordinate; config: range = [102.4, 51.2] m, offset = [-51.2, -25.6] m (the point-cloud range), scale left at its default 2 pi. Table VI shows this step is worth +6.8 DET_l.
- **E3 code form of E1.** For channel index i of one coordinate: divisor `T^(2 floor(i/2) / num_feats)`, `num_feats` = 16, so each coordinate gives 16 channels (8 sin, 8 cos) and each point 32 channels; the paper's d = 32 is therefore the per-point width, not the exponent denominator. The sin / cos layout (interleaved or blocks) could not be settled from the digest.
- **E4 token.** `t_k = W [ E(p'_1,x), E(p'_1,y), ..., E(p'_11,x), E(p'_11,y), c_k ] + b`, input width 11 x 32 + 8 = 360 (config comment), W in R^(256 x 360); `c_k` one-hot class (a single active entry per polyline).
- **E5 encoder.** `S = Enc_6(t_1..t_M)`: 6 stacked `nn.TransformerEncoderLayer` (dropout, activation and norm placement are not set in the file, so PyTorch defaults apply), d_model 256, 4 heads, feed-forward width 256 (the class default is 2048; the config overrides it with a TODO), batch handled sample by sample (no padding inside the encoder), no positional or order encoding of the polylines (a set).
- **E6 fusion (code).** In every BEV encoder layer, after the spatial cross-attention and its LayerNorm: `Q <- Q + MHA(q = Q + pos_bev, k = S, v = S, key_padding_mask = pad(counts))`, then LayerNorm. The residual excludes the positional term; **keys carry no positional encoding** (their position information sits inside the token, as absolute normalised ego coordinates); every one of the 20,000 BEV cells attends to every polyline (no locality prior, no relative geometry); no output projection beyond the one inside `nn.MultiheadAttention`; attention dropout 0.
- The paper states the fusion only as "multi-head cross-attention between the BEV feature queries and the SD map features in each intermediate layer, after each spatial cross-attention" and does not give the query / key / value roles, the residual or the normalisation; those come from the code.

### 2.3 Losses, labels, training, inference

- **Losses.** None for the map branch ("without any additional training signals"). The host's losses are unchanged. TopoNet config: lane classification focal (alpha 0.25, gamma 2) weight 1.5, lane L1 weight 0.025, topology lane-lane and lane-traffic-element focal weight 5 each, detection branch classification 1.0, L1 2.5, GIoU 1.0. **Assignment is Hungarian** in the host (lane assigner with focal cost 1.5 and L1 cost 0.025; detection assigner 1.0 / 2.5 / 1.0). The map gradient arrives only through the BEV features.
- **Labels.** OpenLane-V2 subset A (Argoverse 2) ground truth, 201-point lanes down-sampled to 11 points (host); the SD map needs no labels. SD polylines come from a fresh OSM query per frame (see the code file, section 4).
- **Training recipe (config, digest; the paper text gives none of it).** AdamW lr 2e-4, weight decay 0.01, image backbone lr x 0.1, gradient clipping 35, cosine annealing with linear warm-up (500 iterations, ratio 1/3, min-lr ratio 1e-3), 24 epochs, 1 sample per GPU on 8 GPUs (README: "the paper used 8 GPUs"), no fp16, ResNet-50 (torchvision weights, stage 1 frozen, BN frozen) + FPN, 3 BEV encoder layers, evaluation from epoch 20. GPU type, time, seeds: not found. The paper states only: official implementations, ResNet-50, N = 11, d = 32, L = 6, T = 1000; the baseline's output was changed to TopoNet's 11-point representation for a fair comparison.
- **Inference.** Host's. No map-specific step (no thresholding, no NMS, no map filtering). The map is mandatory: there is no switch to drop it (see section 4).

## 3. Numbers [digest]

Protocol: OpenLane-V2 subset A val, standard split unless "geo-disjoint"; DET_l = mean AP of lane centre-lines over Frechet thresholds {1, 2, 3} m, TOP_ll = lane-lane topology, DET_t = traffic-element AP at 0.75, TOP_lt = lane-traffic-element topology, OLS = mean of DET_l, DET_t, sqrt(TOP_ll), sqrt(TOP_lt) (scores as fractions, result x 100).

### 3.1 Main results and the arithmetic check

**Table I (digest, matches the README table)**

| model | DET_l | TOP_ll | DET_t | TOP_lt | OLS |
|---|---|---|---|---|---|
| Baseline (BEVFormer + deformable DETR) | 17.0 | 2.3 | 48.5 | 16.2 | 30.2 |
| Baseline + SMERF | 26.8 | 3.9 | 48.9 | 19.2 | 34.8 |
| TopoNet | 28.2 | 4.1 | 44.5 | 20.6 | 34.5 |
| TopoNet + SMERF | 33.4 | 7.5 | 48.6 | 23.4 | 39.4 |

Printed relative gains: Baseline DET_l +57.6 %, TOP_ll +73.1 %, OLS +15.3 %; TopoNet DET_l +18.7 %, TOP_ll +83.8 %, OLS +14.2 %. The abstract says "up to 60 %", which only fits DET_l of the baseline; the TOP_ll gains are larger. No seeds, no confidence intervals: single numbers.

**Arithmetic check (computed).** OLS reproduces from the other four columns within rounding for the rows of Tables I, III, IV, VI, VII and for three of the four rows of Table V (16 distinct rows checked, 15 reproduce, e.g. baseline: (17.0 + 48.5 + 15.2 + 40.2) / 4 = 30.2). **One row does not reproduce: Table V, Baseline on the geo-disjoint split, OLS printed 16.9, the printed four columns give about 19.0** (a DET_t of about 32.8 instead of 41.0 would give 16.9). The README repeats 16.9, so it is probably printed that way in the paper; either a typo or the digest has a wrong DET_t. Treat that row with care.

**Table II, split by distance and intersection (digest; columns DET_l / TOP_ll / TOP_lt / OLS; DET_t omitted by the paper; the printed improvement rows were garbled by the tool and are not copied)**

| model | close 0 - 25 m | far 25 - 50 m | non-intersection | intersection |
|---|---|---|---|---|
| Baseline | 20.4 / 4.8 / 16.3 / 32.8 | 16.3 / 2.3 / 14.1 / 29.4 | 20.3 / 3.0 / 16.2 / 31.6 | 14.7 / 2.0 / 11.9 / 27.9 |
| Baseline + SMERF | 22.4 / 6.6 / 18.4 / 35.0 | 26.3 / 3.9 / 17.5 / 34.2 | 29.2 / 5.3 / 19.3 / 36.3 | 21.2 / 3.7 / 14.9 / 32.0 |
| TopoNet | 28.0 / 9.1 / 22.6 / 37.6 | 26.5 / 4.4 / 18.2 / 33.6 | 32.7 / 8.1 / 21.2 / 37.9 | 23.7 / 4.6 / 16.0 / 32.4 |
| TopoNet + SMERF | 32.4 / 12.6 / 24.3 / 41.5 | 33.0 / 7.7 / 21.7 / 39.0 | 37.0 / 11.9 / 23.7 / 42.2 | 28.5 / 8.1 / 19.5 / 37.4 |

OLS drop from close to far: baseline 3.4 without, 0.8 with SMERF (matches the text); TopoNet 4.0 without, 2.5 with (the text says 1.5; the table gives 2.5). The "close / far" split is by longitudinal distance: a centre-line is "close" if all its points have |x| <= 25 m, otherwise "far"; the intersection flag is the dataset label of the nearest ground-truth centre-line (code, digest).

### 3.2 Ablations (all on the baseline host, standard split, unless stated)

**Table III, SD-map representation**

| SD map input | DET_l | TOP_ll | DET_t | TOP_lt | OLS |
|---|---|---|---|---|---|
| none | 17.0 | 2.3 | 48.5 | 16.2 | 30.2 |
| raster map (heat map, one-hot road type, ResNet-50 features) | 20.6 | 2.0 | 49.4 | 15.5 | 30.9 |
| VectorNet-style (polyline graph + GNN) | 19.1 | 2.3 | 50.3 | 16.2 | 31.2 |
| SMERF | 26.8 | 3.9 | 48.9 | 19.2 | 34.8 |

The raster does **not** help topology (TOP_ll 2.0 below the no-map 2.3). The raster branch was a CNN over a low-resolution categorical image; its resolution and extent were not seen, and its configuration is not in the repository, so "raster loses" is a statement about this raster branch, not about rasters in general.

**Table IV, road classes included (OLS 30.2 none, 33.4 road, 34.8 road + ped-way, 34.8 all):** DET_l 17.0 / 20.6 / 22.7 / 26.8; TOP_ll 2.3 / 3.7 / 3.8 / 3.9; DET_t 48.5 / 51.5 / 52.3 / 48.9; TOP_lt 16.2 / 18.0 / 19.8 / 19.2. More classes help the lanes; traffic-element detection drops again with the last group.

**Table VI, encoder components:** map transformer alone 18.2 / 2.6 / 48.1 / 16.8 / OLS 30.9; plus x,y sinusoidal position encoding 20.0 / 3.9 / 49.6 / 18.9 / 33.2; plus normalisation to (0, 2 pi) 26.8 / 3.9 / 48.9 / 19.2 / 34.8. The text says the position encoding is crucial for topology; the depth of the transformer alone is worth +0.7 OLS, coordinate encoding +2.3, normalisation +1.6.

**Table VII, number of heads:** 4 -> OLS 34.8 (DET_l 26.8); 8 -> 34.5 (22.8); 16 -> 33.9 (21.3). The authors explain it by the smaller per-head width with this data size.

**Table V, geographically disjoint split** (train and val share no area; sizes and cities not given): Baseline 5.4 / 0.3 / 41.0 / 5.7 / OLS 16.9 (see the check above); Baseline + SMERF 8.8 / 0.5 / 46.3 / 6.9 / 22.1; TopoNet 14.9 / 1.0 / 34.3 / 7.6 / 21.7; TopoNet + SMERF 17.0 / 1.4 / 35.4 / 8.6 / 23.4. Relative gains stay large for the weak baseline, absolute gains shrink; on TopoNet the OLS gain is +1.7 (standard split +4.9). The same baseline falls from DET_l 17.0 to 5.4 when the split becomes location-disjoint, so the standard split leaks location.

**What was NOT ablated:** SD-map noise, pose error or misalignment, dropped or missing polylines, a wrong map (other place, other frame), map quality (OSM completeness), map age, or the number of polylines. The map is also never removed at test time.

### 3.3 Compute

Not reported (no parameter count, no latency, no FPS, no memory; README: the baseline may need too much memory for most machines, the lighter TopoNet config is suggested). **Computed from the config values** (assumes `nn.TransformerEncoderLayer` with d 256, 4 heads, ffn 256, default LayerNorms; not the authors' number): Linear 360 -> 256 = 92,416; encoder 6 x 395,776 = 2,374,656; three map cross-attention modules x 263,168 = 789,504; three extra LayerNorms 1,536; **about 3.26 M parameters added**, of which 2.47 M in the SD encoder. The cross-attention cost is 20,000 queries x M keys x 4 heads x 3 layers; with M in the tens this is small next to the deformable image attention.

### 3.4 Caveats

OpenLane-V2 numbers depend on the devkit version (OVERVIEW A9 records the same model at TOP_ll 4.1 or 10.9 across devkit versions; here the repository ships its own copy of the devkit and asks for v1.0.0 data); TOP_ll values of 2 - 8 are low and noisy; no seeds; subset A only (Argoverse 2, 2 Hz, 100 x 50 m window); the survey (arxiv 2409.02415) and the OpenLane-V2-OSM paper (local note PL44) report different TopoNet baselines (34.8 against 34.5 here) and a raster fusion for TopoNet at OLS 37.7 against 39.4 for SMERF here, so raster against polyline is **not settled** across papers (different protocols, do not subtract).

## 4. Paper versus code [digest]

| item | paper | code |
|---|---|---|
| venue | ICRA 2024 (notes) | arXiv only on the pages opened |
| encoder width, heads, ffn | H not given, 4 heads implied by Table VII | d 256, 4 heads in the configs; class defaults are 8 heads, ffn 2048, input 30; config ffn 256 with a TODO |
| road classes | six named + catch-all (7) | eight one-hot slots: road, pedestrian group, truck_road, highway, residential, service, bus_way, other |
| d = 32 | "embedding dimension" | 16 channels per coordinate (32 per point) |
| normalisation | to (0, 2 pi) w.r.t. the BEV range | divides by the point-cloud range (102.4 x 51.2 m) but **does not clip**: polylines reaching 80 - 100 m give arguments beyond 2 pi |
| map branch | "after each spatial cross-attention" | op `cross_attn_graph` after the spatial one; only `BEVFormerLayer` implements it, the generic base layer skips it silently |
| pose | "GPS pose" | dataset pose, no noise, no perturbation anywhere |
| dropping the map | not discussed | impossible: no flag; `map_encoder=None` raises, `map_dir_prefix=None` raises |
| temporal BEV | BEVFormer temporal | the history-BEV path (`obtain_history_bev`) is built **without** the map |
| raster baseline | Table III | no raster config in the repository; the embedding modules and a random affine on the raster (`AffineTransformSDMap`, rotation and translation, nearest interpolation, fill 191) exist, whether that was used is not visible |
| geo-disjoint split | Table V | the script only filters stored key lists; the step that decides disjointness is not in the repository |

## 5. Transfer to our pipeline

| question | answer |
|---|---|
| what it needs from us | SD roads as polylines in the ego frame (our simulated ADASIS-like SD map must be available as polylines, not only as a raster) with a class per polyline; vertex tokens (128-d) and their ego positions; **no** image features, lidar or trainable BEV encoder; the pose is the registration problem |
| what it gives us | A6 / A7: a vector encoding of the SD map (11-point polyline tokens, normalised sinusoidal coordinates, class one-hot, set transformer) and a key-padded cross-attention recipe; evidence that coordinate encoding and normalisation, not depth, carry the gain. It gives **no** route attribute, no registration, no map-trust output |
| conflicts | Hungarian matching and DETR queries in the hosts (we do not use the hosts); the encoder imports mmcv registries (maths is plain PyTorch); absolute ego coordinates assume an exact pose; NVIDIA licence is non-commercial (see the code file) |
| reuse mode | re-implement from the description (about 80 lines of PyTorch); idea level for the rest |
| effort | S for a polyline-token SD encoder plus key-padded cross-attention in the route head; M for the data side (polylines with route attributes from our simulated map, perturbation on the fly); L if the registration offset is learned jointly |
| expected gain and risk | not established by this paper for route attribution. Clean-case route success is already 0.885 - 0.889, so headroom is in offset / failure cases and in abstention. Risk: absolute-coordinate tokens identify the place and can be memorised (SMERF's own geo-disjoint result; our splits are not location-disjoint, OVERVIEW A9) |
| feeds which experiment / K id | proposed new K (register first): "SD polyline tokens with route attributes and a null token" (below). Neighbours: K17 (route head), K74 (C03, noise and wrong-map training), K75 (MAT), K99 (pose volume), K185 (route asked along the edges) |

### 5.1 How the route could be marked on the SD tokens (SMERF has no route; this is our design, not the paper's)

SMERF puts every OSM road into the token set and distinguishes only the road class. A route attribute needs three additions, in increasing cost:

1. **Route flag per polyline (road level).** Append one bit "on the route" to the class one-hot (361 inputs instead of 360). The route is a path in the SD graph, so the flag is exact when the route is given as a sequence of SD edges; if it only exists as an ADASIS-like horizon polyline, the loader must map-match it onto the edges once (distance of the sampled points to the polyline). Keep **all** other roads as tokens, including the branches that leave the route, because the lane-level decision needs the alternative at a fork. The soft route probability of a vertex is then the attention-weighted mixture of the flags of the roads it attends to, `p_road(v) = sum_k a(v,k) r_k`, which is also a soft, differentiable road assignment (the nearest thing in this family to MAT's hard assignment, K75). The attention weights are the quantity to supervise or inspect.
2. **Direction and progress along the route.** OSM edges have an arbitrary digitised direction and SMERF stores no one-way flag; the 11-point sequence is read by a linear layer, so a reversed polyline is a different token. For route polylines, orient the points in travel direction and add per-point arc length along the route from the ego (the ADASIS offset), sinusoidally encoded, plus a manoeuvre label at the polyline end (straight, left, right) taken from the route geometry. This is what separates "this lane continues on the route" from "take this lane to turn right in 80 m".
3. **Lane level.** The code keeps neither lane count nor one-way. Keeping the OSM `lanes` and `oneway` tags (a 2025 follow-up adds lane count and width to SMERF, per a search summary, not opened) gives the head a way to relate the lateral index of a perceived lane within its road (cluster the perceived vertices by road and lateral offset) to the number of lanes and the manoeuvre at the end of the road. Where the tags are missing, the attribute stays at road level. Final output as a gated mixture, `p(v) = g(v) p_lane(v) + (1 - g(v)) p_road(v)`, where `g` is a learned "lane resolvable" gate; this is the lane-level-with-road-level-fallback behaviour asked for in OVERVIEW section 3, item 2.

**Abstention and missing map.** SMERF's attention has no way to say "none of these roads": the softmax must put its mass on some polyline, and an empty map gives zero keys (untested here; that attention over fully masked keys returns NaN in older PyTorch versions is from memory, not checked). Add one learned **null token** to the key set (SuperGlue's dustbin idea, C30): mass on the null token is the abstain signal, and an empty or fully masked map degenerates to the null token instead of an error. Train it with the C03 recipe (local note, digest level): shift noise sigma = 1 m, about 10 % element dropout, about 10 % wrong-map samples (other frame or other place); the abstain target for corrupted samples is known by construction.

### 5.2 Misalignment

SMERF does nothing about it: exact dataset pose, absolute normalised coordinates inside the token, learned absolute BEV position as query position, no relative geometry, no augmentation. Our own runs (local notes) show what an exact pose is worth: +0.075 F1 with exact pose and -0.057 unaligned for the map prior at the grid (E40 / E48). What a vector encoding adds for us:

- **Perturbation on the fly.** Rotate / shift the polyline coordinates before tokenising (exact, no resampling artefacts; SMERF's only perturbation code is a nearest-neighbour affine on a raster). The OpenLane-V2-OSM note (PL44, digest level) reports that rotation hurts much more than translation (10 deg rotation: 13.64 - 21.14 % performance loss, against 0.45 - 10.82 % for 0.25 - 2.0 m translation at 0 deg), so rotation must be in the sweep.
- **Differentiable registration through the token.** The coordinate-to-embedding map is smooth, so a predicted SE(2) correction (from a pose volume, K99, or a few learned parameters) applied to the polyline points before the embedding receives gradient from the route loss; with a raster the same needs a differentiable warp.
- **Relative instead of absolute geometry.** Because keys carry absolute ego coordinates, a wrong pose shifts every key against every query. Encoding the polyline points relative to the query vertex (vertex-centred, heading-aligned) removes the global pose from the attention score and leaves the registration to a separate small offset head. This is not in SMERF; it is the main change we would test against it.

### 5.3 Reusable parts for the route head (vertex tokens attending to SD tokens)

| SMERF part | use in our route head | evidence |
|---|---|---|
| normalised sinusoidal coordinates (range = BEV window, 16 ch per coordinate, T = 1000) | encode vertex positions (query side) and SD points (key side) in the same coordinate embedding; check whether the E104b raster tokens already carry a normalised coordinate encoding | Table VI: +2.3 OLS for position encoding, +1.6 for normalisation, +0.7 for the transformer alone |
| polyline = one token (11 points + class, Linear to 256) | alternative to raster tokens; one token per road gives a natural unit for the road-level fallback and for route flags | Table III (this host only) |
| key-padding mask from per-sample counts | batch-safe cross-attention over a variable number of SD tokens | code (digest) |
| 4 heads, post-norm residual, no key position, one cross-attention per BEV layer | small head; 4 beats 8 and 16 at this data size | Table VII |
| self-attention among SD tokens first | context along the road network; cheap, but the paper gives it little credit alone | Table VI |
| all roads with class, not only the route | keeps alternatives visible for fork decisions | Table IV (more classes -> better lanes) |

Wrong or missing SD map in SMERF: not tested, and the code cannot run without a map. Evidence from other papers is in the local notes C03 (wrong-map and dropout sweeps) and PL44 (pose error), at digest level.

## 6. Not found / open

- Number of polylines M per frame, the OSM query radius actually kept (the script crops 80 m x 100 m boxes before simplification; whether points outside the BEV window are removed later was not seen), duplicates of two-way roads (OSM graphs hold both directions; not checked), encoder width H in the paper, loss weights in the paper text, optimiser and epochs in the paper text, GPU type and training time, parameters, latency, memory, seeds and variance.
- Any experiment with noisy, shifted, rotated, partial or wrong SD maps, a missing map at test time, map age, or OSM quality; the cross-attention weights are never shown (what the model attends to is unknown).
- Construction and size of the geographically disjoint split (the script reads stored key lists; the step that makes them disjoint is not in the repository); the unexplained OLS value of Table V, baseline row.
- The raster baseline's resolution, extent and fusion point; whether SMERF was compared with the same hyper-parameters as the raster.
- The venue from a page I could read (ICRA 2024 rests on a search summary).
- Per-file licence headers in the repository (the first file read has none).

## 7. Our experiments that touch it

From the local notes (not re-run here): E104b (route head, cross-attention from vertex tokens to SD raster tokens, 0.69 M parameters, kept; note C09 says the pattern is the same as SMERF's); E40 / E48 (map prior at the grid and learned registration, kept; +0.075 F1 exact pose, -0.057 unaligned); map offsets of 0 - 3 m in route-head training and evaluation (done, per note C03); K185 (route head asked along the edges: 0.798 -> 0.888, control 0.885). **Not run:** polyline tokens instead of raster tokens, route flag / progress / manoeuvre attributes on SD tokens, a null token, wrong-map and element-dropout samples, rotation sweeps, a location-disjoint evaluation of the route head.
