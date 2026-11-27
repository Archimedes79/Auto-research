# GR10 - SAM-Road (with SAM-Road++ and LineGraph2Road in brief): full-text extraction

| field | value |
|---|---|
| paper | Main paper: Segment Anything Model for Road Network Graph Extraction. Hetang et al. (full author list not read; first author per the level-1 note and the repository citation), CVPR 2024 Workshops (2nd Workshop on Scene Graphs and Graph Representation Learning), arXiv 2403.16051. Brief: SAM-Road++ = Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method (Yin, Li, Cao, Yao, Liu, Bai, Zhou, Meng; arXiv 2411.16733v1, 23 Nov 2024; CVPR 2025 per the repository README); LineGraph2Road = Structural Graph Reasoning on Line Graphs for Road Network Extraction (Wei, Jing, He, Suckale; arXiv 2602.23290v2, dated 8 Aug 2026 on the page; no venue listed). |
| axis, tier, K ids | A3 (sparse lane graph; family "graph extraction from dense maps with link verification"), tier 1 core; K48 (nodes + pairwise edge classifier), K13 (candidate edge lists), baseline arm B2 / mix M10 of the concept plan; the level-1 note also lists S6, S3, S2 |
| level-1 note(s) | [GR10](../papers/GR10_sam-road-sam-road-linegraph2road.md) (one note for all three papers) |
| code extraction | [sam_road](../code/sam_road.md) (SAM-Road, MIT). SAM-Road++ (earth-insights/samroadplus) and LineGraph2Road (wzzzzzzy/LineGraph2Road) have no LICENSE file: described only, in the code file |
| sources read | arxiv.org/html/2403.16051 (HTML, four queries: structure, method, section 4 with all tables, a targeted re-read of decoder, Algorithm 1, BFS rule, test-tile count); arxiv.org/html/2411.16733 (HTML, one query: method, dataset, tables); arxiv.org/html/2602.23290 (HTML, two queries: method and tables; appendix 0.I read from character 100000 on; the middle of the page, with the sampling and line-graph definitions, was not returned); GitHub pages and raw files of htcr/sam_road (README, LICENSE, `model.py`, `dataset.py`, `graph_utils.py`, `graph_extraction.py`, `inferencer.py`, `utils.py`, two config files, `cityscale/generate_labels.py`, `cityscale_metrics/` scripts, commit list) and of earth-insights/samroadplus (`model.py`, `dataset.py`, listing) and wzzzzzzy/LineGraph2Road (listing); all read 2026-10-25. Local files read for sections 5 and 7: `literature/OVERVIEW.md`, `papers/GR10_*.md`, `experiment_execution/plans/Concept_Hypothesis_Tree_Lane_Graph_EN.md` and `Experiment_Table_EN.md` (grep only) |
| reading marks | every statement about the papers and the repositories is **digest**: all pages came through the summarising fetch tool (no page was seen raw, no PDF), so numbers may be mis-transcribed; line numbers of the code are the tool's estimates. Where I could cross-check by arithmetic it is said so ("my arithmetic"). Statements about our own pipeline are **checked** (local files). Section 5 is my analysis, not the papers' content. No claim is marked memory unless it says so |
| extracted by, date | Claude (Sonnet 5.5) literature-extraction agent, 2026-10-25 |

## Corrections to the level-1 note

1. **Interior samples are not SAM-Road's.** The note's title line and the catalogue's "why it matters" say the edge classifier works on sampled interior points. In SAM-Road itself the pair token holds features at the two end points plus the 2-D offset and nothing from the interior of the segment (paper section 3.4: no interior points; `TopoNet` code: source feature, target feature, offset). Interior or extension samples belong to **SAM-Road++** (road-mask values along the segment and its extensions) and **LineGraph2Road** (feature samples along the segment). Suggested wording: "Vertex masks + NMS, then a pairwise edge classifier on end-point features (SAM-Road); segment samples in the ++ and LineGraph2Road variants."
2. **"Input: 512 x 512 / 256 x 256"** are training crops. The images are 2048 x 2048 (City-scale) and 400 x 400 (SpaceNet), 1 m per pixel; inference runs a 16 x 16 grid of overlapping windows over each image (256 windows per tile, config `INFER_PATCHES_PER_EDGE: 16`) and averages.
3. **"Differentiable? partial"** is right but understates it: the edge head gets **no gradient at all towards vertex positions**. It is trained by teacher forcing on vertices emulated from the ground-truth graph, the vertex positions are integer pixels from a non-differentiable NMS, and the two heads share only the encoder. There is no end-to-end path from edge loss to the mask head.
4. **SAM-Road++ "node-guided resampling"** is described in the note as "training pairs use the vertices that NMS would choose at inference". Paper (digest): the sources stay GT nodes, each *target* is replaced by the maximum-probability point of the predicted road mask within r = 8 px and keeps its GT connectivity label. Code (digest): targets are snapped to the pixel with the highest summed keypoint + road score inside a **radius-2** window of the predicted mask, in training and at inference. So it is snapping onto predicted-mask peaks, not a run of the inference NMS. The 8 px (paper) against 2 px (code) difference is unresolved.
5. **SAM-Road++ extended-line samples**: the note is right for the paper (n = 15 at each extension, 8 px long, m = 20 along the segment); the code adds a +1 px neighbour in x and in y to every sample (three reads per sample) and indexes the predicted road channel by integer position, giving a 150-dimensional line vector per pair. The ablation Table 4 is not labelled with a dataset; its values equal the SpaceNet rows of Table 2 (my cross-check), so it is a SpaceNet ablation.
6. **Test-tile count:** the note says 27 City-scale test tiles; the SAM-Road page text says 29 of 180 (digest), while LineGraph2Road says 27 and the repository's TOPO driver loops over 27 tiles. Unreconciled; 27 is the more probable figure.
7. **Coupled NMS radii 8 / 16** are credited to LineGraph2Road in the note; they are already SAM-Road's own config values (`ITSC_NMS_RADIUS: 8`, `ROAD_NMS_RADIUS: 16`), although the SAM-Road paper text says d_v = 16 for both masks. LineGraph2Road only names the scheme "Coupled NMS" (its Table 4: +0.26 F1 on City-scale, +0.78 on SpaceNet over plain NMS).
8. **LineGraph2Road edge features** ("two endpoints + 4 interior bilinear samples"): the page I could read says N_sampled = 4 feature positions per edge from the first to the second end point (digest); whether the 4 include the end points is not stated in the part I saw (ablation N = 2: F1 82.25, N = 8: 82.34, N = 4: 82.37). Not verified.
9. **Licence** (catalogue: "not noted"): SAM-Road **MIT, Copyright (c) 2024 htcr**; SAM-Road++ and LineGraph2Road: no LICENSE file in the root listing.
10. **Speed claim:** the abstract says 40 times faster on City-scale; Table 2 gives 231.0 min against 4.6 min, a ratio of about 50 (SpaceNet 112.8 / 8.2, about 14; the text says 10) (my arithmetic). Both are throughput over a whole test set on one RTX 4090, not latency (section 3).
11. **Project documents, not the note:** `Loop16_Triage_Plan_EN.md` and `Experiment_Table_EN.md` (row K85) say that arXiv 2411.16733 is "another paper" than SAM-Road++. The arXiv HTML page of 2411.16733 is the SAM-Road++ paper (title above). The level-1 note has the right id.

## 1. Problem and claim
- Predicted: a vector road-network graph (vertices, undirected edges) for a whole satellite tile, from an RGB image. Not directed, no lanes, no forks as such: a "junction" is a vertex of degree other than two.
- Gap: iterative growth methods (RoadTracer, RNGDet, RNGDet++) are accurate but slow (hundreds of minutes for the test set); segmentation + post-processing is fast but breaks topology. SAM-Road takes a fine-tuned SAM ViT-B image encoder, a tiny dense head for two masks (intersection, road), vertices by NMS on the masks, and a small transformer that scores pairs of nearby vertices as connected or not.
- The claim that matters: on City-scale it is on par with RNGDet++ in TOPO-F1 (77.23 against 78.44), best in TOPO precision and APLS on both data sets, and 40-50 times faster (4.6 min against 231.0 min). The foundation matters most: with ImageNet/MAE weights in the same ViT-B, F1 falls from 77.23 to 31.79 (ablation B).
- Profile worth noting for us: precision far above recall (City-scale P 90.47 / R 67.69, SpaceNet 93.03 / 70.97), i.e. short, correct graphs.

## 2. Method as built

### Pipeline (digest)
```
RGB tile (2048 x 2048 or 400 x 400, 1 m / px)
  -> overlapping windows of 512 x 512 (City) / 256 x 256 (SpaceNet), 16 x 16 windows per tile at inference
  -> SAM ViT-B image encoder (patch 16, 12 layers, window 14 + global at blocks 2,5,8,11, ~80 M params, fine-tuned, lr x 0.1)
        -> image embedding [256, P/16, P/16]            (paper: D_feat 128; code: out_chans 256, projected to 128 inside TopoNet)
  -> geometry decoder: 4 x ConvTranspose2d (stride 2) 256 -> 128 -> 64 -> 32 -> 2        (~0.17 M params)
        -> mask logits [P, P, 2]: channel 0 = intersection (keypoint), channel 1 = road
  -> (inference) average over windows, threshold, NMS -> vertices (x, y) in integer pixels
  -> topology decoder TopoNet, per source vertex: <= 16 nearest vertices within 64 px
        token_k = [proj(f_src), proj(f_tgt_k), tgt_k - src]  ->  Linear -> 3 x TransformerEncoder layer over the 16 targets
        -> 1 logit per pair -> sigmoid
  -> mean over windows, threshold -> edges
```
The encoder output is shared; the topology decoder reads the cached embedding by bilinear sampling.

### Inputs, outputs
- **Input:** RGB patch as raw 0-255 floats; the model normalises inside (SAM pixel mean/std). Vertex coordinates in patch pixels, x = column, y = row (the SpaceNet loader maps y = 400 - row); offsets are in raw pixels (not normalised).
- **Mask output:** [B, H, W, 2] logits, one pixel = 1 m. **Topology output:** logits and probabilities [B, N_sources, N_pairs = 16, 1].
- **Final output:** a pickled graph in Sat2Graph adjacency format (vertices as pixel coordinates, undirected edges).
- **Sizes:** City-scale patch 512, batch 16, 512 source vertices per patch; SpaceNet patch 256, batch 64, 128 sources per patch (so 8,192 and 2,048 candidate pairs per patch).

### Equations
The paper prints **no numbered equations**; only Algorithm 1 (vertex NMS) and prose. I write the implicit formulas as E-numbers of my own so that section 5 can refer to them (reconstructed from the paper text and the code digest, not numbered in the paper):
- E1 (pair token): t_k = ReLU(W_p [ReLU(W_f f(src)); ReLU(W_f f(tgt_k)); d_k]), f(.) = bilinear sample of the embedding map at the pixel position divided by the patch size, d_k = tgt_k - src (2-D), W_f: 256 -> 128, W_p: 258 -> 128.
- E2 (set context): z_1..z_16 = TransformerEncoder(t_1..t_16) with 3 layers, d_model 128, 4 heads, FFN 128, dropout 0.1, ReLU; a key padding mask hides invalid targets; if a whole source has no valid target the mask is flipped to all-valid to avoid NaN.
- E3 (edge probability): p_k = sigmoid(w^T z_k + b). Each pair has its own sigmoid (no softmax over the 16 targets): a source may have several positive targets.
- E4 (vertex NMS, Algorithm 1): keep pixels with score above a threshold, sort by score descending, walk the list and delete every later pixel within radius d_v of a kept one.
- E5 (loss): L = BCE_mean(mask logits, masks) + (sum over valid pairs of BCE(topology logit, label)) / (number of valid pairs). Weights 1 : 1.

### Losses
- Three binary cross-entropy tasks (intersection mask, road mask, edge connection), equal weights, no re-weighting (paper); in code the two masks form one [B,H,W,2] BCE-with-logits (mean over all elements; a sigmoid-focal-loss switch `FOCAL_LOSS` exists and is off in the configs I read) and the topology term is a masked mean over valid pairs.
- **No class balancing anywhere** (no pos_weight; digest of the code). The masks are about 1-3 % positive, so scores are under-confident; this is why the decision thresholds are tuned on validation (0.248 / 0.364 on City-scale).
- **No matching.** Pair labels come from graph reachability (below), not from a Hungarian assignment. Direction: none (undirected).

### Labels / ground truth
- **Masks** (script `cityscale/generate_labels.py`, digest): road mask = every GT edge drawn as a straight line of thickness 3 px; keypoint (intersection) mask = filled discs of radius 3 px at every GT node whose degree is not 2 (end points, junctions); both strictly binary (0 / 255), no blur, no distance transform, 2048 x 2048 PNGs written once.
- **Pair labels** (dataset class `GraphLabelGenerator`, digest): the GT graph is subdivided to a spacing of 4 px; subdivided points near crossovers (4 px) are excluded; every kept point gets a uniform random score in [0.9, 1.0] and every node of degree other than 2 gets a score of 2.0, then the same greedy NMS as at inference (radius 16 on City-scale) is run, and a score above 1.0 means "never suppressed" (junctions survive). This emulates the vertex set the inference NMS would produce, with random choice of where along a road the vertices fall.
- **Sources:** `TOPO_SAMPLE_NUM` (512 / 128) sources per patch are drawn **with replacement** from the NMS survivors with importance weights 0.9 for points within 32 px of an intersection or crossover and 0.1 for all others (hard-region sampling; not stated in the paper text).
- **Candidates and negatives:** per source, the 16 nearest survivors within 64 px (KD-tree; the self hit is dropped); short lists are padded with invalid slots. A candidate is **positive** if a breadth-first search on the (undirected) subdivided GT graph from the source reaches it within 16 hops (64 px / 4 px) without expanding through any other target (targets are "stop nodes": they are reached but not passed). All other candidates in the neighbourhood are **negatives**: there is no sampling ratio, the hard negatives come for free (parallel roads, the vertex behind the next vertex, the far side of a crossing).
- **Augmentation:** random 90 degree rotations (image, masks and graph points), random crop positions, Gaussian jitter of sigma 1.0 px on the graph points (code), no photometric augmentation mentioned.

### Training recipe
- Adam, base lr 1e-3 for the decoders, the encoder at 0.1 times that; weight decay none; paper: constant lr, training until validation plateaus; **config: 10 epochs on City-scale, 30 on SpaceNet, and `MultiStepLR` with milestone 9, gamma 0.1** (code digest; see section 4).
- One RTX 4090. Training time: not found. Seeds and variance: not reported.
- Epoch length in code: City-scale (2048 // patch)^2 x 2500 random crops (40,000 for patch 512); SpaceNet 84,667 (digest).

### Inference (the hidden rule steps)
1. Overlapping windows (config: 16 per edge, margin 64 px on City-scale, 0 on SpaceNet), batches of 64; encoder + mask decoder in float32; the per-window embeddings are cached.
2. Masks of all windows are summed and divided by the number of windows that cover each pixel (average), scaled to 0-255.
3. Vertices: threshold the keypoint mask (City-scale 0.248, SpaceNet 0.195) and run NMS with radius 8; threshold the road mask (0.364 / 0.341) and run NMS with radius 16; merge, rank the keypoint vertices above the road vertices, run NMS with radius 16 again (so the final minimum spacing is 16 px everywhere, as read).
4. Second pass over the windows: the vertices inside a window are found by an R-tree, a KD-tree gives the 16 nearest within 64 px per vertex, pairs are scored with the cached embedding; scores of the same ordered pair from different windows are averaged.
5. Edges: mean score above `TOPO_THRESHOLD` (City-scale 0.500, SpaceNet 0.705); ordered pairs are scored independently, so an edge appears if either direction passes (as I read the code; the writer converts to an undirected adjacency).
6. Thresholds were tuned on validation for F1; plain 0.5 everywhere costs 1.18 F1 / 0.42 APLS (Table 4, H).

### Differentiability of each step
| step | differentiable? | note |
|---|---|---|
| encoder, mask decoder, BCE on masks | yes | pixel-based dense head |
| GT mask rasterisation | n/a (labels) | binary discs and lines, no soft target |
| threshold + greedy NMS (vertices) | no | discrete selection, integer pixel positions; no sub-pixel refinement |
| training vertices | no gradient needed | teacher forcing on GT-derived vertices (random-score NMS), never on predicted ones (SAM-Road); SAM-Road++ snaps targets onto predicted-mask peaks by an argmax (no gradient through the position) |
| bilinear feature sampling at the vertices | yes (features, positions) | positions are constants here, so no gradient reaches them |
| pair token, transformer, sigmoid, BCE | yes | gradient reaches the encoder through the sampled features only |
| pair label (BFS with stop nodes) | n/a (labels) | graph search on GT, not a matching, but a rule on the target side |
| window averaging, thresholds, writer | no | inference rules |

## 3. Numbers
Author-reported, mixed protocols; do not compare with other papers or with ours (SAM-Road's TOPO uses a matching distance of about 10 m, README of the TOPO code, digest; our F1 uses 0.5 - 1.5 m).

**Table 1 (SAM-Road paper): City-scale (180 tiles of 2048 x 2048, 1 m / px, 20 US cities) and SpaceNet (2,549 tiles of 400 x 400; 382 test).** TOPO precision / recall / F1 and APLS:

| Method | City P | City R | City F1 | City APLS | SpaceNet P | SpaceNet R | SpaceNet F1 | SpaceNet APLS |
|---|---|---|---|---|---|---|---|---|
| Seg-UNet | 75.34 | 65.99 | 70.36 | 52.50 | 68.96 | 66.32 | 67.61 | 53.77 |
| Seg-DRM | 76.54 | 71.25 | 73.80 | 54.32 | 82.79 | 72.56 | 77.34 | 62.26 |
| Seg-Improved | 75.83 | 68.90 | 72.20 | 55.34 | 81.56 | 71.38 | 76.13 | 58.82 |
| Seg-DLA | 75.59 | 72.26 | 73.89 | 57.22 | 78.99 | 69.80 | 74.11 | 56.36 |
| RoadTracer | 78.00 | 57.44 | 66.16 | 57.29 | 78.61 | 62.45 | 69.90 | 56.03 |
| Sat2Graph | 80.70 | 72.28 | 76.26 | 63.14 | 85.93 | 76.55 | 80.97 | 64.43 |
| TD-Road | 81.94 | 71.63 | 76.43 | 65.74 | 84.81 | 77.80 | 81.15 | 65.15 |
| RNGDet | 85.97 | 69.78 | 76.87 | 65.75 | 90.91 | 73.25 | 81.13 | 65.61 |
| RNGDet++ | 85.65 | 72.58 | 78.44 | 67.76 | 91.34 | 75.24 | 82.51 | 67.73 |
| **SAM-Road** | 90.47 | 67.69 | 77.23 | 68.37 | 93.03 | 70.97 | 80.52 | 71.64 |

**Table 2: inference time for the complete test set** (16 x 16 windows, one RTX 4090; timing method not fully specified): Sat2Graph 150.6 min / 69.0 min; RNGDet++ 231.0 / 112.8 min; SAM-Road 4.6 / 8.2 min (City-scale / SpaceNet).

**Table 3: window density** (City-scale time, F1, APLS; SpaceNet time, F1, APLS): 16 x 16: 4.6 min, 77.23, 68.37; 8.2 min, 80.52, 71.64. 8 x 8: 3.3 min, 77.20, 67.21; 3.1 min, 80.84, 71.12. 4 x 4: 2.9 min, 77.00, 67.03; 1.7 min, 80.85, 70.88.

**Table 4: ablations on City-scale (F1 / APLS):**

| variant | change | F1 | APLS | change vs A (my arithmetic: F1 / APLS) |
|---|---|---|---|---|
| A | full SAM-Road | 77.23 | 68.37 | - |
| B | no pre-trained SAM (ImageNet-1K / MAE ViT-B) | 31.79 | 12.39 | -45.44 / -55.98 |
| C | no transformer in the topology decoder (dense layer on the pair features) | 73.75 | 59.39 | -3.48 / -8.98 |
| D | no vertex offsets | 77.36 | 66.67 | +0.13 / -1.70 |
| E | no target vertex features | 77.42 | 67.08 | +0.19 / -1.29 |
| F | no intersection mask | 71.94 | 64.62 | -5.29 / -3.75 |
| G | A* path finding on the cost field instead of the learned topology decoder | 69.21 | 63.32 | -8.02 / -5.05 |
| H | fixed 0.5 thresholds | 76.05 | 67.95 | -1.18 / -0.42 |

What decides transfer: (1) the **set context** (transformer over the 16 candidates) is worth far more than the pair features: removing target features or offsets changes F1 by less than 0.2 and APLS by 1.3 - 1.7, removing the transformer costs 3.5 F1 and 9 APLS; (2) the **keypoint (intersection) mask** is worth 5.3 F1; (3) a learned edge head beats a rule on the same masks (G) by 8 F1; (4) the foundation weights are decisive (B). Caveat: the released ablation switches for D and E look buggy (section 4), so these two rows may not measure what they say.

**SAM-Road++ (Table 2 of its paper; F1 / APLS, City-scale; SpaceNet):** SAM-Road 77.23 / 68.37; 80.52 / 71.64. SAM-Road++ 80.01 / 68.34; 81.57 / 73.44. SAM-Road++ pre-trained on its Global-scale set 80.66 / 69.55; 82.07 / 74.35. Global-scale test (3,468 tiles of 2048 x 2048, 13,800 km2; in-domain / out-of-domain 624 / 130 tiles): SAM-Road F1 59.80 / 46.64, APLS 59.08 / 40.51; SAM-Road++ F1 62.33 / 48.34, APLS 62.19 / 43.17.
**SAM-Road++ ablation (its Table 4, SpaceNet by cross-check; APLS / F1):** neither change 71.64 / 80.52; node-guided resampling only 71.90 / 81.77; extended-line only 73.22 / 80.89; both 73.44 / 81.57. The resampling-only row has a higher F1 than the full model; the paper does not discuss it. Gains over the base (my arithmetic): extended-line +1.58 APLS / +0.37 F1, resampling +0.26 APLS / +1.25 F1, both +1.80 APLS / +1.05 F1. No timing is reported for SAM-Road++.

**LineGraph2Road (Table 1, F1 / APLS):** City-scale 82.37 / 68.88 (83.77 / 70.40 with the overpass head); SpaceNet 84.24 / 73.94 (84.08 / 73.36 with overpass); Global-scale in-domain 64.43 / 67.69 (64.05 / 68.70). Ablations (City-scale, F1 / APLS, Table 3): no pre-trained SAM 77.08 / 61.63; no graph transformer 76.12 / 55.84; 3-layer MLP 74.20 / 52.20; N_sampled 8: 82.34 / 68.91; N_sampled 2: 82.25 / 68.11; original graph instead of the line graph 81.35 / 65.72; Table 4: Coupled NMS against plain NMS 82.37 / 68.88 against 82.11 / 68.57 (City-scale). Training: Adam lr 1e-3 (encoder 0.05 times), 10 epochs, four A100.

**Compute.**
- Parameters: image encoder about 80 M and mask decoder about 170 K (paper). Topology decoder: not stated; my count from the code digest: feature_proj 32,896 + pair_proj 33,152 + 3 transformer layers 3 x 99,584 + output 129 = **364,929, about 0.37 M**. The mask decoder count from the code (2 x 2 kernels) is 172,770, which matches the paper's "about 170 K"; with the 3 x 3 kernels the paper text states it would be about 0.39 M (section 4).
- Throughput (my arithmetic from Tables 2 and 3): City-scale 4.6 min / 29 tiles = 9.5 s per 2048 x 2048 tile (10.2 s with 27 tiles); SpaceNet 8.2 min / 382 = 1.3 s per 400 x 400 tile. With 4 x 4 windows the City-scale time falls only to 2.9 min (6.0 s per tile, factor 0.63) although the number of encoder passes falls by 16 times, so most of the time is **not the encoder** but the Python/CPU parts (R-tree, KD-tree, loops, NMS); the paper gives no breakdown (my inference).
- LineGraph2Road measured on an A100: City-scale 16 x 16 windows 87.19 s per image, 5 x 5 windows 10.51 s per image; SAM-Road 551.19 s and SAM-Road++ 576.51 s for the City-scale test set against 283.97 s for LineGraph2Road; on SpaceNet 918.47 s (SAM-Road) against 115.87 s. Different hardware and protocol from the SAM-Road paper (276 s and 492 s on the 4090): do not mix.
- No batch-1 latency, no memory figure, no FPS.

## 4. Paper versus code
(All digest; the code digest and the paper digest can both be wrong where they disagree.)
- **Decoder kernel:** paper text 3 x 3 transposed convolutions; code 2 x 2, stride 2 (256 -> 128 -> 64 -> 32 -> 2, LayerNorm2d after the first, GELU after the first three). The parameter count of about 170 K fits the 2 x 2 kernels (my arithmetic), so the paper text is probably a slip.
- **Learning rate:** paper "constant"; code `MultiStepLR` milestones [9], gamma 0.1 with `TRAIN_EPOCHS: 10` (City-scale), i.e. the drop falls in the last epoch; SpaceNet trains 30 epochs.
- **Feature width:** paper D_feat = 128 for the encoder output; code encoder out_chans 256, projected to 128 inside `TopoNet`.
- **NMS radii:** paper d_v = 16 for both datasets; config 8 for the keypoint mask and 16 for the road mask, final merge NMS at 16.
- **Thresholds:** paper "tuned on validation, 0.5 is similar"; config 0.248 / 0.364 / 0.500 (City-scale) and 0.195 / 0.341 / 0.705 (SpaceNet).
- **Source sampling:** the 0.9 / 0.1 importance weights near intersections and the random-score NMS with junction force-keep are not in the paper text.
- **Coordinate noise:** paper "small Gaussian"; code sigma 1.0 px (also in evaluation).
- **Mask generation:** paper describes it (3 px lines, radius-3 discs for degree other than 2); the code loads precomputed PNGs from `generate_labels.py`.
- **Ablation switches:** `TOPONET_VERSION` options `no_tgt_features` and `no_offset` are written as separate `if` statements, so a later `else` overwrites the earlier branch; the same is in the SAM-Road++ copy. If that is what the released code does, rows D and E of Table 4 cannot have been produced by it (they differ from A by +0.13 / +0.19 F1, which looks like noise). Not verified.
- **Extras in code, not in the paper:** LoRA option, SAM prompt-decoder option, focal loss option, A* graph extraction (the ablation-G path, with a cost field from the masks and the `tcod` library), larger SAM backbones.
- **Debugging leftovers:** `pdb.set_trace()` on NaN in the training step and in a coordinate transform helper.

## 5. Transfer to our pipeline

| question | answer |
|---|---|
| what it needs from us | The frozen cached BEV pyramid (39 ch, 200 x 200, 0.5 m) replaces the SAM embedding: the bilinear sampler and the pair head read it unchanged. The labels need our **directed** GT lane graph (nuScenes map expansion) with fork and merge nodes. No SD map, no image, no lidar, no ego pose. |
| what it gives us | A3: a link-based alternative to the verified-ray tree (concept plan M10, baseline arm B2); a set-context edge head that handles forks and merges as independent pair decisions; two label tricks (junction force-keep NMS with random scores, hard-region importance sampling of sources). |
| conflicts | Greedy NMS and thresholds are rule steps: acceptable as a top-k selection with gradient through the gathered values, not as a training-path argmax. Teacher forcing means no gradient from edges to vertex positions. Undirected edges and a BFS that ignores direction (`mode="all"`). The 64 px candidate radius is about the whole +-50 m field of view. Dependencies: PyTorch Lightning, wandb, python-igraph, rtree, shapely, scikit-learn, tcod (A* only), OpenCV; the `gcar` environment may lack some (no installs without the user's OK). The ViT encoder and its pre-training are not available (frozen BEV only). TOPO / APLS are road-network metrics with about 10 m matching: not ours. |
| reuse mode | MIT: adapt code is allowed with attribution, but the useful parts are small and tied to igraph / rtree; **re-implement** `TopoNet` (about 60 lines), the pair-label generator and the NMS in plain torch / scipy on our lane graph; SAM-Road++ and LineGraph2Road have no licence file: idea only. |
| effort | S: the pair head and the sampler (hours; we already use `grid_sample`). M: directed labels with stop nodes on projected candidates, heat-map candidates, a fair comparison with the tree (days). Blocker: a candidate generator that gives vertices where a shape-preserving graph needs them (5.1). |
| expected gain and risk | Prior evidence from our own work: learned link heads (E42, E45, E46, E51, E56) stayed within 0.01 of the rules, so expect a **tie in S** on clean frames. The place to look is junctions: the champion has J 0.265 against 0.408 at equal sparsity, and a head that sees all competing targets of a source at once is the published remedy for fork / merge confusion (ablation C: -3.5 F1, -9 APLS without the set context). Risks: parallel lanes 3.5 m apart flood the K nearest; chord features miss curved connectors; a second graph builder doubles the code to maintain. |
| feeds which experiment / K id | K48 (nodes + pairwise edge classifier) is the id; baseline B2 of the concept plan. Proposed new techniques, to be registered before use: (a) vertex-worthiness mask built from Douglas-Peucker oracle vertices and junctions with force-keep NMS; (b) directed BFS pair labels with stop nodes on candidate vertices; (c) pair survival = hazard product on a Hermite connector, combined with the set-context head; (d) node-guided snapping (SAM-Road++) as exposure-bias fix. |

### 5.1 The vertex mask replaced by our heat map
- **What SAM-Road's masks are.** Two binary targets: the road mask (3 px lines, the analogue of our Gaussian centre-line heat map, sigma 0.75 m) and a keypoint mask (discs of radius 3 px at nodes of degree other than 2). The vertices are NMS peaks of the *road* mask spaced 16 px apart, plus the keypoint peaks. The keypoint mask is the part that makes junction vertices exist; removing it costs 5.3 F1 (ablation F). Our counterpart of the keypoint mask is the junction head of loop 16.
- **Cheapest port (option V1).** Peaks of our existing heat map by max-pool NMS on the GPU (kernel 2r + 1 cells; r about 5 - 7 m, i.e. 10 - 14 cells), top-k with a cap (100 - 200), vertex confidence = the heat-map value at the peak. Differentiability: the selection is discrete, the gather is not, so the head still gets gradient through the confidence and through bilinear features at the peaks. A soft-argmax in a 3 x 3 window around each peak gives continuous positions, so the pair offsets and chord samples would also carry gradient to the dense head (my proposal; SAM-Road has integer positions only).
- **The flaw of V1 for our sparsity target.** A Gaussian ridge is flat along the lane, so the along-lane position of an NMS peak is set by noise; the spacing is then between r and 2r and not placed where the geometry needs a vertex. At 13.5 vertices per 100 m the mean spacing is 7.4 m (the Douglas-Peucker oracle needs 14.3 per 100 m, 7.0 m). Uniform spacing wastes vertices on straight lanes and misses curvature.
- **Option V2 (the real use of SAM-Road's idea).** Train a second dense channel whose target is discs at the **Douglas-Peucker oracle vertices** of the GT polylines plus fork, merge and end nodes (SAM-Road's degree-other-than-2 rule extended to curvature), and use the SAM-Road trick for priority: junction vertices get a score above 1 in the NMS so they are never suppressed, the others compete by score. The mask then learns where a short graph needs vertices; vertices per 100 m is then set by the target density and not by a radius. This directly serves the project's sparsity metric.
- **Exposure bias (teacher forcing).** SAM-Road trains pairs on vertices emulated from the GT with random scores, which differ from the real inference vertices. Our training forward pass has the real heat-map peaks available, so pairs can be formed from **predicted** candidates in training too (on-policy), with labels from projecting each candidate onto the GT lane graph (point-to-polyline distance, the primitive our losses already use) and the directed walk of 5.2. If one prefers a pure SAM-Road recipe: GT-emulated vertices for the edge head plus SAM-Road++'s snapping of targets onto predicted peaks (+1.25 F1 / +0.26 APLS in its ablation, small).

### 5.2 Directed lane edges, forks and merges
- **Labels.** Replace the undirected BFS by a **directed** walk on the GT lane graph: pair (i -> j) is positive iff j is reachable from i along the travel direction within L metres of path length without passing another candidate vertex (stop nodes, as in SAM-Road); the reverse pair (j -> i) is negative. A fork gives the source two positives, a merge gives the target two positive sources; because every pair has its own sigmoid (E3) and there is no softmax over targets, forks and merges need no special case.
- **Candidates.** Euclidean K nearest is the wrong set for lanes: three parallel lanes 3.5 m apart fill the 16 slots with lateral neighbours. Use a forward cone around the source heading (from the direction channel, say within 60 degrees), a longitudinal radius of about 20 - 30 m instead of 64 px, and K = 6 - 8; keep a small share of lateral candidates as hard negatives. This is the teardrop of the tree in another form.
- **Hard negatives by design.** SAM-Road's stop-node rule makes "the vertex behind the next vertex" a negative automatically; that is the shortcut-edge problem of any chord-based head and the main job of the set context (E2): each candidate token sees the other candidates of the same source. Keep that attention.
- **Merge as closure.** A tree grows forward and finds merges only when two branches end within the merge radius (1.5 m). A pair head scores "tip i -> existing vertex j" directly, so merges (and the weak junction recall of the champion) are a natural place to use it: tips whose ray ends near a foreign vertex get an explicit closure score.
- **Undirected duplicates.** SAM-Road keeps an edge if either direction passes. For directed lanes use the directed score only; the symmetric pair is a negative.

### 5.3 Edge verification on candidate pairs against the verified-ray survival algebra
- **What SAM-Road verifies with.** Only the two end features and the offset (E1); the middle of the segment is never read, so a chord is accepted or rejected on its ends. Its own ablations show that these features matter little (D, E), the set context a lot (C). SAM-Road++ (+1.58 APLS, +0.37 F1 on SpaceNet) and LineGraph2Road (interior feature samples, 4 positions) add samples from the interior and the extensions, which is the published support for reading *along* a candidate.
- **Our algebra.** A ray is read at ordered sample points; a hazard h_k at each, survival S = product of (1 - h_k), validity = survival to the end, length chosen by where survival drops. That is a structured, censored product with a trained hazard; SAM-Road++ feeds the same kind of samples (20 along the segment, 15 on each extension, 3 reads each) to an unstructured MLP / transformer.
- **Hybrid for pairs (proposal (c)).** For a candidate pair, build a Hermite or cubic connector from the two positions and the two direction-field headings (not a chord: lane connectors at junctions curve; E53 showed Bezier connectors are feasible), sample the heat map, the direction field and a few BEV channels at m = 12 - 20 points, and run the *same hazard head* as the ray verifier to get S_ij. Edge probability p_ij = S_ij * g_ij, or log S_ij as an input to the set-context head. The hazard head is shared, so ray verification and pair verification are one trained module, which also keeps the censoring semantics (an edge is valid only if the whole connector survives). The extension samples of SAM-Road++ correspond to evidence beyond the end points (is there a continuation behind j?), which for us is the next ray.
- **Cost.** About 100 - 200 vertices per frame (assumption: 13.5 vertices per 100 m over roughly 1 km of lane centre-line in view) with K = 8 give 800 - 1,600 pairs; 16 samples each by `grid_sample` is 13 - 26 k reads of a few channels. The pair head is about 0.37 M parameters (my arithmetic, section 3); with 1,600 tokens that is about 1.2 GFLOP (my estimate), well under 1 ms of arithmetic on the RTX 3060, so launch overhead decides: expect +1 - 2 ms, to be measured on an idle GPU, batch 1, cached features. SAM-Road's own "speed" does not transfer: it comes from replacing iterative growth over huge tiles, and its time is dominated by the encoder and CPU loops (section 3). Our chain is already 31 ms and the tree is not iterative over images.

### 5.4 Which functions could be ported (MIT, SAM-Road only)
| SAM-Road function (file, approximate line) | use | dependency gap |
|---|---|---|
| `TopoNet` (`model.py`, ~61-120) | pair head: projection, transformer over the targets, NaN-safe padding mask | none (pure PyTorch); add direction channels and connector samples |
| `BilinearSampler` (`model.py`, ~29-58) | sampling at points | we have `grid_sample`; the pixel-to-[-1,1] mapping needs `align_corners=False` care |
| `GraphLabelGenerator` (`dataset.py`, ~60-230) | label recipe: subdivision at fixed spacing, random-score NMS with force-keep, importance-weighted sources, BFS with stop nodes | igraph, rtree, shapely -> scipy `cKDTree` + networkx or a small sparse-graph BFS on our directed lane graph |
| `nms_points` (`graph_utils.py`, ~528) | CPU greedy NMS with force-keep | replace by max-pool NMS on the GPU (V1) or a torch greedy loop for N <= 1000 |
| `bfs_with_conditions` (`graph_utils.py`, ~548) | stop-node BFS | rewrite for directed edges |
| metrics (`cityscale_metrics/`) | none | not wanted (section 5 of the code file) |

### 5.5 Proposed experiment (not run; to be put on a card by the loop owner)
Arms on the same frozen features and the same VAL frames: (0) champion tree; (1) B2 literal: heat-map peaks (V1) + SAM-Road pair head, undirected labels, K nearest; (2) directed labels + forward cone + K = 8 (5.2); (3) arm 2 with the vertex-worthiness mask V2; (4) arm 3 with the survival connector (5.3). Report S and Q, F1@1, J at 6 m, vertices per 100 m, lanes found, ms per frame (idle GPU, batch 1, cached features), two seeds, paired scene bootstrap, noise band 0.010 in S. A prediction I would write down: tie in S for arms 1 - 2 (our link heads were ties), a gain in J and sparsity only if arm 3 beats arm 2; a loss in ms of 1 - 2.

## 6. Not found / open
- Paper: training time and number of epochs (the config gives 10 / 30), the topology decoder's parameter count, the number of heads (config: 4), the positive-to-negative ratio, the exact set of stop nodes in the BFS (digest: other targets; not verified), seeds and variance, the train / validation split (a file `data_split.json` exists, not read), the exact origins of the 16 x 16 windows (digest suggests a linspace over the image minus the margin, giving a stride of about 94 px for 2048 / 512 / margin 64: my arithmetic, assumption), the timing method of Table 2, the per-component time split, the SpaceNet crop details.
- The 29 against 27 test tiles; the 8 px against 2 px snapping radius of SAM-Road++; whether the SAM-Road++ target collection radius is 16 px (paper digest) or the config's 64.
- LineGraph2Road: the middle of the page (line-graph definition, sampling positions, overpass head, coupled NMS definition) was not returned; only the tables and the appendix timing were seen. Its code was not read.
- SAM-Road++ code: `modelinfer.py`, config files and the label script (the README names `generate_labes.py`, absent from the listing) were not read.
- APLS (a Go program in `cityscale_metrics/apls/`) and the TOPO code were not read in detail; TOPO provenance (Biagioni and Eriksson, Sat2Graph) and licence are not stated in the files I saw.
- No behaviour on lanes, on camera / BEV inputs, on directed graphs, or with a frozen backbone is reported; the whole evidence is aerial roads with a fine-tuned SAM.

## 7. Our experiments that touch it
- **K48 is cited by** E44 / E53 (learned split / merge key points, F1 0.34 / 0.38 to 0.55 / 0.61; Bezier connectors 0.71 / 0.64: the nearest relatives of SAM-Road's keypoint mask and of a connector between vertices), E51 / E56 (node-conditioned dense head; direction consistency as acceptance criterion: +0.003 F1; connectors lose 0.026 split F1), E81, E83 and E94 (Experiment_Table).
- **Learned link heads** E42 / E45 / E46 stayed within 0.01 of the rule links; E113 / E116 (junction and piece verifiers on the learned graph) lifted split / merge F1 from 0.385 / 0.428 to 0.437 / 0.504 and precision by 0.043 on clean frames. Node-guided resampling would formalise how those verifiers are trained (on the proposals of the system itself).
- **Not run:** I found no experiment that is a literal SAM-Road arm (baseline B2 / mix M10 of the concept plan) in the files searched (grep for "SAM-Road" and "K48" in the project's Markdown files); the B2 arm is planned there, not recorded as a result.
