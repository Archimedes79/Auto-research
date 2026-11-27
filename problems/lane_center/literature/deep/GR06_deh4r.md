# GR06 - DeH4R: full-text extraction

| field | value |
|---|---|
| paper | DeH4R: A Decoupled and Hybrid Method for Road Network Graph Extraction. Dengxian Gong, Shunping Ji. Accepted to IEEE Transactions on Geoscience and Remote Sensing (README citation: vol. 64, 2026, DOI 10.1109/TGRS.2026.3679934). arXiv 2508.13669 (cs.CV): v1 submitted 2025-08-19, v2 revised 2026-03-30 (the text read here). v1 was not read. |
| axis, tier, K ids | A3 (sparse lane graph), tier 1 core; K51 (candidate vertices first, growth from dangling ends; id from the core table in `../OVERVIEW.md`). Level-1 pipeline stage tags: S1, S4, S7 |
| level-1 note(s) | [GR06](../papers/GR06_deh4r.md) (mostly right; corrections in the list below) |
| code extraction | none (the coordinator reserved code extractions for the most meaningful papers). Repository facts, licence and a start list for a later extraction are in section 4. **The repository URL on the level-1 note has moved: github.com/godx-7/DeH4R** |
| sources read | arxiv.org/abs/2508.13669 (abstract page); arxiv.org/html/2508.13669v2 (HTML, nine separate queries: structure and introduction, III-A / III-B, III-C to III-E, IV-A to IV-C, Table I to III, IV-E ablations, IV-F limitations, Algorithms 1 and 2, figures 3 to 5 and the AVP parameterisation); GitHub pages and API listings of the repository; raw files `label_generator.py`, `matcher.py`, `criterion.py`, `data_preprocess.py`, `config/cityscale/final_sam2.yml`, `infer.py`, `model.py` (only for the statements in section 4 and for the numbers marked "config"); all read 2026-10-25. Not read: `utils.py` (52 KB, holds the decoding and growing code), `dataset.py`, `train.py`, `metrics_*`, the PDF (binary), Hugging Face pages, `scripts/`. Local files read for sections 5 and 7: `../OVERVIEW.md`, `../papers/GR06_deh4r.md`, `VM12_l2g-map.md` (for the soft-association idea) |
| reading marks | every statement about the paper and the repository is **digest**: all pages came through the summarising fetch tool (no page was seen raw), so numbers may be mis-transcribed; where I could check arithmetic it is said so. Line numbers quoted from code are the tool's counts and may be off by a line or two. Statements marked "(inference)" and all of section 5 are my reading, not the paper's content. Statements about our own pipeline are taken from the OVERVIEW and the level-1 note (**checked** as files, not re-run) |
| extracted by, date | extraction agent (Claude Sonnet 5.5, sub-agent of the literature session), 2026-10-25 |

**Corrections to the level-1 note** (the note is accurate on architecture, losses, datasets and the main numbers; these points are missing or changed):
1. Repository: `github.com/7777777FAN/DeH4R` now resolves to **`github.com/godx-7/DeH4R`** (the API returns that full name for the old URL). The note and the code queue say licence "not noted": there is **no LICENSE file** and the GitHub API licence field is null (section 4).
2. The note lists the validity threshold 0.5 / 0.45 only. Expansion rounds use a **higher** threshold, 0.7 (CityScale) and 0.65 (SpaceNet) (paper IV-C; config key `TRACE_THR` 0.7).
3. Table IV: the note says "row mapping inferred". It still is (the check marks did not render), but the paper's prose supports the mapping (expansion alone beats RNGDet++ and trails decoding). The table also has IoU and time columns the note omits: decoding only has the best IoU (56.42) and the shortest time (12.05 min); expansion adds only +0.19 F1 and +0.66 APLS to decoding for +1.29 min and -0.63 IoU (section 3).
4. Speed claim: the 13.34 min against 136.0 min is the whole CityScale test set against RNGDet++ only. SAM-Road needs 10.34 min on CityScale, i.e. DeH4R is slower there and faster on SpaceNet (15.24 against 18.53 min); the paper calls it "on par" with SAM-Road.
5. Code differs from the text in two places the note repeats: the N(0,1) jitter is applied to the **candidate position** in the label generator (offsets are recomputed from the noisy position), not to the adjacent vertices; and the coordinate loss in the code is a squared error (MSE), the paper writes a non-squared L2 norm (section 4).
6. Not an error of the note, but a discrepancy to keep in mind when quoting: the abstract digest states +4.62 APLS and +10.18 IoU over RNGDet++, while Table I gives 72.38 - 67.72 = 4.66 and 55.79 - 45.61 = 10.18. The 4.62 may be a mis-transcription by the tool or a rounding in the paper (unresolved; quote the table).

## 1. Problem and claim
- Task: aerial RGB tile -> road network graph (vertices on road centre-lines, undirected edges) on CityScale (2048 x 2048 tiles, 1 m / pixel) and SpaceNet (400 x 400, resampled to 1 m / pixel). Not a driving-BEV task, no lanes, no direction.
- Gap: segmentation + vectorisation loses topology; graph-growing (RoadTracer, VecRoad, RNGDet++) is accurate but runs a full network pass per step on heavily overlapping crops; graph-generating (Sat2Graph, SAM-Road and others) is fast and parallel but its vertex set is fixed after detection and cannot be completed.
- Claim: decouple the problem into four parts that share one backbone pass (candidate vertex detection CVD, adjacent vertex prediction AVP, initial graph construction IGC by decoding, graph expansion GEP from degree-1 vertices). The hybrid is reported as best on CityScale (TOPO-F1 81.21, APLS 72.38, IoU 55.79) and SpaceNet (86.04 / 73.37 / 48.54) at about ten times the speed of RNGDet++ (Tables I and III).
- What the ablations say about the claim (section 3): most of the quality comes from dense candidates + adjacent-vertex prediction + decoding; the dangling-end expansion adds a small, one-time gain (Table IV).

## 2. Method as built

### Pipeline (digest)
```
RGB patch 512 x 512 (CityScale) / 256 x 256 (SpaceNet), sliding windows over the tile
  -> SAM2 Hiera-B+ (fine-tuned, lr x 0.1) -> features 1/4, 1/8, 1/16, 1/32, 256 ch
  -> FPN: 1/32 upsampled and added to 1/16; 1/8 and 1/4 keep their lateral features
  -> mask head: 4 transposed convolutions -> 3 maps (keypoint, sampling-point, road)   [code: 3 channels]
  -> candidates C: local maxima of each map above a threshold, NMS per map, final NMS  (hard, no gradient)
  -> AVP, per candidate: ROI l x l (l = 3) grid of positions around it, bilinear grid_sample
       on the FPN scales, summed over scales -> f_roi (3 x 3 x d) -> 9 tokens as decoder memory
       -> 10 learned queries (hidden 128, 8 heads, FFN 256, 3 decoder layers) -> linear -> 4 numbers per query:
          2 class logits (valid neighbour / no neighbour) + 2 coordinates (relative offset)
  -> windows overlap: a candidate gets N x K predictions (K = covering windows); because the queries have a
       fixed order they are averaged index-wise, no matching across windows
  -> IGC (decoding): connect two candidates whose predicted neighbours point at each other (cost z, Eq. 1)
  -> GEP (expansion): seeds = degree-1 vertices; AVP on the seeds; insert new valid vertices with their edge,
       merge into an existing vertex within D_merge; <= 3 rounds or until no degree-1 vertex is left
```
Shapes in the code (digest): AVP input keypoints [B, N_points, 2] (padded), ROI features [B, N_points, C, 3, 3], output [B, N_points, 10, 4]; N_points 220 per 512 patch (CityScale), 64 per 256 patch (SpaceNet).

### The decoupling in four sentences
1. CVD is a plain segmentation head: it knows nothing about edges. 2. AVP is a per-vertex regressor: given a position and a 3 x 3 patch of multi-scale features at that position, it lists up to 10 neighbours as offsets with a valid flag (a set, trained with Hungarian matching). 3. IGC turns the two into a graph with a hand-written agreement test. 4. GEP re-uses the same AVP on the dangling ends, so a vertex the detector missed can still appear. AVP never sees the graph built so far, only features at one position: seeds of later rounds need no hidden state.

### Inputs, outputs
- Input: RGB tile, mean / std normalised. Output: graph (vertex list in global pixel coordinates, undirected edges), saved as a pickle (code).
- Candidate vertices carry no heading or direction. AVP offsets are relative to the candidate (the paper does not say so outright; the label generator confirms it, section 4). The class output means "valid neighbour" (road) against "no neighbour" (non-road, used like padding).
- Step size: not defined in the paper; there is no fixed step. The config bounds a neighbour offset to 30 px (`NORM_D` 30; section 4). Number of neighbours per vertex: up to N_q = 10, no cap per seed beyond that, no per-step insertion limit stated.

### Equations (as numbered in the paper; digest)
- **E1** (IGC cost): z = s1 + s2 + w (1 - cos th1) + w (1 - cos th2), w = 10 (taken from Sat2Graph). For candidates v_n and v_c with predicted neighbours u_n and u_c: s1 = |v_n - u_c| and s2 = |v_c - u_n| are the distances from each predicted neighbour to the existing candidate found within a search radius r; th1 and th2 are the angles between the predicted edges {v_n, u_n}, {v_c, u_c} and the candidate edge {v_n, v_c}. Two candidates are linked when their predictions "mutually correspond within a predefined tolerance" and z is minimal. Not stated: r, the tolerance, greedy or Hungarian selection.
- **E2** L_seg = BCE(S_k) + BCE(S_s) + BCE(S_r) over the keypoint, sampling-point and road masks.
- **E3** L_coord = |n_i - n*_sigma(i)| (L2 norm, not squared), only on matched pairs, averaged over the matched set.
- **E4** L_class = CE(c_i, c*_sigma(i)) over all N queries, unmatched queries get the non-road label.
- **E5** L = 1 x L_seg + 1 x L_class + 10 x L_coord.
- E6, E7: the APLS definition (path-length similarity); not read in detail. TOPO: precision |C|/|P|, recall |C|/|Q|, F1 from seed-vertex pairs averaged over several sampled seeds; the matching radius and hole size are not given.
- **Algorithm 1** (candidate extraction from a mask M with threshold tau; digest, the tool's condensed numbering): negate the mask; mark pixels that equal the minimum of their 8-neighbourhood (a local maximum of M); mark the zero-valued plateau; erode the plateau by the neighbourhood (outside counts as 1); keep pixels that are both a local extremum and in the eroded plateau, and where M exceeds tau.
- **Algorithm 2** (inference, digest): split the tile into windows; run CVD per window to get masks and features; extract candidates; in decode or combined mode predict neighbours for all candidates and decode the initial graph; in combined mode loop while num_exp < exp_limit: S = degree-1 vertices of G; stop if S is empty; predict neighbours for S, update G, num_exp += 1; in expansion-only mode start with S = all candidates (all degree 0) and the same loop. The tool noted two likely typos in the printed listing (G not initialised in expansion mode; one line increments exp_time instead of num_exp).

### Losses and assignment
- Three BCE terms on the masks (unweighted, no focal or Dice, no masking of empty samples), CE on the query classes, L2 on the offsets. **Assignment is Hungarian** (set prediction as in DETR): predicted neighbours are matched to the ground-truth neighbours of the candidate; unmatched queries are trained as "no neighbour". The matching cost is not in the paper; in the code it is a weighted sum of the Euclidean offset distance and the negative class probability (section 4).
- Differentiable? Only inside the heads. Candidate extraction (local extrema + NMS), the Hungarian assignment (computed without gradient, a discrete choice), the decoding and the expansion are non-differentiable rule steps. Training uses ground-truth vertices as candidates (teacher forcing with jitter); the graph building is **not on the training path**, so no gradient connects detection errors, decoding and expansion. No rollout training; exposure bias is not discussed.

### Labels / ground truth
- Paper: keypoints = road intersections and terminals; sampling points = points at regular intervals along segments between any two keypoints, including the keypoints; road mask = road surface. Interval, radius and Gaussian-or-binary are not given in the paper. (Code: section 4.)
- AVP samples: every ground-truth vertex in the patch together with its direct neighbours forms one sample (the vertex is the candidate, the neighbours are the targets). Samples are shuffled, the first N_points are kept (220 CityScale, 64 SpaceNet), empty ones are masked. Augmentation: random 90-degree rotations, N(0,1) noise (paper: on the adjacent vertices).

### Training recipe
- AdamW, base lr 0.001, divided by 10 at epochs 7, 11, 15 (CityScale, 20 epochs) and 10, 20, 25 (SpaceNet, 30 epochs); Hiera-B+ fine-tuned at 0.1 x the base lr; N_q 10, l 3. Hardware: server with 8 RTX 3090, each training run on 4 GPUs. Training batch size: not in the paper (config digest: 4, 32 workers). Training time: not found. The authors' SAM-Road reproduction was trained on the training set only (the original also used validation).

### Inference rules (the hidden rule steps)
| item | paper (IV-C / III-D) | config `final_sam2.yml` (digest) |
|---|---|---|
| candidate mask thresholds | tau not given | keypoint 0.35, sampling-point 0.20, road 0.25 |
| NMS | radii not given | keypoint 4 px, sampling-point 10 px, road 12 px; final NMS on the union 12 px; keypoint and sampling-point candidates get a fixed score 1.1, road candidates keep their map score (so road points lose ties) |
| validity of a predicted neighbour | 0.5 (CityScale), 0.45 (SpaceNet) | `VALIDITY` 0.5 |
| validity during expansion | 0.7 (CityScale), 0.65 (SpaceNet) | `TRACE_THR` 0.7 |
| merge distance in expansion | D_merge = 10 px | `BRIDGE_D` 10 (and `EDGE_ENDPOINT_D` 10) - mapping is inference |
| decoding | w = 10, search radius r not given | `ANGLE_DISTANCE_WEIGHT` 10, `SNAP_DIST` 12, snapping on |
| rounds | 3 (both data sets) | `EXTEND_TIMES` 3; search radius `SERCH_D` 32 |
| window layout | stride and overlap not given | `PATCHES_PER_EDGE` 16, `SAMPLE_MARGIN` 32, batch 4 |
| stop | no degree-1 vertex left or 3 rounds | same |

### Dynamic update of the graph (GEP)
- Seeds are the degree-1 vertices (in expansion-only mode the first round starts from all candidates, degree 0). Each seed runs through AVP. A valid predicted vertex that is not in the graph is inserted together with its edge to the seed; a valid predicted vertex within D_merge = 10 px of an existing vertex is merged into it (Fig. 4: insertion and merge). The new dangling ends become the seeds of the next round, so the frontier moves at most one predicted offset per round (with the config bound of 30 px, at most about 90 px in three rounds; inference).
- Edges that already exist are never removed: the paper states that expansion cannot correct wrong connections made in decoding, and that growing only from degree-1 vertices does poorly at sharp-turn vertices that should form a T-junction (Fig. 7c).
- Inference (not stated): a seed's prediction normally contains its own existing neighbour; the merge rule would absorb that back-edge into the parent, which avoids a special case for "do not grow backwards". The paper text I saw does not say this.

## 3. Numbers (digest; table numbers as printed on the v2 HTML page)

Table I (best in bold in the paper, not visible here; dagger = reproduced by the authors with public code; DeH4R* = SAM ViT-B backbone). CS = CityScale, SN = SpaceNet. P, R, TOPO-F1, APLS, IoU.

| method | CS P | CS R | CS F1 | CS APLS | CS IoU | SN P | SN R | SN F1 | SN APLS | SN IoU |
|---|---|---|---|---|---|---|---|---|---|---|
| Seg-UNet | 75.34 | 65.99 | 70.36 | 52.50 | - | 68.96 | 66.32 | 67.61 | 53.77 | - |
| Seg-DRM | 76.54 | 71.25 | 73.80 | 54.32 | - | 82.79 | 72.56 | 77.34 | 62.26 | - |
| Seg-Improved | 75.83 | 68.90 | 72.20 | 55.34 | - | 81.56 | 71.38 | 76.13 | 58.82 | - |
| Seg-DLA | 75.59 | 72.26 | 73.89 | 57.22 | - | 78.99 | 69.80 | 74.11 | 56.36 | - |
| RoadTracer | 78.00 | 57.44 | 66.16 | 57.29 | - | 78.61 | 62.45 | 69.90 | 56.03 | - |
| Sat2Graph (dagger) | 80.15 | 71.75 | 75.37 | 63.53 | 45.50 | 85.93 | 76.55 | 80.97 | 64.43 | - |
| TD-Road | 81.94 | 71.63 | 76.43 | 65.74 | - | 84.81 | 77.80 | 81.15 | 65.15 | - |
| RNGDet++ (dagger) | 85.65 | 72.58 | 78.44 | 67.72 | 45.61 | 91.37 | 75.24 | 82.52 | 67.81 | 40.70 |
| SAM-Road (dagger) | 91.27 | 65.70 | 76.20 | 66.54 | 33.87 | 91.48 | 75.49 | 82.72 | 71.10 | 48.08 |
| DeH4R* (SAM ViT-B) | 83.25 | 76.79 | 79.71 | 70.97 | 54.94 | 89.01 | 81.69 | 85.20 | 72.68 | 47.33 |
| DeH4R (SAM2) | 82.50 | 80.18 | 81.21 | 72.38 | 55.79 | 86.22 | 85.86 | 86.04 | 73.37 | 48.54 |

Table II (CityScale, backbone swap; P / R / F1 / APLS / IoU): Sat2Graph DLA-34 80.15 / 71.75 / 75.37 / 63.53 / 45.50, ViT-B 83.08 / 69.36 / 75.64 / 67.55 / 52.72; RNGDet++ R-101 85.65 / 72.58 / 78.44 / 67.72 / 45.61, ViT-B 84.82 / 71.95 / 77.68 / 65.61 / 43.26; SAM-Road R-101 90.13 / 67.14 / 76.73 / 65.89 / 48.22, ViT-B 91.27 / 65.70 / 76.20 / 66.54 / 33.87; DeH4R R-101 77.93 / 81.61 / 79.61 / 69.92 / 49.98, ViT-B 83.25 / 76.79 / 79.71 / 70.97 / 54.94. With the same ViT-B backbone DeH4R is +3.5 F1 and +4.4 APLS over SAM-Road (my arithmetic), so the gain is not only the SAM2 backbone.

**Table III, inference time** (minutes, whole test set): RNGDet++ 136.0 (CityScale) / 145.0 (SpaceNet); SAM-Road 10.34 / 18.53; DeH4R 13.34 / 15.24. No FPS, no batch size for the baselines; DeH4R batch 4 / 16; all inference results "produced by us" ran on GPU #0 of the 8 x RTX 3090 server (hardware of the baselines' timings not stated). Ratios to RNGDet++: 10.2 x and 9.5 x (my arithmetic). Per tile, assuming the whole test set was timed: 13.34 min / 27 tiles = about 30 s per 2048 x 2048 tile; 15.24 min / 382 = about 2.4 s per 400 x 400 image (my arithmetic, assumption). This is aerial imagery with a SAM2 backbone and sliding windows, **not comparable to our 31 ms per frame**.

### Ablations (CityScale; time in minutes)
| Table | rows (TOPO-F1 / APLS / IoU / time) |
|---|---|
| IV, strategy (row mapping inferred from the prose) | decoding + expansion 81.21 / 72.38 / 55.79 / 13.34; decoding only 81.02 / 71.72 / 56.42 / 12.05; expansion only 79.18 / 68.70 / 55.27 / 18.33 |
| V, expansion rounds | 1: 81.14 / 71.62 / 56.01 / 12.55; 2: 81.17 / 71.61 / 55.87 / 13.01; 3: 81.21 / 72.38 / 55.79 / 13.34; 4: 81.22 / 72.31 / 55.73 / 13.65; 5: 81.24 / 72.41 / 55.70 / 13.99; 6: 81.26 / 72.41 / 55.68 / 14.30; 7: 81.26 / 72.28 / 55.66 / 14.82 |
| VI, ROI size (P / R / F1 / APLS / IoU / time) | 1x1: 79.28 / 83.19 / 81.07 / 71.54 / 54.98 / 13.87; 3x3: 82.50 / 80.18 / 81.21 / 72.38 / 55.79 / 13.34; 5x5: 79.15 / 83.07 / 80.96 / 71.43 / 54.93 / 14.83; 7x7: 79.46 / 82.83 / 81.00 / 71.26 / 55.15 / 15.08; 9x9: 78.95 / 83.23 / 80.91 / 71.71 / 54.97 / 15.05 |
| VII, backbone (P / R / F1 / APLS / IoU) | SAM ViT-B 83.25 / 76.79 / 79.71 / 70.97 / 54.94; SAM2 with the 1/16 scale only 79.34 / 83.34 / 81.19 / 71.79 / 55.13; SAM2 multi-scale 82.50 / 80.18 / 81.21 / 72.38 / 55.79 |

What the ablations decide (my reading, with the arithmetic):
- **Expansion is a small add-on.** On top of decoding it gives +0.19 F1, +0.66 APLS, -0.63 IoU for +1.29 min (+11 % time). Expansion alone costs 18.33 min (slower than both together, because it starts from all candidates with degree 0) and is 2.03 F1 / 3.68 APLS worse than the hybrid. The authors attribute the IoU drop to the extra vertices.
- **Rounds:** F1 is flat from 1 to 7 rounds (81.14 to 81.26); APLS jumps by 0.77 between 2 and 3 rounds and is flat afterwards; each round costs about 0.3 to 0.5 min (3 to 4 % of the total). Three was chosen; no run-to-run variance is reported, so a 0.77 APLS step cannot be told from noise.
- **ROI size:** 1 x 1 is as good as 3 x 3 in F1 (-0.14), precision and recall move in opposite directions (79.28 / 83.19 against 82.50 / 80.18), i.e. a threshold shift at constant F1. The authors speculate noise in cropped features. In the code the ROI positions are integer pixel neighbours at full resolution (section 4), so on the 1/16 scale a 3 x 3 window mostly reads the same cell; this would explain the flat curve (inference).
- **Backbone:** multi-scale (1/4, 1/8) adds +0.59 APLS and +0.66 IoU over the 1/16 scale alone at equal F1.
- Single runs, no seeds, no confidence intervals anywhere.

Caveats: numbers of other papers are on other protocols and must not be compared with ours; the TOPO radius, the APLS and IoU settings other than the 3-pixel line width, and the baselines' timing hardware are not stated; the SAM-Road baseline is reproduced and trained on the training set only; DeH4R is recall-heavy (R 80.18 against P 82.50, SAM-Road P 91.27 / R 65.70), i.e. the opposite end of the operating curve from "better short graphs than false ones" unless the thresholds are raised.

## 4. Paper versus code
**Repository and licence.** `https://github.com/godx-7/DeH4R` (old URL `github.com/7777777FAN/DeH4R` redirects; account renamed). Created 2025-08-19, 10 commits on `main`, last push 2026-05-25 (commit 53c490e66f77fe960d76424440a99ca50e612942, "resolved an issue in data_preprocess.py where CityScale masks were vertically flipped and caused training anomalies"); the release of the full code is commit c726b70c1903c92ac546d504dc4d98d3ab37b15f (2026-03-30); 24 stars. Acknowledgements: SAM-Road, Sat2Graph, RNGDet++, DETR, SAM, SAM2. **Licence: none.** The top-level listing (20 entries, GitHub contents API) has no LICENSE, LICENCE or COPYING file; the API licence field is null; the raw `.../main/LICENSE` returned HTTP 404. Consequence: **read only** (describe, never copy). `matcher.py` carries a Facebook copyright header (DETR heritage); the `sam/` and `sam2/` sub-folders were not opened. Weights: the README names Hugging Face `godx7/DeH4R` (not opened); data from Google Drive links (CityScale, SpaceNet). nuScenes-style restrictions do not apply; the aerial data sets were not checked for licences.

**Differences between the text and the code I saw (digest):**
- Masks: the code draws binary discs of radius 3 px (not Gaussians). Keypoint mask = vertices of degree other than 2, plus degree-2 vertices where the two incident edges bend by more than 15 degrees; sampling-point mask = **every vertex of the ground-truth graph** (the paper says "regular intervals along segments"; the spacing of the pickled CityScale graph was not seen); road mask = lines of width 3 px. The vertical-flip fix of 2026-05-25 concerned CityScale masks; the current preprocessing flips rows only for SpaceNet.
- AVP targets (`label_generator.py`): ground-truth edges longer than 30 px are dropped and vertices left without neighbours are dropped. Offsets of the neighbours relative to the (jittered) candidate are divided by `NORM_D` = 30, so targets lie in [-1, 1]; the model squashes the offsets with tanh. Jitter: Gaussian, standard deviation 1 px, applied to the candidate position (paper: on the adjacent vertices). A candidate within 32 px of the patch border whose neighbour lies outside the patch is discarded for that patch (marked invalid). Padded slots are class "none", offset 0, invalid.
- Hungarian cost (`matcher.py`, config `SET_COST_CLASS` 1, `SET_COST_BOX` 5): 5 x Euclidean offset distance minus 1 x class probability; computed without gradient; no mask for padded targets inside the function. Loss (`criterion.py`): CE over the 10 queries with the no-object class weighted 0.1 (`EOS_COEF`, not in the paper), coordinate loss = squared error summed over matched pairs divided by their number (L1 and cosine variants are commented out), no deep supervision; loss weights 1 / 1 / 10 as in the paper (the config keys `MASK_LOSS_COEF`, `PROB_LOSS_COEF`, `COORD_LOSS_COEF`). The tool flagged a key mismatch in the returned loss dictionary of `criterion.py` (digest, unverified).
- Candidate extraction (`infer.py`): the maps of the covering windows are averaged pixel-wise; each map is divided by max(map max, 0.001) (order relative to thresholding not verified); thresholds and NMS radii in the table of section 2. Predictions of the same candidate from several windows are collected as a list and combined in `merge_across_patch_predicts` (not read).
- The ROI grid (`model.py`) is built from integer pixel positions r - 1 .. r + 1 around the candidate in the full-resolution patch, normalised by (patch size - 1) but sampled with `align_corners=False` (the tool flagged the mismatch); the high-resolution levels (1/4, 1/8) are summed onto the 1/16 ROI features. The coordinate head is tanh; the class output is a softmax with the "neighbour" class first.
- `infer.py` computes the number of batches as ceil(num_patch // batch_size); the tool flagged that the floor division would drop a partial last batch (unverified).
- The config's expansion parameters (`BRIDGE_D`, `EDGE_ENDPOINT_D`, `SERCH_D`, `SNAP_DIST`) are more numerous than the paper's single D_merge; which one is the paper's merge distance is inference.

**Start here for a later code extraction** (names as seen; nothing below was read in full):
1. `utils.py` (52 KB): `graph_growing` (the expansion loop, seed choice, `EXTEND_TIMES`, `TRACE_THR`, `SERCH_D`, `BRIDGE_D`), `GTE_decode` (the Sat2Graph-style decoding, cost z, `SNAP_DIST`), `merge_across_patch_predicts` (cross-window mean), `get_patch_info_one_img` (stride).
2. `label_generator.py`: class `LabelGnerator` (sic), `sample_patch` (AVP targets, border rule, jitter, offset normalisation) and `_kick_off_invalid_edges`; `data_preprocess.py` for the three masks.
3. `model.py`: `TopoDecoder` (ROI grid `_get_valid_grid`, `cal_ROI_feature`, query decoder), `DeH4R.forward` / `infer_topo`; `matcher.py` and `criterion.py` for the Hungarian set loss.
4. `infer.py`: `infer_one_img` (candidate extraction, per-window AVP, decode, trace) and the timing lines; `metrics_cityscale/` for TOPO and APLS (not opened).

## 5. Transfer to our pipeline

| question | answer |
|---|---|
| what it needs from us | Not image features, not a trainable encoder: its input is "a feature map + candidate positions". It runs on our frozen cached BEV (39 ch, 200 x 200, 0.5 m) after replacing the SAM2 backbone and the mask head by our dense lane head (heat map + 2-ch direction). No ego poses, no SD map, no lidar |
| what it gives us | Axis A3 (and the interface of A2 to A3). Ideas: (1) a keypoint-type candidate map as a source of seeds and of fork / merge positions, (2) a per-vertex set of neighbour offsets, (3) **mutual-agreement (reverse-pointing) cost between two vertices as a link score**, (4) a second, cheap, batched completion pass from dangling ends |
| conflicts | Hungarian set loss (training of AVP); local-extremum + NMS candidate extraction (hard selection); rule-based decoding and merge; undirected, road-level graph (no lane, no direction, no fork versus merge); no heading on candidates; dense vertices, no sparsity; teacher forcing only (exposure bias); image domain and SAM2 backbone; recall-heavy operating point; all radii (NMS 4 to 12 px, merge 10, snap 12, search 32) are for roads at 1 m / px and would merge lanes 3.5 m apart; no licence (read only) |
| reuse mode | Idea only / re-implement from the description (the head is small); no code may be copied |
| effort | S (hours to a day): keypoint-type label channels on the dense head; the cost z as a soft mutual-agreement score on the output of a head we already have. M (days): a neighbour-set head with successor and predecessor slots, ROI read from the BEV pyramid, trained matching-free. L (weeks, probably not worth it): the full hybrid with a soft merge, trained through 2 - 3 expansion rounds. Blocker: a matching-free assignment of the neighbour slots in crowded configurations (parallel lanes, junction mouths), which A4 lists as open |
| expected gain and risk | Gain on S: expected tie on clean frames. Possible gain in junction recall J (the weak point at equal sparsity: 0.265 against 0.408) only if missing seeds are the bottleneck; this must be tested with oracle seeds first. Risk: dense candidates raise vertices per 100 m (champion 13.5, control 21.8) and add false lanes, the opposite of our standing rule; DeH4R's own ablation says the dangling-end growth is worth only +0.19 F1 over decoding |
| feeds which experiment / K id | K51 (as in the OVERVIEW). New techniques to register before use: reverse-consistency link score (soft mutual agreement), directed successor / predecessor neighbour slots, oracle-seed bound for the tree |

### 5.1 Side by side with the verified-ray tree
| aspect | DeH4R | ours (loop 16 builder) |
|---|---|---|
| grown unit | vertex = position; undirected edge | tip = position + heading; directed edge, DAG |
| where new vertices come from | (a) candidates detected independently of any parent (NMS on three masks), (b) up to 10 regressed neighbour offsets per vertex from the 3 x 3 features at that vertex | b children per tip from a 12-ray teardrop read by bilinear `grid_sample` from the BEV pyramid |
| validity | class output per query (threshold 0.5 / 0.7), plus mutual agreement in decoding | hazard / survival along the ray: evidence over the whole segment |
| evidence along an edge | none described in the paper text: an edge exists when the two endpoints agree or when a seed inserts its neighbour (endpoint evidence only) | every ray is verified step by step |
| step length | regressed, bounded by 30 px in the config | set by the verification |
| training | teacher forcing on ground-truth vertices with 1 px jitter; Hungarian set loss; graph building is off the training path | matching-free point-to-polyline distance both ways; losses through the tree |
| sparsity | none: every ground-truth graph vertex is a sampling point | explicit; 13.5 vertices per 100 m |
| forks and merges | undirected vertices of degree 3 or more: a candidate lists several neighbours (keypoint mask = degree other than 2, or a bend above 15 degrees) | junction head + fork relocation; directed |
| non-differentiable parts | candidate extraction, Hungarian, decoding, merge / insert | none on the training path |

### 5.2 What maps to a 200 x 200 BEV with directed lane graphs and forks / merges
1. **Keypoint-type heat map as fork / merge / end / bend candidates.** The label is cheap (degree and bend angle of the ground-truth graph; binary discs of radius 3 px, BCE). In the lane setting split it by type: fork (out-degree 2 or more), merge (in-degree 2 or more), lane start / end (also at the BEV border), bend above about 15 degrees; use a Gaussian in metres like our lane heat map (sigma 0.75 m). This is close to what the junction head and fork relocation already do (E44: split / merge F1 0.34 / 0.38 to 0.55 / 0.61 with key-point heads); the DeH4R definition adds bends and ends as separate candidates. Compare labels, do not add a second head blindly.
2. **Mutual agreement as a soft link score.** E1 asks: if I stand at the child and ask where the neighbours are, does one of them point back to the parent, with the same heading? For a directed tip-to-child edge: run a small head at the child, read the predicted predecessor offsets, and score s = exp(-z / tau) with z = distance(predicted predecessor, parent) + angle term using a signed cosine (the paper's undirected 1 - cos would not separate opposite lanes, which are about 3.5 m apart with reversed heading). This is a second, independent opinion on every child and every merge, differentiable, and attacks "false lanes" directly. It is a verification of the endpoint, not of the segment: it complements the ray hazard, it does not replace it.
3. **Successor and predecessor slots.** DeH4R's 10 exchangeable queries per vertex become two sets for us (successors, predecessors); a fork vertex has two successor slots valid, a merge vertex two predecessor slots. Mutual agreement then runs between a successor slot of A and a predecessor slot of B.
4. **Matching-free training of the slots.** Replace Hungarian by the project's own loss: every valid predicted neighbour is pulled to the nearest ground-truth neighbour polyline (precision side), every ground-truth neighbour is pulled towards the nearest predicted valid point (coverage side), validity trained against the distance to the polyline. Known risk: several queries collapse onto one neighbour; the merge radius (1.5 m) removes duplicates after the fact, and rMCL-style winner-take-all with a learned score (K38) is the existing remedy for the children of a tip. DeH4R shows that even the newest hybrid stays with Hungarian for this step; no matching-free recipe for crowded configurations is published (A4 gap).
5. **Dense seeds + tip growth.** The hybrid idea in our terms: extra roots from a candidate map for lanes the ego-rooted tree never reaches (side arms, adjacent lanes), then tip growth fills gaps. The ablation warns that the growth step is a minor part of the gain (Table IV); in our pipeline the growth is already the main engine, so the open question is the opposite one: are missing roots the bottleneck of J? Test: give the tree **oracle seeds** from ground-truth fork / merge positions (a sanity bound, not an upper bound); if J does not move, candidates are not the problem and the idea stops there.
6. **Units.** Convert from 1 m / px aerial roads to 0.5 m cells and 3.5 m lane spacing: NMS 1 to 2 m (not 4 to 12), merge 1.5 m and snap 1.0 m (ours), maximum neighbour offset: 30 m is 60 cells (the BEV half extent is 100 cells), but our vertices are about 7 m apart on average, so a bound of 10 to 15 m is the natural scale. Oriented ROI: candidates in our case have a heading from the direction channels, so the 3 x 3 sampling pattern can be rotated along it (DeH4R's candidates have none), which should lighten the decoder.
7. **ROI read.** The 3 x 3 integer-pixel pattern reads almost one cell on a coarse scale (inference from the code digest and the flat Table VI); our 12-ray teardrop is a richer read of the same kind. Nothing to adopt except the multi-scale sum (1/4, 1/8 added to 1/16: +0.59 APLS, +0.66 IoU in Table VII), which our BEV pyramid already provides.

### 5.3 Can the expansion step be batched and made differentiable?
- **Batched:** yes, and the paper already does it: AVP treats every vertex independently, seeds are padded into one batch (N_points up to 220 per patch), and three rounds are three batched passes. With frozen cached features there is no backbone pass at all; per round the cost is one 3 x 3 x 3-scale `grid_sample`, 3 decoder layers (hidden 128) and 10 queries for a few hundred vertices at most. The measured part of the paper's time (about 0.3 to 0.5 min per extra round over 27 tiles, Table V) is dominated by Python-side graph updates in a CPU R-tree, not by the network (inference from the code digest, not measured).
- **Differentiable:** not as published. A differentiable version is plausible by three replacements: (i) seed selection by degree 1 becomes a soft weight w_i = 1 - min(1, sum of incident edge validities), or simply "all vertices" because predictions near an existing neighbour merge anyway and the vertex budget of a BEV frame is small; (ii) the insert-or-merge branch becomes a soft association of each predicted vertex to the existing vertices with a dustbin column (new vertex), as in the L2G-style design of [VM12](VM12_l2g-map.md) section 5.2 and SuperGlue's dustbin ([C30](../papers/C30_superglue-sinkhorn-optimal-transport-with.md)); merged positions are weighted means; (iii) the seeds of round k+1 are the predicted positions of round k, so the `grid_sample` read at them is differentiable in the offsets of round k, which gives a gradient through a 2 to 3 round unroll (the same mechanism as the 5-step unroll of DAGMapper, [GR05](../papers/GR05_dagmapper.md)). DeH4R itself offers none of this: it trains one step on ground-truth seeds. The price is exposure to its own errors, which is exactly what training through the unroll cures.
- **Our tree already is a differentiable, verified version of the same expansion**, with heading and without NMS. So the question for us is not "adopt DeH4R's growth" but "does a second, parallel pass with a different head (neighbour sets + mutual agreement) improve J or precision at equal vertices per 100 m".

### 5.4 Effort to re-implement (from the description)
Head: ROI sampler (about 30 lines), 3 decoder layers at 128 hidden with 10 or 2 x 10 queries (about 0.5 M parameters, my estimate), mask or keypoint channels on the existing dense head; loss: matching-free; decode: the cost z and a soft threshold. Roughly M (days) for a head + eval on cached features, plus S for the oracle-seed test. Not worth building the DeH4R decoding and expansion rules themselves.

## 6. Not found / open
- Paper: sampling interval of the sampling points, keypoint radius, tau, NMS radii, the search radius r and the tolerance of decoding, FPN channel width, window stride and overlap, TOPO matching radius and hole size, IoU buffer, parameter count, FPS, memory, training time and batch size, per-component timing (CVD, AVP, decoding, expansion), seeds and variance, any robustness or out-of-distribution test in the main text (the code has an `--OOD` option and a `globalscale` data set, not discussed here), the cost of the Hungarian matching, whether edges are directed (the paper does not say; undirected is implied by mutual correspondence).
- Not read: the PDF and the appendix (binary), v1, `utils.py` (decoding and growing code), `dataset.py`, `train.py`, the metric folders, Hugging Face weights.
- Digest risks: Table IV row mapping (inferred; consistent with the prose), the abstract's 4.62 APLS (against 4.66 from Table I), all code line numbers.

## 7. Our experiments that touch it
- None ran DeH4R, a candidate-vertex + neighbour-set head, or a mutual-agreement link score. Level-1 note: E94 - E99 (trace network and similar events) partial; E42 / E45 / E46 / E51 / E56 (learned link heads) stayed within 0.01 of the rules and were dropped (DeH4R's decoding is a hand rule on predicted neighbours, so the same expectation applies to a literal port). From the OVERVIEW: E44 key-point heads for split / merge (F1 0.34 / 0.38 to 0.55 / 0.61), E53 Bezier connectors (0.71 / 0.64), E94 trace network (F1@1 0.851), and the loop-16 builder (K01 - K03, K30 - K51). Not tried: oracle seeds for the tree, reverse-consistency score, directed successor / predecessor slots, dense candidates as extra roots.
