# GR09 - Sat2Graph: full-text extraction

| field | value |
|---|---|
| paper | Sat2Graph: Road Graph Extraction through Graph-Tensor Encoding. Songtao He, Favyen Bastani, Satvat Jagwani, Mohammad Alizadeh, Hari Balakrishnan, Sanjay Chawla, Mohamed M. Elshrif, Samuel Madden, Amin Sadeghi. ECCV 2020 (arXiv comment "ECCV 2020"). arXiv 2007.09547, v1 only (submitted 2020-07-19, 3,225 KB). |
| axis, tier, K ids | A2 (dense lane-centre field), tier 1 core; K83 (named in the OVERVIEW core table as "read, not used"; `register.md` not opened by me). Related ids that will touch it: K62 (our dense U-Net head), K48 (SAM-Road: vertex mask + pairwise edge classifier). |
| level-1 note(s) | [GR09](../papers/GR09_sat2graph.md) (corrections in the last section of this file) |
| code extraction | none written (decision of the coordinator: code extractions only for the most meaningful papers). Repository: https://github.com/songtaohe/Sat2Graph (the URL in the note is correct). Where a later code extraction should start: section 4. |
| sources read | arxiv.org/abs/2007.09547 (abstract page); ar5iv.labs.arxiv.org/html/2007.09547 (table of contents only); arxiv.org/html/2007.09547v1 (HTML; seven separate queries: Sections 3.1 and 3.2 with both equations, Sections 4.1 - 4.6, Table 1 - 3, Section 5, Section 6, figure captions); github.com/songtaohe/Sat2Graph (README, directory pages `model/`, `prepare_dataset/`, `metrics/`, `metrics/topo`, `metrics/apls`, commit list); raw files `LICENSE`, `model/decoder.py`, `model/dataloader.py`, `model/model.py`, `model/infer.py`, `model/train.py`, `model/douglasPeucker.py`, `model/common.py`, `prepare_dataset/graph_ops.py`. The GitHub API commit listing returned HTTP 403 (not used). The PDF was not opened and no supplement was looked for. All read 2026-10-25. |
| reading marks | **digest** for every statement about the paper and the code: every page came through the summarising fetch tool (no page was seen raw), so numbers can be mis-transcribed; line counts and "about" figures of code files are the tool's estimates. The LICENSE text was returned in full by the tool (standard MIT wording). Section 5 is my own analysis (not the paper's content); arithmetic done by me is marked. Figures (Figure 2 workflow, Figure 3 curves, Figure 4 - 5 images) were not visible. |
| extracted by, date | Claude (Sonnet 5.5) literature-extraction agent, sub-agent of the literature session, 2026-10-25 |

## 1. Problem and claim
- Predict a road graph (vertices with positions, undirected edges) from an aerial RGB tile at 1 m per pixel. Prior work is either pixel-wise segmentation followed by thinning and tracing (loses topology, cannot separate stacked roads) or an iterative agent that adds one edge at a time (slow, hard to train, error accumulation).
- Claim: if the graph is written into a dense multi-channel raster (the graph-tensor encoding, GTE), a plain non-recurrent segmentation network trained with a per-cell loss predicts the graph structure in one pass; a short rule-based decoder turns the tensor back into a graph.
- The claim that matters for us: a dense head can carry vertex positions and the geometry of the outgoing edges per vertex in fixed channel slots, without matching and without recurrence; the weak point is that the labels are not unique along long segments (the paper says so itself).

## 2. Method as built

### Pipeline (digest)
```
aerial RGB tile (1 m / px; every model is trained with a 352 x 352 receptive field)
  -> DLA backbone (Deep Layer Aggregation, residual blocks as aggregation function)
  -> graph tensor T, (W/lambda) x (H/lambda) x (1 + 3 D_max), lambda = 1 m, D_max = 6  -> 19 channels per cell
       p_v                     vertexness, one channel
       for slot i = 1..6:      p_ei (edgeness), dx_i, dy_i (offset to the neighbour vertex)
       slot i covers the edge directions (i-1)*60 .. i*60 degrees seen from the vertex
  -> decode (rule-based): local maxima of p_v above p_thr = vertices;
     for every vertex v and slot i choose the candidate vertex u minimising E1, reject if more than 15 m away
  -> undirected road graph
```
Channel order inside a group (edgeness first, then the two offsets) is as the text lists it; absolute channel indices are my numbering. Output stride, up-sampling scheme, parameter count and output activations are **not found** in the paper.

### Inputs, outputs
- Input: aerial image (RGB), 1 m per pixel. Output: tensor above, then a graph (vertex list + edges). The graph is **undirected**: Section 5 of the paper says that for an edge (va, vb) it is enough to encode one link, from va to vb or from vb to va, and that road directions can be attached as attributes afterwards (Table 3 below).
- One cell (lambda x lambda = 1 m x 1 m) holds at most one vertex. Whether and how a vertex position inside the cell is stored or refined: **not in the paper** (vertices are the local maxima of the vertexness map, nothing more is said). Units and normalisation of the offsets: **not in the paper** (the code uses a division by 25, section 4).
- At most one outgoing edge per slot, i.e. per 60-degree sector, at each vertex. Vertices of degree above 6 are said to be very rare.

### Encode (ground-truth tensor from a graph)
1. Interpolate every road segment with the fewest evenly spaced intermediate points that keep the spacing under d metres. d = 20 in the setup; the paper says 15 - 25 m works best (a very small d degenerates into segmentation, a very large d cannot follow curves; it also keeps the offset lengths regular, which stabilises training).
2. **Overlapping (stacked) roads, bridges:** no extra channel and no layer index. After interpolation an iterative conflict-resolution step shifts the end vertices of the two overlapping edges until any two vertices of the two edges are more than 5 m apart; the two edges then cross near their middles, without a vertex at the crossing. Because edges are stored as offsets and not as drawn lines, two crossing edges do not collide in the tensor. My reading (the paper gives no explicit rule): intersection versus overpass is decided by whether a vertex exists at the crossing, and the network has to tell the two from image context. Algorithm details of the shift: not in the paper.
3. Write vertexness 1 at the vertex cells, 0 elsewhere; for each vertex, put each edge into the slot whose sector contains the edge direction, with edgeness 1 and the offset to the neighbour. Which endpoint stores an edge: see "Corrections" and section 4 (the text only says one of the two is enough).

### Equations (as in the paper; digest, symbols as written)
- **E1 (decoder link cost).** d(v, i, u) = || (v_x + dx_i, v_y + dy_i) - (u_x, u_y) || + w * cos_dist( (dx_i, dy_i), (u_x - v_x, u_y - v_y) ). v: vertex, i: slot, u: nearby candidate vertex, (dx_i, dy_i): predicted offset of the slot, cos_dist: cosine distance of the two vectors, w = 100. The u with the smallest d is taken; candidates beyond 15 m are rejected. If cos_dist is 1 - cos (the usual definition; not stated), the cosine term is 1.5 m-equivalent at 10 degrees and 13.4 at 30 degrees (my arithmetic).
- **E2 (loss).** L(T, T^) = sum over all cells (x, y) of [ CE(p_v, p^_v) + T^_{x,y,1} * sum_{i=1..D_max} ( CE(p_ei, p^_ei) + L2((dx_i, dy_i), (dx^_i, dy^_i)) ) ]. Hatted = ground truth; T^_{x,y,1} is the ground-truth vertexness, so edgeness and offset terms count only at cells that hold a ground-truth vertex.

### Losses, assignment
- Three terms, unit weights written in E2; the paper names no coefficient between them. No matching of any kind in the loss: the sector index is a deterministic function of the edge angle, so target slots are fixed by geometry (no Hungarian, no nearest-prediction assignment). Reason given for masking edge terms to vertex cells: the encoding is not unique along long segments (the interpolated vertices could sit at different places), but it is consistent at intersections and overpasses (the single quote: "locations of intersection vertices are always consistent"); inconsistent long segments have simple topology and the decoder can repair them.
- No auxiliary segmentation output or loss in Sat2Graph (the joint segmentation + orientation training belongs to the Seg-Orientation baseline in the paper). The code differs (section 4).

### Training recipe (digest)
TensorFlow, one V100, 300k iterations (about 120 epochs), learning rate 0.001 halved every 50k iterations, validation every 5k iterations and the best validation model kept. Augmentation: random brightness, hue and colour temperature, random tile rotation, random masks on the image. All models (also baselines) use the same 352 x 352 receptive field. Optimiser, batch size, training time, parameter count: **not found**.

### Inference (all rule-based)
1. Vertices: local maxima of the vertexness map above p_thr (no NMS radius, no sub-cell refinement given).
2. Edges: for each vertex and each slot with edgeness above p_thr, take the u minimising E1, reject beyond 15 m. The paper gives no numeric p_thr; Section 4.5 says only that vertex and edge thresholds are set to the same value in the basic setup, and it sweeps the threshold for the precision-recall curves.
3. The paper names **no** post-merge, spur removal, isolated-piece removal or simplification step. The released code has several (section 4).

## 3. Numbers (digest; table numbers as on the HTML page)

**Data.** City-scale: 20 US cities, 720 km2, OpenStreetMap ground truth, Google static map imagery at 1 m per pixel, 180 tiles of 2 km x 2 km: 144 train (80 %), 9 validation (5 %), 27 test (15 %). SpaceNet roads: 2,549 non-empty tiles of 0.4 km x 0.4 km resized to 1 m per pixel, split 80 / 15 / 5 % (train / test / validation); the official test ground truth is not public.

**Table 1, best achievable TOPO F1 and APLS (all %; precision and recall are the ones at the best F1, given for reference only):**

| method | city P | city R | city F1 | city APLS | SpaceNet P | SpaceNet R | SpaceNet F1 | SpaceNet APLS |
|---|---|---|---|---|---|---|---|---|
| RoadTracer (CVPR-18) | 78.00 | 57.44 | 66.16 | 57.29 | 78.61 | 62.45 | 69.60 | 56.03 |
| Seg-UNet | 75.34 | 65.99 | 70.36 | 52.50 | 68.96 | 66.32 | 67.61 | 53.77 |
| Seg-DRM (ICCV-17) | 76.54 | 71.25 | 73.80 | 54.32 | 82.79 | 72.56 | 77.34 | 62.26 |
| Seg-Orientation (ICCV-19) | 75.83 | 68.90 | 72.20 | 55.34 | 81.56 | 71.38 | 76.13 | 58.82 |
| Seg-DLA (ours) | 75.59 | 72.26 | 73.89 | 57.22 | 78.99 | 69.80 | 74.11 | 56.36 |
| Sat2Graph-DLA (ours) | 80.70 | 72.28 | 76.26 | 63.14 | 85.93 | 76.55 | 80.97 | 64.43 |

**Table 2, TOPO F1 (city-wide) for the weight w of E1:** w = 1: 75.87; 5: 76.28; 10: 76.62; 25: 76.72; 75: 76.55; 100: 76.26; 150: 75.68 (all %). The Table 1 headline (76.26) equals the w = 100 cell, so the headline is not the best w (+0.46 at w = 25); the whole range spans 1.04 F1.

**Table 3, share of edges that need a repair (caption missing in the source; counting method not stated):**

| D_max | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|
| fixed with the undirected property | 8.62 % | 2.81 % | 1.18 % | 0.92 % | 0.59 % |
| fixed with the interpolatable property | 0.013 % | 0.0025 % | 0.0015 % | 0.0013 % | 0.0013 % |

Reading (the paper states the two properties, not the procedure): an edge that finds its sector at one endpoint already occupied may be stored at the other endpoint (undirected); if neither endpoint can store it, the edge is split by a midpoint vertex (interpolatable). So D_max = 6 costs about one edge in a hundred an endpoint swap, and almost none a split.

**Ablations that decide transfer.**
- Adding GTE to a U-Net backbone lifts TOPO F1 from 70.36 (the Seg-UNet row) to 76.40 (text); to the shallow-decoder Seg-DRM from 73.80 to 74.66 (text, the authors blame the shallow decoder). The 76.40 is higher than the DLA headline 76.26 and than the DLA w = 100 cell; the paper does not discuss it.
- Segmentation with the same DLA backbone: 73.89; GTE with the same backbone: 76.26 (city F1), APLS 57.22 -> 63.14: the gain is in topology-sensitive APLS more than in F1.
- Stacked roads: precision 83.11 %, recall 49.81 %; only 0.37 % of intersections are wrongly predicted as over- or underpasses. Failure cases: complicated stacked-road scenes (Figure 5 d - e), and small roads under wide highways are missed.

**Metrics.** TOPO: subgraphs around seed points are compared, nodes within a graph distance of a threshold from the seed (300 m in the paper's example); precision, recall and F1 averaged over random seeds; the authors implemented it strictly. APLS: shortest-path quality between location pairs; no numeric threshold in the paper text. The repository's TOPO README (digest) gives: propagation distance 300 m for large tiles and 150 m for small tiles, propagation interval 5 m, matching distance 10 m, matching angle 30 degrees, one-to-one matching; the APLS metric is a Go program with parameters set for 2048 x 2048 tiles in metres.

**Compute.** Training: one V100, 300k iterations. Inference time, memory and parameter count: **not stated anywhere in the paper**.

**Caveats.** Aerial imagery at 1 m per pixel, US cities, OSM labels (noisy, undirected); TOPO and APLS are road-network metrics with 300 m propagation and 10 m matching: they are not comparable with our point F1 at 0.5 / 1 / 1.5 m, J (fork / merge F1 at 6 m) or route success.

## 4. Paper versus code
Repository https://github.com/songtaohe/Sat2Graph, `master`, newest commit seen 7ca4a345a2ceae4c62c120661608a60e9e962be4 (2023-04-09, "Update README.md"); 140 commits, 220 stars when read. **Licence seen:** the `LICENSE` file is the **MIT License, "Copyright (c) 2020 Songtao He"** (the GitHub sidebar says MIT too). The README adds that all pretrained models, the dataset and the docker container are for **non-commercial academic use only**. Stack: TensorFlow 1.x with Python 2 idioms (`iteritems`, integer `/`, `scipy.ndimage.imread`, `np.float`), `rtree`, OpenCV, `tflearn` imports; not runnable on PyTorch 2.8 / Python 3.12 without a rewrite. Code was read only through the summarising tool, no clone, no download.

Differences between text and code that matter (all digest):
- **Label tensor (dataloader).** Vertexness is a hard 0/1 in a 3 x 3 window around each vertex, no Gaussian; the same window carries the slot flag and the offset (neighbour minus vertex) divided by 25.0, not clipped. The slot is the angle of the edge direction (shifted by pi) in units of 60 degrees, modulo 6. The neighbour dictionary of the graph is built undirected, so my inference is that every edge is written at **both** endpoints (in opposite slots); I did not run it. Patches are sampled with a bias: 40 % centred on overpass points, 20 % on complicated intersections, 20 % on parallel roads, 20 % uniform; the paper text mentions only random masks as augmentation.
- **Interpolation and stacked roads (prepare_dataset).** Interpolation density 0.00020 degrees, about 22 m by the tool's arithmetic, against d = 20 m in the text. The stacked-road shift is a vector-adjustment loop (crossing radius 9.5 pixels, node moves of 1.5 then 1.0, up to 40 iterations, a node-deletion step every fifth iteration) instead of a plain rule that vertices of overlapping edges be more than 5 m apart.
- **Output and loss (model).** 28 raw channels: a two-class softmax for vertexness (2), for each of 6 slots a two-class edgeness softmax (2) plus the offset (2), and a segmentation head (2); the published model is trained jointly with segmentation. Terms: vertexness CE averaged over all cells, weight 1; edgeness CE per slot multiplied by a soft mask (0.01 on background, 1 at vertex cells), averaged, divided by 6, **times 10**; squared offset error with the same soft mask, divided by 6, **times 1000**; segmentation CE **times 0.1**. So the masking is soft rather than the strict indicator of E2, and the weights that the paper does not give exist. Adam; learning rate 0.001, halved every 50k steps, stop at 300,002 steps; no best-model selection in `train.py` (checkpoints every 10k steps, validation loss only printed), batch size 2 there.
- **Decoder (decoder.py).** The vertexness map is smoothed (sigma 1) and **divided by its own maximum** before thresholding, so the threshold is relative to the tile; peaks from a 3 x 3 neighbourhood; thresholds 0.05 for vertex and edge in `train.py` / `infer.py`. A slot is accepted if vertexness times edgeness exceeds the product of the two thresholds and the edgeness exceeds its threshold. The endpoint is the vertex plus 25 times the offset. With snapping on, a search window of +-20 pixels is scanned in three passes (vertices with distance below the offset length, vertices below half of it, then predicted edge-endpoint peaks), cost = distance + 100 * (1 - cos) terms, rejection beyond 15; `infer.py` passes an angle weight of 50, not the paper's 100. Then graph clean-up that the paper does not describe: removal of connected pieces shorter than a length threshold (200, or 100 for small tiles), removal of short spurs at degree-3 nodes (50 / 25), removal of the longer edge of near-straight detours (cosine above 0.984) repeated until nothing changes, shaving of short chains from dead ends (50), and in `infer.py` Douglas-Peucker simplification of every degree-2 chain with tolerance 2.5. Hidden rule steps everywhere on the inference side.
- **Tiling (infer.py).** 352 x 352 windows with stride 88 over a 2048 x 2048 image (padded by 32), outputs blended by a weighted average of the 28 channels with weights 0.001 / 0.5 / 1.0 / 1.5 by distance from the window border (0, 32, 56, 88 pixels): a dense-tensor overlap merge, not mentioned in the paper.
- **Start here for a later code extraction (nothing was ported or copied):** (1) `model/decoder.py`: `DecodeAndVis`, `detect_local_minima`, `graph_refine`, `graph_refine_deloop`, `graph_shave` (tensor to graph); (2) `model/dataloader.py`: `Sat2GraphDataLoader.preload` / `getBatch` (label tensor: window, slot rule, /25); (3) `prepare_dataset/graph_ops.py`: `graphDensify`, `locate_stacking_road`, `apply_adjustment`, `graphGroundTruthPreProcess` (interpolation and stacked-road shift); (4) `model/model.py`: `Sat2GraphModel.SupervisedLoss` and `SoftmaxOutput` (channel layout, loss weights); (5) `model/infer.py` (tiling and blending), `model/douglasPeucker.py` (`simpilfyGraph`) and `metrics/topo/` (`main.py`, `topo.py`).

## 5. Transfer to our pipeline

| question | answer |
|---|---|
| what it needs from us | The frozen cached BEV (200 x 200 at 0.5 m, 39 channels) as head input only; no ego pose, SD map, image feature or lidar. For labels: the lane polylines we already rasterise for the heat map, turned into a vertex set with successor (and predecessor) offsets. |
| what it gives us | Axis A2 -> A3 bridge: one dense pass that outputs vertex candidates and, per vertex, the geometry of its outgoing edges. Most useful as a **dense junction / fork detector with child geometry** (the paper's own consistency argument holds at intersections, not along long segments), and as proposal source for the verified-ray builder. |
| conflicts | (1) The graph is undirected; a lane graph is a directed DAG. (2) World-fixed 60-degree sectors put the two children of a lane fork in one slot (see 5.3). (3) Vertex phase along a long lane is not recoverable from the image (label aliasing, see 5.4). (4) Peak finding, threshold, the argmin link, the 15 m cut and the clean-up of the code are rule steps; fine at inference, not on our training path. (5) 1 m aerial pixels and a 352 m receptive field against our 0.5 m cells and a 100 m window: border effects. (6) Code is TensorFlow 1 / Python 2 (MIT, usable, but nothing runs). Licence is no obstacle for an idea-level re-implementation. |
| reuse mode | Idea only / re-implement from the description (encoding rule, loss structure, link cost E1). No code port. |
| effort | S: offline label audit (slot collisions, vertex phase) on TRAIN, no GPU. M: a GTE head next to the heat head on the frozen BEV plus a soft decode feeding the assembler, and a candidate-feed test in the builder. L: replacing the verified-ray builder (not recommended, see 5.5). Blocker: a slot rule that survives lane forks. |
| expected gain and risk | Target is the open junction gap at equal sparsity (junction recall J 0.265 against 0.408, lanes found 0.695 against 0.800, `../OVERVIEW.md` 1.5). Gain plausible for J and for candidate recall at fork mouths; no gain expected from vertex-on-straight prediction. Risks: label aliasing, border cells, density above the champion's if the vertex set is chosen badly, a second head that costs training budget. |
| feeds which experiment / K id | K83 exists (read, not used). Proposed new technique ids, to be registered before use: GTE head with heading-relative successor / predecessor slots; soft decode (soft-argmax vertex, softmin or radius linking, slot validity p_v * p_e) feeding the assembler; GTE peaks as builder seeds. |

### 5.1 Could our dense head output a GTE-like tensor on the 200 x 200 grid?
Yes in shape: (1 + 3 D) x 200 x 200, where the last layer of the existing small U-Net (heat map + 2 direction channels now) widens to 10 channels for D = 3 or 19 for D = 6; the added cost is a few thousand weights in the output convolution (my estimate, not measured). Changes needed:
- Cell size: at 0.5 m the "one vertex per cell" rule is no constraint. Replace the 3 x 3 hard window (about +-1 m at 1 m per pixel) by the Gaussian blob of the existing heat map (sigma 0.75 m) or a 5 x 5 window, so the vertex blob has the same metric size.
- Offsets: store them in metres divided by a fixed scale (the code divides by 25); at 0.5 m an offset of 20 m is 40 cells, so slot offsets are long-range regressions; keep the scale near the longest edge in the label set.
- The window is 100 m wide and offsets can leave it; Sat2Graph skips vertices within 16 pixels of the border in the data loader (digest) and recovers them by overlapping tiles at inference. We cannot tile an ego-centred frame, so cells near the border need either a mask or offsets that point outside (the label exists, the successor is simply not visible).
- The vertex set of the label has to be chosen (5.4). Edge and offset terms only at vertex cells is the right mask for us too (it is also what keeps the head from being trained on undefined values).

### 5.2 What is differentiable at the decode step
| decode step | paper | gradient? | replacement for us |
|---|---|---|---|
| vertexness to peaks | local maxima above p_thr (code: smoothing, divide by max, 3 x 3 test) | the comparison is a hard mask; gradient exists through the value at a selected cell (top-k with gradient through selected items is allowed by our rules) and none to unselected cells | top-k on the max-pooled map; do not divide by the tile maximum (a relative threshold makes the output depend on the other cells) |
| vertex position | the cell centre, no sub-cell refinement | no positional gradient; error up to 0.25 m at 0.5 m cells | soft-argmax in a 5 x 5 window (a weighted centroid, not an argmax) |
| slot validity | edgeness above p_thr (code: vertexness times edgeness) | the product p_v * p_ei is smooth | use it as a validity weight in the losses |
| segment end point | v + offset_i | smooth in the offset (and in v if soft-argmax is used) | feed the segment (v, v + offset_i) with weight p_v * p_ei to our matching-free point-to-polyline losses (both directions); this is a graph-level loss with no linking step |
| link to a vertex (E1 argmin, 15 m cut) | nearest candidate by E1 | argmin and threshold: no | softmin over candidates, or no link at all: the assembler (merge radius 1.5 m, snap 1.0 m) joins end points that fall within its radii |
| clean-up of the code (components, spurs, de-loop, shave, Douglas-Peucker) | not in the paper | rule steps | inference only, if at all |

The paper's training does not need any of this: E2 is dense and matching-free, so the training path has no decode and no rule step. Only if we want gradient from graph-level losses (sparsity, point-to-polyline, junction F1) do the first four rows matter, and they can all be made smooth except the top-k selection. Whether the champion's assembly (merge radius, snap) counts as a rule step on the training path is not something I checked.

### 5.3 Forks and merges of a directed lane DAG
- Sat2Graph: at most one outgoing edge per 60-degree sector, sectors fixed in the image frame, six slots; branches in different sectors are free, branches in one sector are repaired by the interpolatable property (a midpoint vertex), and by storing the edge at the other endpoint.
- For lanes the sector rule collides: the two children of a fork leave the mouth at a small angle (for illustration, a 3.5 m lane offset reached over 15 m is atan(3.5 / 15) = 13 degrees, my arithmetic), far inside one 60-degree sector, and in a fixed world frame the slot of a lane depends on the road orientation, which makes the label depend on the compass direction for no reason. An undirected encoding also loses the travel direction that our dense head already has.
- Design to test (my proposal, not run): two groups of slots per vertex in the vertex's own frame, successors (out-edges) and predecessors (in-edges), D = 3 each (the real maximum degrees of our GT graphs have to be counted first); slot order = signed bearing relative to the local travel heading (left, straight, right), a deterministic sort of the GT children, so the label is unique and matching-free. A **fork** is a vertex with at least two valid successor slots, a **merge** a vertex with at least two valid predecessor slots; a smooth score is p_v times a noisy-OR style "at least two valid slots" expression, which gives J (fork / merge F1 at 6 m) a dense differentiable detector. Merges cost nothing in the successor encoding (two parents point at the same child), forks nothing in the predecessor encoding; having both is redundant and gives a consistency check, which is what the released data loader appears to do for undirected edges (inferred above).
- Crossing lane centre-lines in a junction (a left turn crossing the oncoming lane) have no common vertex; this is the same situation as an overpass in the paper, and the 5 m conflict rule maps to: vertices of crossing lanes at least a few metres from the crossing. With our vertex spacing of about 7 m the rule rarely binds; the dense heat map blurs a crossing, the offsets do not.
- The existing 2-channel travel direction is then redundant: it is (up to normalisation) the weighted mean of the successor offsets, so it can stay as an auxiliary target.

### 5.4 Vertex density (sparsity) of the decoded graph
- Paper: vertices every d = 20 m (15 - 25) on a chain between intersections, so about 5 to 6 interpolated vertices per 100 m of straight road (my arithmetic: spacing below 20 m over 100 m) plus the intersection vertices; the paper reports no vertex count. The code additionally simplifies chains with Douglas-Peucker (tolerance 2.5 m), which removes interpolated points on straights. For comparison, our champion has 13.5 vertices per 100 m, the control 21.8, the Douglas-Peucker oracle 14.3. A Sat2Graph-style graph would be sparser on straights, with chords up to 20 m; on curves it needs shorter chords (the paper's argument for not taking d large), so d must adapt to curvature to avoid a loss in point F1.
- The issue is the label, not the decode: evenly spaced interpolation anchored at a segment end gives a phase that a 100 m window often cannot see (the anchoring junction is outside), so the vertexness along a straight lane is a smear. The paper's masking only protects the edge terms. For us the vertex set of the label should be defined by geometry (Douglas-Peucker at the tolerance of the 14.3 oracle plus fork / merge points plus a forced vertex at the border crossing), not by arc-length phase; whether that gives stable peaks is the first thing the audit in 5.6 has to show.
- Cap the number of peaks (top-k); the decoder in the code has none for large tiles.

### 5.5 What replaces the verified-ray builder, or feeds it
GTE does not replace it. It has no verification of what lies between a vertex and its predicted end point (the endpoint comes from one offset regression and one edgeness value; our hazard / survival read along the ray is exactly that missing check), no step-length selection, and it inherits the straight-segment aliasing. What it can do:
- **Seed tips** at fork and merge mouths: GTE peaks with p_v above a threshold as extra tips for the builder, or as the targets of the fork relocation step.
- **Replace the fixed fan by data-driven proposals:** the teardrop of rays (12 in the brief for this task; the level-1 note says 9 bearings, I could not check which is current) could be narrowed to rays near the slot offsets of the GTE vertex, which cuts the number of rays read per tip (speed) while the ray verification keeps deciding validity and length.
- **Dense prior for the junction head:** vertexness times the number of valid successor (or predecessor) slots is a dense fork / merge score.
- It reads the BEV once per frame for the whole graph, so the extra cost at inference is the wider output layer and a max-pool plus top-k (below a millisecond by my estimate, not measured), against the per-tip ray reading of the builder.

### 5.6 Proposed small tests (not run; to be put on a card by the loop owner)
0. **Offline label audit (S, no GPU):** on TRAIN GT count (a) slot collisions for world-fixed 6 x 60 degrees, 12 x 30 degrees and heading-relative 3 slots (the analogue of Table 3), (b) maximum in- and out-degree, (c) vertex-set stability for arc-length phase d in {10, 15, 20} m versus Douglas-Peucker vertices, as vertices per 100 m and as the shift of the vertices when the window is moved by 5 m.
1. **GTE auxiliary head (M)** on the frozen cached BEV with E2 structure; start from the code's mask and weights rescaled to our offset normalisation (they are untested in our setting); report J against the junction head, recall of GT fork / merge vertices by peaks at 6 m, and ms per frame.
2. **Candidate-feed test (M):** seed the builder with GTE peaks; report S, Q, vertices per 100 m, J, lanes found, ms per frame, two seeds, paired scene bootstrap, noise band 0.010 in S. My expectation to be written down by the owner, not a result: a tie in S on clean VAL, a gain in J only if the audit shows stable peaks.

## 6. Not found / open
- Numeric p_thr, the NMS or peak-window size, sub-cell vertex handling, offset units / normalisation (in the paper), the output activations, DLA output stride and parameter count, optimiser, batch size, training time, inference time, memory.
- The conflict-resolution algorithm of the encode step beyond the 5 m goal; the counting method behind Table 3 (the source table has no caption); the exact p_thr / w used for each row of Table 1 (the w = 100 cell matches the headline).
- The thresholds of TOPO beyond the "300 m, for example" of the text; APLS thresholds in the paper.
- Supplement, project page and the PDF were not opened; figure content (Figure 2 workflow, Figure 3 curves) not seen.
- The claim in the level-1 note that small intersection angles and curves are weak points is second-hand (RNGDet); I found no such statement in the Sections 4.5, 4.6 and 5 queries; I did not query the related-work section for it.
- Whether the released checkpoints, the "global" models (2021) and the docker decoder differ from `model/decoder.py` (the README lists newer models; nothing read).

## 7. Our experiments that touch it
Per the level-1 note (the experiment table was not opened by me): E31 / E32 (dense grid + tracer won over a matched sparse decoder, +0.058 F1, +0.18 route F, `../OVERVIEW.md` A2); the dense heat map + direction head (K62, 2.7 M U-Net) exists. No experiment has used a graph-tensor encoding, vertex-slot channels or successor offsets; nothing about Sat2Graph was run.

## Corrections to the level-1 note
1. "each edge encoded once from one endpoint": the paper says only that one link per edge is enough (from either end) and uses this as a capacity argument (an edge that does not fit in a sector at one end may be stored at the other, Table 3, 0.92 % of edges at D_max = 6). The text does not say that one end is the default. The released data loader (digest, inferred from an undirected neighbour dictionary) seems to write every edge at both ends.
2. "Loss: ... edge terms only where GT has a vertex": true for the paper's E2 (unit weights, no coefficients). The released code uses a soft mask (0.01 floor), weights 10 (edgeness) and 1000 (offset), 0.1 for a joint segmentation head, none of which is in the paper; the note lists none of this because the paper does not.
3. "Decode: ... nearest-vertex matching": it is an argmin of E1 per slot (no one-to-one matching). The code adds snapping passes and a clean-up chain (components, spurs, de-loop, shave, Douglas-Peucker) that the note and the paper do not mention.
4. "the encoding lifts a U-Net from 70.36 to 76.40 TOPO-F1 (Table 1 and text)": 70.36 is the Seg-UNet row of Table 1; 76.40 appears in the text only, and it is above the DLA headline (76.26) and close to the best cell of Table 2 (76.72 at w = 25); the Seg-DRM pair 73.80 -> 74.66 is not in the note.
5. "(a) angular-sector slots as a matching-free way to give each tip b children": the slots are fixed in the image frame and 60 degrees wide, so the children of a lane fork share one slot; heading-relative slots are needed (5.3). The note's "our fan has 9 bearings" disagrees with the 12 rays of the task brief (not checked).
6. "Failure modes: ... the RNGDet text adds (second-hand)": the paper's own failure cases are complicated stacked-road scenes and small roads under wide highways; no statement on small intersection angles was found in the sections read.
7. Licence: the note leaves it open; it is MIT for the code (`LICENSE`, "Copyright (c) 2020 Songtao He"), while pretrained models, dataset and docker container are non-commercial academic use only (README).
8. Confirmed (no change): 19 channels, D_max = 6, d = 20 m, 5 m separation, w = 100 and the 15 m rejection, backbone DLA with a 352 receptive field, 300k iterations, lr 1e-3 halved every 50k, one V100, Table 1 numbers, stacked-road precision / recall, compute not stated, undirected with directions as attributes.
