# E273 The lane graph from pixels to map ids as ONE differentiable function: pixel maps, grid-cell nodes, Hermite links, soft sections, map assignment (full plan)

Card of 2026-11-04, **revision 6 = the full plan; nothing is started, nothing trains** (Mirko: "Plan for now, do not yet start a training, because I cannot use my computer, if you do"). Steps 0 and 1 are CPU only (light, low priority); every GPU step waits for Mirko's word. Revisions 1-5 (vertex soft graph, lane-piece decoder, ports, port graph; git a256c38 ... 1da51e9) are replaced by this one; the measurements they produced are kept in section 1.

Mirko's brief and decisions of 2026-11-04 (in short): a pyramidal head (markings and road boundaries at 0.5 m, lane centres at 1 m, roads at 2 m); an end-to-end differentiable graph that understands connectivity and neighbourhood; the map graph encoded with one id per lane, splits and merges create new ids; an assignment of map ids to perception ids; differentiable up to and including the classification; "pixel map modification first, then queries with Hermite, then connection, then assignment"; **D55: the rule is end-to-end differentiable, matching in the loss is allowed**; the lane GT is built by the GT chat; a local evaluation (overlay pictures, videos, numbers) and one result folder per experiment.

## 0. The card (what the tools run)
```json card
{
 "id": "E273",
 "card": "",
 "title": "grid-cell nodes + Hermite links + soft sections + map assignment on a pyramid of pixel maps, one function, vs the E272 geometric assignment (lane level) and the best graph recall_sp0_dp",
 "kind": "block",
 "deterministic": false,
 "metric": "lane_acc_err",
 "direction": "higher",
 "band": 0.03,
 "prediction": {"value": 0.10, "vs": "g"},
 "arms": {
  "g": {"recipe": "control of the assignment, no new training: the E272 chain (seed of the same number) read out at 11.2 vertices per 100 m, its START-value geometric assigner, applied to the lane-level map tokens of section 2.5 with the same map errors"},
  "d": {"recipe": "reference of the graph, no new training: recall_sp0_dp (the prototype of record, D7 builder + rule assembly), stored network outputs, evaluate.py settings, lane suite run graphs_recall_sp0_dp_carry_20261104_181557"},
  "p": {"recipe": "the whole E273 function (section 2); step 2 pixel maps (1 m branch from the E271 head), step 3 nodes, step 4 links + Hermite curves, step 5 sections, step 6 map encoder + assignment (earlier stages frozen), step 7 everything jointly at a lower learning rate; the map gets simulated errors in training"}
 },
 "control": "g",
 "variant": "p",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "lane_acc_err", "vs": "g", "max_loss": 0.03},
  {"metric": "lane_acc_exact", "vs": "g", "max_loss": 0.01},
  {"metric": "topo_F1", "vs": "d", "max_loss": 0.03},
  {"metric": "F1@1.0", "vs": "d", "max_loss": 0.03}
 ],
 "budget_gpu_h": 10,
 "kill": "a loss that is not finite; after step 3 the node F1 (2 m) more than 0.05 below the E271 read-out's F1 at the same tolerance, or the median node position error above 0.5 m; after step 4 junction F1 not above d's 0.273 or F1@1.0 more than 0.05 below d; after step 5 more than 2 delivered pieces per GT piece; a step longer than 1.5 x its estimate (section 6)",
 "evaluate": {"split": "val"}
}
```
Verdict rule (by hand, as E271 / E272: `card.validate` demands the project's graph headline metrics and cannot read the assignment numbers; label "by hand: <reason>"). Headline `lane_acc_err` = the weighted share of the delivered nodes within 1 m of a GT lane whose most probable map piece is the map piece of that GT lane, under the map-error cell "1-2 m" (one fixed perturbation set), mean of the two seeds. **KEEP** > +0.03 over g with the guards held: exact-map `lane_acc_exact` not below g - 0.01; the graph (the project's `lcr_topo.topo_F1` and `lcr.F1@1.0`, lane suite) not below d - 0.03. **TIE** inside the band with the guards held: the function stays as the base to iterate on. **CUT** < -0.03 or a guard broken: back to the step that broke it, not a new card. Predictions: `lane_acc_err` +0.10 over g (geometry alone fails when the map is 1-2 m off; E272's map was exact); junction F1 from 0.27 (d) to about 0.45, topo F1 level with d (0.740). Reported, not judged: every layer of the pixel pyramid, lanes found, junction P / R, duplicate share, pieces per GT piece, id switches, the road-level assignment, the route attribute, ms per frame per stage.

## 1. Where we are (2026-11-04, all measured on VAL, 873 frames; one-off probes on the VAL GT, scratch in `temp/e273_probes/`, the oracle becomes a test in step 0)
- **E271** (closed, KEEP by hand): 1 m centre-lane head on the 39 frozen GaussianCaR channels + differentiable sparsification: cov5 0.721 / 0.728, prec1 0.871 / 0.873 at 10.3 vertices per 100 m. **E272** (closed, TIE): head + hold + geometric assignment as one trained function; the hold gate closes on clean frames, the assignment learned nothing beyond geometry because the map was exact.
- **The best graph** `recall_sp0_dp` (lane suite, run `graphs_recall_sp0_dp_carry_20261104_181557`): Q 0.742, F1@1.0 0.878, lanes found 0.781, topo F1 0.740 (P 0.811, R 0.682), duplicates 0.077, **junction F1 0.273** (P 0.213, R 0.382). The slow control (E94 trace network): topo F1 0.763, junction F1 0.333, 21.8 vertices per 100 m, 628 ms. Errors by situation: the quarter of frames with the most GT lane length holds 51 % of the error, the quarter with the most predicted pieces 48 % (fragmentation). Its connections are rules (path union, fork pruning), and a loss on the delivered graph reaches none of its node positions, validity or links (E269 measured exactly zero).
- **Why a grid of centre pixels cannot carry the connections:** at 1 m, 12.1 % of the lane cells hold two or more pieces and 1.9 % crossing pieces (headings apart by more than 30 degrees), 18.4 % / 3.7 % at 2 m; almost all of it is connectors.
- **Connectors dominate the graph:** 54 % of the pieces in the window are connectors (median 19.8 m, their ends 19.1 m apart, 47 % farther than 20 m); the middle of a connector collides with another piece's middle in 31 % of the cases (lane pieces: 0 %); 61 % of the connectors join a lane end to a lane start, 35 % a lane to the window edge, 3 % are chained. Their shape is cheap: a cubic Hermite / Bezier between the two end poses with two handle lengths fits 99.5 % of all pieces within 0.5 m.
- **GT status.** The lane GT is its own version of autogt (`autogt/lane_gt.py`, tag `autogt/lane-l1`, signature 42a3d08f8f, commit cf0ac51, local): pieces with ids that change only at splits and merges, successors, left / right neighbours, lane index and count, events (split, merge, stop, start, end, window, cross) with in / out ids and directions, roads, dividers, road boundary with corners; **l3** (tag `autogt/lane-l3`, signature 3576fa1f7e, commit 4df1032, local) adds the drivable-area and intersection polygons (`g["areas"]`, not dilated), the two additions of this plan (`exit` events, `segments` per piece) and review fixes (joined-piece stops now name their continuation, aligned direction rows, `opposite` on both sides, short parts with a natural end are kept); the front GT (c643a7539d) is untouched. The full store (850 sequences) builds with l3 in the background (about 45 min, nice 15). The piece id `piece` is the same in every frame (use it for the assignment targets); `id` differs only for a part that comes back into the window (`<piece>#1`). Against our `make_graph_gt.py` its points agree within 0.3 m (99.9 % / 99.6 %); its 2 m minimum piece length drops only stubs.
- **Literature:** the field trains query-based lane graphs end to end with matching only in the loss (TopoNet, MapTRv2, LaneSegNet, CGNet, RATopo); D55 allows that; the design below does not need it (targets by location) and keeps it as the fallback (section 7).

## 2. The pipeline (one function, five stages)
```
frozen GaussianCaR 39 ch, 200 x 200 @ 0.5 m, window |x|, |y| < 50 m, lane frame (x right, y forward)
  A  pixel maps (one encoder, three taps)      0.5 m: lane divider, road divider, road boundary (logit + truncated distance)
                                               1 m:   centre lane (p, cos, sin, ox, oy, gate = E271's 6 ch) + event heats (split, merge, stop, exit, start, end)
                                               2 m:   road, intersection (polygon rasterised at 2 m); the 2 m feature map is the node grid
  B  nodes: one per 2 m cell (50 x 50 = 2500), 1 slot, type flags, pulled onto the line (offset, direction), refined once on the fine features
  C  links: node -> node (short, forward, <= 8 m: all cells; long, through intersections: the top-K entries x top-K exits); curve = cubic Hermite from the two node poses
  D  sections (the ids): absorbing walk backwards on the link matrix -> every node gets a distribution over section starts
  E  map side: map pieces -> graph transformer; nodes x map pieces cross-attention + dustbin (+ soft registration, + structure term)
       -> P(map piece | node), P(road | node), P(route | node), per section by pooling
```

### 2.1 Stage A: the pixel maps (Mirko's pyramid)
- One shared stride-2 U-Net (E271's head, 2.05 M parameters, GroupNorm) widened by three taps. The 1 m tap starts from the E271 head's weights, so the centre-lane channels keep their numbers.
- **0.5 m** (200 x 200): lane divider, road divider, road boundary, each as a line logit and a truncated distance (2 m, sub-cell position); 6 channels. The stop line is an optional later layer.
- **1 m** (100 x 100): E271's six channels (p, cos, sin, ox, oy, keep gate) and six event heats (split, merge, stop = lane end in front of an intersection, exit = lane start after it, start, end) with Gaussian sigma 1.5 m. They give the nodes dense help and give the pictures their layer.
- **2 m** (50 x 50): road (drivable area) and intersection: the GT polygons rasterised at 2 m, a cell counts when any part of it is covered (that is the slight enlargement); Mirko told the GT chat to leave the dilation out of the GT, so a further dilation is a loader parameter, default 0 m; the road-level graph is a v2 item (section 8).
- Targets: the lane GT vectors rasterised at load time (lines with exact distance, centre pieces, events, areas): nothing is stored besides the vectors. Losses: focal BCE + Dice on masks, L1 on truncated distances, E271's `grid_loss` unchanged on the 1 m channels, focal BCE on the event heats, BCE + Dice on road and intersection.
- 8 heading bins per cell (revision 3) are dropped: once the connectors carry no nodes, lanes never share a 2 m cell (section 2.8).

### 2.2 Stage B: nodes on the 2 m grid
- **Cells and slots.** 2500 cells, one node slot per cell. Measured: with the intersection connectors suppressed, 99.4 % of the occupied cells hold exactly one lane point (section 2.8); a second slot is not needed.
- **Token** of a cell = the 2 m feature + the pooled 1 m / 0.5 m features of its 2 x 2 sub-cells + the road and intersection channels + a Fourier code of the cell centre; then **2 layers of neighbourhood attention** (9 x 9 cells = +-8 m) so that a node can tell a split from a normal point by what lies ahead.
- **Outputs per cell:** existence e; type flags (multi-label: stop, exit, split, merge, start, end, window-in, window-out; a point can be exit and merge at once); offset o (2, bounded to the cell: 1 m x tanh); direction h (cos, sin, over 360 degrees); section-start probability s; two scores u_out and u_in (is this node an end of a long link, section 2.3). The node sits at p = cell centre + o with weight e.
- **Pull onto the line (refinement).** A second pass samples the 1 m and 0.5 m feature maps bilinearly at p, updates the token with a residual MLP and re-predicts o, h and the flags. This is where a query is "pulled to the line, to an end point, to a split / merge point".
- **Targets by location, no matching** (rule in section 3.3): the item of a cell is the point of a GT piece nearest to the cell centre inside the half-open cell; a piece's end points (events) win their cell; connector points inside intersections give no item, event points are always kept. Cells without an item learn e = 0. Losses: focal BCE on e; BCE on the flags and s (masked by e); L1 on o and cosine on h (masked).
- Cost: 2500 tokens, 64-d, local attention only: a few ms (estimate).

### 2.3 Stage C: links and Hermite curves (the connection matrix)
- **Short links** (the lane itself): node i links to j in its forward half-plane within 8 m: about 7 candidates per node (measured), computed densely for all cells, the weights e multiply the loss. Logit = MLP of [token i, token j, offset of j in i's frame (along, across, distance), heading change (sin, cos), evidence: the centre probability sampled at 8 points of the Hermite curve (mean, min)]. The curve is a cubic Hermite segment from (p_i, h_i) to (p_j, h_j) with both handles = chord / 3 and nothing learned: measured, nodes taken every 4 m reproduce the GT lanes within the 0.25 m floor of the measurement, and the decoded lanes cover 99.98 % of the GT lane points within 0.3 m.
- **Long links** (through an intersection, the stop -> exit connectors): the endpoints are chosen by the learned scores u_out (top-32 cells) and u_in (top-32): an index choice with the gradient through the values (the D7 SELECTION convention, named in 2.7); the GT chat counts at most 22 stop events per frame on mini. Pairs within 100 m and a 75 degree forward cone. Measured: 8.6 entries per frame, the true partner lies in the cone in 99.4 %, 7.3 true exit candidates per entry (107 if every node were a candidate). The curve is a Hermite curve with two learned handle scales (a, b) = 0.9 x sigmoid(MLP of the pair), initialised at 1/3; the evidence (centre probability along 16 points: mean, minimum, longest gap) enters the logit: missing paint inside an intersection is normal, the weight on it is learned.
- **Smoothness is by construction:** every piece of the graph is a chain of Hermite segments that share position and direction at every node (G1); the sharp corner of a road boundary is a node with a free direction change; long links are the smooth curves already measured (0.26 m p90 against the GT connectors with fitted handles). Extra: a curvature penalty on the handle scales.
- **Degrees.** A node expects one successor, a split two, an end none: the type flags and a soft degree term (expected out-degree = sum of the link probabilities against the GT degree) keep the matrix consistent; links are independent sigmoids (RATopo-style extra positives or Sinkhorn are the fallbacks, section 7).
- **Targets:** consecutive items along a GT piece are linked; the last item of a piece links to the first item of each successor, through suppressed connector parts (long links); positives are mapped to cells by location. Loss: focal BCE over all candidate pairs; for positive long links a Chamfer distance between the sampled curve and the GT connector points (it trains the handle scales).

### 2.4 Stage D: sections (one id per lane, new ids at splits and merges)
- Section start probability s(n) (a node head, supervised: 1 at the first node of a GT piece, i.e. after a split, a merge, a start or a window entry). Predecessor distribution Pb[n, m] = softmax of the link logits into n over the candidates m, plus "none".
- **Absorbing walk backwards:** membership M = (I - diag(1 - s) Pb)^(-1) diag(s), rows normalised: M[n, k] = the probability that the section of n starts at node k. A dense solve on the K = 512 most probable nodes (an index choice, 3 x the measured 160 nodes per frame) or pointer doubling; differentiable.
- Measured with noisy probabilities (true link 0.97, true start 0.95, false start 0.01): the most probable start is the right one for 99.8 % of the nodes, 97.7 % for nodes more than 30 nodes after their start (oracle probe).
- Loss: for sampled node pairs the same-section probability sum_k M[i, k] M[j, k] against the GT piece ids (BCE), plus the cross entropy of the start index. Delivered pieces = the argmax sections (output step).

### 2.5 Stage E: the map and the assignment
- **The map** is the lane GT graph of the frame (pieces after the id rule, with successor, predecessor, left / right ids, lane index and count, road id) with simulated errors: a scene-constant translation (sigma 1-2 m) and yaw (2 sigma deg), thinned shape points (Douglas-Peucker 0.5-2 m), 5 % of the pieces dropped, 5 % wrong lane counts. It is the lane-level map (ADASIS "SD+" level); the road level is the `road` ids of the pieces and the `roads` of the GT (derived from the lanes: two nuScenes maps have no usable road layer). Built in `training/grid/map_sim.py`; the correct correspondence is known by construction.
- **Map encoder:** each piece resampled every 2 m; token features: position, tangent, arclength fraction, kind, lane count, lane index, number of successors / predecessors, route flag (the id is a label, not a feature); 3 graph-transformer layers whose attention runs along the map's own edges (successor, predecessor, left, right, opposite) with an edge-type bias; piece embedding by attention pooling. Taken from SMERF / SEPT / TopoSD (map as tokens) and MAT (path-aware attention).
- **Assignment:** cross-attention from the nodes (rows: the perception structure is kept) to the map pieces (columns) plus a dustbin: logit_ij = q_i . k_j / sqrt(d) + b_geo(i, j); b_geo is E272's geometric term (squared distance to the nearest token of piece j, heading difference), so the start equals the control g. Two rounds of soft registration (weighted Procrustes over the soft pairs, weights e) move the map tokens. **Structure term:** two nodes joined by a link must land on map pieces joined by the map's own edges: -log(P_i^T C P_j) weighted by the link probability, C = identity + the map's successor adjacency. Per section: P(map piece | section) = sum over nodes weighted by M. Road level: P(road | node) = sum of P(piece | node) over the pieces of that road (the fallback when the lane level fails); route: P(route | node) = sum over the route pieces.
- **Loss:** NLL of the node's GT piece (its own piece id; the dustbin when the piece was dropped from the map or the node has no GT) + the structure term + route BCE.

### 2.6 Training: blocks first, then joint (as E272)
| step | trained | frozen | loss | estimate |
|---|---|---|---|---|
| 2 pixel maps | encoder, three taps (1 m from E271) | - | the pixel losses | 3000 steps |
| 3 nodes | token stage, local attention, node heads, refinement | stage A | node losses | 3000 steps |
| 4 links | link MLP, handle scales, u scores | A, B | link BCE, degree, Chamfer | 3000 steps |
| 5 sections | s head, membership | A, B, C | pair BCE, start CE | 1500 steps |
| 6 map + assignment | map encoder, assignment | A-D | assignment NLL, structure, route | 1500 steps |
| 7 joint | everything | - | all of it, lower learning rate | 2000 steps |
The hold of E272 stays out (its gate closes on clean frames); temporal fusion returns on the nodes later (section 8).

### 2.7 What is differentiable (audit)
- **Forward path, all soft:** pixel maps -> node tokens -> local attention -> node outputs (existence, flags, offset, direction) -> refinement (bilinear sampling) -> link logits -> Hermite curves (closed formulas of node poses and handle scales) -> section membership (a linear solve) -> assignment (attention + Procrustes) -> P(map piece), P(road), P(route). A loss on any output reaches every earlier stage.
- **Index choices (named; the gradient flows through the values of the chosen items, the D7 SELECTION convention):** (1) the top-32 entries and exits of the long links by u_out / u_in; (2) the K = 512 active nodes of the section solve by e. The candidate windows of the short links are fixed geometry. A dense alternative exists at about 25 x the pairs and is not planned.
- **Output step (named):** e > 0.5, link probability > 0.5, argmax of the section membership; the delivered graph (nodes, Hermite chains, pieces, edges) is read from them on the CPU (`qgraph_eval.py`). No loss goes through it.
- **Matching:** none in the first version (targets by location, one slot per cell); allowed by D55 if a gate fails.

### 2.8 Measured sizes (the representation oracle on the VAL GT, 218 frames, GT = `graph_gt_val.pkl`; connectors suppressed, their window ends and dead ends kept)
| quantity | value |
|---|---|
| nodes per frame (lane points + events) | 159 (27 are events); 158 occupied cells of 2500 |
| cells with two or more nodes | 0.6 % (0.0 % without the window / dead-end connector events) |
| links per frame | 156: 145 short, 11 long |
| out-degree of a node | 0: 3.7 %, 1: 94.7 %, 2: 1.5 %, 3+: 0.1 % (in-degree alike) |
| short candidates per node (forward, 8 m) | 7.2; true short link missed by the candidates: 0.01 % |
| long links: entries per frame; true partner inside the 100 m / 75 degree cone | 8.6; 99.4 % |
| candidates per entry: every node / exit nodes only | 107 / 7.3 |
| Hermite decode of the GT links, GT points within 0.3 m of the decoded curves | lane pieces 99.98 %, connectors 93.4 % (crude 5 x 5 handle grid; the rest is overlapping duplicate connectors of the old GT and connectors with no node at one end) |
| decoded points within 0.3 m of the GT | 98.3 % |
| section start by the absorbing walk, noisy probabilities | 99.8 % right; 97.7 % at 30 or more nodes from the start |
With the connectors kept as nodes 18.5 % of the occupied cells would hold two nodes and 4 % three: the intersection suppression is what makes one slot enough.

## 3. Ground truth: what exists, what is requested, how the targets are made
### 3.1 One vector source
Everything (pixel maps, nodes, links, sections, map tokens) is drawn from the lane GT of the autogt chat (`autogt.lane_gt`, frame "lane", window 100 x 100 m): pieces (ids, polylines every 0.5 m, successor / predecessor, left / right, lane index and count, road), events, roads, lines (dividers, boundary with corners), areas (l2: drivable, intersection). Only vectors are stored; rasters and targets are made in the data loader; the 39 frozen channels and the poses come from `BuilderData` as before, the sample tokens from `feats/{train,val}_tokens.json`.
### 3.2 Requests to the GT chat (small, light)
- **R1 exit events (done in l3):** the mirror of `stop`: the start of a lane whose predecessor connector lies inside an intersection (fields `on`, `s`, `n_pred`, `in`, `out`, `dirs_in`, `dirs_out`; `pos` is the joint vertex). Needed for the type "exit" of the nodes.
- **R2 (done in l2 / l3):** area polygons (`areas.road`, `areas.intersection`, polygons with holes, not dilated).
- **R3 segments per piece (done in l3):** `segments = [{token, kind, i0, i1, in_intersection}]` (contiguous index ranges of the part's points per original map piece, with the test result; False for lanes), so that the loader can drop the points of connectors inside intersections and keep the points of connectors outside them (16-20 % of the connectors are plain continuations).
- The open parameters stay the GT chat's defaults (boundary layers, corner 30 degrees, stop rule "intersection_branch").
### 3.3 Target builder (lane side, `training/grid/qtargets.py`): the rules
| item | rule |
|---|---|
| node of a cell | per piece and 2 m half-open cell the point nearest to the centre; the first and last point of a piece win their cell; at most one item per cell (event first, else nearer to the centre; the dropped share is reported, expected < 1 %) |
| suppression | points of connector segments inside an intersection give no item; event points are always items |
| flags | an item within 1 m of a GT event gets its flag(s); window events at the border |
| offset, direction | item position minus cell centre; the piece tangent at the item |
| links | consecutive items of a piece; the last item of a piece to the first item of each successor through suppressed parts; distance above 8 m = long |
| sections | the GT piece id (ids change only at splits and merges); the section start = the first item of a piece; s = 1 there |
| map | the pieces of the frame, errors by `map_sim.py`; the target of a node = its own piece id, the dustbin if dropped |
| pixel layers | lines, centre pieces, event heats, areas rasterised from the vectors with exact distance |
### 3.4 Checks of step 0 (before anything is trained)
1. The representation oracle: GT items and GT links through the Hermite decoder and the section solve reproduce the GT graph: lane points 99.9 %, connector points at least 95 % (0.934 on the old GT with a crude handle grid), section starts right for 100 % with clean links and at least 97 % with the noisy probabilities, at most 1 % of the cells with two items.
2. The new layers lie on the lane graph (median under 0.25 m); the round trip vectors -> rasters -> vectors stays within 0.3 m (the check the GT chat left to us).
3. The target builder reproduces the counts of section 2.8 on the lane GT store.
4. Pictures with the GT projected into the six cameras (the GT chat's `lane_graph_view`).

## 4. Local evaluation (new local scorer and pictures; `evaluation/evaluate.py` and the global protocol untouched)
Frames: VAL 873, the order of E271 / E272 and the lane suite; map cells: exact, error 1-2 m (one fixed set), no map. `evaluation/qgraph_eval.py` (the graph part through the lane suite, so d and p are scored by the same code; the GT of the suite is the old graph GT, which agrees with the lane GT within 0.3 m), pictures `tools/viz/qgraph_view.py` (from `grid_view.py`), both local to this card.
| level | numbers | why |
|---|---|---|
| 0.5 m lines | F1 of line cells within 0.5 and 1.0 m, per class; mean distance of matched cells | a half-cell shift of a 1-cell line is a full miss in IoU |
| 1 m centres | centre-pixel F1 at 1 m, cov5 / prec1 / vertices per 100 m (E271's), event-heat F1 at 3 m per type | continuity with E271 / E272 |
| 2 m roads | IoU of drivable and intersection | standard for areas |
| nodes | F1 of existence within 2 m, median position and direction error, event recall and precision per type within 3 m (stop, exit, split, merge, start, end, window) | split and merge positions are a map convention: 3 m tolerance |
| links | precision / recall of short and long links on matched nodes, degree accuracy | the connection matrix itself |
| graph | the lane suite on the delivered graph: F1@1.0, lanes found, topo F1 (P / R), junction F1 (P / R), duplicates, Q, R@1 far; pieces per GT piece, id switches per 100 m | the same code that measured d today |
| assignment | `lane_acc` (exact, 1-2 m), dustbin precision / recall, per-GT-lane purity, road-level accuracy, Association P-R of MAT | the card's headline |
| drive | route F1 per node, route success (`p8_lib.route_success`) | the planner-level attribute |
| speed | ms per frame per stage (idle GPU, batch 1, cached features) | project rule |
**Sense checks before any number is believed (layers 0-2):** the GT through the scorer gives 1.0 everywhere; known degradations move the numbers as expected (lines shifted by 0.25 / 0.5 / 1 m, 10 % of the nodes dropped, links cut, two sections' ids swapped, the map moved by 1 / 2 m); null rows (empty output, all-dustbin), the control g and the reference d; d's numbers of today reproduced through the new scorer (Q 0.742, junction F1 0.273).
**Pictures per frame** (one figure, all BEV panels on the 0.5 m canvas; 1 m maps scaled x2 and 2 m maps x4 by nearest neighbour):
1. the camera images with the GT projected (centres, dividers, boundary; six cameras, the front one large);
2. 1 m centre pixels: GT as a faint grey underlay, correct cells in a neutral dark tone, false positives a light red tint, misses a light blue tint (alpha about 0.35);
3. 2 m road and intersection pixels, the same scheme; 0.5 m lines, the same scheme with one hue per class;
4. nodes by type (a marker per flag), Hermite links drawn, GT graph thin grey underneath, false nodes / links tinted red, missed ones blue, junctions circled faintly;
5. ids: GT side, each map piece in a clean colour; system side, each node in the expected colour sum_j P(j | i) colour_j, dustbin grey (an unsure assignment blends the colours, as Mirko expects); section switches marked.
Videos: the same figure per keyframe (2 Hz) as mp4 for four VAL scenes: the two junction scenes (scene-0221, scene-0108), one straight (scene-0092), one night scene; d in the same layout for the side-by-side.

## 5. Result folder (one per experiment, `results/E273/`)
```
results/E273/
  README.md          the index: question, verdict, the numbers table (from the cards, never typed), key pictures, how to rebuild everything   (git)
  <run id>/package.json, provenance.json     from autoresearch/tools/package.py                                                          (git)
      names: code tag lcr/<run id> + commit; the head / graph code bundle hash (training/grid/*.py copied into bundle/, not in git); front GT version and signature
             (c643a7539d, gt.json); lane GT version and signature (autogt/lane-l3, 3576fa1f7e); the map-error set and its seed; training version
             (trainer commit, config.json, steps per phase, seeds); dataset (nuScenes v1.0-trainval, data/splits.json hash, frame counts); evaluation code version
             (commit + hash of qgraph_eval.py and lane_suite.py); the numbers (quantitative); the derived files with the command that rebuilds them
  figures/, videos/  pictures and mp4 (not in git: binaries), the qualitative part
```
The lane project has no `results/` folder yet; `package.py` runs per problem and may need a `package` entry in `project.json` (a protected file: a D entry and a grant first).

## 6. Steps, gates, cost (nothing past step 1 runs before Mirko's word)
| step | what | where | gate / done when | cost |
|---|---|---|---|---|
| 0 | (R1, R3 are in l3, the l3 store is building); target builder, map simulator, oracle | `training/grid/qtargets.py`, `map_sim.py`, `training/data/lane_gt_data.py`, tests | section 3.4 passes | CPU, about 3 days of coding (estimate), light runs |
| 1 | scorer and pictures first | `evaluation/qgraph_eval.py`, `tools/viz/qgraph_view.py` | GT scores 1.0, degradations respond, d reproduced, g scored | CPU, about 1 day (estimate) |
| 2 | pixel maps | `training/grid/pyramid.py`, `training/train/qgraph_train.py` | G1 (gradient into every tap, one batch overfits); centre-pixel F1 at 1 m not below the E271 dense head - 0.02 | GPU about 0.5 h / seed |
| 3 | nodes | `training/grid/qnodes.py` | node F1 (2 m) not more than 0.05 below the E271 read-out at the same tolerance; median position error <= 0.5 m; event recall reported | GPU about 0.5 h / seed |
| 4 | links + Hermite | `training/grid/qlinks.py` | junction F1 above 0.273; F1@1.0 and topo F1 not more than 0.05 below d | GPU about 0.5 h / seed |
| 5 | sections | `training/grid/sections.py` | at most 2 delivered pieces per GT piece; section starts right for at least 95 % of the nodes | GPU about 0.3 h / seed |
| 6 | map + assignment | `training/grid/map_graph.py`, `qassign.py` | at the start equal to g; `lane_acc_err` above g | GPU about 0.3 h / seed |
| 7 | joint, both seeds, scoring, pictures, videos, result folder | `results/E273/` | verdict by the rule of section 0 | GPU about 1 h / seed + 0.5 h scoring |
Estimate from the step times of E271 / E272 (0.6 s per step for a frozen-head stage, 2.1 s for the joint stage): about 5 GPU hours per seed with development, 10 in all; replaced by measured values after step 2. Code rules: a test and a README row per module in the same commit, the grid tests registered in `run_all_tests.sh`, an `ar-critic` pass (mode code) on the shared code before step 2, commit by path, scratch to `temp/`. Dependencies on other chats: the GT chat (the l3 lane store); the BEV system chat owns the GPU slots (trainsuite queue, `run.py`).

## 7. Risks and fallbacks
| risk | effect | what we do |
|---|---|---|
| intersection mask wrong, or the node head cannot tell connector from lane points | nodes inside intersections, two items per cell | the mask is a dilated input channel and a loss; a second slot is the reserve; measured on the oracle first |
| weak evidence inside junctions: long links hard to score | junction F1 stays near d's 0.27 | the learned weight on missing evidence; the map prior through the assignment (the structure term and a map-conditioned long-link logit); RATopo-style extra positives for the link loss (3 per GT link; in their paper TopoNet's lane-lane topology rose from 10.9 to 29.5 on OpenLane-V2 subset A, not our metric) |
| splits come out with three successors | degree errors | Sinkhorn row / column normalisation with a dustbin (PolyWorld, SuperGlue) instead of independent sigmoids |
| soft sections blur at junctions | colours blend, pieces fragment | by design the blend is visible in the pictures; the pair loss and the pieces-per-GT-piece gate; a learned-embedding head as the fallback (it needs a clustering output step) |
| the by-location targets are unstable near cell borders | offsets flip between neighbouring cells | the rule is deterministic (nearest to the centre, half-open cells); neighbour-cell positives as in RATopo if needed; D55 allows a matching loss |
| the whole design loses to a query decoder | the base is wrong | arm Q as a REFERENCE row only: CGNet (ECCV 2024, nuScenes, Apache-2.0 `xiaomi/cgnet`): read its tables first; if its checkpoint runs, score its graphs with the lane suite next to d; build a query decoder on our pyramid only if p misses its gates (RATopo recipe for its link matrix) |
| E32 repeats (a DETR-style decoder from scratch lost 0.058 F1 to the grid method) | worse than the dense method | not applicable to the grid nodes: they start from dense pixel maps and fixed positions, not from learned queries |
| the front GT or the lane store changes under us | numbers not comparable | the package names both signatures; compare only runs with the same signatures |

## 8. Assumptions I chose (the defaults of this plan; each can be changed before step 0)
1. One node slot per 2 m cell (measured). 2. Connector points inside intersections give no nodes; entry (stop) and exit markers are always nodes. 3. Short links are Hermite segments with fixed chord / 3 handles, long links with two learned handle scales. 4. Sections by the absorbing walk. 5. Matching-free targets by location first; matching in the loss (D55) and the query-decoder arm Q only as fallback and reference. 6. The map is the lane-level nuScenes lane graph with simulated errors; the road level is derived from the lanes (the nuScenes road layer is unusable on two of four maps). 7. The GT chat's defaults for the open parameters (boundary = edge of the drivable area including driveways and parking, corner angle 30 degrees, a stop point with two successors only for the intersection branch). 8. Out of this card (v2): temporal fusion on the nodes (the hold), a perceived road graph without a map, stop lines, the six-camera visibility masks. 9. No GPU step before Mirko's word; steps 0-1 are light CPU work at low priority.

## 9. What others do (local catalogue in `literature/`)
| work | what it does | what we take |
|---|---|---|
| TopoNet, MapTRv2, LaneSegNet, LaneGAP, CGNet (ECCV 2024) | query-based lane graphs, Hungarian matching in the loss, successor matrix between queries; CGNet: junction-aware queries, Bezier-space continuity, iterative topology refinement | arm Q as the reference and the fallback; junction awareness = our event heats and node flags; Bezier / Hermite continuity = our curves |
| RATopo (ACM MM 2025) | one-to-many supervision for the topology matrix (3 positives per lane), reordered decoder, parallel cross-attention | the link-loss fallback |
| SAM-Road, MaGRoad | key points from a dense map, transformer scoring of links, edge score from path statistics | the evidence along the curve in the link logit |
| CenterNet, CondLaneNet | key points as heat maps, targets by location, no matching | nodes by location |
| PolyWorld, SuperGlue, LightGlue, scene-graph matching (2604.27821) | Sinkhorn degree matching with a dustbin, sharp corners from vertex connections | the degree fallback |
| MAT / OMA (ICLR 2026), SMERF, SEPT, TopoSD | map as polyline tokens, path-aware attention, map noise in training | the map encoder and the error simulation |
| MGMap, PseudoMapTrainer (ICCV 2025) | multi-scale BEV features; finer grids help lines, not areas | the pyramid taps and their resolutions |
Gap: nobody assigns perceived lane-level ids to a lane-level map graph with a soft, structure-aware assignment and trains it through the perception; published association is road-level (MAT) or hard.

## 10. Summary
Open.
