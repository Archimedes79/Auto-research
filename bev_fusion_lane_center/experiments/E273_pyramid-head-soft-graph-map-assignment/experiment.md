# E273 Pyramid head -> port graph (pixels, then smooth connections) -> map lane graph -> assignment, as ONE differentiable function (plan)

Card of 2026-11-04, **plan only, nothing started** (Mirko: "Plan for now, do not yet start a training"). Mirko's brief of 2026-11-04: a pyramidal head (lane markings and road boundaries at 0.5 m, lane centres at 1 m, roads at 2 m); an E2E differentiable graph that understands connectivity and neighbourhood; the map graph encoded sparsely with one id per lane (splits and merges create new ids); an assignment of map ids to perception lanes (cross-attention that keeps the structure of the input and estimates the id from the map); E2E differentiable up to and including the classification; a local evaluation with overlay pictures and videos; one result folder per experiment. Second message of the day: "the graph is not good. What would be a good base?" -> section 1b (revision 2 of this card, same day: the vertex-level soft graph of revision 1 is replaced by a lane-piece decoder; revision 3: intersections; revision 4: Mirko's two layers, centre and port pixels, then smooth connections between ports, replace the anchors). One card, as for E272: the phases are inside it, each with its own gate and a numbers row.

## 0. The card (what the tools run)
```json card
{
 "id": "E273",
 "card": "",
 "title": "pyramid head (centre and port pixels) + port graph (smooth connections between ports, one id per piece) + map lane graph encoder + structure-keeping assignment, one function, vs the E272 chain (geometry-only assignment) and the best graph recall_sp0_dp",
 "kind": "block",
 "deterministic": false,
 "metric": "lane_acc_err",
 "direction": "higher",
 "band": 0.03,
 "prediction": {"value": 0.10, "vs": "g"},
 "arms": {
  "g": {"recipe": "control of the assignment, no new training: the E272 chain (seed of the same number) read out at 11.2 vertices per 100 m, its START-value geometric assigner, applied to the map LANE tokens of section 2.3 (lane-level ids) with the same map errors"},
  "d": {"recipe": "reference of the graph, no new training: recall_sp0_dp (the prototype of record, D7 builder + rule assembly), stored network outputs, evaluate.py settings"},
  "p": {"recipe": "the whole E273 function; phase 1 pyramid head (1 m branch from the E271 head), phase 2 port graph (head frozen), phase 3 map encoder + assignment (head and port graph frozen), phase 4 everything jointly at a lower learning rate; maps with simulated errors in training"}
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
 "budget_gpu_h": 14,
 "kill": "a loss that is not finite; after phase 1 the 1 m cov5 more than 0.03 below the E271 head; after phase 2 F1@1.0 of the delivered pieces more than 0.05 below d or junction_F1 not above d's 0.273; a phase longer than 1.5 x its estimate (section 6)",
 "evaluate": {"split": "val"}
}
```
Verdict rule (by hand, as E271 / E272: `card.validate` demands the project's graph headline metrics and cannot read the assignment numbers): headline `lane_acc_err` = weighted share of the delivered piece points within 1 m of a GT lane whose most probable map lane is the map lane of that GT lane, under the map-error cell "1-2 m" (fixed perturbation set, section 4), mean of the two seeds. **KEEP** > +0.03 over g with the guards held: exact map `lane_acc_exact` not below g - 0.01; the graph (the project's `lcr_topo.topo_F1` and `lcr.F1@1.0`, lane suite) not below d - 0.03. **TIE** inside the band with the guards held: the function stays as the base to iterate on. **CUT** < -0.03 or a guard broken: back to the phase that broke it, not a new card. Predictions: `lane_acc_err` +0.10 over g (geometry alone fails when the map is 1-2 m off, E272's map was exact); graph: junction F1 from 0.27 (d) to about 0.45 and topo F1 level with d (0.740) - the piece decoder makes split and merge points explicit (piece ends), the rule assembly of d creates them by accident.

Reported, not judged: the 0.5 m and 2 m layers (no control exists yet), lanes found, junction P / R, duplicate share, the route attribute, ms per frame, hold on / off.

## 1. Where we are (2026-11-04)
- **E271** (closed, KEEP by hand): 1 m centre-lane grid head on the 39 frozen GaussianCaR channels + differentiable sparsification: VAL cov5 0.721 / 0.728, prec1 0.871 / 0.873 at 10.3 vertices per 100 m (2 seeds).
- **E272** (closed, TIE by hand): head -> sparsify -> hold -> assignment as ONE trained function: cov5 +0.006 (mean of 2 seeds), the hold gate closes on clean frames, the assignment learned nothing beyond geometry (the SD map is GT-derived and exact).
- **The best graph, measured today** (`recall_sp0_dp`, lane suite `graphs:recall_sp0_dp quant run`, VAL 873 frames, run `graphs_recall_sp0_dp_carry_20261104_181557`): Q 0.742, F1@1.0 0.878, lanes found 0.781, topo F1 0.740 (P 0.811, R 0.682), duplicate share 0.077, **junction F1 0.273 (P 0.213, R 0.382)**: the lanes are where they should be, the graph between them is not. Where the errors are (`probe situations`): the quarter of frames with the most GT lane length holds 51 % of the error, the quarter with the most predicted pieces 48 % (fragmentation), slow ego speed (waiting at junctions) 40 %, far range 39 %. On the two VAL scenes with the most splits and merges (scene-0221, scene-0108; video `runs/prototype/recall_sp0_dp_junctions/`): Q 0.709, topo F1 0.634, lanes found 0.677.
- **Why a grid alone cannot fix it** (`temp/overlap_probe.py`, VAL GT): at 1 m, 12.1 % of the lane cells hold two or more lane pieces and 1.9 % hold pieces whose headings differ by more than 30 degrees (crossing connectors); at 2 m 18.4 % / 3.7 %. A head with one heading per cell cannot represent them, and they sit exactly at splits, merges and junctions; after a split the two branches share cells for metres before they separate, so the split point is placed late.
- **Intersections in the GT** (`temp/junction_probe.py`, `temp/connector_oracle.py`, VAL 873 frames; Mirko's question of 2026-11-04 "how to handle intersections: weak information, connection points far away, large overlapping anchors, smoothness"): connectors are 54 % of the pieces in the window (10,830 against 9,104 lane pieces), median 19.8 m long, their two ends median 19.1 m apart (47 % more than 20 m). Midpoint anchors collide for connectors (31 % have another piece's midpoint within 2 m, 20 % even with the same heading) and never for lane pieces (0.000). 61.3 % of the connectors join the end of a lane piece to the start of a lane piece, 35.0 % join a lane piece to the window edge, 3.3 % are chained to another connector, 0.4 % other. -> Section 2.2: connectors are not anchored objects but relations between ports.
- **No E2E graph with connectivity exists yet.** The D7 builder (behind d) is not E2E: rule assembly and SELECTION steps, and E269's gradient measurement found exactly zero gradient from a downstream loss into its node positions, validity and links. The grid chain is E2E but has no edges.

## 1b. Which base for the graph (the answer to "what would be a good base?")
| candidate | graph today (VAL) | E2E to the id | ids per lane piece | junctions | speed | verdict |
|---|---|---|---|---|---|---|
| D7 builder + rule assembly (`recall_sp0_dp`, d) | Q 0.742, topo F1 0.740, junction F1 0.273 | no (assembly rules; zero gradient into positions / validity / links, E269) | no (vertex trees, pieces made by the assembly) | weak: one direction field, junctions by accident | 21-27 ms | **reference and fallback**, its graphs a teacher; making it E2E (E270, 4-5 days) keeps the representation that fails at junctions |
| E94 trace network (`control`) | Q 0.737, topo F1 0.763, lanes found 0.800, junction F1 0.333, at 21.8 vertices per 100 m (twice the budget), duplicate share 0.129 (lane suite today) | no (sequential tracing with rules) | no | a little better than D7, still poor | 628 ms | too slow, not a base; even the best tracer finds only a third of the junctions |
| grid chain + vertex soft graph (revision 1 of this card) | vertices only (cov5 0.72), no edges yet | yes, but the ids need a clustering read-out | no | one heading per cell: the 12 % / 2 % cells above | fast | keeps the junction problem; the id level is a rule step |
| **pyramid head + port graph (this card, revision 4)** | to be built | **yes, up to the id and the map id** | **yes: one query = one piece, a split or merge starts new pieces** | piece ends are outputs; pieces may overlap freely | estimate 5-10 ms for the decoder | **the base** |

Why the piece decoder: Mirko's id rule (one id per lane, splits and merges create new ids) is exactly the definition of a lane piece (nuScenes lane / lane connector, OpenLane-V2 lane segment), so the natural token is a piece, not a cell or a vertex: ids are outputs, not a read-out; two pieces can overlap or cross without conflict; successor and neighbour relations are relations between tokens (TopoNet, TopoMLP, LaneSegNet); and the assignment becomes piece graph against map piece graph, both with the same semantics. The dense pyramid stays as the first stage because dense evidence plus a tracer beat a sparse decoder in this project (E31 / E32: a 4.3 M DETR-style decoder from learned queries was 0.058 F1 below): the decoder here does not find lanes from nothing, it starts from ports the dense maps propose and checks smooth connections between them against the dense evidence (revision 4, section 2.2).

**Matching-free training (the project rule: no matching):** ports are key points (CenterNet style: heat maps, targets by location); a candidate connection is positive when its two ports are the two ends of one GT piece (by location, no assignment problem to solve). Revisions 2 and 3 used one anchor per piece in its middle; revision 4 drops the anchors (connector midpoints collide, 31 %, and the port graph does not need them). If this fails its gate, the fallback is bipartite matching in the LOSS only (the literature's standard; the forward path stays differentiable) - that needs Mirko's decision (PROBLEM.md says "no matching").

## 2. Design (one function, five blocks)
```
frozen GaussianCaR 39 ch, 200 x 200 @ 0.5 m
   | shared encoder (stride 1 / 2 / 4 / 8), FPN-like decoder with three taps
   +--> [M] 0.5 m: lane divider, road divider, road boundary, stop line (logit + truncated distance each) + line orientation (cos 2a, sin 2a)   10 ch, 200 x 200
   +--> [C] 1 m:   layer 1 of the graph, two kinds of pixels: CENTRE pixels (E271's 6 ch + 8 heading bins, overlaps allowed) and PORT pixels
                   (heat + sub-cell offset + heading per port type: split, merge, lane start, lane end, window edge)                            ~25 ch, 100 x 100
   +--> [R] 2 m:   road: drivable, carriageway centre, travel direction, lane count                                                             5 ch, 50 x 50
[C] ports --> [G] layer 2, the port graph: every (out-port, in-port) pair within 60 m and heading-compatible is a candidate connection; its curve
              = a cubic Bezier between the two port poses (+ inner control points for long pieces), checked IN PARALLEL by sampling the centre
              pixels along it + attention among all ports; outputs P(connection), the handle lengths, the inner control points.
              One accepted connection = one lane piece = one id; a split = a port with 2+ outgoing connections, a merge = 2+ incoming.
map lane graph (one id per lane piece) --tokens--> [E] map encoder: graph transformer on the map's own edges (along, successor, neighbour)
[G] x [E] --> [A] assignment: cross-attention pointer, rows = perception pieces (structure kept), columns = map pieces + dustbin
             geometric bias (E272's assigner, identity start), 2 rounds of soft registration (weighted Procrustes), structure term (graph matching)
          --> per perception piece: P(map piece id), P(route), lane index; per map piece: presence
```

### 2.1 Pyramid head [M] [C] [R]
- One encoder, three decoder taps (FPN / U-Net like, about 3-4 M parameters, GroupNorm as E271); the 1 m tap starts from the E271 head's weights (same stem, same 6 channels), the 0.5 m tap adds a full-resolution skip from a light stem on the 0.5 m input, the 2 m tap reads the stride-4 decoder level.
- Why these resolutions fit the targets: markings and boundaries are 0.1-0.2 m wide lines whose lateral position matters to 0.2 m (0.5 m cells + truncated distance give sub-cell accuracy); centre lanes are 3.5 m apart (1 m cells + offset, proven by E271); roads are 7-20 m wide areas (2 m cells are enough and cheap). PseudoMapTrainer (ICCV 2025) reports that a finer grid helps dividers but not area classes: the same trade-off.
- New at 1 m for the graph (layer 1 of section 2.2): 8 heading-binned centre heats (a cell can be on in two bins: crossing and diverging pieces are separable, the 1.9 % / 12.1 % cells of section 1) and the port heats with offset and heading (split, merge, lane start, lane end / dead end, window edge); key-point heads of this kind gave split / merge F1 0.55 / 0.61 in E44.
- New at 0.5 m: the stop line (autogt `stop_line`), the strongest visible cue where a lane enters a junction.
- Targets: from **autogt** on a new grid (`GridSpec(-50, 50, -50, 50, 0.5)`, the lane project's 100 x 100 m window) for the 10,120 frames of our split: `lane_divider`, `road_divider` (+ `_dist`), `drivable_area` (boundary = its 0.5 crossing), `centreline` (+ `_dir`) for the alignment check; the 1 m graph targets from the GT lane graph (`graph_gt_*.pkl`, pieces = nuScenes lanes and lane connectors). 2 m road targets: `drivable_area` pooled to 2 m, and the carriageways of the SD simulator (`sd_lib.build_sd`: centre line, direction, lane count) rasterised at 2 m.
- Losses: per line class focal BCE + Dice on the 1-cell line + L1 on the truncated distance; orientation cosine on line cells; E271's `grid_loss` unchanged at 1 m; focal BCE for the heading bins and the port heats, L1 for the port offsets, cosine for the port headings; BCE + Dice for drivable, BCE for the carriageway centre, cosine for direction, Poisson NLL for the lane count.

### 2.2 The port graph [G]: layer 1 = pixels, layer 2 = smooth connections (revision 4, Mirko's formulation of 2026-11-04)
Mirko's reading of revision 3: "three kinds of pixels first (lane centre, split / merge, end points); then a second layer that connects one end point with another over a smooth connection". Measured on the VAL GT, this is simpler than the anchors of revision 3 and fits the data:
- **Every piece of the GT graph is one smooth curve between its two end poses** (`temp/port_curve_oracle.py`, VAL, 1 frame in 4): a cubic Bezier with the two handle lengths fitted reproduces 99.5 % of the lane pieces and 99.5 % of the connectors within 0.5 m (max distance p90: lane pieces 0.11 m, connectors 0.19 m; lane pieces of 50-150 m 0.21 m). So the whole graph is: **ports** (position + heading) and **connections** (a pair of ports + two numbers).
- **Port types** (`temp/port_types.py`, end ports of the GT pieces): pass-through (one in, one out) 31 %, merge 21 %, window edge 16-18 %, split 9 %, dead end 3 %, lane start without predecessor 3 %. A pass-through port is a cut of the map (often invisible on the road, e.g. a straight connector through a junction): by Mirko's id rule (new ids only at splits and merges) it is dropped, the pieces on both sides become one. Chains merged that way (median 3 pieces, 59 m) fit one Bezier within 0.5 m in 86 % (p90 0.59 m): long connections get 1-3 inner control points.

**Layer 1 - pixels (1 m, part of the pyramid head)**
- CENTRE pixels: lane probability, sub-cell offset, heading in 8 bins (a cell can be on in two bins: overlapping and crossing pieces stay separable).
- PORT pixels: one heat map per port type (split, merge, lane start, lane end / dead end, window edge) with sub-cell offset and heading. A split port is where one lane ends and two or more begin; nuScenes places it by convention (often no paint), so it is scored with 3 m tolerance.
- At 0.5 m the stop line helps the ports at junction mouths; at 2 m the road layer gives the context.

**Layer 2 - connections ("tracing", but all at once)**
- Ports = the peaks of the port heats (an index choice; low threshold: a false port only costs compute because it gets no connection; in training also the GT ports). About 30-80 per frame.
- Candidates: every (out-port, in-port) pair within 60 m whose headings are compatible (an index choice); a few hundred per frame.
- Each candidate is a curve: cubic Bezier between the two port poses; handle lengths a, b from a small MLP of the pair; for pairs longer than 30 m 1-3 inner control points, refined towards the centre pixels.
- Score of a candidate, IN PARALLEL for all of them: (1) evidence along the curve: the centre pixels (heading bin matching the curve's tangent) sampled bilinearly at 32 points, summarised by mean, minimum and a soft gap length (MaGRoad's path statistics; D7's ray verification without the hard choice); (2) context: one attention layer over all ports and candidates (a port with an obvious straight continuation makes a weak alternative less likely; a split port expects two outgoing connections); (3) the map prior through the assignment (section 2.4). Output: P(connection), with a soft degree loss (each in-port of type lane start / merge / split gets the number of incoming / outgoing connections its type implies).
- This is why it is fast: no step-by-step tracing (E94: one network call per step, 628 ms); every candidate is checked at once (estimate: a few ms).
- **Ids**: one accepted connection = one lane piece = one id; split = a port with 2+ outgoing connections, merge = 2+ incoming. Ids are outputs, no read-out.
- **Neighbours**: the accepted pieces as tokens (feature = the pair feature + curve samples); a pair head between pieces gives left / right neighbour (parallel, same direction, 2.5-4.5 m apart).
- **Smoothness**: by construction (every piece is a Bezier / B-spline curve, G1-continuous at its ports because the curve leaves and enters a port along the port's heading); a curvature penalty on the inner control points.
- **Differentiable**: the evidence is sampled bilinearly from the centre pixels along the curve, the curve depends on the port positions, headings and the handle lengths, so a loss on a connection reaches the pixels and the ports. Only the port peaks and the candidate list are index choices; the output thresholds (P > 0.5) are the named output step.
- **Targets** (no matching): ports from the GT pieces after dropping the pass-through ports (type, position, heading); a candidate pair is positive if its two ports are the two ends of one GT piece (GT ports within 2 m of the predicted ports), its handle lengths and inner points from the Bezier fit of that piece; neighbours from parallel GT pieces (`sd_lib._par` rule).
- **Not covered (named)**: connectors chained to another connector without a port between them (3.3 % of the connectors) are one piece here; pieces with both ends outside the window (0.3 %); a missed port loses its connections (watched: port recall is reported).
- Taken from: SAM-Road (key points from a dense map + a transformer that scores links between them), MaGRoad (edge score from statistics along the candidate path), E44 / E53 (split / merge key points + Bezier connectors: split F1 at 3 m from 0.22 to about 0.45 with rules; here the same pieces are learned), CenterNet (key points as heat maps, no matching), D7 (ray verification along a candidate).

### 2.3 Map lane graph encoder [E] (one id per lane, splits and merges create new ids)
- **The map**: the nuScenes lane graph of the frame (lanes and lane connectors, `training/data/prep/make_graph_gt.py`: one token per piece, edges end -> start). A connector is its own piece, so every split and every merge starts new ids, as Mirko asks. This is a lane-level map (ADASIS v3 / "SD+" level, not a road-level SD map); the road-level carriageways of `sd_lib` stay as the fallback level.
- **Map errors** (otherwise the assignment is geometry alone, the E272 finding): scene-constant translation (sigma 1-2 m) and yaw (2 sigma deg) as the protocol's map-error modes, shape points thinned (Douglas-Peucker 0.5-2 m), 5 % of the pieces dropped, 5 % of the lane counts wrong. The association labels stay exact because the map pieces are made from the GT pieces.
- **Tokens**: each piece resampled every 2 m; features: position, tangent, arclength / piece length, kind (lane / connector), number of successors / predecessors, lane index from the right and lane count of its carriageway, route flag; the piece id is a label, not a feature (ids are arbitrary).
- **Encoder**: a 3-layer graph transformer whose attention runs only along the map's own edges (along the piece, successor / predecessor, left / right neighbour, opposite) with an edge-type bias (Graphormer style) and a random-walk structural encoding; a piece embedding per id by attention pooling over its tokens (about 20-60 pieces in a 100 m window).
- Taken from: SMERF and SEPT (SD map encoded by a transformer over polyline tokens; SEPT compares raster and graph encodings), MAT of OMA (path-aware attention inside a path, spatial attention across paths), TopoSD.

### 2.4 Assignment [A] (structure from perception, id from the map) and the drive attribute
- **Pointer cross-attention**: logits_ij = q_i . k_j / sqrt(d) + b_geo(i, j), rows = perception pieces, columns = map pieces; b_geo = mean over the piece's points of E272's geometric term -|p - q|^2 / sigma^2 - lam (1 - h . t) against the map piece's nearest points (so the start equals the geometric assigner and phase 3 starts at g's level); dustbin logit c_bin + MLP(query). Output per perception piece: P(map piece | piece) over the map pieces + "no lane". The rows are the perception pieces: their structure (geometry, successor and neighbour matrices) is kept, only the column space comes from the map.
- **Soft registration inside the assignment** (2 rounds): translation + yaw by weighted Procrustes over the soft pairs, the map tokens moved, the logits recomputed. Differentiable, a few lines.
- **Structure term** (soft graph matching): -sum_ij S_ij log(P_i^T C P_j) with C = identity + the map's successor adjacency, and the same for the neighbour matrices: two pieces the perception links must land on map pieces the map links. **Arm for later**: 3 Sinkhorn iterations with dustbins (SuperGlue).
- **Classification level**: P(map piece id) per perception piece, from it P(route) = sum over the route pieces, the lane index from the right; per map piece: presence. "How to drive" = the route attribute on every perceived piece.

### 2.5 Loss and training phases (blocks first, then joint, as E272)
| phase | trained | frozen | loss | steps (estimate) |
|---|---|---|---|---|
| 1 pyramid head | encoder + three taps (1 m from E271) | - | the three head losses | 3000 |
| 2 port graph | connection scorer, handle / inner-point MLP, neighbour head | head (its port heats are trained in phase 1) | connection BCE + soft degree loss, handle and inner-point L1, curvature penalty, neighbour BCE | 3000 |
| 3 map + assignment | map encoder, assignment | head, decoder | assignment NLL (map piece of the query's GT piece; dustbin for a query without one) + structure term + route BCE | 1500 |
| 4 joint | everything | - | all of it, lower learning rate | 2000 |
The hold of E272 stays out of the first version (its gate closes on clean frames); temporal fusion comes back on the piece queries (query propagation, StreamMapNet style) with a degraded-frame evaluation.

### 2.6 Not differentiable (named, on the output or index only)
The port peaks (an index choice; the values carry the gradient); the candidate port pairs within 60 m (an index choice); the output threshold (connection 0.5). Nothing else on the training path: no argmax, no Hungarian matching, no NMS, no assembly rules.

## 3. What others do (2025-2026 scan; local catalogue in `literature/`)
| work | what it does | what we take |
|---|---|---|
| MAT / OMA (2507.07487, ICLR 2026; nuScenes + OSM, code github.com/WallelWan/OMA-MAT) | associates HD centre lines to SD road links: path-aware attention inside a path, spatial attention across paths, Association P-R, 34 ms | path-aware attention for the map encoder, its P-R metric; their OSM roads for nuScenes are a later option (download needs an OK) |
| SMERF (2311.04079), SEPT (2505.12246), TopoSD (2411.14751) | SD map as polyline tokens through a transformer, cross-attention into perception; SEPT compares raster and graph encodings; TopoSD trains with map noise | token encoding of the map, noise in training |
| LaneSegNet, TopoMLP, TopoNet, TopoLogic, LaneGAP | lane segments as queries, successor matrix between them, paths as graph targets | the piece decoder and its relation heads |
| CenterNet, SAM-Road, MaGRoad | key points as heat maps without matching; a transformer scoring links between key points; an edge score from statistics along the candidate path | ports as key points, the connection scorer |
| MapTRv2, Sparse4D | point queries refined by deformable attention | the polyline refinement |
| SuperGlue, LightGlue; scene-graph matching with Sinkhorn and dummy columns (2604.27821) | soft assignment with a dustbin, one-to-one by Sinkhorn | dustbin now, Sinkhorn as a later arm |
| DOGE (2511.19850) | road graph as Bezier graph optimised by differentiable rendering against masks | a later refinement step (pieces -> raster -> loss against the dense maps) |
| MGMap (CVPR 2024), PseudoMapTrainer (ICCV 2025) | multi-scale BEV features with mask guidance; resolution helps dividers, not areas | the pyramid taps, the per-class resolution choice |
Gap the scan found: nobody assigns perceived lane-level ids to a lane-level map graph with a soft, structure-aware assignment and trains it through the perception; published association is road-level (MAT) or hard (Hungarian).

## 4. Local evaluation (new local scorer and pictures; `evaluation/evaluate.py` and the global protocol untouched)
Frames: VAL 873, the same frames and order as E271 / E272 and the lane suite; map cells: exact, error 1-2 m (one fixed perturbation set, seed fixed in the scorer), no map. Scorer `evaluation/pyramid_eval.py` (the graph part through the existing lane suite `graphs:` path, so d and p are scored by the same code), pictures `tools/viz/pyramid_view.py` (from `grid_view.py`), both local to this card.

**Numbers (what makes sense, with the reason)**
| level | number | why this one |
|---|---|---|
| 0.5 m lines | precision / recall / F1 of line cells within 0.5 m and 1.0 m tolerance, per class (lane divider, road divider, boundary); mean distance error of matched cells | pixel IoU of a 1-cell line punishes a half-cell shift as a full miss; tolerance F1 is what map papers report for lines |
| 1 m centres | cov5, prec1, vertices per 100 m (E271's, comparable); heading-bin F1 in crossing cells; port F1 within 3 m per type (split, merge, start, end, window edge) | continuity with E271 / E272; the crossing cells are the new part |
| 2 m roads | IoU of drivable; F1 of carriageway centre cells within 2 m; lane-count accuracy | IoU is the standard for areas (order of 0.8 for drivable on nuScenes in the BEV literature; to be checked against the papers before it is quoted) |
| graph | lane suite on the delivered graph: F1@1.0, lanes found, topo F1 (P / R), junction F1 (P / R), duplicate share, Q, R@1 far; successor-edge and neighbour-edge precision / recall between pieces | the same code that measured d today (junction F1 0.273, topo F1 0.740) |
| assignment | `lane_acc` (exact and 1-2 m), dustbin precision / recall for pieces without a GT piece, per-GT-lane purity (share of its length on one map id), id switches per 100 m, Association P-R of MAT | the card's headline; purity and switches are what the coloured picture shows |
| drive | route F1 per piece point, route success (`p8_lib.route_success`) | the planner-level attribute |
| speed | ms per frame per block (idle GPU, batch 1, cached features) | project rule |

**Sense checks before any number is believed (layers 0-2)**: the GT through the scorer gives 1.0 everywhere; known degradations move the numbers as expected (shift the GT lines by 0.25 / 0.5 / 1 m, drop 10 % of the pieces, cut pieces in two, swap the ids of two pieces, shift the map by 1 / 2 m: F1, junction F1, purity and `lane_acc` must fall by a predictable amount); the GT ports and GT pairs through the curve code reproduce the GT graph (oracle of the representation, layer 2); null rows (empty output, all-dustbin), the geometry-only arm g and the reference d; the alignment check of the new autogt layers against the lane GT (median distance of `centreline` cells to `g05` < 0.25 m, 98 % within 0.5 m), otherwise the frames are not used.

**Pictures per frame** (one figure, all BEV panels on the 0.5 m canvas; 1 m maps scaled x2 and 2 m maps x4 by nearest neighbour so the cells line up exactly):
1. the camera images with the GT projected (centres, dividers, boundaries on the ground plane z = 0; the six cameras, the front one large);
2. 1 m centre pixels: GT as a faint grey underlay, correct system cells in a neutral dark tone, false positives as a light red tint, misses as a light blue tint (alpha about 0.35);
3. 2 m road pixels, the same scheme;
4. 0.5 m lines, the same scheme with one hue per class;
5. graph: GT pieces thin grey, system pieces dark; false pieces / edges light red, missed ones light blue; split / merge markers, the false and missed junctions circled faintly;
6. ids: GT side, each map piece in a clean colour; system side, each perceived piece in the expected colour sum_j P(j | i) colour_j (an unsure assignment blends the colours), dustbin grey; id switches marked.
Videos: the same figure per keyframe (2 Hz) as mp4 for 4 VAL scenes: the two junction scenes of today (scene-0221, scene-0108), one straight (scene-0092), one night scene; d is drawn in the same layout for the side-by-side.

## 5. Result folder (one per experiment, `results/E273/`)
```
results/E273/
  README.md          index: question, verdict, the numbers table (from the cards, never typed), the key pictures, how to rebuild everything   (git)
  <run id>/package.json, provenance.json        from autoresearch/tools/package.py                                                             (git)
      names: code tag lcr/<run id> + commit, head code bundle hash (training/grid/*.py copied into bundle/, not in git),
             autogt version + signature (gt.json), training version (trainer commit, config.json, phase steps), dataset (nuScenes v1.0-trainval,
             data/splits.json hash, frame counts), evaluation code version (commit + hash of pyramid_eval.py and lane_suite.py), the numbers, the derived files
  figures/, videos/  pictures and mp4 (not in git: binaries), listed in package.json with the command that rebuilds them
```
The lane project has no `results/` folder yet; `package.py` runs per problem (it may need a `package` entry in project.json, a protected file: then a D entry first).

## 6. Detail plan (order, gates, cost; nothing runs before Mirko's word)
| step | what | where | gate / done when | cost |
|---|---|---|---|---|
| 0 GT | autogt on the 100 x 100 m grid for the split's 10,120 keyframes; 2 m road rasters from `sd_lib`; the 1 m graph targets (heading bins, ports by type after dropping the pass-through ports, Bezier fits with inner points, positive pairs, neighbour pairs) from the GT lane graph | `data/autogt/<signature>/` (not in git), a target builder in `training/grid/` | alignment check; the representation oracle (GT ports + GT pairs through the curve code) reproduces the GT graph: every piece within 0.5 m (measured already for single pieces: 99.5 %; merged chains need the inner points: 86 % with none), junction F1 > 0.95 (layer 2) | CPU, one process, about 2-3 h (machine must be free) |
| 1 scorer + pictures first | `pyramid_eval.py`, `pyramid_view.py` on GT, on g and on d | evaluation/, tools/viz/ | G0 and the degradation checks pass; d's numbers of today reproduced (layer 1) | CPU, < 1 h |
| 2 pyramid head | phase 1 | `training/grid/pyramid.py`, `train/pyramid_train.py` | G1 (gradient into every tap, overfit one batch); cov5 not below E271 - 0.03 | GPU about 2 h / seed |
| 3 port graph | phase 2 | `training/grid/port_graph.py` | G1; F1@1.0 not more than 0.05 below d, junction F1 above d's 0.273 (else 2b: teacher, then the matching fallback with Mirko) | GPU about 2 h / seed |
| 4 map + assignment | phase 3 | `training/grid/map_graph.py`, `assign.py` extended | at the start equal to g (identity start); the map-error cell moves | GPU about 1 h / seed |
| 5 joint + score | phase 4, both seeds, scorer, pictures, videos, result folder | `results/E273/` | verdict by the rule of section 0 | GPU about 1.5 h / seed, scoring 0.5 h |
Total about 13-14 GPU hours for two seeds plus 3-4 CPU hours; one critic pass (mode code) on the new shared code before phase 2.

## 7. Open for Mirko (before step 0)
1. The base: the piece decoder on the pyramid (recommended), with `recall_sp0_dp` as the reference and fallback. OK?
2. Matching-free (ports as key points, pairs by location); bipartite matching in the loss only as the fallback (needs a D entry, PROBLEM.md says "no matching"). OK?
3. The map graph = the nuScenes lane graph with simulated errors (lane level, one id per lane / connector piece), or a road-level map only (OSM through OMA: a download)?
4. Roads at 2 m = drivable area + carriageway centre, direction, lane count; lines at 0.5 m = lane divider, road divider, edge of the drivable area (nuScenes has no solid / dashed type). OK?
5. The GT build (CPU, 2-3 h) and the trainings need the machine: when?

## 8. Summary
Open.
