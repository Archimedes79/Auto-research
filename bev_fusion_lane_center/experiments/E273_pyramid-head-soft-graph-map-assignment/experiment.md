# E273 Pyramid head -> soft lane graph -> map lane graph -> assignment, as ONE differentiable function (plan)

Card of 2026-11-04, **plan only, nothing started** (Mirko: "Plan for now, do not yet start a training"). Mirko's brief of 2026-11-04: a pyramidal head (lane markings and road boundaries at 0.5 m, lane centres at 1 m, roads at 2 m); an E2E differentiable graph that understands connectivity and neighbourhood; the map graph encoded sparsely with one id per lane (splits and merges create new ids); an assignment of map ids to perception lanes (cross-attention that keeps the structure of the input and estimates the id from the map); E2E differentiable up to and including the classification; a local evaluation with overlay pictures and videos; one result folder per experiment. One card, as for E272: the phases are inside it, each with its own gate and a numbers row.

## 0. The card (what the tools run)
```json card
{
 "id": "E273",
 "card": "",
 "title": "pyramid head + soft lane graph + map lane graph encoder + structure-keeping assignment, one function, vs the E272 chain (geometry-only assignment) on the same frames",
 "kind": "block",
 "deterministic": false,
 "metric": "lane_acc_err",
 "direction": "higher",
 "band": 0.03,
 "prediction": {"value": 0.10, "vs": "g"},
 "arms": {
  "g": {"recipe": "control, no new training: the E272 chain (seed of the same number) read out at 11.2 vertices per 100 m, its START-value geometric assigner, applied to the map LANE tokens of section 2.3 (lane-level ids) with the same map errors"},
  "p": {"recipe": "the whole E273 function; phase 1 pyramid head (1 m branch from the E271 head), phase 2 soft graph (head frozen), phase 3 map encoder + assignment (head and graph frozen), phase 4 everything jointly at a lower learning rate; maps with simulated errors in training"}
 },
 "control": "g",
 "variant": "p",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "lane_acc_err", "vs": "g", "max_loss": 0.03},
  {"metric": "lane_acc_exact", "vs": "g", "max_loss": 0.01},
  {"metric": "cov5", "vs": "g", "max_loss": 0.02},
  {"metric": "prec1", "vs": "g", "max_loss": 0.02}
 ],
 "budget_gpu_h": 12,
 "kill": "a loss that is not finite; the phase-1 cov5 at 1 m more than 0.03 below the E271 head after phase 1; a phase longer than 1.5 x its estimate (section 6)",
 "evaluate": {"split": "val"}
}
```
Verdict rule (by hand, as E271 / E272: `card.validate` demands the project's graph headline metrics): `lane_acc_err` = weighted share of the read-out vertices within 1 m of a GT lane whose most probable map lane is the map lane of that GT lane, under the map-error cell "1-2 m" (fixed perturbation set, section 4), mean of the two seeds. **KEEP** > +0.03 over g with the guards held (exact map: `lane_acc_exact` not below g - 0.01; `cov5` / `prec1` at 1 m not below g - 0.02). **TIE** inside the band: the function stays as the base to iterate on only if its graph numbers (section 4, reported, not judged) are better than "no edges". **CUT** < -0.03 or a guard broken: back to the phase that broke it, not a new card. Prediction +0.10: geometry alone fails when the map is 1-2 m off (E272: it was exact, so geometry was enough); the soft registration and the structure term should recover most of it.

Reported, not judged: the 0.5 m and 2 m layers (no control exists yet), the graph metrics against the champion's (`champion` Q 0.735, topo F1 0.741 with rule steps), the route attribute, ms per frame, hold on / off.

## 1. Where we are (2026-11-04)
- **E271** (closed, KEEP by hand): 1 m centre-lane grid head on the 39 frozen GaussianCaR channels + differentiable sparsification: VAL cov5 0.721 / 0.728, prec1 0.871 / 0.873 at 10.3 vertices per 100 m (2 seeds).
- **E272** (closed, TIE by hand): head -> sparsify -> hold -> assignment as ONE trained function: cov5 +0.006 (mean of 2 seeds), the hold gate closes on clean frames, the assignment learned nothing beyond geometry (the SD map is GT-derived and exact).
- **Best graph so far:** with connectivity, the D7 builder of loop 16 (`training/models/p16_builder.py`, registry `champion`: Q 0.735, topo F1 0.741, lanes found 0.695) - but its assembly (path union, fork pruning, numba) and its SELECTION steps are rules, so it is not E2E up to the graph. The grid chain is E2E but delivers vertices without edges (no connectivity at all). E269 / E270 (soft graph tokens for D7) were proposed, never run. **So: no E2E graph with connectivity exists yet; this card builds the first one on the grid chain.**
- What carries over: the 1 m head and sparsification (E271), the chain loader and ego step (E272), the GT lane graph with directed path distances (`p16_data.gt_tensors`: `g1`, `dg`), the split / merge key points (`js`, `jm`), the SD simulator (`training/data/sd_lib.py`), the geometric assigner (`training/grid/assign.py`), the pictures (`tools/viz/grid_view.py`), `autoresearch/tools/package.py`.

## 2. Design (one function, five blocks)
```
frozen GaussianCaR 39 ch, 200 x 200 @ 0.5 m
   | shared encoder (stride 1 / 2 / 4 / 8), FPN-like decoder with three taps
   +--> [M] 0.5 m: lane divider, road divider, road boundary (logit + truncated distance each) + line orientation (cos 2a, sin 2a)      8 ch, 200 x 200
   +--> [C] 1 m:   centre lane: p, heading, offset, keep gate (E271's 6 ch) + 8-d lane-instance embedding                               14 ch, 100 x 100
   +--> [R] 2 m:   road: drivable, carriageway centre, travel direction, lane count                                                       5 ch, 50 x 50
[C] --sparsify--> soft vertices (weight w_i)  --tokens--> [G] soft perception graph: graph transformer on k-NN neighbourhoods
                                                         edge heads: successor s_ij, left / right neighbour, same lane; start / end per token
map lane graph (one id per lane piece) --tokens--> [E] map encoder: graph transformer on the map's own edges (along, successor, neighbour)
[G] x [E] --> [A] assignment: cross-attention pointer, rows = perception tokens (structure kept), columns = map lanes + dustbin
             geometric bias = E272's assigner (identity start), 2 rounds of soft registration (weighted Procrustes), structure term
          --> per vertex: P(map lane id), P(route), lateral offset, lane index; per map lane: presence
```

### 2.1 Pyramid head [M] [C] [R]
- One encoder, three decoder taps (FPN / U-Net like, about 3-4 M parameters, GroupNorm as E271); the 1 m tap starts from the E271 head's weights (same stem, same 6 channels), the 0.5 m tap adds a full-resolution skip from a light stem on the 0.5 m input, the 2 m tap reads the stride-4 decoder level.
- Why these resolutions fit the targets: markings and boundaries are 0.1-0.2 m wide lines whose lateral position matters to 0.2 m (0.5 m cells + truncated distance give sub-cell accuracy); centre lanes are 3.5 m apart (1 m cells + offset, proven by E271); roads are 7-20 m wide areas (2 m cells are enough and cheap). PseudoMapTrainer (ICCV 2025) reports that a finer grid helps dividers but not area classes: the same trade-off.
- Targets: from **autogt** on a new grid (`GridSpec(-50, 50, -50, 50, 0.5)`, the lane project's 100 x 100 m window) for the 10,120 frames of our split: `lane_divider`, `road_divider` (+ `_dist`), `drivable_area` (boundary = its 0.5 crossing), `centreline` (+ `_dir`) for the alignment check. 2 m road targets: `drivable_area` pooled to 2 m, and the carriageways of the SD simulator (`sd_lib.build_sd`: centre line, direction, lane count) rasterised at 2 m, so the road head predicts exactly the layer the road-level fallback of the assignment uses.
- Losses: per line class focal BCE + Dice on the 1-cell line + L1 on the truncated distance (the distance carries the sub-cell position); orientation cosine on line cells; E271's `grid_loss` unchanged at 1 m; BCE + Dice for drivable, BCE for the carriageway centre, cosine for direction, Poisson NLL for the lane count. Optional consistency term (off at the start): centre-lane mass outside the drivable area.

### 2.2 Soft perception graph [G] (connectivity and neighbourhood)
- **Tokens** = the cells of the 1 m sparse map: in training every cell with w > 1e-3 (the numerical cut of E272), at inference the read-out set; features = the 1 m decoder feature sampled at the vertex, Fourier position, heading, w, the instance embedding.
- **Neighbourhoods**: k = 16 nearest tokens within 6 m (an index choice, no gradient needed, like D7's SELECTION; the values on the chosen pairs carry the gradient). A 3-layer graph transformer (d = 64) with a relative-position bias in the token's own frame (along, across, heading difference): this is where "neighbourhood" is learned.
- **Edge heads** per pair (i, j): successor s_ij (sigmoid: a fork has two successors, a merge two predecessors), left neighbour, right neighbour, same lane; per token: start and end probability.
- **Targets from the GT graph** (no matching: each token takes the nearest GT 1 m point within 1 m): successor if the directed path distance `dg` from a_i to a_j is in (0.5, 3] m and within 0.7 m of the straight distance; neighbour if the GT pieces run side by side (lateral 2.5-4.5 m, same direction, `sd_lib._par` rule), side by the sign of the cross product; same lane if the same GT piece. Loss weighted by w_i w_j (detached), plus a degree term (out-degree 1, 2 at a GT split).
- **Lane ids of the perception graph**: the delivered ids are a read-out (chains of same-lane edges, cut at tokens whose successor mass splits or whose predecessor mass merges: a split or merge starts new ids, as in the map); named rule step on the output, used for the pictures and the per-lane metrics, not on the training path.
- Taken from: SAM-Road (dense map -> node tokens -> transformer link prediction; their node selection is NMS, ours is the soft weight), TopoNet / TopoLogic (successor relation between lane tokens), D7's split / merge key points (E44), LaneGAP (paths through the graph as the topology target).

### 2.3 Map lane graph encoder [E] (one id per lane, splits and merges create new ids)
- **The map**: the nuScenes lane graph of the frame (lanes and lane connectors, `training/data/prep/make_graph_gt.py`: one token per piece, edges end -> start). A connector is its own piece, so every split and every merge starts new ids, as Mirko asks. This is a lane-level map (ADASIS v3 / "SD+" level, not a road-level SD map); the road-level carriageways of `sd_lib` stay as the fallback level.
- **Map errors** (otherwise the assignment is geometry alone, the E272 finding): scene-constant translation (sigma 1-2 m) and yaw (2 sigma deg) as the protocol's map-error modes, shape points thinned (Douglas-Peucker 0.5-2 m), 5 % of the pieces dropped, 5 % of the lane counts wrong. The association labels stay exact because the map lanes are made from the GT pieces.
- **Tokens**: each piece resampled every 2 m; features: position, tangent, arclength / piece length, kind (lane / connector), number of successors / predecessors, lane index from the right and lane count of its carriageway, route flag; the piece id is a label, not a feature (ids are arbitrary).
- **Encoder**: a 3-layer graph transformer whose attention runs only along the map's own edges (along the piece, successor / predecessor, left / right neighbour, opposite) with an edge-type bias (Graphormer style) and a random-walk structural encoding; a lane embedding per id by attention pooling over its tokens (about 20-60 lanes in a 100 m window).
- Taken from: SMERF and SEPT (SD map encoded by a transformer over polyline tokens; SEPT compares raster and graph encodings), MAT of OMA (path-aware attention inside a path, spatial attention across paths), TopoSD.

### 2.4 Assignment [A] (structure from perception, id from the map) and the drive attribute
- **Pointer cross-attention**: logits_ij = q_i . k_j / sqrt(d) + b_geo(i, j) with b_geo = -|p_i - q_j|^2 / sigma^2 - lam (1 - h_i . t_j) (E272's assigner: the start equals the geometric assigner, so phase 3 starts at g's numbers); lane logit = logsumexp over the lane's tokens; dustbin logit c_bin + MLP(token). Output per perception token: P(map lane | token) over the K lanes + "no lane". The rows are the perception tokens, so the structure of the input is kept and only the column space comes from the map (the "special MLP" of Mirko's sketch is the token MLP that produces q_i and the dustbin score).
- **Soft registration inside the assignment** (2 rounds): translation + yaw by weighted Procrustes over the soft pairs (sum w_i P_ij (p_i, q_j)), the map tokens moved, the logits recomputed. Differentiable, a few lines; it replaces the separate registration for this chain.
- **Structure term** (soft graph matching): for successor pairs, -s_ij w_i w_j log(P_i^T C P_j) with C = identity + the map's successor adjacency (same lane or the map's next lane); the same for left / right neighbours with the map's neighbour matrices. Successive vertices cannot jump between unrelated lanes without paying for it.
- **Arm for later (not in the first version)**: 3 Sinkhorn iterations with a dustbin row and column (SuperGlue; the scene-graph matching of 2604.27821 uses dummy columns for a partial view, as here).
- **Classification level**: P(map lane id) per vertex, from it P(route) = sum over the route lanes, the lane index from the right, the lateral offset; per map lane: presence = 1 - exp(-mass / expected). "How to drive" = the route attribute on every perceived lane vertex (the route = the map lanes of the driven path, as the route GT of the project).

### 2.5 Loss and training phases (blocks first, then joint, as E272)
| phase | trained | frozen | loss | steps (estimate) |
|---|---|---|---|---|
| 1 pyramid head | encoder + three taps (1 m from E271) | - | the three head losses | 3000 |
| 2 soft graph | graph transformer, edge heads, embedding | head | edge BCE + degree + embedding pull / push | 1500 |
| 3 map + assignment | map encoder, assignment | head, graph | assignment NLL (map lane of the nearest GT piece, dustbin farther than 2 m) + structure term + route BCE | 1500 |
| 4 joint | everything | - | all of it, lower learning rate | 2000 |
The hold of E272 stays out of the first version (its gate closes on clean frames); it comes back with a degraded-frame evaluation.

### 2.6 Not differentiable (named, on the output or index only)
The k-NN index choice (values carry the gradient); the read-out of vertices (threshold + 3 x 3 maximum), of the delivered edges (s_ij > 0.5) and of the perception lane ids; the numerical cut w > 1e-3. Nothing else on the training path: no argmax, no Hungarian matching, no NMS.

## 3. What others do (2025-2026 scan; local catalogue in `literature/`)
| work | what it does | what we take |
|---|---|---|
| MAT / OMA (2507.07487, ICLR 2026; nuScenes + OSM, code github.com/WallelWan/OMA-MAT) | associates HD centre lines to SD road links: path-aware attention inside a path, spatial attention across paths, Association P-R, 34 ms | path-aware attention for the map encoder, its P-R metric for per-lane association; their data (OSM roads for nuScenes) is a later option (download needs an OK) |
| SMERF (2311.04079), SEPT (2505.12246), TopoSD (2411.14751) | SD map as polyline tokens through a transformer, cross-attention into perception; SEPT compares raster and graph encodings; TopoSD trains with map noise | token encoding of the map, noise in training |
| SuperGlue, LightGlue; scene-graph matching with Sinkhorn and dummy columns (2604.27821) | soft assignment with a dustbin, one-to-one by Sinkhorn | dustbin now, Sinkhorn as a later arm |
| SAM-Road, TopoNet, TopoLogic, LaneGAP | node tokens from a dense map, link prediction, topology between lane tokens, paths as graph targets | edge heads and targets of [G] |
| DOGE (2511.19850) | road graph as Bezier graph optimised by differentiable rendering against masks | a later refinement step for [G] (graph -> raster -> loss) |
| MGMap (CVPR 2024), PseudoMapTrainer (ICCV 2025) | multi-scale BEV features with mask guidance; resolution helps dividers, not areas | the pyramid taps, the per-class resolution choice |
Gap the scan found: nobody assigns perceived lane-level ids to a lane-level map graph with a soft, structure-aware assignment and trains it through the perception; published association is road-level (MAT) or hard (Hungarian).

## 4. Local evaluation (new local scorer and pictures; `evaluation/evaluate.py` and the global protocol untouched)
Frames: VAL 873, the same frames and order as E271 / E272; map cells: exact, error 1-2 m (one fixed perturbation set, seed fixed in the scorer), no map. Scorer `evaluation/pyramid_eval.py`, pictures `tools/viz/pyramid_view.py` (from `grid_view.py`), both local to this card.

**Numbers (what makes sense, with the reason)**
| level | number | why this one |
|---|---|---|
| 0.5 m lines | precision / recall / F1 of line cells within 0.5 m and 1.0 m tolerance, per class (lane divider, road divider, boundary); mean distance error of matched cells | pixel IoU of a 1-cell line punishes a half-cell shift as a full miss; tolerance F1 is what map papers report for lines |
| 1 m centres | cov5, prec1, vertices per 100 m (E271's, comparable); F1 within 1 m | continuity with E271 / E272 |
| 2 m roads | IoU of drivable; F1 of carriageway centre cells within 2 m; lane-count accuracy | IoU is the standard for areas (comparable to the BEV segmentation literature, order of 0.8 for drivable on nuScenes; to be checked against the papers before it is quoted) |
| graph | the project's `graph_eval` on the read-out graph: point F1 at 0.5 / 1 / 1.5 m, lanes found, split / merge F1 within 3 m, route F (APLS-like); successor-edge precision / recall on vertex pairs; neighbour-edge F1 | comparable with `champion` (topo F1 0.741) and the control |
| assignment | `lane_acc` (exact and 1-2 m), dustbin precision / recall for vertices farther than 2 m, per-lane purity (share of a GT lane's vertices on its majority map id) and id switches per 100 m, Association P-R of MAT | the card's headline; purity and switches are what the coloured picture shows |
| drive | route F1 per vertex, route success (`p8_lib.route_success`) | the planner-level attribute |
| speed | ms per frame per block (idle GPU, batch 1, cached features) | project rule |

**Sense checks before any number is believed (layers 0-2)**: G0 the GT through the scorer gives 1.0 everywhere; known degradations move the numbers as expected (shift the GT lines by 0.25 / 0.5 / 1 m, drop 10 % of the lanes, swap the ids of two lanes, shift the map by 1 / 2 m: F1, purity and `lane_acc` must fall by a predictable amount); null rows (empty output, all-dustbin) and the geometry-only arm g; the alignment check of the new autogt layers against the lane GT (median distance of `centreline` cells to `g05` < 0.25 m, 98 % within 0.5 m), otherwise the frames are not used.

**Pictures per frame** (one figure, all BEV panels on the 0.5 m canvas; 1 m maps scaled x2 and 2 m maps x4 by nearest neighbour so the cells line up exactly):
1. the camera images with the GT projected (centres, dividers, boundaries on the ground plane z = 0; the six cameras, the front one large);
2. 1 m centre pixels: GT as a faint grey underlay, correct system cells in a neutral dark tone, false positives as a light red tint, misses as a light blue tint (alpha about 0.35, so the errors are visible but do not dominate);
3. 2 m road pixels, the same scheme;
4. 0.5 m lines, the same scheme with one hue per class;
5. sparse graph: GT edges thin grey, system edges dark; false vertices / edges light red, missed ones light blue; split / merge markers;
6. ids: GT side, each map lane id in a clean colour; system side, each vertex in the expected colour sum_j P(j | i) colour_j (an unsure assignment blends the colours, as Mirko expects), dustbin grey; perception-id switches marked.
Videos: the same figure per keyframe (2 Hz, the features exist at keyframes only) as mp4 for 4 VAL scenes (straight, junction, merge, night), `--video`.

## 5. Result folder (one per experiment, `results/E273/`)
```
results/E273/
  README.md          index: question, verdict, the numbers table (from the cards, never typed), the key pictures, how to rebuild everything   (git)
  <run id>/package.json, provenance.json        from autoresearch/tools/package.py                                                             (git)
      names: code tag lcr/<run id> + commit, head code bundle hash (training/grid/*.py copied into bundle/, not in git),
             autogt version + signature (gt.json), training version (trainer commit, config.json, phase steps), dataset (nuScenes v1.0-trainval,
             data/splits.json hash, frame counts), evaluation code version (commit + hash of pyramid_eval.py), the numbers, the derived files
  figures/, videos/  pictures and mp4 (not in git: binaries), listed in package.json with the command that rebuilds them
```
The lane project has no `results/` folder yet; `package.py` runs per problem (it may need a `package` entry in project.json, a protected file: then a D entry first).

## 6. Detail plan (order, gates, cost; nothing runs before Mirko's word)
| step | what | where | gate / done when | cost |
|---|---|---|---|---|
| 0 GT | autogt on the 100 x 100 m grid for the split's 10,120 keyframes; 2 m road rasters from `sd_lib` | `data/autogt/<signature>/` (not in git) | alignment check of section 4 | CPU, one process, about 2-3 h (machine must be free) |
| 1 scorer + pictures first | `pyramid_eval.py`, `pyramid_view.py` on GT and on E272 (g) | evaluation/, tools/viz/ | G0 and the degradation checks pass; pictures of g look right | CPU, < 1 h |
| 2 pyramid head | phase 1 | `training/grid/pyramid.py`, `train/pyramid_train.py` | G1 (gradient into every tap, overfit one batch); cov5 not below E271 - 0.03 | GPU about 2 h / seed |
| 3 soft graph | phase 2 | `training/grid/graph.py` | edge targets checked on GT (GT tokens give successor F1 near 1.0) | GPU about 1 h / seed |
| 4 map + assignment | phase 3 | `training/grid/map_graph.py`, `assign.py` extended | at the start equal to g (identity start), map-error cell moves | GPU about 1 h / seed |
| 5 joint + score | phase 4, both seeds, scorer, pictures, videos, result folder | `results/E273/` | verdict by the rule of section 0 | GPU about 1.5 h / seed, scoring 0.5 h |
Total about 11-12 GPU hours for two seeds plus 3-4 CPU hours; one critic pass (mode code) on the new shared code before phase 2.

## 7. Open for Mirko (before step 0)
1. The map graph = the nuScenes lane graph with simulated errors (lane level, one id per lane / connector piece). OK, or a road-level map only (OSM through OMA: a download)?
2. Roads at 2 m = drivable area + carriageway centre, direction, lane count. OK?
3. Lines at 0.5 m = nuScenes lane divider, road divider, and the edge of the drivable area as the boundary (nuScenes has no solid / dashed type). OK?
4. The GT build (CPU, 2-3 h) and the trainings need the machine: when?

## 8. Summary
Open.
