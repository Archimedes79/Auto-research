# Project context for the literature research (read this first)

Purpose: the research must judge techniques against OUR pipeline, interfaces and environment. Everything below is measured or fixed in this project (source: `report_final/text/01_task.md`, `decision_report.md`, `Hypothesis_Tree_Spec_and_Tests_EN.md`, scripts). Date: 2026-10-23.

## Task
From six cameras and radar sweeps only (no lidar) produce, for every nuScenes keyframe, the **lane-centre-line graph** (polylines + successor edges, incl. splits and merges) in the ego BEV frame (+-50 m, 200 x 200 cells of 0.5 m), assign a **navigation route** to it (route over lanes of a map graph / SD map, no pose given), and hand it to a **learned planner consumer** (path points at 5 ... 40 m). The whole chain should be one differentiable network, trained / refined end to end (no argmax on the training path; selections with gradient through the selected items are allowed; Hungarian matching is NOT used in our losses).

## Data and labels
nuScenes first three blobs: 255 scenes (Boston, Singapore; day and dry only), scene-disjoint: TRAIN 212 scenes (8,414 samples), VAL 22 (873 frames), TEST 21 (833 frames). Ground truth = nuScenes **map-expansion** lane and lane-connector centre-lines with connectivity (piece graph in travel direction). Route label = lane sequence driven in the next <= 6 s. Failures (night, blur, radar off), map offsets and pose jumps are **synthetic**.

## Foundation and interfaces (what any block must plug into)
- **Frozen camera-radar BEV network: GaussianCaR** (ICRA 2026, map-segmentation-pretrained). Cached features per frame: decoder feature (32 PCA channels) + 7 map-layer logits (= 39 channels) at **200 x 200 cells, 0.5 m, +-50 m**, float16 (also a second 32-ch fused feature). No lidar. The foundation is frozen; only small heads are trained.
- **Dense head** (small U-Net, 2.7 M): lane heat map (Gaussian, sigma 0.75 m) + travel direction (2 ch). Trace-network input = 42 channels (39 + heat + 2 direction), pyramid f0 200 x 200 (64 ch), f1 100 x 100 (96), f2 50 x 50 (128).
- **System B (learned chain, loops 10 - 15):** input memory (ego-warped input of the previous frame) -> trace network E94 (1.67 M; seeds = peaks of the heat map; per step a fan of 27 bilinear samples (9 bearings +-45 deg x ranges 2 / 4 / 6 m) + 7 cross-section points, two transformer blocks, heading bin + step class {2, 4, 6 m} + residual, events continue / end / fork / merge, merge pointer; ring wavefront; teacher-forced imitation) -> E120 position refinement (point-to-polyline loss both ways, no matching) -> route head E104b (vertex tokens cross-attend to SD-map raster tokens, 0.69 M) -> planner consumer E117 (93 k parameters, edges + route probability + ego motion -> path). Vertex output = positions (ego metres), 128-d step token, validity / keep probability, edges.
- **Coordinate / sampling convention:** metres in the ego frame, bilinear `grid_sample` at arbitrary positions on the BEV pyramid.
- **Frozen interface facts that limit reuse:** features are *cached BEV rasters*, not image features and not a trainable BEV encoder; radar and camera are already fused. Methods that need perspective image features, a trainable BEV encoder, lidar, or end-to-end training of the backbone cannot be used as published; their *head / decoder / loss / training ideas* may be.

## Environment (for "can we run their code")
WSL2 conda env `gcar`: Python 3.12.15, PyTorch 2.8.0+cu129, numpy 1.26.4, scipy 1.17, shapely 2.0.7, networkx 3.6, nuscenes-devkit; **no mmcv / mmdet / mmdet3d / mmengine, no pytest**. One NVIDIA RTX 3060 12 GB. Two jobs at a time at most. Reference run times: E94 trace network 10 k iterations about 85 min; E120 about 70 min. Standing rule: **no downloads without asking** (reading web pages is fine; no git clone, pip, file downloads).

## Our metrics
Point F1 at 0.5 / 1 / 1.5 m (pooled P and R of dense 0.5 m points; headline F1@1), lanes found, false length fraction, direction accuracy, vertex error, split / merge F1 at 6 m (junction keypoints), route success (coverage and precision >= 0.8 within 1.5 m) at map error 0 / 1 / 2 / 3 m, planner path error at 5 - 40 m, temporal consistency. Selection on VAL, TEST for reporting (21 scenes = noisy); route-head seed noise up to 0.03.

## Measured state (TEST unless marked)
E94 trace network F1@1 0.851 (VAL 0.883), F1@0.5 0.661; E120 refinement 0.863 / 0.694 (vertex error 0.54 -> 0.43 m), keep gate 0.872; rule system A 0.867 loose / 0.875 final graph; split / merge F1 0.36 / 0.46 (E94) against 0.69 / 0.65 (A with key points + connectors); 22 vertices per 100 m against 14.3 for a Douglas-Peucker oracle (0.3 m); two thirds of E94 edges sit in the 6 m step class; seeds (192 peaks) limited E94; route success 0.853 / 0.843 / 0.832 / 0.820 (B, seed 0) against 0.876 / 0.879 / 0.876 / 0.858 (A); planner path error 0.34 m at 20 m (A: 0.36 m); planner loss through the positions gives no gain (E121), planner loss into the route head no gain (E119).

## What the project already tried (verdicts; do not just re-suggest these)
| tried | result | verdict |
|---|---|---|
| detection-pretrained CRN vs map-pretrained GaussianCaR (E27) | F1@1 0.624 vs 0.867 | map-pretrained foundation kept |
| decoders FCN / U-Net / ConvNeXt + Transformer (E17, E18) | U-Net +0.09 - 0.12 over FCN, transformer decoder -0.11 | U-Net kept |
| dense heat map + tracer vs sparse DETR-style decoder with matching (E31, E32) | grid +0.058 F1, +0.18 route F | grid kept, matched sparse decoder dropped |
| cascade refiner, temporal feature fusion (E39, E41) | +0.001 F1 / +0.014 route F; +0.008 F1 | small, kept |
| learned split / merge key points (E44), Bezier connectors (E53) | split / merge F1 0.34 / 0.38 -> 0.55 / 0.61 -> 0.71 / 0.64 | kept (final graph of A) |
| learned link heads, TopoLogic-style scoring, node-conditioned head, path repair (E42, E45, E46, E51, E56) | within 0.01 of the rules | dropped (the rules already decide) |
| learned trace network, near-to-far order, ranges, graph crop (E94 - E99) | F1 0.851 vs 0.867; seeds limited it | partial (system B) |
| junction diagnosis, connector head, junction / piece verifier on the learned graph (E113, E116) | 1.8 x as many junctions as exist; verifier split / merge 0.385 / 0.428 -> 0.437 / 0.504 | partial |
| map prior at the grid, learned registration (E40, E48) | +0.075 F1 exact pose, -0.057 unaligned | kept (registration) |
| navigation attribute in the learned chain: step-token head, graph transformer, correlation-volume localisation, cross-attention (E102 - E104) | cross-attention 0.853 - 0.820 | cross-attention kept |
| position refinement decoder (E120), planner loss through positions (E121) | F1@1 +0.012; planner no gain | refinement kept optional |

## Our design under test (for orientation; see the spec and concept notes)
Verified-ray hypothesis tree: from a tip (position + heading) a teardrop of 12 rays (short in all directions, long and narrow forward up to 24 m; samples on rings 4 / 8 / 12 / 18 / 24 m) samples BEV features; every ray is verified sample by sample (survival = cumulative product, endpoint = most probable last verified sample, validity = survival x direction prior); the b >= 2 best endpoints (after redundancy suppression) become children in a fixed-shape tree with a beam; cascade per level (generate -> refine -> rescore with attention over siblings / ancestors); tree -> DAG by clustering + edge split; time recursion (ego-warped previous set, log-odds fusion). Alternative family: spray points, snap to lanes, link with a learned link head. Files: `reports/Hypothesis_Tree_Spec_and_Tests_EN.md`, `reports/Growth_and_Link_Architectures_EN.md`, `reports/Concept_Hypothesis_Tree_Lane_Graph_EN.md`, earlier review `report_final/text/lit_p16_tree.md` (abstract level, ~75 papers).
