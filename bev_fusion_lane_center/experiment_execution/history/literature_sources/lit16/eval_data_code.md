# Cluster EV - datasets, exact metric definitions, evaluation protocols, code and licence inventory (loop 16)

Date: 2026-10-23. Agent: EV (one of 8 parallel literature agents). Reading only (WebSearch / WebFetch). Nothing downloaded, no git / pip / curl.
This cluster does not use the card format; it uses tables and short notes (Parts A - D), then the compatibility notes required by SCHEMA.md.

Tags used in this file
- [P] = seen on a web page in this session. Caveat: WebFetch hands back a summary written by a small model, not the raw page. Numbers, pins and licence names below are as that summary reported them. "not on page" means "not in the summary". For code files (raw.githubusercontent.com) the code text was summarised the same way; treat code-level statements as "read via summary", not as audited.
- [S] = only a search-result snippet was seen.
- (from memory) = not seen on any page in this session. "not found" = searched, nothing found.
- Last-push dates and SPDX licence names for repositories come from the GitHub REST API search endpoint (JSON, read through WebFetch; field `pushed_at`, `license`). GitHub's licence detector says "NOASSERTION" when the licence file is custom.

---

## 1. Coverage statement

- Searched / opened: nuScenes devkit and schema pages; Motional news page; OpenLane-V2 repo (README, docs/metrics.md, docs/submission.md, data/README.md, setup.py, requirements.txt, devkit evaluate.py of master AND of tag v2.0.0, PR #77 diff, issue #76); OpenLane-V2 paper (arXiv HTML 2304.10440); Argoverse 2 paper (arXiv HTML 2301.00493) and argoverse.org; Sat2Graph, RNGDet++, SAM-Road, SAM-Road++ READMEs; SpaceNet 3 page; Global-Scale paper (arXiv 2411.16733); NAVSIM (README, docs/metrics.md, install.md, scorer code, paper arXiv 2406.15349); nuPlan devkit README, metrics description page, nuPlan paper (arXiv 2106.11810); OpenScene README; MapTR (paper, install.md, eval code), StreamMapNet (paper, README); RNTR (paper arXiv 2402.08207); SeqGrowGraph (arXiv HTML); LaneGAP (arXiv HTML); CGNet (arXiv HTML); LaneGNN (ar5iv 2302.06175); STSU (ar5iv 2110.01997); APLS paper (ar5iv 1807.01232); TopoMLP paper (arXiv HTML); stability paper (arXiv HTML 2510.10660); about 50 GitHub repository pages and 8 GitHub-API queries (licence and last-push fields).
- Could NOT open: the official nuScenes terms-of-use page, the nuScenes download page and the nuPlan page (they are script-rendered; WebFetch returned an empty page); web.archive.org (blocked by the tool). The nuScenes licence statement below therefore rests on the nuScenes paper, a later paper by the owners, a Motional page and search snippets, not on the terms page itself (see A.2). One PDF fetch (APLS) was unreadable and the tool saved a binary copy under its own tool-results folder; later fetches avoided PDFs.
- Depth: metric definitions at "M" level (code or definition sections read) for OpenLane-V2 (old and new devkit), MapTR chamfer AP, RNTR landmark / reachability, APLS, TOPO / JTOPO (LaneGAP), SDA / GEO (LaneGNN, CGNet), mAS, PDMS / EPDMS, STSU metrics. nuPlan closed-loop weights: not seen (from memory only). CGNet's full metric definitions are in a supplement that was not opened; its released evaluation code (`topo_eval/`) was read instead, as was the released landmark / reachability code of SeqGrowGraph.
- Repositories: README level ("C") for all methods that have a repository; install documents (docs/install.md, setup.md) of several mmdet3d-based repositories were NOT opened (pins marked "not on page").
- Correction made late in the session: SeqGrowGraph DOES have code (MIV-XJTU/SeqGrowGraph, found by a GitHub name search after web search and the paper HTML showed no link); it also contains RNTR and LaneGraph2Seq configs and the released landmark / reachability evaluation code, which was read (B.4).
- Repositories / code not found: DAGMapper (no code found), Topo2Seq (no code found; a GitHub name search also found none), SMART (project page says code "coming soon"), TopoBDA (only a LiDAR fork of the OpenLane-V2 devkit and a data zip; no model code), LaneGAP (the repository holds README, assets and LICENSE only; its news item says the code was released together with MapTRv2), standalone RNGDet (github.com/TonyXuQAQ/RNGDet returned 404; the RNGDet++ repository contains both scripts).

---

## 2. Findings that matter for the design (read this first)

1. OpenLane-V2 numbers are NOT comparable across devkit versions. Devkit v1.0.0 / v2.0.0 ("x.0") and v1.1.0 / v2.1.0 ("x.1") compute TOP_ll / TOP_lt differently (details in B.1). The same model gives TOP_ll 4.1 vs 10.9 (TopoNet, R50, subset A, README tables) and 7.1 vs 21.7 (TopoMLP, R50, subset A); OLS 35.6 vs 39.8 and 38.3 vs 44.1. The shift differs per model (x2.7 and x3.1 on TOP_ll), so no constant conversion exists. The DET_l / DET_t values of these model pairs are identical across versions (the PR diff changed the topology functions and one dtype declaration in the matching helper; the detection logic itself was not rewritten). Some later papers do not state their version (UniTopo page: not stated). Only compare to numbers that state v2.1 / v1.1, or re-run the devkit ourselves.
2. All published lane-graph metrics need one of three things that our raw output (vertices, edges, validity) does not provide directly: (a) instances with a confidence and a fixed point count (OLS, MapTR-type AP), (b) a 1:1-matched dense point graph with sub-graph walks (GEO / TOPO / JTOPO), (c) landmark vertices by degree (RNTR landmark / reachability). All three are obtainable with small adapters (B.8, effort S - M). The adapters contain choices (how to cut pieces, how to turn validity into a confidence, directed or undirected) that change the numbers, so they must be fixed once and written down. Even a metric with one name has several protocols: TOPO / JTOPO use 0.15 m / 0.45 m / 7.5 m in the LaneGAP text but 0.2 m / 0.8 m / 40 m (inferred units) in the released CGNet code, and SDA is an F1 at 1 m there; the CGNet evaluation folder is Apache-2.0 and pure python, so it is the cheapest reference to vendor (B.3). Evaluation-time matching in all of them is greedy / nearest, never Hungarian in the loss, so there is no conflict with our matching-free training.
3. nuScenes-based numbers are inflated by location overlap: StreamMapNet reports that 85 % (appendix) of nuScenes validation locations also occur in training; MapTR (ResNet-50, 24 epochs, 30 m range) drops from 48.7 on the original split to 20.9 on the geographically disjoint split. Our TRAIN / VAL / TEST are scene-disjoint, not location-disjoint, so the same inflation can be present. A location-overlap measurement of our split is a cheap check (effort S) and should precede any claim against published numbers.
4. Our fork / merge metric (F1 at 6 m) is looser than the published landmark protocol: RNTR / SeqGrowGraph use thresholds 0.5 - 5.0 m (landmarks) and 0.5 - 2.5 m (reachability) (paper text; the released code uses 5 thresholds each and non-one-to-one nearest matching, see B.4); CGNet / LaneGAP use a 7.5 m sub-graph around junctions. Report a threshold curve in addition to the 6 m point.
5. Code reality for Python 3.12 / PyTorch 2.8 / no mmcv: practically every lane-topology and vector-map repository pins Python 3.8, PyTorch 1.9 and mmcv-full 1.x / mmdet 2.x / mmdet3d 1.0.0rc6 (TopoNet, LaneSegNet, TopoStreamer, StreamMapNet, SQD-MapNet, MapTR, CRN, BEVFusion ...). None runs as is. The repositories that run (or nearly run) are GaussianCaR (Python 3.12.3, PyTorch 2.8.0, Apache-2.0), the nuScenes devkit (tested on 3.12), the multiple-choice-learning losses, and DETR-family cores (with the pure-PyTorch fallback of deformable attention). The OpenLane-V2 devkit pins numpy < 1.24, scipy == 1.8.0, ortools == 9.2.9972, shapely == 2.0.0, so it cannot be installed unchanged next to scipy 1.17 / numpy 1.26; its evaluate.py is plain numpy / scipy code and can be vendored.
6. Licence picture: all candidate evaluation datasets are non-commercial (CC BY-NC-SA 4.0 for OpenLane-V2, Argoverse 2, OpenScene; nuScenes: free for academic use, commercial licence on request). GPL-3.0 repositories (RNGDet++, CenterLineDet, Topo-boundary / iCurb, StreamMapNet, SQD-MapNet, InstaGraM) and the repositories without any licence file (TopoLogic, TopoPoint, TopoFG, UniTopo, TopoStreamer, SeqGrowGraph, DeH4R, PlanTF, PLUTO, Diffusion Planner, aMCL) are "read and re-implement" sources, not copy sources (Part D).
7. No radar anywhere except nuScenes: the OpenLane-V2 package lists camera images and an SD map only; Argoverse 2 and nuPlan / NAVSIM list cameras and lidar. Our camera-radar chain can therefore be trained and run only on nuScenes. OpenLane-V2 subset B (nuScenes frames) is usable only as an independent ground truth for the same frames, if the frames can be matched (A.3).

---

## 3. PART A - Datasets and licences

### A.1 Overview table

| dataset | size / scope [P] | sensors | ground truth relevant here | how a centre-line graph with connectivity is obtained | licence as stated | fit with our nuScenes map-expansion labels (TRAIN 212 / VAL 22 / TEST 21) |
|---|---|---|---|---|---|---|
| nuScenes (+ map expansion) | 1,000 scenes of 20 s (paper abstract); keyframes at 2 Hz (schema page); about 40 k annotated keyframes, 1.4 M 3D boxes, 1.4 M camera images (cameras 12 Hz), 1.3 M radar point clouds (paper, Table 1); official split 700 / 150 / 150 (RNTR page; the nuScenes paper page gives no split counts); keyframe share by location: Boston 55 %, SG-OneNorth 21.5 %, SG-Queenstown 13.5 %, SG-HollandVillage 10 % (paper, section 2). Map expansion: 4 maps (singapore-onenorth, singapore-hollandvillage, singapore-queenstown, boston-seaport); v1.0 2019-07-01 (11 semantic layers), v1.1 ego poses fixed, v1.2 arcline paths + lane connectivity, v1.3 2020-11-23 (BitMap, lidar basemap, one broken lane removed) | 6 cameras, 5 radars (77 GHz FMCW, 13 Hz, range up to 250 m, paper Table 2), 1 lidar | 3D boxes (23 classes, 8 attributes); map-expansion layers: drivable_area, road_segment, road_block, lane, ped_crossing, walkway, stop_line, carpark_area (polygons), road_divider, lane_divider, traffic_light (lines), lane_connector (polygon, lookup layer), `arcline_path_3` (dict by lane token), `connectivity` (dict token -> incoming / outgoing lists) | Nodes = `lane` and `lane_connector` tokens, edges = `connectivity` outgoing; geometry = `discretize_lanes(tokens, resolution_meters)` from the arcline path (default render resolution 0.5 m). This is our label route (piece graph in travel direction) | see A.2 (official page unreadable; second-hand statements) | Source of our labels (identical frames). Our 255 scenes = first three data blobs (3 x 85 scenes is an inference from "850 trainval scenes in 10 blobs", from memory); day and dry only |
| OpenLane-V2 subset B (nuScenes) | 1,000 segments, 700 / 150 / 150; 6 cameras 1600 x 900; mean 20 s; 2 Hz; per frame about 24 centerlines (train 24.32 / val 24.80 / test 23.82), 3.6 traffic elements, 1.8 connections per centerline; range x +-50 m, y +-25 m | camera images + `sdmap.json` (road, cross_walk, side_walk polylines in BEV space); no radar / lidar listed in data/README | lane centerlines [n,3] (z = 0 in B), `topology_lclc` [n,n] adjacency (directed), traffic elements (2D boxes, 13 attributes), v2.x: lane segments (centerline + left / right lane line + line types + is_intersection_or_connector), areas (ped crossing, road boundary), `topology_lsls`, `topology_lste` | Centerlines regressed from HD-map lane boundaries; lanes with one predecessor or successor are merged; split into instances only at intersections, forks, merges (paper). Stored with 201 points, 11 subsampled for evaluation. Topology = adjacency matrix between instances | data CC BY-NC-SA 4.0, code Apache-2.0; users must agree to the nuScenes and Argoverse 2 terms (README / paper) | Same raw frames as nuScenes. `meta_data.source` and `source_id` give the original dataset and original segment id; frame matching by timestamp is plausible but NOT documented on the pages read (not found). Differences to our labels: centreline derived from boundaries (ours: stored arcline path), merged chains (ours: lane / connector pieces), narrower range (y +-25 m vs ours +-50 m), no radar in the package |
| OpenLane-V2 subset A (Argoverse 2) | 1,000 segments, 700 / 150 / 150; 7 cameras 2048 x 1550; mean 15 s; 6 cities (Austin, Detroit, Miami, Pittsburgh, Palo Alto, Washington D.C.); about 26.3 centerlines per frame | cameras + SD map; no radar | as subset B (real z) | as B | as B | Not usable with our camera-radar chain (no radar). Only for studying label conversion or for numbers quoted by others |
| Argoverse 2 (Sensor / Lidar / Motion Forecasting + map) | Sensor: 1,000 scenarios x 15 s, 700 / 150 / 150; Lidar: 20,000 x 30 s, about 6 M frames; Motion forecasting: 250,000 x 11 s (58 GB) | 7 ring cameras 20 fps + 2 stereo, two 32-beam lidars 10 Hz, 6-DOF ego pose; no radar mentioned | HD map: lane segments with 3D left / right boundaries and attributes (is_intersection, lane type, mark types, left / right neighbours, successor ids), drivable area polygons (1 cm quantised), pedestrian crossings, ground-height raster 30 cm (sensor set only); local map = 100 m dilation of the ego trajectory; 3D boxes 30 categories (26 evaluated) | Lane-segment graph from successor ids (predecessors by inverting; the attribute table lists no predecessors); centreline not stored, the API infers it at a chosen resolution | data and docs CC BY-NC-SA 4.0; code / API MIT; non-commercial = not primarily intended for commercial advantage or monetary compensation; faces and plates blurred, re-identification prohibited (argoverse.org) | No radar -> cannot feed GaussianCaR. Second domain for label-conversion code only |
| CityScale (Sat2Graph) | 20 US cities, 180 tiles of 2 km x 2 km (720 km2), 1 m / pixel; 144 train / 9 val / 27 test tiles | aerial imagery (Google Static Maps API) | road graph from OpenStreetMap | graph = OSM road network (undirected; from memory) | Sat2Graph README: MIT licence, but "pretrained models, dataset and docker container are for non-commercial academic use only" | Different sensor (aerial), road graph not lane graph, no travel direction. Idea-level only |
| SpaceNet 3 roads | 4 AOIs (Las Vegas, Paris, Shanghai, Khartoum), about 3,011 km2 of raster (sum of the page's four values), over 8,000 km of road centerlines, 400 m tiles; resampled to 1 m / pixel in the SAM-Road++ comparison; 2,549 non-empty tiles split 80 / 15 / 5 in Sat2Graph | WorldView-3 30 cm pan-sharpened | LineString road centerlines (OSM-based schema, hand corrections), attributes road type, surface, bridge, lane number | undirected road graph from centreline strings | CC BY-SA 4.0 (spacenet.ai page and APLS paper); AWS public bucket | as CityScale |
| Global-Scale (SAM-Road++) | 3,468 tiles of 2048 x 2048 px at 1 m / pixel; 2,375 train / 339 val / 624 test (in-domain) + 130 out-of-domain (Hong Kong, Shenzhen, Lucerne); "over 13,800 km2"; all continents except Antarctica | Google Static Map imagery | OpenStreetMap road graphs, second screening for completeness | OSM -> graph; conversion steps not on page | no licence on the arXiv page; code repository has no LICENSE file; data on a Baidu Pan share | as CityScale |
| nuPlan / OpenScene | nuPlan: over 1,300 h, 15,000+ logs, 4 cities (Las Vegas, Pittsburgh, Boston, Singapore) (devkit changelog); the paper plans 1,500 h, sensor data only for a subset (200+ TB). OpenScene: redistribution, 120 h of sensor data at 2 Hz, more than 10x smaller | lidar, cameras, localisation (paper); no radar listed | per-city semantic maps with a query API (map layers not on the pages read), auto-labelled agent tracks | via the nuPlan map API (lane / connector baseline paths; from memory, not seen) | OpenScene: code Apache-2.0, data CC BY-NC-SA 4.0 plus the nuPlan Dataset License Agreement for Non-Commercial Use; nuPlan devkit README shows an Apache-2.0 badge (GitHub detector: NOASSERTION) | No radar; different maps; not usable as input for our chain |
| NAVSIM (navtrain / navtest) | navtrain 103 k samples, navtest 12 k, navmini 396 scenarios; about 450 GB standalone download (paper) | 8 cameras 1920 x 1080 + merged lidar of 5 sensors (paper); no radar | non-reactive simulation on OpenScene logs; PDMS / EPDMS scoring | trajectory benchmark, no centre-line graph labels | code Apache-2.0; "datasets inherit their own distribution licences"; LICENSE file hosted on a Motional S3 bucket (not read) | Not a data source for us; its metric is discussed in B.6 |

### A.2 nuScenes licence - what could and could not be read

- Official terms-of-use page: not readable (script-rendered, empty result). The statements below are the best available; they come from the nuScenes paper, a later paper by the dataset's owners, a Motional page, the OpenLane-V2 sources and search-result snippets of nuscenes.org.
- nuScenes paper (ar5iv 1903.11027, section 1.1) [P]: the data is published under CC BY-NC-SA 4.0, which permits non-commercial research use. The licence text itself is not on the page.
- "nuScenes Revisited" (arXiv 2512.02448, section I) [P]: the dataset allows free use by academics and startups and offers commercial licences for larger industry partners; nuImages uses the same licence; no licence name on the page. This is broader than "strictly non-commercial" and is not reconciled with the other statements on the pages read.
- Motional news page [P]: nuScenes is free of charge for academic use; licensing exists for commercial purposes. The page does not define the terms, name a contact, or say whether redistribution is allowed.
- Search-result snippets of nuscenes.org pages [S]: free to use strictly for non-commercial purposes; non-commercial defined as not primarily intended for or directed towards commercial advantage or monetary compensation; commercial users are asked to contact the owners (address nuScenes@motional.com). Not confirmed on an opened page.
- OpenLane-V2 paper and README [P]: nuScenes (and Argoverse 2) are distributed under CC BY-NC-SA 4.0.
- AWS Open Data Registry entry [P]: the licence field says "Commercial" and links to the terms page; the entry carries no terms. This label contradicts the other statements and is the only source of it.
- nuScenes devkit README [P]: users must create an account and agree to the nuScenes Terms of Use before download; the page names no licence for code or data; GitHub's detector reports a custom licence (NOASSERTION). (From memory: Apache-2.0 for the devkit code.)
- Map expansion: no separate terms were found; it is delivered by the same download page (from memory).
- The Hugging Face derived-dataset cards seen in search results disagree (CC BY-NC-SA vs CC BY-NC) [S]; they are third-party and were not used.

### A.3 Compatibility notes with our labels (analysis)

- Granularity: our label graph is a piece graph (lane and lane_connector pieces, successor edges). OpenLane-V2 merges 1:1 chains and splits "only at intersections, forks and merges" (paper). The exact rule (does a lane -> connector boundary at an intersection always cut?) is not on the pages read. The safest adapter for our own labels is: cut at every fork / merge and at every lane <-> connector boundary, merge only 1:1 lane -> lane chains. Flag as unverified until compared with OpenLane-V2 subset B ground truth on shared frames.
- Range: OpenLane-V2 annotation window x +-50 m, y +-25 m (paper) vs ours +-50 m x +-50 m (200 x 200 cells of 0.5 m). MapTR-type evaluation uses 60 x 30 m or 100 x 50 m windows; RNTR uses -48 .. 48 m by -32 .. 32 m (RNTR page) or -48 .. 48 by -30 .. 30 (SeqGrowGraph page). Every comparison needs cropping of prediction and ground truth to the foreign window.
- Frames: OpenLane-V2 keyframes are at 2 Hz like nuScenes; segment ids and timestamps are in the info files; `source` / `source_id` exist. Whether our 255 scenes overlap OpenLane-V2 subset B val (150 scenes) is unknown; nuScenes official val scenes are scattered over the data blobs (from memory), so only part of our scenes would lie in B val. Evaluating "OLS-lite" on our scenes with OpenLane-V2 ground truth is possible only on frames where the labels exist, and the OpenLane-V2 data download needs the user's permission (standing rule: no downloads without asking).
- Sensors: no public dataset in this list except nuScenes offers radar. OpenLane-V2 offers an SD map (`sdmap.json`: road, cross_walk, side_walk elements with 2D points in BEV space) which could serve as a second source for the route head's SD-map raster; our current SD map is simulated (project history).
- Location leakage: see finding 3. For a location-overlap check of our split one needs the nuScenes log -> map-location field (scene -> log -> location, from memory) and the ego positions; effort S.
- Aerial graph datasets: top-down, 1 m / pixel (SpaceNet native 30 cm), undirected road graphs, no radar / camera. They could only benchmark a growth or link head in isolation (input = raster + heat map). Not recommended for us: the semantic gap (no direction, no fork / merge semantics at lane level) is large.
- nuPlan / NAVSIM: scoring needs agents, a simulator and nuPlan maps; our planner outputs a spatial path (5 - 40 m), not a time-indexed trajectory. Not computable on nuScenes frames (B.6).

---

## 4. PART B - Exact metric definitions

### B.1 OpenLane-V2: DET_l, DET_t, TOP_ll, TOP_lt, OLS; old vs new devkit

Sources: docs/metrics.md [P], evaluate.py of master (x.1) [P], evaluate.py at tag v2.0.0 (x.0) [P], PR #77 diff [P], issue #76 [P], paper [P], TopoMLP paper [P]. Lane-segment variant (v2.x): metrics.md [P].

Version history [P, README]: dataset v0.1 2023-01-15; v1.0 + devkit v1.0.0 2023-02-15; v2.0 + devkit v2.0.0 2023-07-21 (v2.x updated the APIs and materials for lane segments and the SD map); subset_B released 2023-08-28; devkit v2.1.0 and v1.1.0 2023-11-01 together with the evaluation-metric update ("x.1" metrics recommended, issue #76; code change = PR #77). v1.1.0 is the long-term-support release for the OpenLane Topology task; v2.1.0 also covers it. Whether a dataset "v2.1" exists: not on page. The Map Element Bucket was updated in October 2023 (LaneSegNet README).

Definitions common to both versions (code of master, same constants in v2.0.0 for the detection part):

| item | definition |
|---|---|
| centerline matching | discrete Frechet distance between predicted and GT centerline (3D points; subset B has z = 0). THRESHOLDS_FRECHET = [1.0, 2.0, 3.0] (unit metres by context, not stated on the page). Pairs whose Chamfer distance is not below 3.0 are masked first; the Frechet matrix is computed with `relax=True` (the paper's example: threshold relaxed to 1.2 t for far lanes, 1.1 t for nearer lanes; the code meaning of `relax` is not defined in the file) |
| points | GT stored with 201 points per centerline, 11 subsampled for evaluation (paper datasheet); the submission page recommends about 10 points per lane line and 20 per area |
| DET_l | predictions sorted by confidence, greedily matched to GT, each GT used once; AP with 11-point interpolation (recalls 0, 0.1 ... 1.0, divided by 11) in the code (the paper says "area under the PR curve"); mean over the three thresholds |
| DET_t | same procedure, IoU distance (1 - IoU) with THRESHOLDS_IOU = [0.75]; mean over the 13 attribute classes |
| edge threshold | THRESHOLD_RELATIONSHIP_CONFIDENCE = 0.5: a predicted link exists only if its score exceeds 0.5; only such candidates enter the ranking |
| per-vertex AP | candidates = predicted neighbours with score > 0.5, sorted by descending score; precision at each true-positive rank, summed, divided by the number of GT neighbours; empty GT and empty prediction = 1, only one empty = 0 |
| TOP_ll | per-vertex AP over rows (one direction) and over columns (other direction) of the lane x lane matrix, averaged (directed graph) |
| TOP_lt | undirected bipartite lane x traffic-element matrix (rows = lanes, columns = traffic elements); each row is scored over its traffic-element neighbours and each column over its lane neighbours, same per-vertex AP; averaged over the three lane thresholds and the one traffic-element threshold |
| OLS | mean(DET_l, DET_t, sqrt(TOP_ll), sqrt(TOP_lt)) in the code of both versions; metrics.md writes f(.) without defining it |
| range | annotation window x +-50 m, y +-25 m |

The part that changed (x.0 -> x.1):

| aspect | x.0 (v1.0.0, v2.0.0) | x.1 (v1.1.0, v2.1.0) |
|---|---|---|
| operating points | outer loop `for r in range(10)`: 10 confidence thresholds built from the pooled recall percentiles 10 % ... 100 % of the detection PR curve; only predictions with confidence >= threshold_r that are matched TPs enter `gt_pred` | no r loop; every matched prediction is kept, independent of its confidence |
| entries of the predicted topology matrix without a matched prediction | filled with `1 - GT`: a GT-connected pair gets 0, a GT-disconnected pair gets 1 (a full-confidence false link) | filled with `(1 - GT) * (0.5 + eps)`: connected pair 0, disconnected pair about 0.5, i.e. just above the 0.5 edge threshold and ranked last |
| aggregation | mean of all per-vertex APs over r, the three distance thresholds and all samples | mean over distance thresholds, frames, directions |
| loop style | Python loops over GT index pairs | vectorised indexing of the sub-matrix of matched predictions |
| what this does to a score | unmatched ground-truth lanes inject full-confidence false links, so TOP depends strongly on detection recall at each r | unmatched lanes are almost harmless; TOP numbers rise (2 - 3x in the examples) |
| consequence for comparison | numbers of the two versions are different quantities | same |

What TopoMLP reports about the old behaviour [P]: unmatched lanes / elements default to confidence 1.0, so many false positives rank high and the score depends on pushing confidences towards 1.0 or 0.0 around 0.5; the authors propose an adjusted TOP-dagger = TOP x N_TP / (N_TP + N_FP) in their section 4.6. The maintainers call the old behaviour a loophole (issue #76) and warn that TOP_ll and TOP_lt "might show inconsistencies with previous versions". The full mechanism of the inflation is not reproduced here.

Same model, two versions (README tables, subset A, ResNet-50, 24 epochs):

| model | metric version | DET_l | TOP_ll | DET_t | TOP_lt | OLS |
|---|---|---|---|---|---|---|
| TopoNet | first README table, no version label (x.0 era) | 28.6 | 4.1 | 48.6 | 20.3 | 35.6 |
| TopoNet | v1.1 devkit | 28.6 | 10.9 | 48.6 | 23.8 | 39.8 |
| TopoMLP | v1.0.0 | 28.5 | 7.1 | 49.5 | 23.4 | 38.3 |
| TopoMLP | v2.1.0 | 28.5 | 21.7 | 49.5 | 26.9 | 44.1 |
| TopoMLP, subset B | v1.0.0 / v2.1.0 | 26.3 | 7.2 / 21.2 | 57.8 | 17.0 / 19.4 | 38.0 / 43.6 |

DET_l and DET_t are identical across versions in these rows, consistent with the PR diff (topology functions rewritten; in the detection helper `_tpfp` only a dtype declaration changed, effect not shown on the page). Other numbers seen with a stated version: TopoPoint (v2.1.0, A) DET_l 31.4, DET_t 55.3, TOP_ll 28.7, TOP_lt 30.0, OLS 48.8, DET_p 52.6; TopoFG (v2.1.0, A) OLS 48.0, DET_l 33.8, DET_t 47.2, TOP_ll 30.8, TOP_lt 30.9; TopoLogic README (A) OLS 41.6 -> 44.1 and TOP_ll 18.6 -> 23.9 between its two tables (version labels on the page are unclear); UniTopo abstract: TOP_ll 30.1 (A) / 31.8 (B), devkit version not stated on the page. The TopoMLP paper and README differ in the last digit (28.3 / 7.2 / 50.0 / 22.8 / 38.2 vs 28.5 / 7.1 / 49.5 / 23.4 / 38.3).

Lane-segment variant (v2.x, OLUS): DET_ls = AP with distance 1/2 (Frechet(centerline) + Chamfer(left) + Chamfer(right)), thresholds {1, 2, 3}; DET_a = Chamfer-based AP for pedestrian crossings and road boundaries (thresholds not stated); OLUS = 1/5 [DET_ls + DET_a + DET_t + f(TOP_lsls) + f(TOP_lste)]. LaneSegNet's "mAP" is the mean of lane-segment AP and pedestrian-crossing AP (README); its README says v2.1.0 fixed a loophole in the TOP metric (README numbers, devkit v2.1.0: mAP 33.5, TOP_lsls 25.4). Challenge history: CVPR 2023 OpenLane Topology (EvalAI, closed 2023-06-01, server reopened), CVPR 2024 Autonomous Grand Challenge Mapless Driving (2024-03-01 to 2024-06-01, Hugging Face server).

Submission format [P, docs/submission.md and data/README.md]: pickle, frame key = (split, segment_id, timestamp); per frame `lane_centerline` (id, points [n,3], confidence), `traffic_element`, `topology_lclc` [n,n] float, `topology_lcte` [n,k] float (OLUS: `lane_segment`, `area`, `topology_lsls`, `topology_lste`). Confidence must exist for every predicted instance; the topology matrix holds scores in [0,1].

### B.2 nuScenes vector-map AP (MapTR protocol)

Sources: MapTR paper [P], MapTR eval code (tpfp_chamfer.py, mean_ap.py) [P, via summary], StreamMapNet paper and README [P].

| item | definition |
|---|---|
| classes | pedestrian crossing, lane divider, road boundary; mAP = mean over the three |
| perception range | x in [-15, 15] m, y in [-30, 30] m (60 x 30 m); StreamMapNet adds 100 x 50 m |
| threshold set | Chamfer thresholds {0.5, 1.0, 1.5} m at 60 x 30 m; {1.0, 1.5, 2.0} m at 100 x 50 m (StreamMapNet); AP per class = mean over thresholds. In the code `eval_map` takes one threshold per call; the aggregation over the list is outside the file seen |
| Chamfer distance | symmetric: mean over predicted points of the distance to the nearest GT point, plus mean over GT points of the distance to the nearest predicted point, divided by 2 (scipy cdist, Euclidean) |
| sampling | GT resampled to 100 points per instance (`num_sample=100`, linspace along length); predictions are used with their own points (20 per instance by default in training, Appendix B of the paper) unless a flag is set |
| matching | greedy by prediction score, each GT once (standard detection practice; the matching function itself was not in the summary). The score function also gates pairs: only pairs whose 1 m buffers intersect are scored |
| AP interpolation | 'area' mode hard-coded |
| training-time matching | Hungarian assignment with permutation-equivalent point matching (not part of the metric) |
| numbers | MapTR-tiny R50 camera: 24 epochs AP_ped 46.3 / AP_div 51.5 / AP_bnd 53.1 / mAP 50.3; 110 epochs mAP 58.7 (paper). README of the repository: tiny GKT 24 epochs 50.0, 110 epochs 59.3; MapTRv2 R50 24 epochs 61.4; MapTRv2* with centerline class 54.3 (24 epochs) |
| splits | original nuScenes split vs geographically disjoint split of StreamMapNet (nuScenes 700 / 150 scenes from Roddick and Cipolla; Argoverse 2 re-divided). Area overlap train / val: nuScenes original 85 % (0.79 of 0.92 km2), new 11 %; Argoverse 2 original 54 %, new 0 %. MapTR ResNet-50, 24 epochs, 30 m: 48.7 (original) vs 20.9 (new). StreamMapNet README (60 x 30 m): nuScenes new split AP 34.1, old split 63.4; Argoverse 2 new split 58.3 |

Direction note: Chamfer AP is direction-agnostic (polylines are compared as point sets with permutation-equivalent ordering in training); OpenLane-V2's Frechet distance respects direction. A reversed lane is a hit for MapTR-AP and a miss for DET_l.

### B.3 Lane-graph point-level metrics: GEO, TOPO, JTOPO, SDA, APLS, Graph IoU

| metric | definition as seen | parameters | source |
|---|---|---|---|
| TOPO (aerial, original) | sub-graphs sampled around a seed on GT and prediction (seed matched to the closest seed node); nodes within graph distance of a threshold (example 300 m); precision / recall / F1 of the sub-graph pair averaged over random seeds; Sat2Graph uses a "very strict" version of the earlier description | 300 m only as an example; matching tolerance not on page | Sat2Graph (ar5iv) |
| TOPO (lane graph, directed and undirected) | both graphs interpolated so connected vertices are 0.15 m apart; candidate pairs within 0.45 m; maximal one-to-one matching; each matched pair compared through the sub-graph reachable within 7.5 m; precision = matched sub-graph vertices / predicted vertices, recall = / GT vertices; the undirected version is computed after converting to undirected graphs | 0.15 m, 0.45 m, 7.5 m | LaneGAP (arXiv HTML) |
| JTOPO | same, but only the sub-graphs within 7.5 m of the GT junction points (merge and fork points) are used | 7.5 m | LaneGAP |
| TOPO (LaneGNN) | sub-graphs by walking a maximum distance D = 50 m from each matched vertex pair, scored with GEO | D = 50 m; hop limit none | LaneGNN (ar5iv) |
| GEO | graphs densely interpolated, 1:1 matching of vertices, precision / recall (edges ignored: local correctness only) | matching radius not on page (LaneGAP implies 0.45 m) | LaneGNN, CGNet |
| SDA_R | accuracy of predicted lane splits / junction points within a circle of radius R around GT splits; reported as SDA20 / SDA50 (R in pixels, aerial) or SDA4.5 / SDA9.0 (metres) in other papers; no exact formula on the pages | R | LaneGNN, CGNet, search summary |
| APLS | 1 - (1/N) sum over node pairs (a, b) in GT of min(1, abs(L(a,b) - L(a',b')) / L(a,b)); a' = proposal node closest to a; missing path = 1; control nodes = intersections, endpoints, midpoints injected every 50 m; snapping buffer 4 m by default; part 1 snaps GT nodes to the proposal graph, part 2 the reverse, final score = harmonic mean; path length in metres; directedness not on page | 4 m, 50 m | SpaceNet 3 paper (ar5iv 1807.01232) |
| Graph IoU | both graphs rendered as binary images, pixels within 5 px of the graph set to 1, IoU over the non-zero pixels | d = 5 pixels | LaneGNN |
| CGNet table 1 (nuScenes val, 60 x 30 m, point level) | GEO F1 / TOPO F1 / JTOPO F1 / APLS / SDA, e.g. CGNet 110 epochs 63.9 / 53.2 / 43.3 / 41.4 / 14.5 against MapTR (re-implemented) 61.2 / 50.0 / 40.7 / 38.4 / 13.4 (percent); the detailed metric definitions are in the supplement (not opened); perception range x +-15 m, y +-30 m, BEV grid 0.3 m | | CGNet |
| LaneGAP (directed, nuScenes) | Junction TOPO F1 0.564, TOPO F1 0.529 (scale 0 - 1); Argoverse 2 0.616 / 0.565; 50 x 100 m range 0.392 / 0.388 | | LaneGAP |
| CGNet released evaluation code (`projects/mmdet3d_plugin/datasets/topo_eval/`: geotopo.py, topo_metrics.py, topo_evaluator.py, apls.py, apls_utils.py; Apache-2.0 repository; read via summary) | Input = two networkx DiGraphs with node attribute `pos`. GEO / TOPO / JTOPO (geotopo.py): coordinates are multiplied by 10 and truncated to a 0.1-unit grid (`nx_to_geo_topo_format`); defaults radius 8, interpolation 2, sub-graph distance 400, graph mode 'direct' (only forward edges stored); bounds x in [-150, 150), y in [-300, 300) in these scaled units. If the input is in metres (inferred from the bounds equalling +-15 m by +-30 m) this means 0.8 m match radius, 0.2 m interpolation spacing, 40 m sub-graph radius, which differs from the LaneGAP text (0.45 m, 0.15 m, 7.5 m). Matching: greedy one-to-one, candidate pairs sorted by distance. GEO precision = matched / predicted nodes, recall = matched / GT nodes. TOPO precision = (matched / predicted nodes) x mean sub-graph precision, evaluated on every 10th matched pair; recall analogous. JTOPO = the same restricted to matches at GT junctions, junction = GT node with more than one neighbour in the stored neighbour list, so in 'direct' mode only forks count (merges only in undirected mode). F1 = 2PR / (P + R); empty prediction -> zeros. SDA (topo_metrics.py `calc_sda`, threshold default 1): junction points = nodes with out-degree >= 2 or in-degree >= 2; Hungarian one-to-one matching (linear_sum_assignment); TP = matched pairs below the threshold; the result is an F1 (not an accuracy), NaN if the GT has no junction. APLS (apls.py): undirected, weight = length, `max_snap_dist = 0.5` (SpaceNet default 4), `linestring_delta = 1` (commented-out 50), `min_path_length = 5`; pure python (networkx, scipy.spatial, shapely, utm), legacy networkx edge-attribute fallbacks. Graph IoU: edges drawn with width `lane_width = 1` on an `area_size = [30, 60]` canvas |  | CGNet code |

Scale warning: LaneGAP quotes 0 - 1, CGNet 0 - 100; TOPO radii differ (7.5 m, 50 m, 300 m); aerial APLS values (Sat2Graph 63.14 City-scale) and lane-graph APLS values (CGNet 41.4, LaneGNN 0.202) come from different graph types.

STSU / TPLR family (nuScenes centreline graphs from a front camera): precision / recall - each estimate is matched to the GT centreline with minimum L1 on Bezier control coefficients, then interpolated to dense points; true positive = point within a threshold of its matched line; thresholds uniform in [0.01, 0.1] normalised units in 0.01 steps (0.01 = 50 cm per the paper; metre values only if the scaling is linear); detection ratio = unique GT centrelines matched by at least one estimate / all GT centrelines; connectivity from incidence matrices, C-IOU = TP / (TP + FP + FN); BEV range x -25 .. 25 m, z 1 .. 50 m, 25 cm; no cycle-based measure (STSU page). Used by RNTR, LaneGraph2Seq, SeqGrowGraph (mean P / R / F, connectivity P / R / F, detection ratio: SeqGrowGraph table 3: M-F 70.6, C-F 65.8, Det 67.2; LaneGraph2Seq M-F 64.1, C-F 63.2, Det 64.5).

### B.4 RNTR landmark and reachability metrics (nuScenes)

Source: RNTR paper (Sec. III-I, IV-B) [P]; SeqGrowGraph reuses them (tables 1, 2, 5, 6) and adds Junction Landmark / Junction Reachability that merge continuous nodes before scoring.

| item | definition |
|---|---|
| landmarks | road landmarks named in the paper: crossroads, stop-lines, fork points, merge points. For the semi-autoregressive model, key points = vertices with out-degree > 1, in-degree > 1, or in-degree 0 (roots). End points are not listed explicitly |
| landmark matching | each predicted landmark matched to the nearest GT landmark; true positive if within the threshold, else false positive; unmatched GT landmark or one beyond the threshold = false negative |
| landmark thresholds | paper: 0.5 to 5.0 m in 0.5 m steps (Sec. III-I and IV-B); reported value = mean over thresholds. Released code: 5 default thresholds (see the code row) |
| reachability | two landmarks reachable if a path exists; with matched predicted pair (A', B') and GT pair (A, B), a true positive is a predicted path whose Chamfer distance to any GT A -> B path is below the threshold; false negative = GT path without a matched prediction; the text gives no formal false-positive definition |
| reachability thresholds and scope | [0.5, 1.0, 1.5, 2.0, 2.5] m; only vertex pairs within at most 5 edges |
| precision / recall / F1 | F1 = harmonic mean (Eq. 8) |
| range and splits | target area -48 .. 48 m (x), -32 .. 32 m (y) in the RNTR text (SeqGrowGraph states -48 .. 48 and -30 .. 30); input 128 x 352 pixels; nuScenes 700 / 150 / 150; "default" split has lanes that overlap between train and val, the PON split removes the overlap |
| numbers (SeqGrowGraph table 1, landmark P / R / F and reachability P / R / F, default split) | RNTR 57.1 / 42.7 / 48.9 and 63.7 / 45.2 / 52.8; LaneGAP 49.9 / 57.0 / 53.2 and 74.1 / 34.9 / 47.5; TopoNet 52.5 / 47.1 / 49.6 and 46.9 / 10.8 / 17.5; SeqGrowGraph 63.6 / 50.8 / 56.4 and 75.5 / 61.4 / 67.8. PON split: SeqGrowGraph 43.5 / 33.3 / 37.7 and 63.6 / 36.9 / 46.7 (RNTR 34.9 landmark F reproduced). The same method drops about 20 points between the two splits |
| released evaluation code (MIV-XJTU/SeqGrowGraph, files `seq_grow_graph/nus_reach_metric.py` and `seq_grow_graph/transforms/roadnet_reach_dist_eval_new.py`, commit c148057, read via summary; no licence file in the repository) | (1) Landmarks = nodes returned by `get_nodechains_dpt(1)` (the degree rules are inside classes not in the file read). (2) Landmark matching: every predicted keypoint is assigned to its NEAREST GT keypoint (cdist, no Hungarian) and is unmatched if beyond the threshold; the assignment is NOT one-to-one, so several predictions may share one GT landmark and all count as true positives (duplicates are not penalised in precision). TP = matched predicted nodes; FP = predicted - TP; FN = GT nodes - GT nodes that received at least one match. (3) Reachability: chains with at most `max_dpt = 5` edges grouped by (start, end) node pair; a GT pair is considered only if both its nodes were matched, and the node matching inside `eval_reach` uses a hard-coded threshold of 5 (grid units), independent of the loop threshold; a predicted chain is a true positive if its distance to the nearest GT chain is below the threshold; FN = GT chains not matched; predicted chains whose end points match no GT pair are never counted as false positives. (4) Chain distance = symmetric nearest-point (Chamfer-style) mean with every point distance multiplied by a direction factor tan(-d) + tan(1) + 1 where d is the product of local direction vectors (factor 1 when aligned). (5) Default thresholds in the code: landmark [1, 3, 5, 8, 10], reachability [1, 2, 3, 4, 5], in grid units; the metric key label is t x `xbound` interval, and with the stated 0.5 m discretisation this gives 0.5 - 5.0 m and 0.5 - 2.5 m (the interval value itself was not in the files read, so this conversion is an inference). The paper text describes 10 and 5 thresholds; the code defaults are 5 and 5. (6) Only samples present in the prediction file are evaluated; missing samples are skipped silently and do not count as false negatives, so an empty or missing prediction is not punished. (7) Output keys: landmark_precision, landmark_recall, landmark_f_score, reach_precision, reach_recall, reach_f_score (means over thresholds). Dependencies: the metric wrapper is an mmengine `BaseMetric` registered in mmdet3d; the evaluator classes are separate and need the repository's own graph classes (`EvalSeq2Graph`, `EvalSuperBzNode`). The repository also holds `nuscenes_pon_split.py` (the PON split) and older evaluator variants (`roadnet_reach_dist_eval.py`, `_aaai.py`) |

### B.5 Stability metrics (mAS)

Source: arXiv 2510.10660 [P] (HTML page read up to 100,000 of 134,406 characters).

| item | definition |
|---|---|
| frame pairing | for each anchor frame t, one frame t+k with k drawn uniformly from [1, M], M = 2 by default; the earlier polyline is moved into the current ego frame via the world frame (T_world->t+k times T_t->world); clipped to the perception range, resampled along x with N = 100 points |
| matching | Hungarian matching of predictions to GT in each frame (cost "geometric and semantic similarity", exact form not on page); two predictions matched to the same GT instance in both frames are paired. No distance threshold; Chamfer not used |
| presence stability | 1 if the confidence is on the same side of a threshold tau in both frames, 0.5 ("flickering") otherwise; tau not on page |
| localisation stability | mean absolute y-difference of the two resampled polylines times beta = 15.0; the printed formula cannot be the intended score (it grows with the deviation); the text speaks of an exponential mapping to [0,1] while the appendix says stability reaches zero when the deviation equals beta, so the intended form is unclear |
| shape stability | curvature kappa = mean turning angle (rad) of the resampled polyline; Shape = 1 - abs(kappa_{t+k} - kappa_t) / pi |
| per-element and aggregate | Stability = Presence x [omega x Loc + (1 - omega) x Shape], omega = 0.7; class stability = mean over instances; mAS = mean over classes |
| data | nuScenes validation set named in the text; classes not listed; range x +-15, y +-30 m (standard map range); the benchmark release was announced |
| also seen | MapTracker README [P]: consistency-aware mAP = existing mAP plus consistency checks on temporally re-processed ground truth; no formula on the page |

### B.6 Planning and closed-loop scores

PDM Score in NAVSIM (v1) [P: paper, docs/metrics.md]: PDMS = (NC x DAC) x (5 EP + 5 TTC + 2 C) / 12, per frame, averaged over frames, range [0,1]. NC no at-fault collision {0, 1/2, 1} (collisions with vehicles, pedestrians, bicycles = 0; static objects 0.5; not at fault when ego is static or rear-ended); DAC drivable-area compliance {0,1}; EP ego progress = progress along the route centre divided by an upper bound from PDM-Closed, clipped to [0,1] (low or negative scores discarded when the bound is under 5 m); TTC {0,1}; C comfort {0,1}. Simulation: 4 s horizon, 10 Hz, non-reactive agents, LQR controller, kinematic bicycle model. Weights EP 5, TTC 5, C 2 (paper; equal weights keep the top ranks). In the current code [P, scorer file]: TTC horizon window 1.0 s with offsets 0, 0.3, 0.6, 0.9 s; progress threshold 5.0 m; DAC fails if fewer than 4 ego corners lie in drivable polygons; driving direction horizon 1.0 s with 2.0 m / 6.0 m thresholds (1.0 / 0.5 / 0.0). EPDMS (NAVSIM v2): multiplicative NC, DAC, DDC (driving direction, {0,1/2,1}), TLC (traffic light compliance, {0,1}); weighted TTC 5, EP 5, HC history comfort 2, LK lane keeping 2 (0.5 m deviation, 2.0 s window, skipped at intersections), EC extended comfort 2 (in the code listed but excluded from the aggregate); each term filtered so a penalty is disabled when the human log violates it. Comfort thresholds not read for NAVSIM. Navtrain / navtest filtering: frames removed when constant velocity scores PDMS > 0.8 or the human scores < 0.8; afterwards constant velocity 22 %, human 95 %. Reported correlation with closed loop: PDMS better than open-loop L2 (coefficients only in figures).

nuPlan scores [P: metrics_description page; weights from memory]: multipliers (no at-fault collisions with allowed counts VRU 0, vehicle 0, object 1; drivable-area compliance with 0.3 m maximum violation; driving-direction compliance with 2 m / 6 m over 1 s; making progress with ratio threshold 0.2); weighted metrics: ego progress along expert route (ratio, 0.1 m floor), time to collision within bound (horizon 3.0 s, step 0.1 s, least min TTC 0.95 s), speed-limit compliance (over-speed up to 2.23 m/s), comfort (longitudinal acceleration -4.05 .. 2.40 m/s2, lateral 4.89 m/s2, yaw acceleration 1.93 rad/s2, yaw rate 0.95 rad/s, longitudinal jerk 4.13 m/s3, jerk magnitude 8.37 m/s3). The page does not give the weights and does not define CLS-NR / CLS-R / OLS; from memory the weights are 5 / 5 / 4 / 2 and CLS-NR = non-reactive agents (log replay), CLS-R = reactive agents (IDM). Search snippets confirm that the multiplier / weighted structure is shared by CLS-NR and CLS-R and scores run 0 - 100 [S]. Published numbers seen: PlanTF Val14 OLS 89.18 / NR-CLS 84.83 / R-CLS 76.78; Diffusion Planner (no refinement) Val14 89.87 / 82.80, with refinement 94.26 / 92.90 (NR / R).

nuScenes open-loop planning (UniAD, VAD, SparseDrive READMEs) [P]: L2 at 1 s / 2 s / 3 s and collision rate; VAD-Base L2 0.41 / 0.70 / 1.05 m, collision 0.07 / 0.17 / 0.41 %; UniAD avg L2 0.90 m, collision 0.29 %; SparseDrive-B avg L2 0.58, collision 0.06 % (paper); the SparseDrive README says the collision metric was re-implemented because some cases were missed, so its released-checkpoint collision numbers (0.097 %) differ from the paper. Averaged-versus-per-time L2 protocol differences between ST-P3 and UniAD / VAD: from memory.

### B.7 Mapping table: their metric <-> our metric

Our metrics (PROJECT_CONTEXT): point F1 at 0.5 / 1 / 1.5 m (pooled P and R of dense 0.5 m points), lanes found, false length fraction, direction accuracy, vertex error, split / merge F1 at 6 m, route success (coverage and precision >= 0.8 within 1.5 m) at map error 0 / 1 / 2 / 3 m, planner path error at 5 - 40 m, temporal consistency.

| their metric | closest metric of ours | what differs | comparable? |
|---|---|---|---|
| OLS / DET_l (Frechet AP, thresholds 1 / 2 / 3 m, 11 pts, confidence-ranked, greedy match) | point F1@1 and lanes found | instance-level and confidence-ranked vs pooled and unranked; direction-aware (Frechet); thresholds 1 / 2 / 3 m; range 100 x 50 m | no (different quantity); computable after adapter G2 |
| TOP_ll (adjacency AP, edge threshold 0.5, version-dependent) | none; closest: split / merge F1 (junction topology) and route success (reachability) | adjacency between instances; depends on instance granularity and on the version | no |
| DET_t, TOP_lt, DET_a, OLUS terms | none | traffic elements, areas: not in our task | not computable |
| MapTR mAP (Chamfer 0.5 / 1 / 1.5 m, 60 x 30 m, 100-point GT) | point F1 at 0.5 / 1 / 1.5 m | same threshold values, but AP ranks instances by confidence and matches instances, ours pools points; MapTR has no lane-graph class (MapTRv2* adds a centerline class) | partly (same thresholds, different aggregation) |
| STSU M-Pre / M-Rec (pooled, thresholds 0.5 - 5 m), detection ratio, C-IOU | point F1 (pooled P and R), lanes found | STSU averages over thresholds up to 5 m, front camera, 50 m range | closest family |
| GEO F1 (1:1 matching, 0.15 m interpolation, about 0.45 m) | point F1@0.5 | 1:1 matching vs nearest-point precision / recall | similar idea, not equal |
| TOPO / JTOPO F1 (sub-graph within 7.5 m, 0.45 m matching) | split / merge F1 at 6 m (junction key points) plus point F1 | sub-graph connectivity around every matched vertex vs key-point detection | complementary |
| SDA_R / SDA (CGNet code: F1 of Hungarian-matched junction points at threshold 1, junction = in- or out-degree >= 2) | split / merge F1 at 6 m | CGNet SDA is also an F1 over forks and merges but at 1 (metres, inferred) instead of 6 m; LaneGNN's SDA_R is an accuracy within a radius in pixels | similar; ours is looser, compute ours at 1 m for a like-for-like value |
| RNTR landmark P / R (forks, merges, roots, 0.5 - 5 m) | split / merge F1 at 6 m | thresholds averaged over 0.5 - 5.0 m vs one 6 m radius; RNTR includes roots | ours is looser; add curve |
| RNTR reachability P / R (path between landmarks within 5 edges, Chamfer 0.5 - 2.5 m) | route success (coverage and precision >= 0.8 within 1.5 m) | pairs of landmarks vs a driven route over a map graph | related, not equal |
| APLS | route success / reachability | shortest-path length ratio over control nodes, 4 m snap buffer | related |
| Graph IoU | point F1 | raster IoU at 5 px | related |
| mAS (presence, localisation, shape) | temporal consistency | needs ego-warp pairs, confidence threshold, Hungarian | related; ours to be defined |
| MapTracker consistency-aware mAP | temporal consistency | formula not read | unknown |
| NAVSIM PDMS / EPDMS, nuPlan CLS | none | needs agents, simulator, nuPlan maps and speed profile | not computable on nuScenes |
| nuScenes open-loop L2 / collision (UniAD, VAD) | planner path error at 5 - 40 m | time-indexed (1 / 2 / 3 s) vs distance-indexed; the ego-status paper (see lit_p16_tree) shows L2 hardly reflects map quality | convertible with a speed profile; weak signal |
| (no counterpart) | vertex error (0.54 -> 0.43 m), direction accuracy, false length fraction | published protocols have no vertex-position error | ours only |

### B.8 Computing their metrics from OUR output graph (vertices, edges, validity)

Adapter definitions (all CPU, numpy / networkx / shapely, no new dependency):

- G1 dense polyline: resample every edge at a fixed spacing (0.15 m for TOPO / GEO, 0.5 m for our F1, arclength n-point resampling for AP metrics). Exists in our F1 code.
- G2 pieces (instances): threshold vertex validity (tau = 0.5 for metrics that need a binary graph; keep all vertices and carry confidence for AP-type metrics); a vertex is a junction if in-degree is not 1 or out-degree is not 1; a piece is a maximal chain between junctions, endpoints included; confidence of a piece = min over its vertices of the validity (alternatives: mean, geometric mean - fix one); polyline resampled to 11 points (OpenLane-V2 evaluation), 20 or 100 points (MapTR type), 201 for a GT store. Apply the same function to the GT piece graph with the cut rule from A.3.
- G3 adjacency matrix: A[i,j] = score if the last vertex of piece i is the first vertex of piece j (shared junction vertex or direct edge); score = min(confidence_i, confidence_j) or the edge probability if the model has one. For OpenLane-V2, pieces meeting at a junction are predecessors / successors of each other by construction.
- G4 landmarks: vertices with out-degree > 1 (forks), in-degree > 1 (merges), in-degree 0 (roots) on the validity-thresholded graph; optionally merge close vertices first (SeqGrowGraph "junction" variants).
- G5 weighted digraph for APLS, reachability and TOPO sub-graphs: edge weight = metric length.
- G6 temporal pairs: warp pieces of frame t into frame t+k with the nuScenes ego poses; Hungarian or nearest-Chamfer matching to GT.

| their metric | needs | adapter | cost | pitfalls |
|---|---|---|---|---|
| DET_l (OLS part) | instance polylines with confidence, GT instances, Frechet | G2 | S - M (vendor evaluate.py, replace `similaritymeasures` / ortools; or write 40 lines of discrete Frechet) | confidence semantics; cropping to x +-50, y +-25; GT cut rule; z = 0 |
| TOP_ll (x.1) | G3 score matrix; calibrated at 0.5 | G2 + G3 | S after DET_l | only candidates > 0.5 are ranked; our validity must be calibrated; report which devkit version (x.1 only) |
| OLS-lite | DET_l, TOP_ll only | (DET_l + sqrt(TOP_ll)) / 2 as a proxy | S | not OLS; label as "lane-only" |
| MapTR-type AP on a centreline class | G2 (20 / 100 points), chamfer thresholds | G2 | S | direction-agnostic; window 60 x 30 m |
| RNTR landmark P / R / F | G4, nearest matching, thresholds 0.5 .. 5.0 m | G4 | S | roots at the crop border; threshold average; the released code does not require one-to-one matching (duplicate predictions all count as true positives) and skips frames without predictions, so report our variant explicitly and score every frame |
| RNTR reachability | G5, paths up to 5 edges between landmark pairs, Chamfer 0.5 .. 2.5 m | G4 + G5 | S - M | combinatorics of simple paths; define false positives yourself |
| TOPO / JTOPO | G1 at 0.15 m, 0.45 m one-to-one matching, 7.5 m sub-graphs, junction subset | G1 + G5 | M | directed and undirected variants differ; scale 0 - 1 vs 0 - 100 |
| GEO | G1 at 0.15 m, 1:1 matching | G1 | S | matching radius not stated on pages (0.45 m in LaneGAP) |
| SDA | junction points of prediction and GT, radius R | G4 | S | R definition |
| APLS | G5, control nodes (junctions, ends, midpoints every 50 m), 4 m snap | G5 | S - M | directed vs undirected; missing path = 1 |
| mAS | G6, confidence threshold tau, 100-point resampling along x | G2 + G6 | M | formula inconsistency in the paper; tau not given |
| STSU metrics | G1 + matching of estimates to GT lines | G1 | S | normalised thresholds |
| DET_t, TOP_lt, DET_a, PDMS, EPDMS, CLS | traffic elements / areas / agents / simulator | none | not computable | -- |

### B.9 Comparability traps (checklist)

1. OpenLane-V2 TOP_ll / TOP_lt / OLS: x.0 vs x.1 (B.1). DET_l / DET_t look version-independent (identical in the README pairs; the PR rewrote only the topology code).
2. TOP is computed on a 0.5 edge threshold and ranks only candidates above it; a model with uncalibrated edge scores is penalised or flattered.
3. MapTR AP: 60 x 30 vs 100 x 50 m, threshold sets differ, split original vs geographically disjoint (48.7 vs 20.9 for MapTR).
4. RNTR-family: "default" vs PON split (about 20 points of landmark F).
5. TOPO / JTOPO / GEO: radii (0.45 m, 7.5 m, 50 m, 300 m), directed vs undirected, scale 0 - 1 vs 0 - 100.
6. SDA: radius in pixels (SDA20 / SDA50, aerial) vs metres (SDA4.5 / SDA9.0).
7. Instance granularity: pieces vs merged paths (LaneGAP argues paths; RNTR re-segments at 20 / 30 / 40 m optionally).
8. Direction: Frechet (OpenLane-V2) and directed TOPO respect direction; Chamfer AP and undirected TOPO do not.
9. Windows: x +-50 / y +-25 m (OpenLane-V2), 60 x 30 m (MapTR family), -48 .. 48 by -32 .. 32 m (RNTR), x +-15 / y +-30 (stability), ours +-50 m square.
10. Papers that do not state the devkit version (UniTopo) cannot be placed on the x.0 / x.1 axis.
11. Landmark / reachability: the paper text and the released code differ in the number of thresholds (10 and 5 in the text, 5 and 5 as code defaults); the code matches non-one-to-one, counts only predicted chains whose end points match a GT pair as candidates for false positives, and silently skips frames that have no prediction. Numbers computed with our own stricter variant (one-to-one, every frame scored) are not comparable with the published ones.
12. TOPO / JTOPO exist in two protocol forms: the LaneGAP text (0.15 m interpolation, 0.45 m matching, 7.5 m sub-graph) and the CGNet code (0.2 m, 0.8 m, 40 m under the unit inference of B.3; TOPO multiplied by the matched fraction; directed JTOPO counts forks only). SDA is an F1 with Hungarian matching at threshold 1 in the CGNet code but an accuracy within a radius R (pixels or metres) in LaneGNN. APLS: SpaceNet default snap 4 m, control points every 50 m (harmonic mean of two directions) vs CGNet code snap 0.5, spacing 1, undirected. Numbers from different papers under the same metric name are not comparable unless the code is the same.
13. Instance-level AP metrics (OpenLane-V2, MapTR) need a confidence per instance; pooled point F1 (ours) and STSU-style P / R do not. An empty graph gets AP 0 but is skipped by the released landmark code.

---

## 5. PART C - Code and licence inventory

Verdict key (our environment: Python 3.12, PyTorch 2.8, no mmcv / mmdet / mmdet3d): runs as is | light port (named blocker, effort S < 1 day, M days) | port ideas only | no code. "Last push" = `pushed_at` from the GitHub API, checked 2026-10-23. "Weights" = pretrained checkpoints offered on the page. All pins [P] unless marked.

### C.1 Graph growers, road-graph and lane-graph methods

| method | repo | licence | framework / pins | last push | weights | dataset scripts | verdict |
|---|---|---|---|---|---|---|---|
| RNGDet / RNGDet++ | TonyXuQAQ/RNGDetPlusPlus (standalone RNGDet repo 404; both included) | GPL-3.0; README adds "not allowed for commercial purposes, only academic research" | Docker (Ubuntu 20.04, CUDA 11.1, tested on 4 x RTX 3090); no PyTorch / mmcv pin; TOPO and APLS scripts "not runnable in docker" | 2026-02-25 | yes (cloud.tsinghua.edu.cn after Google Drive links failed) | prepare_dataset/preprocessing.bash; sampler parameters edge_move_ahead_length 30, noise 8 | port ideas only (aerial image-crop agent; GPL) |
| CenterLineDet | TonyXuQAQ/CenterLineDet | GPL-3.0 | Docker; separate `docker_py2` image for evaluation; no PyTorch / mmcv pin | 2026-02-25 | "check ./data" (links not shown) | nuScenes dir set in container build script; no converter shown | port ideas only |
| iCurb | TonyXuQAQ/iCurb (README only; code moved to TonyXuQAQ/Topo-boundary) | iCurb: none stated; Topo-boundary: GPL-3.0 (API) | not on page | 2026-02-25 | not on page | not on page | port ideas only |
| SeqGrowGraph | MIV-XJTU/SeqGrowGraph (also holds RNTR and LaneGraph2Seq configs: `ar_rntr.py`, `ar_lanegraph2seq.py`) | none (API: no licence; README: none) | Python 3.8, PyTorch 2.0.1 + CUDA 11.7, mmdetection3d v1.4.0, mmengine, mmcv >= 2.0.0rc4, mmdet >= 3.0.0, mmsegmentation, einops, bezier; flash-attention file included | 2025-11-14 | segmentation backbone (Cityscapes DeepLab-V3-Plus R50) + a trained checkpoint on ModelScope (misstl/SeqGrowGraph; the page implies it is the SeqGrowGraph model) | tools/create_data_pon_centerline.py (nuScenes PON split), `nuscenes_converter_default_centerline.py`; Argoverse 2 dataset class present in the tree, not described in the README; GT-graph construction not described | port ideas only (no licence, mmdet3d 1.4 stack); its evaluation code is the best reference for the landmark / reachability protocol (B.4) |
| RNTR | fudan-zvg/RoadNet (folders RoadNetwork-1.8.1 and RoadNetwork-2.0.1) | MIT | 1.8.1 branch: Torch < 2.0; 2.0.1 branch: Python 3.8, PyTorch 2.0.1 + CUDA 11.7, mmdetection3d v1.4.0, mmengine, mmcv >= 2.0.0rc4, mmdet >= 3.0.0, mmsegmentation, einops, bezier; LSS-style BEV encoder; acknowledges MMDetection3D, PETR, DETR3D, BEVDet | 2025-09-10 | no (only a segmentation backbone; release note lists checkpoints as future work) | tools/create_data_pon_centerline.py (PON split); GT-graph construction not described | port ideas only (sequence format re-implementable, effort S - M); the same models also live in MIV-XJTU/SeqGrowGraph (no licence, torch 2.0.1 + mmdet3d 1.4) |
| LaneGraph2Seq | same repo (fudan-zvg/RoadNet; README: "RNTR (TPAMI 2025 / ICCV 2023) and LaneGraph2Seq (AAAI 2024)") | MIT | as RNTR | 2025-09-10 | no | as RNTR | port ideas only |
| DAGMapper | none found | -- | -- | -- | -- | -- | no code |
| DeH4R | godx-7/DeH4R | none (API: null; README: none) | uv environment, requirements.txt (not shown); no PyTorch pin; SAM and SAM2 base checkpoints needed | 2026-05-25 | yes (Hugging Face godx7/DeH4R) | data_preprocess.py; metrics_cityscale / globalscale / spacenet folders | port ideas only (no licence -> no copying; aerial) |
| Sat2Graph | songtaohe/Sat2Graph | MIT, but README: models, dataset, docker "non-commercial academic use only" | framework not stated on page (TensorFlow, from memory) | 2023-04-09 | yes (Google Drive; download script no longer works) | City-scale, SpaceNet splits via Drive link | port ideas only |
| SAM-Road (+ SAM-Road++) | htcr/sam_road, earth-insights/samroadplus | MIT; SAM-Road++: no licence file | PyTorch ("latest"), PyTorch Lightning, wandb, Go (APLS only); SAM submodules; no requirements file | 2024-08-10; 2025-03-01 | SAM-Road yes (Hugging Face congrui/sam_road); SAM-Road++ no | generate_labes.py | light port of the graph head only (vertex peaks by NMS + edge transformer; SAM encoder not needed), effort M |
| Relationformer | suprosanna/relationformer | Apache-2.0 | environment not on page; 4 branches (vessel_graph, road_network, road_network_rgb, scene_graph); Deformable-DETR attention with custom op (from memory) | 2023-06-04 | not mentioned | per branch | light port (verify the custom op; replace by the grid_sample fallback), effort M |
| InstaGraM | juyebshin/InstaGraM | GPL-3.0 | Python 3.8, PyTorch 1.10.2 + cu113, torch-scatter; HDMapNet-based | 2025-04-01 | links point to repo root (unclear) | vis_label.py, export_gt_to_json.py | port ideas only (GPL) |

### C.2 Lane topology and centre-line methods (OpenLane-V2 / nuScenes)

| method | repo | licence | framework / pins | last push | weights | dataset scripts | verdict |
|---|---|---|---|---|---|---|---|
| TopoNet | OpenDriveLab/TopoNet | Apache-2.0 | Python 3.8, PyTorch 1.9.1 + cu111, torchvision 0.10.1, mmcv-full 1.5.2, mmdet 2.26.0, mmsegmentation 0.29.1, mmdet3d 1.0.0rc6 | 2025-07-02 | yes (Hugging Face, subsets A and B) | via OpenLane-V2 v1.0.0 preprocessing; `data_dict_subset_{A,B}_{train,val}.pkl` | port ideas only (scene-graph NN head small, re-implement S - M) |
| TopoMLP | wudongming97/TopoMLP | Apache-2.0 | built on mmdet3d; versions in docs/setup.md (not opened) | 2026-07-13 | yes (subset A R50 `topomlp_setA_r50_wo_yolov8_e24.pth` in release v1.0) | docs/setup.md | port ideas only (MLP heads trivial, S) |
| LaneSegNet | OpenDriveLab/LaneSegNet | Apache-2.0 | as TopoNet (mmcv-full 1.5.2, mmdet 2.26.0, mmdet3d 1.0.0rc6, PyTorch 1.9.1) | 2026-02-25 | yes (subset A) | tools/data_process.py -> `*_lanesegnet.pkl` (not the OpenLane-V2 `*_ls.pkl`) | port ideas only (lane attention, M) |
| TopoLogic | Franpin/TopoLogic | none (API: null) | Python 3.8, PyTorch 1.9.1 + cu111; mm* not on page | 2025-05-26 | yes (subset A R50) | follows OpenLane-V2 v1.0.0 | port ideas only; geometric link score re-implementable in S (we tried similar scoring: E42, E45, dropped) |
| CGNet | XiaoMi/CGNet | Apache-2.0 | Python 3.8 conda; bundled mmdetection3d folder; GKT op from MapTR; requirement.txt: torch 1.9.1 + cu111, mmcv-full 1.4.0, mmdet 2.14.0, mmsegmentation 0.14.1, networkx 2.2, Shapely 1.8.1, scipy 1.8.0, numpy 1.23.4, Rtree 1.0.1, nuscenes-devkit 1.1.9, av2 0.2.1, torch-geometric 2.1.0 | 2024-12-16 | yes (Google Drive, `cgnet_ep110.pth`) | tools/create_data.py --version v1.0 --canbus; eval `--eval chamfer openlane topology` | model: port ideas only. `topo_eval/` folder (GEO / TOPO / JTOPO / SDA / APLS / Graph IoU, Apache-2.0, pure python on networkx graphs): light port, effort S - M (old networkx / scipy / rtree pins; a removed `scipy.ndimage.imread` import; the parameters differ from the LaneGAP text, B.3) |
| TopoPoint | Franpin/TopoPoint | none | Python 3.8, PyTorch 1.9.1 + cu111 | 2025-12-13 | not mentioned | OpenLane-V2 v1.0.0 | port ideas only |
| TopoFG | GXmmm18/TopoFG | none | as TopoPoint | 2025-11-29 | yes (A, Google Drive; label in table inconsistent) | OpenLane-V2 v1.0.0 | port ideas only |
| UniTopo | homothetic/UniTopo | none | Python 3.8.10, CUDA 11.8; built on TopoNet, TopoMLP, TopoLogic | 2026-05-09 | yes (A and B, Google Drive) | OpenLane-V2 layout | port ideas only |
| TopoBDA | none for the model (project page: LiDAR fork artest08/OpenLane-V2 branch lidar_addition + openlanev2_lidar.zip) | footer "all rights reserved", no licence | -- | -- | no | LiDAR data zip | no code |
| Topo2Seq | none found | -- | -- | -- | -- | -- | no code |
| LaneGAP | hustvl/LaneGAP (README, assets, LICENSE only; news says code released together with MapTRv2 in hustvl/MapTR) | Apache-2.0 (API) | -- | 2024-09-11 | none listed | -- | no code in this repo; unverified inside MapTR |
| SMART | none (project page: "code coming soon") | -- | -- | -- | -- | -- | no code |

### C.3 Vector-map, streaming and query-propagation methods

| method | repo | licence | framework / pins | last push | weights | dataset scripts | verdict |
|---|---|---|---|---|---|---|---|
| MapTR / MapTRv2 | hustvl/MapTR (MapTRv2 on branch maptrv2) | MIT | Python 3.8, PyTorch 1.9.1 + cu111, mmcv-full 1.4.0, mmdet 2.14.0, mmsegmentation 0.14.1, mmdet3d bundled, custom op geometric_kernel_attn (GKT), gcc >= 5 | 2025-03-03 | yes (Google Drive) | docs/prepare_dataset.md (not opened) | port ideas only; Chamfer-AP evaluation = light port (S) |
| StreamMapNet | yuantianyuan01/StreamMapNet | GPL-3.0 | Python 3.8, PyTorch 1.9.0 + cu111, mmcv-full 1.6.0, mmdet 2.28.2, mmsegmentation 0.30.0, mmdet3d v1.0.0rc6 | 2024-03-03 | yes (many, Google Drive) | tools/nuscenes_converter.py --newsplit; argoverse_converter.py | port ideas only (GPL) |
| SQD-MapNet | shuowang666/SQD-MapNet | GPL-3.0 | as StreamMapNet | 2024-10-06 | yes (2 nuScenes configs) | as StreamMapNet | port ideas only; stream query denoising re-implementable (training-only trick, S - M) |
| MapTracker | woodfrog/maptracker | README: GPL-3.0 (LICENSE, LICENSE_GPL); GitHub detector: NOASSERTION | versions in docs/installation.md (not opened); builds on StreamMapNet | 2025-10-07 | yes (per data guide) | docs/data_preparation.md | port ideas only; consistency-aware mAP code may be extractable (light port, GPL) |
| TopoStreamer | YimingYang23/TopoStreamer | none | Python 3.8, PyTorch 1.9.0 + cu111, mmcv-full 1.5.2, mmdet 2.26.0, mmsegmentation 0.29.1, mmdet3d 1.0.0rc6; 4 x A100 40 GB or V100 32 GB | 2025-07-31 | not mentioned | tools/data_process.py, tracking script | port ideas only |
| Sparse4D (v1 - v3) | HorizonRobotics/Sparse4D | MIT | not on page (requirement.txt, docs/quick_start.md); mmdet3d only in acknowledgements; deformable-aggregation CUDA op (from memory) | 2026-05-21 | yes (GitHub releases) | not on page | port ideas only (instance bank, top-k propagation) |
| StreamPETR | exiawsh/StreamPETR | GitHub: NOASSERTION (custom) | not on page (docs/setup.md); mmdet3d-based | 2024-06-26 | yes | docs/data_preparation.md | port ideas only |

### C.4 Detection-transformer cores and line / lane detectors

| method | repo | licence | framework / pins | last push | weights | verdict |
|---|---|---|---|---|---|---|
| DAB-DETR | IDEA-Research/DAB-DETR | Apache-2.0 | tested Python 3.7.3, PyTorch 1.9.0, CUDA 11.1; base model needs no extra ops; DAB-Deformable-DETR needs a compiled op | 2023-06-02 | yes | light port (base model is plain PyTorch; small old-API fixes are likely, from memory), effort S |
| Deformable DETR | fundamentalvision/Deformable-DETR | Apache-2.0 | Python >= 3.7, PyTorch >= 1.5.1, compiled MultiScaleDeformableAttention op; the file `ms_deform_attn_func.py` also holds `ms_deform_attn_core_pytorch` (F.grid_sample, marked "for debug and test only") | 2024-05-16 | yes | light port: use the grid_sample path (slow, but fits our 200 x 200 / 100 x 100 / 50 x 50 pyramid), effort S |
| DINO | IDEA-Research/DINO | Apache-2.0 | tested Python 3.7.3, PyTorch 1.9.0, CUDA 11.1; compiled op in models/dino/ops; detrex port exists | 2024-07-31 | yes | light port (denoising, mixed query selection, look-forward-twice are plain PyTorch), effort M |
| CLRNet | Turoad/CLRNet | Apache-2.0 | Python >= 3.8, PyTorch >= 1.6 (tested 1.6), CUDA 10.2 tested, `python setup.py build develop` (compiled extension); mmcv not mentioned | 2024-07-16 | yes (releases) | light port (Line IoU loss, ROIGather re-implementable; NMS via torchvision), effort S - M |
| LaneATT | lucastabelini/LaneATT | MIT | PyTorch == 1.6 (CUDA 10.2), NMS in lib/nms compiled | 2022-10-09 | yes (zip via gdown) | light port (replace the NMS build), effort S |
| L-CNN | zhou13/lcnn | MIT | PyTorch (no pin), conda; APH evaluation needs MATLAB; sAP / mAPJ scripts in Python | 2024-08-12 | yes (Hugging Face) | light port (LoI line verification), effort S - M |
| HAWP | cherubicXN/hawp | MIT | Python 3.9, PyTorch 1.12.0 + cu116, ninja (custom ops not confirmed), Dockerfile | 2024-02-26 | yes (downloads.sh) | light port, effort S - M |

### C.5 Multiple-choice learning

| method | repo | licence | framework / pins | last push | weights | verdict |
|---|---|---|---|---|---|---|
| rMCL | Victorletzelter/code-rMCL (toy-rMCL, audio-rMCL) | MIT | PyTorch, version not stated; no requirements file seen | 2026-02-26 | not mentioned | runs as is for the toy loss; reimplement the scoring head and loss (S) |
| aMCL | Victorletzelter/annealed_mcl | none (API: null) | Python 3.9.20 (synthetic, UCI); Python 3.8 + PyTorch 2.0.0 + CUDA 11.7 + Lightning (source separation) | 2025-05-26 | UCI checkpoints on Google Drive | runs as is (loss code is small) but no licence: re-implement from the paper (S) |

### C.6 End-to-end driving and planners

| method | repo | licence | framework / pins | last push | weights | dataset scripts | verdict |
|---|---|---|---|---|---|---|---|
| UniAD | OpenDriveLab/UniAD | Apache-2.0 | v2.0: "mmdet3d 1.x and torch 2.x" (TODO names mmdet3d 1.0.0rc6, torch 2.0.1 + cu118); install in INSTALL.md | 2025-10-29 | yes (2 stages) | DATA_PREP.md (nuPlan and NAVSIM integrated) | port ideas only |
| VAD | hustvl/VAD | Apache-2.0 | docs/install.md (not opened) | 2026-01-31 | yes (Tiny, Base) | docs/prepare_dataset.md | port ideas only; vectorised planning-constraint losses re-implementable (S) |
| SparseDrive | swc-17/SparseDrive | MIT | not on page; mmdet3d in acknowledgements | 2026-09-16 | yes (Small, stage 1 and 2) | not on page | port ideas only |
| PlanTF | jchengai/planTF | none (API: null) | Python 3.9, nuplan-devkit (editable), PyTorch Lightning; torch not stated; feature cache 1 M frames, "20+ hours" | 2024-07-11 | yes (OneDrive) | caching via nuPlan scenario filter | port ideas only (no licence) |
| PLUTO | jchengai/pluto | none | Python 3.9, nuplan-devkit; README: training part "not fully tested" | 2024-07-15 | yes (OneDrive) | nuPlan cache scripts | port ideas only |
| Diffusion Planner | ZhengYinan-AIR/Diffusion-Planner | none | Python 3.9, nuplan-devkit, requirements_torch.txt; torch not stated | 2026-03-10 | yes (Hugging Face) | data_process.py | port ideas only |

### C.7 BEV encoders, fusion and foundation

| method | repo | licence | framework / pins | last push | weights | dataset scripts | verdict |
|---|---|---|---|---|---|---|---|
| GaussianCaR | santimontiel/gaussiancar | Apache-2.0 | Python 3.12.3, PyTorch 2.8.0, Lightning 2.5.0, Hydra, uv + Docker; example log CUDA 12.9; FP16 / BF16 unsupported (PTv3 module has no backward support), train in FP32 | 2026-10-07 | yes (Hugging Face santimontieleu/gaussiancar: vehicle and map) | tools/create_data.py (labels/ JSON) | runs as is (our foundation) |
| BEVFusion (MIT-HAN-lab) | mit-han-lab/bevfusion | Apache-2.0 | Python 3.8, PyTorch 1.9 - 1.10.2, mmcv 1.4.0, mmdet 2.20.0, OpenMPI 4.0.4, mpi4py 3.0.3, torchpack, bev_pool CUDA op; repository archived (2025-01-12) | 2024-07-31 | yes (Dropbox; map seg mIoU 62.95 C+L, 57.09 camera only) | mmdet3d info files + map expansion | port ideas only |
| CRN | youngskkim/CRN | MIT | PyTorch 1.9.1 + cu111, Lightning 1.6.0, mmcv 1.6.0, mmsegmentation 0.28.0, mmdet 2.25.2, bundled mmdet3d | 2023-12-07 | yes (BEVDepth R50, CRN R18, R50; BEV segmentation code and checkpoints listed but unchecked) | scripts/gen_info.py, gen_depth_gt.py, gen_radar_bev.py, gen_radar_pv.py | port ideas only; radar-to-BEV scripts may be light-port candidates; E27 already compared CRN (F1@1 0.624 vs 0.867) |

### C.8 Evaluation tooling

| tool | repo | licence | framework / pins | last push | verdict |
|---|---|---|---|---|---|
| OpenLane-V2 devkit (metrics) | OpenDriveLab/OpenLane-V2 | Apache-2.0 | devkit 2.1.0; install_requires numpy >= 1.22.0, < 1.24.0; scipy == 1.8.0; ortools == 9.2.9972; shapely == 2.0.0; similaritymeasures; scikit-learn; opencv-python; openmim; ninja; jupyter; no python_requires; baseline plug-in tested Python 3.8.15, PyTorch 1.9.1, GCC 5.4.0 on mmdet3d v1.0.0rc6 | 2025-07-02 | light port: vendor `openlanev2/centerline/evaluation/evaluate.py` and relax the pins (S - M) |
| nuScenes devkit | nutonomy/nuscenes-devkit | GitHub: NOASSERTION (custom); README gives no name | tested Python 3.9 and 3.12 (v1.2.0, 2025-08-28), pip package | 2026-08-06 | runs as is |
| Argoverse 2 API | argoverse/av2-api | MIT | conda + Rust files; versions not on page | 2026-02-19 | not needed |
| NAVSIM | autonomousvision/navsim | Apache-2.0 | conda env from environment.yml (versions not shown); needs nuPlan maps and OpenScene data | 2025-10-27 | not usable on nuScenes |
| nuPlan devkit | motional/nuplan-devkit | README badge Apache-2.0; GitHub: NOASSERTION | Python 3.9 badge; requirements_torch.txt | 2025-08-27 | not usable on nuScenes |
| OpenScene | OpenDriveLab/OpenScene | Apache-2.0 code; data CC BY-NC-SA 4.0 + nuPlan non-commercial agreement | -- | 2026-01-13 | not needed |

### C.9 Verdict summary

- runs as is: GaussianCaR, nuScenes devkit, rMCL toy code (aMCL code runs but has no licence).
- light port (named blocker): OpenLane-V2 evaluation (pins), CGNet `topo_eval` metric code (old networkx / scipy), Deformable DETR (grid_sample path), DAB-DETR, DINO (compiled ops), CLRNet and LaneATT (NMS build, old PyTorch), L-CNN, HAWP (old PyTorch / ops), SAM-Road (graph head only), Relationformer (custom op, unverified), MapTR Chamfer-AP code.
- port ideas only: RNGDet(++), CenterLineDet, iCurb, RNTR, LaneGraph2Seq, SeqGrowGraph, DeH4R, Sat2Graph, InstaGraM, TopoNet, TopoMLP, LaneSegNet, TopoLogic, CGNet, TopoPoint, TopoFG, UniTopo, MapTR, StreamMapNet, SQD-MapNet, MapTracker, TopoStreamer, Sparse4D, StreamPETR, UniAD, VAD, SparseDrive, PlanTF, PLUTO, Diffusion Planner, BEVFusion, CRN.
- no code: DAGMapper, TopoBDA (model), Topo2Seq, SMART, LaneGAP (standalone repository).
- Blocker patterns: (1) mmcv-full 1.x / mmdet 2.x / mmdet3d 1.0.0rc6 with PyTorch 1.9 (mmcv 1.x has no build for PyTorch 2.8 - from memory); RNTR 2.0.1 needs the mmengine / mmcv 2.x / mmdet 3.x / mmdet3d 1.4 stack with PyTorch 2.0.1 (mmcv 2.x has no wheel for 2.8 - from memory); (2) custom CUDA ops (GKT, deformable attention, bev_pool, NMS) compiled against old PyTorch; (3) image-feature BEV encoders (LSS, BEVFormer, GKT) that we do not need because our features are cached; (4) licences (Part D).

---

## 6. PART D - What the licences imply (plain statements of what the pages say; not legal advice)

| artefact | what the pages say | plain implication for what we publish |
|---|---|---|
| nuScenes data and map expansion | the nuScenes paper says CC BY-NC-SA 4.0 (non-commercial research use); a later paper by the owners says free for academics and startups with commercial licences for larger partners; a Motional page says free for academic use; search snippets of the site say strictly non-commercial, commercial users to contact the owners; the official terms page was not readable | Metrics, figures and text based on it can be published for research with attribution. Derived data (our graph labels, cached BEV features of nuScenes frames) are adaptations of non-commercial data: share under the same terms if at all, not for commercial use. Whether raw-data redistribution is allowed: not found. Check the official terms before releasing features or labels |
| OpenLane-V2 data | CC BY-NC-SA 4.0; users must also accept the nuScenes and Argoverse 2 terms | same as above; also requires the user's permission to download (project rule) |
| OpenLane-V2 code | Apache-2.0 | may be copied / vendored with notice (permissive); pins make it uninstallable unchanged |
| Argoverse 2 | data and documentation CC BY-NC-SA 4.0; code MIT; non-commercial defined; no re-identification | as nuScenes |
| nuPlan / OpenScene / NAVSIM | OpenScene data CC BY-NC-SA 4.0 + nuPlan Dataset License Agreement for non-commercial use; code Apache-2.0; NAVSIM "datasets inherit their own licences" | non-commercial only |
| SpaceNet | CC BY-SA 4.0 | attribution and share-alike; commercial use not excluded by the page |
| CityScale, Global-Scale | Sat2Graph README: dataset and models non-commercial academic use only; Global-Scale: no licence statement; imagery from a map provider's API, labels from OpenStreetMap | treat as non-commercial; the imagery terms are not on the pages read |
| Permissive code (MIT, Apache-2.0): MapTR, RoadNet, SAM-Road, Sparse4D, SparseDrive, CRN, BEVFusion, GaussianCaR, VAD, UniAD, TopoNet, TopoMLP, LaneSegNet, CGNet, DAB-DETR, Deformable DETR, DINO, CLRNet, LaneATT, L-CNN, HAWP, Relationformer, rMCL, OpenLane-V2 devkit, NAVSIM, OpenScene code, av2-api, Sat2Graph code | MIT / Apache-2.0 named on the pages | code may be reused and redistributed if the notices are kept (from memory of these licences, not on the pages) |
| GPL-3.0 code: RNGDet++, CenterLineDet, Topo-boundary, StreamMapNet, SQD-MapNet, InstaGraM (MapTracker: README says GPL, detector says NOASSERTION) | GPL-3.0 named on the pages; RNGDet++ adds academic-only wording | copying their code into our repository would make the combined work subject to GPL-3.0 when distributed (from memory of the licence text); re-implementing an idea from the paper is not copying. Safe route: read, re-implement, cite |
| No licence file: TopoLogic, TopoPoint, TopoFG, UniTopo, TopoStreamer, SeqGrowGraph, DeH4R, aMCL, PlanTF, PLUTO, Diffusion Planner, SAM-Road++ | no licence on the page and "null" in the GitHub API | no permission to copy is stated (from memory: the default under copyright law is "all rights reserved"); use as a reading source only |
| Custom / unclear licence: nuScenes devkit, nuPlan devkit, StreamPETR, MapTracker | GitHub detector reports NOASSERTION; READMEs name Apache-2.0 for nuPlan, none for the nuScenes devkit | read the licence file before vendoring any file |
| Pretrained weights | Sat2Graph: non-commercial academic only; RNGDet++: not for commercial purposes; others: no separate weight licence on the pages | weights trained on non-commercial data follow the data terms; do not assume more than the repository licence |
| Papers | arXiv listings show a non-exclusive licence; GaussianCaR's arXiv page lists CC BY-NC-SA 4.0 for the paper, Apache-2.0 for the code | cite, do not reproduce long passages or tables |

---

## 7. Cluster compatibility notes

(a) Interface contracts for evaluation
- What the literature expects from a method: (1) a set of instances, each with a point array of fixed length (11 / 20 / 100) and a scalar confidence, plus an adjacency score matrix (OpenLane-V2, MapTR-type); (2) a point graph that can be walked (TOPO / JTOPO / GEO / APLS); (3) landmarks and paths (RNTR family); (4) a trajectory with time stamps (NAVSIM, nuPlan, UniAD / VAD).
- What we have: vertices (ego metres), a 128-d token, validity / keep probability, edges, plus a route probability and a planner path. All four contracts can be reached through adapters G1 - G6 (B.8); (4) needs a speed profile and is only partly meaningful on nuScenes.
- Output interface for training-time blocks stays unchanged; evaluation adapters live outside the network.

(b) Known adapters and conflicts
- graph -> pieces (G2): conflict-free, but the cut rule and the confidence aggregation change the numbers; fix and document once.
- confidence: OpenLane-V2 TOP needs probabilities calibrated around 0.5; our keep gate (0.872 at E120) and edge scores should be checked for calibration before TOP_ll is reported.
- direction: Frechet-based and directed metrics penalise reversed travel direction; Chamfer-based AP does not. Report both, since direction accuracy is one of our reported metrics.
- z: OpenLane-V2 uses 3D points; append z = 0 (subset B has z = 0 itself).
- radar: only nuScenes; any metric that needs images from other datasets is irrelevant.
- Conflict: PDMS / EPDMS / CLS need agents, a simulator and nuPlan maps; not available for our frames.

(c) Combinations the papers demonstrate
- TopoLogic's geometric link score is added to TopoNet / TopoMLP style detectors (post hoc); UniTopo builds on TopoNet, TopoMLP and TopoLogic code; TopoStreamer builds on LaneSegNet; CGNet reuses MapTR's GKT op; SQD-MapNet and MapTracker build on StreamMapNet; LaneGAP was released together with MapTRv2 code; DeH4R and SAM-Road share the SAM-based CityScale / SpaceNet pipeline; SeqGrowGraph, RNTR and LaneGraph2Seq share the landmark / reachability protocol (fudan-zvg/RoadNet holds RNTR and LaneGraph2Seq; MIV-XJTU/SeqGrowGraph holds configs for all three plus the evaluation code).
- No paper evaluates a graph-growth method with OpenLane-V2 OLS and the landmark protocol on the same output; if we want both, the adapters are ours.

---

## 8. Reuse versus re-implement

| item | decision | reason / effort |
|---|---|---|
| OpenLane-V2 DET_l / TOP_ll evaluation | vendor `evaluate.py` of master (x.1 only), relax pins, add our G2 / G3 adapters | Apache-2.0 code, plain numpy / scipy; effort S - M |
| Chamfer AP (MapTR type) | re-implement (30 lines) | MIT code, but wrapped in mmdet3d dataset classes; effort S |
| Landmark / reachability | re-implement from B.4; read the SeqGrowGraph evaluation files as a reference only | reference code has no licence and needs the repository's own graph classes; effort S - M; implement both the paper variant (one-to-one) and the released-code variant (nearest, non-one-to-one) and say which is reported |
| TOPO / JTOPO / GEO / SDA / APLS | vendor CGNet `topo_eval/` (Apache-2.0, pure python / networkx) with fixes, or re-implement from B.3; choose one parameter set and state it | the LaneGAP text (0.15 / 0.45 / 7.5 m) and the CGNet code (0.2 / 0.8 / 40 m, inferred units) differ; the older aerial reference code is Python 2 era (CenterLineDet uses a `docker_py2` evaluation image); effort S - M |
| mAS | re-implement with a fixed tau and the intended localisation decay | formula inconsistency in the paper; effort M |
| Deformable-DETR / DAB-DETR / DINO ideas | use the pure-PyTorch path (grid_sample) or our own bilinear sampling | we already sample the pyramid bilinearly |
| multiple-choice learning (rMCL, aMCL) | re-implement the loss (S) | tiny; aMCL repository has no licence |
| SAM-Road graph head | consider as a baseline for the "spray and link" family | MIT; effort M; encoder replaced by our pyramid |
| mm*-based lane-topology repositories | read, do not run | environment conflicts and licences (Part D) |
| GaussianCaR | keep as the foundation | runs in our environment |

Evaluation additions in order of value per effort: (1) location-overlap check of our split (S); (2) landmark P / R / F at 0.5 - 5 m next to the 6 m point (S); (3) Chamfer AP on pieces (S); (4) lane-only OLS-lite with the x.1 definition (S - M); (5) TOPO / JTOPO with LaneGAP parameters (M); (6) APLS (S - M); (7) stability metric (M).

---

## 9. Open questions and gaps

- nuScenes official terms (research and commercial, map expansion, redistribution of derived features): page unreadable; needs a manual look by the user.
- Whether OpenLane-V2 subset B frames can be matched one to one to nuScenes sample tokens (the info files carry `source`, `source_id`, `timestamp`; the matching rule is not on the pages read) and how many of our 255 scenes lie in subset B val.
- The exact OpenLane-V2 rule for cutting lanes at intersections (merged chains vs lane / connector boundaries).
- The exact mechanism of the old-TOP inflation (TopoMLP section 4.6 and PR #77 describe the fill value; the full argument was not reproduced).
- UniTopo, TopoLogic README tables: which devkit version each table uses.
- TOPO matching radius in the original aerial definition; GEO radius in CGNet (supplement not opened).
- nuPlan score weights and CLS-NR / CLS-R definitions (page silent; from memory only); NAVSIM comfort thresholds.
- Licence of the map expansion alone; licence of imagery in CityScale and Global-Scale.
- Last-commit dates: `pushed_at` is the last push to any branch, not necessarily to the default branch.
- Which repositories use mmcv ops only in the BEV encoder (and could run decoder-only) was not audited at import level.
- SeqGrowGraph evaluation: the node-degree rules behind `get_nodechains_dpt` and the grid interval (`xbound`) were in classes not read, so the metre conversion of the code thresholds is an inference; whether the paper tables were produced with `roadnet_reach_dist_eval_new.py` or one of the older variants (`_aaai.py`, plain) is not stated on the pages read.
- Whether LaneGAP's TOPO / JTOPO code is public: the standalone repository is empty, the listing of the MapTR `maptrv2` datasets folder shows no such file (listing was partial). CGNet's repository does contain its own TOPO / JTOPO / SDA / APLS code (B.3), but whether it is the code behind the published LaneGAP numbers is unknown (the parameters differ from the LaneGAP text).
- nuScenes licence: the "free for academics and startups" wording (Revisited paper) versus "strictly non-commercial" (site snippet) is unresolved.

---

## 10. Search log

Fetches (WebFetch) and searches (WebSearch), one line each; GitHub-API queries use `api.github.com/search/repositories?q=repo:...`.
1. WebFetch nuscenes.org/terms-of-use (empty); nuscenes.org/nuscenes (empty); nuscenes.org/nuimages (empty); nuscenes.org/nuplan (empty); web.archive.org/... (blocked)
2. WebSearch "nuScenes terms of use non-commercial CC BY-NC-SA 4.0 Motional commercial license map expansion"; "nuScenes Terms of Use non-commercial use Motional AD LLC" (allowed domains nuscenes.org, github.com, motional.com)
3. WebFetch github.com/nutonomy/nuscenes-devkit; raw docs/schema_nuscenes.md; raw map_api.py; motional.com news page; AWS open-data registry yaml; arXiv 1903.11027
4. WebFetch github.com/OpenDriveLab/OpenLane-V2 (README); raw docs/metrics.md; issue #76; PR #77 files; raw evaluate.py (master); raw evaluate.py at tag v2.0.0; raw docs/submission.md, devkit.md, annotation.md, getting_started.md, statistics.md; raw data/README.md; raw requirements.txt, setup.py; docs tree listing; arXiv HTML 2304.10440
5. WebFetch argoverse.org/about.html; arXiv HTML 2301.00493; github.com/argoverse/av2-api
6. WebFetch Sat2Graph, RNGDetPlusPlus, RNGDet (404), SAM-Road, SAM-Road++ (earth-insights/samroadplus), DeH4R, CenterLineDet, iCurb, Relationformer, InstaGraM READMEs; spacenet.ai roads page; arXiv HTML 2411.16733; WebSearch "Global-Scale road network dataset ... Sat2Graph City-scale ... SpaceNet"
7. WebFetch NAVSIM README, raw docs/metrics.md, raw docs/install.md, raw pdm_scorer.py, arXiv HTML 2406.15349; nuPlan devkit README, readthedocs metrics_description (metrics_intro 404), arXiv HTML 2106.11810; OpenScene README; WebSearch "nuPlan simulation score weighted average ..." 
8. Metric sources: arXiv HTML MapTR 2208.14437; raw MapTR tpfp_chamfer.py and mean_ap.py; StreamMapNet arXiv HTML 2308.12570; RNTR arXiv HTML 2402.08207; SeqGrowGraph arXiv HTML 2507.04822v1; LaneGAP arXiv HTML 2303.08815; CGNet arXiv HTML 2407.11337; LaneGNN ar5iv 2302.06175; STSU ar5iv 2110.01997; APLS ar5iv 1807.01232 (PDF version unreadable); Sat2Graph ar5iv 2007.09547; TopoMLP arXiv HTML 2310.06753; stability arXiv HTML 2510.10660; WebSearch "lane graph extraction JTOPO junction TOPO split detection accuracy SDA"
9. Repositories fetched: TopoNet, TopoMLP, LaneSegNet, TopoLogic, CGNet, TopoPoint, TopoFG, UniTopo, LaneGAP (README raw + tree), MapTR (+ raw docs/install.md), StreamMapNet, MapTracker, SQD-MapNet, TopoStreamer, Sparse4D, StreamPETR, DAB-DETR, Deformable-DETR (+ raw ms_deform_attn_func.py), DINO, CLRNet, LaneATT, lcnn, hawp, UniAD, VAD, SparseDrive, planTF, pluto, Diffusion-Planner, gaussiancar, bevfusion, CRN, fudan-zvg/RoadNet (+ RoadNetwork-2.0.1 tree + raw get_started.md), code-rMCL, annealed_mcl
10. WebSearch for repositories not found or unclear: "SeqGrowGraph ... github"; "LaneGraph2Seq github"; "TopoPoint github Franpin"; "TopoFG ... github"; "UniTopo ... github"; "TopoBDA ... github"; "Topo2Seq ... github"; "SMART scalable map prior ... github"; "TopoStreamer ... github"; "GaussianCaR ... github" (then arXiv abs / HTML for the link); "InstaGraM github"; "DAGMapper github"; "Resilient Multiple Choice Learning rMCL github"; "Annealed Multiple Choice Learning ... code"; "github Victorletzelter rMCL"
11. WebFetch junjieye.com/smart (SMART project page, after redirect), artest08.github.io/TopoBDA.github.io, arXiv abs 2602.08784 / 2311.01052 / 2407.15580, github.com/Victorletzelter?tab=repositories
12. GitHub-API search queries (8 queries) for licence and `pushed_at` of about 50 repositories; later: name searches `SeqGrowGraph OR Topo2Seq OR TopoBDA OR DAGMapper OR LaneGraph2Seq OR TopoStreamer in:name` (found MIV-XJTU/SeqGrowGraph, YimingYang23/TopoStreamer, artest08/TopoBDA.github.io only) and a broad description query (useless, 26,100 hits).
13. CGNet evaluation code: api.github.com contents listings of `projects/mmdet3d_plugin/datasets` and its `topo_eval` and `map_utils` subfolders; raw `topo_eval/geotopo.py`, `topo_evaluator.py`, `topo_metrics.py`, `apls.py` (imports and defaults only), `requirement.txt`.
14. Late additions: WebFetch github.com/MIV-XJTU/SeqGrowGraph; api.github.com git tree of that repository; raw `seq_grow_graph/transforms/roadnet_reach_dist_eval_new.py` and `seq_grow_graph/nus_reach_metric.py` at commit c148057 (a guessed path `nus_metric_new.py` on main returned 404); ar5iv 1903.11027 and arXiv HTML 2512.02448 (nuScenes licence and size statements); extended WebSearch on the nuScenes terms; MapTR `maptrv2` datasets folder listing; api.github.com code search (HTTP 401, needs authentication); CVF page of SeqGrowGraph (HTTP 403).
