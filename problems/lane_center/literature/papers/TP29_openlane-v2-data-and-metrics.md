# TP29 - OpenLane-V2: data and metrics

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: partly (the evaluation file of the master branch was read through a digest, version of that file not confirmed as v2.1; supplements not read). Pipeline stage: D E.*

## Papers

OpenLane-V2: A Topology Reasoning Benchmark for Unified 3D HD Mapping (Wang et al., NeurIPS 2023 D&B, arXiv 2304.10440 v3) https://arxiv.org/abs/2304.10440 ; repository https://github.com/OpenDriveLab/OpenLane-V2 (issue #76 for the metric fix)

## Idea

Two subsets (A: Argoverse 2, B: nuScenes), 1,000 scenes each (700 / 150 / 150 train / val / test segments), 2 Hz keyframes; per frame about 24 - 26 centerlines (90 % of frames > 10 centerlines), average 1.8 - 1.9 connections per centerline (up to 7 at complex junctions); range x +-50 m, y +-25 m; each lane stored with 201 points and evaluated with 11. Topology is a per-frame directed adjacency matrix: (i,j) positive iff the end point of lane i meets the start point of lane j; lanes with a single predecessor / successor are merged, lanes are split only at intersections, forks and merges (= a junction-vertex graph with whole pieces as edges). Subset B has z = 0.

## Formulation

DET_l = mean AP over discrete Frechet-distance thresholds {1, 2, 3} m, relaxed with distance to ego; DET_t IoU 0.75 over 13 attributes; TOP_ll / TOP_lt = vertex-level mAP on matched graphs: predictions are matched to GT by Frechet distance (per threshold 1 / 2 / 3 m); per GT vertex the candidate neighbours with confidence > 0.5 are ranked, precision is accumulated at every hit and divided by the true neighbour count (0 if one side is empty, 1 if both are); averaged over rows and columns (directed for lane-lane), thresholds and samples. Master-branch code as read through a digest (`_mAP_topology_lclc`): predicted lanes that match no GT are dropped from the matrix (no dummy vertices), cells of unmatched GT lanes get the fallback (1 - gt) x (0.5 + eps), i.e. 0 where an edge exists (a miss) and just above the cut-off where none exists; there is no explicit N_TP / (N_TP + N_FP) factor. Consequence: TOP measures link quality among MATCHED lanes only; false-positive lanes are punished by DET_l, not by TOP. OLS = mean of DET_l, DET_t, sqrt(TOP_ll), sqrt(TOP_lt) (sqrt on the 0-1 scale, reproduced numerically by several papers). Versions: devkit v1.0 / v2.0 (loophole: unmatched predictions get confidence 1.0) versus v1.1 / v2.1 (fixed; README recommends the .1 metrics; lane-segment tasks TOP_lsls and the OLUS score only in v2.x). Other nuScenes centerline-graph metrics: CGNet / LaneGAP / LaneDiffusion use point-level TOPO F1, JTOPO F1 (junction sub-graphs), APLS, SDA with BEV 30 x 60 m; RNTR / SeqGrowGraph use landmark and reachability P/R/F; TopoGPT G-F1, T-F1, JT-F1.

## Input

n/a

## Output

n/a

## Training

n/a

## Differentiable?

n/a

## Sensors and data

subset A 7 cameras 2048 x 1550 (Miami, Pittsburgh, Detroit, Washington D.C., Austin, Palo Alto; night 3.0 %, rain 1.1 %); subset B 6 cameras 1600 x 900 (Boston 55 %, Singapore 45 %; night 11.7 %, rain 17.4 %). Licence of the data CC BY-NC-SA 4.0, code Apache-2.0 (README).

## Reported results

TopoNet baseline (old metric) A: OLS 35.4, DET_l 29.2, TOP_ll 4.1; B: 33.2, 24.3, 2.5.

## Compute

n/a

## Code

see papers: devkit in the repository.

## Fit to our project

reuse_idea - effort S - contract: our piece graph (nuScenes map expansion, travel direction) contracts to an OpenLane-V2 style junction-vertex graph by merging degree-(1,1) vertices; DET_l / TOP evaluation could then be run on our output as an external comparison; the TOP definition (ranking of neighbours) is not what our split / merge F1 measures.

## Needs

piece polylines with 11 points; adjacency by end / start coincidence.

## Conflicts

TOP depends on the metric version; lane-segment labels exist only for subset A; train / val geographic overlap inflates results (SMART, SCORE show 20 - 30 % drops on a geo-disjoint split).

## Tried in our experiments

none
