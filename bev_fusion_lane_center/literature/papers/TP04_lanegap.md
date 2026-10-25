# TP04 - LaneGAP (path-wise modelling, Path2Graph)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M+C (landing page only, no technical content)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S7 S6 E.*

## Papers

Lane Graph as Path: Continuity-preserving Path-wise Modeling for Online Lane Graph Construction (Liao et al., ECCV 2024) https://arxiv.org/abs/2303.08815

## Idea

Instead of predicting pieces (segments between junctions) and then linking them, predict whole root-to-leaf PATHS of the directed lane graph with a MapTR-like decoder and recover the graph with a simple merge step; forks and merges then appear where paths share vertices, and no link head exists.

## Formulation

Graph2Path (labels): find roots (in-degree 0) and leaves (out-degree 0), DFS to every root-to-leaf path. Path2Graph (inference): discretise each predicted path to points; add points as vertices and consecutive points as edges; merge vertices of different paths closer than 0.15 m (= the graph interpolation step); the merged vertex inherits predecessors and successors. Decoder: MapTR-style, 6 deformable layers, learnable path queries (count not stated), polyline default with 30 points (coordinates normalised to (0,1)); Bezier alternative B = Gamma x V (Bernstein matrix). Loss: Hungarian; L = L_cls (focal) + L_path (sum of L1 over matched points) + L_auxseg (CE on BEV segmentation, train only), unit weights as written. Metrics: TOPO (precision / recall / F1 of the directed graph; vertices every 0.15 m, match threshold 0.45 m, traversal radius 7.5 m) and Junction TOPO (same, restricted to sub-graphs around GT junctions).

## Input

six cameras, ResNet-50 + GKT, BEV 200 x 100 at 0.3 m.

## Output

set of ordered paths -> directed graph.

## Training

Hungarian one-to-one; 110 epochs on nuScenes (AdamW 6e-4 cosine, batch 32, 8 RTX 3090), 24 epochs for ablations.

## Differentiable?

partial - Hungarian; Path2Graph is a rule (merge by distance), no gradient.

## Sensors and data

camera (also LiDAR-only and fusion variants); nuScenes x in [-15, 15], y in [-30, 30] m; Argoverse 2; OpenLane-V2 (mIoU).

## Reported results

nuScenes (Junction F1 / TOPO F1, directed): HDMapNet 0.507 / 0.513, STSU 0.419 / 0.395, MapTR 0.496 / 0.465, LaneGAP 0.564 / 0.529 (110 ep, 15.6 FPS). 24 ep ablations: aux seg 0.529 / 0.450 -> 0.539 / 0.461; polyline points 10 / 30 / 40 -> junction F1 0.532 / 0.539 / 0.557; Bezier points 3 / 5 / 10 -> 0.481 / 0.510 / 0.501. Modality 24 ep (junction / TOPO): vision 0.539 / 0.461, LiDAR 0.600 / 0.509, vision + LiDAR 0.615 / 0.531. Post-processing on GT inputs (Table 10): pixel-wise (HDMapNet type) 0.773 / 0.788, piece and path 1.000 / 1.000 (lossless). Large range 50 x 100 m: junction 0.392, TOPO 0.388. Planning with VAD: average L2 0.70 m (LaneGAP), 0.74 (MapTR), 0.78 (no lane graph).

## Compute

35.9 M parameters, 16.5 FPS network only / 15.6 FPS end-to-end.

## Code

https://github.com/hustvl/LaneGAP (landing page only, Apache-2.0; the path-wise centerline configs are announced in the news of https://github.com/hustvl/MapTR, branch `maptrv2`, MIT per its README page, mmdetection3d based; branch not opened)

## Fit to our project

reuse_idea - effort S - Path2Graph is exactly our tree -> DAG conversion by clustering (0.15 m) and the Junction-TOPO metric is a ready junction-level evaluation next to our split / merge F1; Graph2Path gives path-wise labels from our piece graph.

## Needs

ordered point sequences per path; a merge threshold; paths covering every edge.

## Conflicts

number of root-to-leaf paths grows multiplicatively with forks (path count per scene not reported; unknown at +-50 m); a path view duplicates shared edges, so per-edge validity / route attributes need de-duplication after merging.

## Tried in our experiments

none (the tree -> DAG step of the hypothesis tree is the analogue; see `Hypothesis_Tree_Spec_and_Tests_EN.md`).
