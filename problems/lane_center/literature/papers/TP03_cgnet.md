# TP03 - CGNet (junction-aware queries, Bezier-space continuity, iterative adjacency; nuScenes centerline graph)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked (supplement not opened; image resolution and pair-scoring details not stated; the paper does not say which points count as junctions (fork, merge, endpoint), whether R = 9 is in cells, or whether the heat map is output at inference; JTOPO radius not given; SDA = junction localisation within radius R). Pipeline stage: S6 S1 L.*

## Papers

Continuity Preserving Online CenterLine Graph Learning (Han et al., ECCV 2024) https://arxiv.org/abs/2407.11337

## Idea

Three additions to a MapTR-style centerline-segment decoder on nuScenes: (JAQ) a junction heat-map branch whose features enhance the positional part of the queries; (BSC) a training-only head that decodes the Bezier curve of a CONNECTED PAIR of segments from the concatenated embeddings (continuity at joints); (ITR) an adjacency matrix refined once per decoder layer by GCN + GRU.

## Formulation

N = 50 queries (content Q^c + positional Q^p), each one segment of n = 20 ordered points; 6 deformable layers; GKT BEV. JAQ: F_jp = conv-BN stack of F_bev; heat map Z, focal loss with GT junction map dilated by radius R = 9; Q~p = LinearAttention(Q^p, F_jp, F_jp), concatenated with Q^c. BSC: for connected matched pairs (E_i, E_j): E^ = B(E_i concat E_j), B = pseudo-inverse of the Bernstein matrix (m x 2n); MLP predicts control points (default 4); L_bezier = L1 to least-squares-fitted control points of the joined GT polylines; removed at inference. ITR: Lambda^l = GCN(E^l, A^(l-1)); A^l = MLP(GRU(Lambda^l, Lambda^(l-1))); A^0 = 0; 6 iterations, weights shared (pair scoring inside the MLP not detailed). Loss: 2 L_cls + 5 L_poly + 1 L_topo + 0.005 L_dir + 0.01 L_bezier + 0.1 L_ja; Hungarian (focal + L1); L_topo = BCE averaged over the T^2 adjacency entries, no class balancing. CODE AS READ (CGNet_head.py): hierarchical MapTR-style point queries (num_query = vectors x points per vector, instance + point embedding); junction GT = the dataset's line.key_points rendered as 1 in a BEV-sized map, dilated with a 9 x 9 elliptical kernel (cv2), CornerNet-style modified focal loss (exponents 2 and 4), loss weight 0.1 (which points are "key points" is decided in the dataset code, not visible); topology branch per decoder layer: vertex feature = mean of the point embeddings of a query, one GNN step (adjacency and its transpose, edge weight 0.8), GRU across layers, then TopologyHead = per-vertex MLP + concatenation of every pair + association classifier + sigmoid (a pairwise MLP; the GRU does not output the matrix directly); loss = nn.BCELoss on the submatrix of MATCHED queries only (sm_pred[pos_inds][:, pos_inds] against gt[pos_gt_inds][:, pos_gt_inds]), weight 1, i.e. valid-pairs-only supervision (the setting RATopo later shows to be better than full supervision); BSC: all matched pairs (i, c) with target 1: concatenate the point embeddings of both queries (2n of them), multiply by the 4 x 2n pseudo-inverse Bernstein matrix (cubic, 4 control points) to get 4 control-point embeddings, MLP(C, C/2, 2) -> control points; inference returns the adjacency as is (threshold in the bbox coder, not seen).

## Input

six cameras -> BEV 100 x 200 cells (0.3 m; x in [-15, 15] m, y in [-30, 30] m).

## Output

50 segments (20 points) + 50 x 50 adjacency probabilities; junction heat map (aux).

## Training

Hungarian one-to-one; 24 epochs (ablations) or 110; AdamW 6e-4 cosine, batch 32, A100.

## Differentiable?

partial - Hungarian assignment; BSC and JAQ are differentiable.

## Sensors and data

camera; nuScenes and Argoverse 2 centerline graphs (point-level metrics GEO / TOPO / JTOPO F1, APLS, SDA; segment-level IoU, mAPcf, DET_l, TOP_ll).

## Reported results

nuScenes 24 ep (GEO F1 / TOPO F1 / JTOPO F1 / APLS / SDA): CGNet 54.7 / 42.2 / 34.1 / 30.7 / 8.8, MapTR 53.3 / 39.7 / 32.4 / 26.8 / 7.6, TopoNet 50.8 / 39.6 / 31.9 / 26.2 / 5.6; 110 ep CGNet 63.9 / 53.2 / 43.3 / 41.4 / 14.5. Module ablation (TOPO / JTOPO / APLS / SDA): base 39.7 / 32.4 / 26.8 / 7.6; +JAQ 40.5 / 33.1 / 28.5 / 7.7; +BSC 41.4 / 33.4 / 29.1 / 8.3; +ITR 40.1 / 32.7 / 28.6 / 7.8; all 42.2 / 34.1 / 30.7 / 8.8. Control points 4 / 6 / 8: TOPO 42.2 / 40.8 / 40.4. ITR iterations 1 / 3 / 6: 39.6 / 41.3 / 42.2. Dilation R 7 / 9 / 11: 40.4 / 42.2 / 40.9.

## Compute

38.8 M parameters, 241.9 GFLOPs, 11.2 FPS (RTX 3090); MapTR baseline 36.0 M, 216.7 G, 12.7 FPS.

## Code

https://github.com/XiaoMi/CGNet (README: Apache-2.0, bundles an mmdetection3d folder, compiled GKT op; Python 3.8)

## Fit to our project

reuse_idea - effort S - junction heat map as feature enhancement is the principle of our key-point head; the Bezier-continuity auxiliary is the principle of our connectors (E53). The decoder and adjacency GRU are tied to a 50-query set.

## Needs

a set of matched segment embeddings (pairs S) for BSC; junction GT points; segments with ordered points.

## Conflicts

Hungarian one-to-one matching; N = 50 queries on 30 x 60 m only (cannot be assumed at +-50 m); BSC needs pairs of pieces, not vertices.

## Tried in our experiments

E44 (learned key points) and E53 (Bezier connectors): kept, large gain (split / merge F1 0.34 / 0.38 -> 0.71 / 0.64).
