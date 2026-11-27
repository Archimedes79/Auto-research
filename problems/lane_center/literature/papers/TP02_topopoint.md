# TP02 - TopoPoint (endpoint detection coupled to lane tokens)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked (paper and head file; digest flagged typos: Eq. 4 and 8, lambda_pl missing, init order of alpha / lambda ambiguous; paper text and released head file differ on the snapping step). Pipeline stage: S1 S6.*

## Papers

TopoPoint (Fu et al., arXiv 2025, arXiv:2505.17771; the page title was not seen) https://arxiv.org/abs/2505.17771

## Idea

The de-duplicated start / end points of all ground-truth centerlines are made a separate detection target with their own 200 point queries (= the vertices of the piece graph). Points and lanes live in one 6-layer decoder; a self-attention with an additive distance bias couples them, a bipartite GCN passes features along the predicted point-on-lane incidence, and at inference lane endpoints within 1.5 m of a detected point are averaged with it.

## Formulation

queries N_p = 200 (3D learnable reference points), N_l = 300, N_t = 100. PLMSA: logits of concat[Q_p; Q_l] get the additive bias [[0, M_pl], [M_pl^T, M_ll]], M = f_map(D), f_map(x) = exp(-x^p / (lambda * sigma)) (the TopoLogic mapping); D_ll = sum |end_i - start_j| over predicted lanes of the previous layer; D_pl = min(sum |p_i - start_j|, sum |p_i - end_j|); no bias between points. PLGCN: A_pl = l1 G_pl + l2 M_pl; Q_p += GCN(Q_l, A_pl); Q_l += GCN(Q_p, A_pl^T); GCN(X, A) = sigmoid(norm(A) X W). Heads: G_pl = sigmoid(MLP(Q_p) MLP(Q_l)^T) (point lies on lane), G_ll and G_lt analogous. PLGM (inference only): keep points and lanes with score > 0.3; for each point collect lane endpoints within delta = 1.5 m; replace them by E = (P + sum of matched endpoints) / (n + 1) (greedy, no Hungarian). Loss: lt, lp, ll weights 1, topology weights 5 (lpl missing); focal + L1; DET_p = mean AP over thresholds {1, 2, 3} of point distance. CODE AS READ (topopoint_head.py; the data pipeline, transformer and relation heads are in other files): points and lanes are matched by the same HungarianAssigner (cost cls 1.0 + L1 5.0 + GIoU 2.0); classification loss in this file is softmax cross-entropy with background weight 0.1 and L1 regression weight 5.0 (not focal; topology loss classes not in the file); GT endpoints: in the dataset file (used for evaluation) the first and last point of every GT centerline (11-point version) are collected and de-duplicated by EXACT equality (torch.unique; OpenLane-V2 pieces share identical coordinates at joints), no class label, single-lane endpoints are kept (a filter keeping only points shared by several lanes is commented out); the training-side point labels and the point-lane adjacency come from the pipeline, not visible; point-lane target = GT point-lane adjacency scattered into a (points x lanes) zero matrix by the Hungarian indices and supervised on the FULL matrix, while the lane-lane loss is computed on the matched sub-matrix only (valid-only supervision, the variant RATopo later shows to be better than full supervision); defaults num_query = num_pts_query = 100 (config values not seen). The inference snapping in get_lanes does NOT use detected points: it keeps lanes with score > 0.3, scores links by exp(-d^2 / 11.5275) with d = L1 distance end_i -> start_j, keeps links above 0.7 (d below about 2.0 m), then runs two passes that replace the end of lane i and the starts of its linked lanes by their arithmetic mean (second pass on the transposed links averages starts with ends), and recomputes the topology; the 1.5 m point threshold of the paper text does not appear in this file.

## Input

BEVFormer BEV 200 x 100 x 256, front-view features for traffic elements.

## Output

200 3D points, 300 lanes (11 x 3), incidence N_p x N_l, adjacency N_l x N_l, lane-traffic matrix.

## Training

end-to-end, 24 epochs, AdamW 2e-4 cosine, R50, multi-view x0.5 resolution, 8 V100; assignment method not stated (DETR-type from memory).

## Differentiable?

partial - set assignment (not stated) and the PLGM snapping (post-process, no gradient).

## Sensors and data

camera; OpenLane-V2 A and B; BEV 200 x 100.

## Reported results

subset A ablation (TOP_ll / DET_p / OLS): reproduced baseline 23.4 / 44.5 / 43.4; +FVScale 23.8 / 44.8 / 46.0; +PLMSA 27.2 / 49.8 / 47.6; +PLGCN 28.0 / 51.8 / 48.3; +PLGM 28.7 / 52.6 / 48.8. Main: A OLS 48.8, TOP_ll 28.7 (TopoLogic 44.1, 23.9); B OLS 49.2, TOP_ll 28.3, DET_p 45.1 (TopoLogic 42.3, 21.6, 39.2). Query counts (N_p, N_l) 100/200 -> 200/200 -> 200/300 -> 300/300: DET_p 49.7, 51.8, 52.6, 51.4.

## Compute

not stated.

## Code

https://github.com/Franpin/TopoPoint (README: PyTorch 1.9.1, CUDA 11.1; no licence line; no checkpoints; data per OpenLane-V2 v1.0.0, metrics v2.1.0)

## Fit to our project

reuse_idea - effort M - principle (explicit vertices, distance-biased coupling of vertex and edge tokens, snapping) maps onto the key-point head and the DAG clustering; the query decoder itself does not.

## Needs

lane pieces with start / end; a point detector; incidence labels; Hungarian-type matching for the point set (not stated).

## Conflicts

needs a joint token decoder with self-attention (not a dense heat map); thresholds 0.3 / 1.5 m fixed; authors: relies on accurate calibration.

## Tried in our experiments

E44 (learned key points) kept; E113 / E116 (junction diagnosis, verifier) partial.
