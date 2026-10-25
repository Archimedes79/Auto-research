# TP06 - UniTopo (connected lanes as unified queries)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked (paper + head file; N_P = 11 from the code, decoder depth and the edge threshold are in files / configs not seen). Pipeline stage: S6 S4 T.*

## Papers

Unified Modeling of Lane and Lane Topology for Driving Scene Reasoning (Li et al., IEEE TCSVT 2026) https://arxiv.org/abs/2605.08911

## Idea

Topology is not derived from pairs of detected pieces. A CONNECTED LANE (end of piece i joined to start of piece j, resampled to N_P points) is a geometric object like any other lane, predicted by its own 300 "connection queries" through the same lane head. A link then exists where a connection query matches a pair of piece queries, found by nearest average L1 distance of the first / second half to the pieces.

## Formulation

lane queries 300 and connection queries 300 (random learnable, no anchors), shared head (existence + N_P x 3 regression). GT connected lane = merge of the two GT pieces at the junction, resampled; Hungarian matching per query group with the same cost as the loss; training uses K = 6 replicated query groups (one-to-many style, shared at inference). TAM (topology-aware attention): D = min(f_g(L, Lc_first), f_g(L, Lc_second)), f_g = average L1; S = sigmoid(MLP(D)); A = softmax(f_q(Q) f_k(Qc)^T / sqrt(C) + log S); Q~ = LN(A f_v(Qc) + Q). Pair score: Q_ll_ij = concat(MLP(Qc_ij + Q_i), MLP(Qc_ij + Q_j)); G_ij = sigmoid(MLP(Q_ll_ij)); unmatched pair uses only the two lane features. Loss L = 1.0 L_lane + 1.0 L_traffic + 5.0 L_ll + 5.0 L_lt (focal for topology; lane cls 1.5, reg 0.025). CODE AS READ (unitopo_head.py): 200 lane queries per group (default), 11 points per lane (33 values for 3D); the connection GT is built by concatenating the previous lane with the next lane minus its first point and keeping every second point (11 points); connection labels are all class 0; Hungarian assigner with focal cost 1.5 and L1 cost 0.025; the pairing of a connection query to its two pieces is done in the loss with torch.cdist (p = 1, sum not average) between 6 sub-sampled lane points and the first 6 / last 6 points of the path, taking the minimum for each half; lane classification focal (alpha 0.25, gamma 2); the relation loss classes are in the relation-head file (not seen); the relation losses of the one-to-many group are multiplied by zero in the code (the extra groups supervise detection only); the inference decoding (score and adjacency thresholds) is in the bbox coder (not seen); GeoDist is not in the head file.

## Input

ResNet-50 / FPN, 3-layer BEVFormer-style encoder, BEV 200 x 100 over x +-51.2 m, y +-25.6 m.

## Output

piece polylines, connected-lane polylines, link probabilities (edge threshold not stated).

## Training

Hungarian per group; 24 epochs (48 for the long Swin-B run), AdamW 2e-4, batch 8, 8 A100.

## Differentiable?

partial - Hungarian and the argmin that pairs a connection query with two pieces (hard selection).

## Sensors and data

camera; OpenLane-V2 A and B; lane segment too (TOP_lsls 31.4, mAP 34.4 vs LaneSegNet 25.4 / 32.6).

## Reported results

subset A ablation (TOP_ll / TOP_lt / DET_l / DET_t / OLS): TopoNet 10.9 / 23.8 / 28.6 / 48.6 / 39.8; matched-parameter baseline 12.4 / 26.0 / 30.0 / 50.4 / 41.6; +connection queries 23.3 / 27.9 / 29.4 / 49.4 / 45.0; +TAM 25.0 / 28.8 / 30.2 / 51.1 / 46.2; +groups 29.6 / 31.2 / 29.8 / 49.3 / 47.3; +GeoDist (TopoLogic) 30.1 / 31.4 / 31.4 / 52.4 / 48.7. Queries 200 / 300 / 400: TOP_ll 27.0 / 30.1 / 31.6. Groups 4 / 6 / 8 / 10: 28.4 / 30.1 / 29.8 / 30.0. BEV: 1 layer 200 x 100 26.2, 3 layers 100 x 50 29.2, 3 layers 200 x 100 30.1. Subset B R50: TOP_ll 31.8, OLS 47.6. Swin-B 48 ep: A TOP_ll 35.1, OLS 51.0; B 40.7, 54.6.

## Compute

67.70 M parameters, 77.05 GFLOPs, 4.21 FPS (TopoNet 65.96 M, 63.07 G, 4.57 FPS).

## Code

https://github.com/homothetic/UniTopo (README: Python 3.8.10, CUDA 11.8; no licence line; built on TopoNet, credits TopoMLP / TopoLogic; weights for A and B)

## Fit to our project

reuse_idea - effort L - "predict the turning movement (two consecutive edges through a vertex) as one object" is the opposite of pairwise link scoring; it is the closest published relative of growing edges with fork / merge events, but requires a query decoder with Hungarian groups.

## Needs

pair-level GT objects; query groups; hard pairing step.

## Conflicts

Hungarian matching; 600 + 100 queries; code is a TopoNet derivative.

## Tried in our experiments

none directly; E42 .. E56 (pairwise heads) point the same way (pairwise scoring adds nothing over geometry).
