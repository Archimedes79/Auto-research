# CGNet refinement head: deep read for E355 (2026-11-15)

Marks: all pages went through the summarising fetch tool, so every fact is **digest**. [P] https://arxiv.org/html/2407.11337; [C] https://github.com/XiaoMi/CGNet, `projects/` (Apache-2.0); [T] `papers/TP03_cgnet.md`; [A] my arithmetic. Data: nuScenes / Argoverse2 centreline graphs (P Tables 1, 2), not OpenLane-V2. No E355 card on disk, so section 7 follows the brief.

## 1 The head (P Sec 3.3 Eq. 10-11; C `dense_heads/CGNet_head.py`)
- 50 segment queries x 20 points, 256 d, 6 decoder layers, N = 50 [P Sec 4.3; C config].
- Per layer: v_i = mean of query i's point embeddings; GCN step with A^(l-1) (FFN, then W v + 0.8 (A v W_f + A^T v W_b), residual); GRU with state carried across layers; pair head e = sigmoid(MLP 256-128-64), A_ij = sigmoid(MLP([e_i, e_j]), 128-64-1). Directed, unnormalised, A^0 = 0, weights shared by all layers.
- Refined: the adjacency (every layer, supervised) and the head's own node state (GRU memory). The decoder is not conditioned: the head loops over the stacked decoder outputs afterwards, so A^k reaches round k+1 only by message passing and the GRU, no attention [C, structure].
- Rounds, TOPO / JTOPO / APLS / SDA (P Table 5): 1 = 39.6 / 31.5 / 27.6 / 7.1; 3 = 41.3 / 33.3 / 29.9 / 8.3; 6 = 42.2 / 34.1 / 30.7 / 8.8. Round 1 equals the baseline (MapTR 39.7, TopoNet 39.6, Table 1). Not stated: whether fewer rounds means a shorter decoder; seeds.

## 2 Junction map, Bezier head
- Junction target: dataset key points on the BEV grid, dilated (R = 9; 7 / 9 / 11 give 40.4 / 42.2 / 40.9 TOPO, Table 5); focal loss weight 0.1 [P Eq. 6; C]. Branch: two 3x3 conv-BN-ReLU + 1x1 conv on the 200 x 100 x 256 map [C `modules/transformer.py`].
- Inference: only its features are used (2x2 max-pool, linear attention, residual onto the positional query). The heat map is returned but unused; no junction point is extracted [C]. P is silent.
- Bezier: training only [P]. Per GT-connected matched pair, both queries' point embeddings are concatenated, mapped by a pseudo-inverse Bernstein matrix to 4 control-point embeddings, MLP to 4 control points, L1 to fitted GT control points, weight 0.01 [P Eq. 8-9]. "On feature" 42.2 vs "on lane" 40.1; 4 / 6 / 8 points 42.2 / 40.8 / 40.4.

## 3 Compute (P Tables 1, 4)
| row | params | GFLOPs |
|---|---|---|
| baseline (MapTR) | 36.0 M | 216.7 |
| + junction branch | 37.8 M | 241.6 |
| + Bezier | 36.0 M | 216.7 |
| + refinement | 36.9 M | 217.0 |
| full CGNet | 38.8 M | 241.9 |

Table 4 check marks are lost on the page; the row mapping is inferred and fits the code sizes [A]. FPS 11.2 vs MapTR 12.7, RTX 3090 (Table 1). Split by backbone / BEV encoder / decoder: not stated. Config: ResNet-50 + FPN, one BEVFormer encoder layer (P says GKT: paper-code difference), 0.3 m cells [C].
- **Refinement share: +0.3 of 241.9 GFLOPs (0.1 %), +0.9 of 38.8 M parameters (2.3 %).** The junction branch costs +24.9 (10 %).
- Scaling [A, code dimensions]: per round 0.07 GMAC at N = 50 (6 rounds about 0.4, matches 0.3), 0.18 at N = 100, 0.79 at N = 250 (pair MLP 0.52); 3 rounds at 250 = 2.4 GMAC.

## 4 Loss
`nn.BCELoss` on the matched x matched sub-matrix only, weight 1, every layer; no focal, sampling or balancing [C; P Eq. 16]. Hungarian one-to-one; unmatched queries get "no lane" in the class loss and nothing in the adjacency loss, but their unsupervised rows still enter the next round's GCN [C]. No NMS; at inference queries are kept by score > 0.3, max 50, edges by A > 0.9, connected endpoints averaged [C `nms_free_coder.py`, config].

## 5 Pruning between rounds
Confirmed absent: no detach, top-k or threshold; all 50 queries and the soft matrix pass all 6 rounds [C]. Thresholds act at the output only.

## 6 Cost of a cheap version
- 6 -> 3 rounds: -0.9 TOPO (-0.8 JTOPO, -0.8 APLS); 6 -> 1: -2.6 (-2.6, -3.1) (Table 5).
- Refinement alone gives +0.4 over baseline, the full model +2.5 (Table 4): rounds pay with the other parts.
- Bezier head: free at inference, +1.7 alone. Junction branch: +0.8 alone for +25 GFLOPs. No leave-one-out ablation.
- Not ablated: resolution, query count, points per query, GCN / GRU. R 7 / 11 change TOPO by -1.8 / -1.3, as much as the 1 -> 3 step; no seeds.

## 7 Reuse for us
1. Pair head spec above (8.3 k MACs per pair); restrict to k nearest neighbours.
2. Share weights over rounds: extra rounds cost no parameters.
3. Message passing with A and A^T plus GRU: about 40 lines, sparse on the lattice.
4. 2 - 3 rounds, same BCE each round; gain is front-loaded (+0.85 per round to 3, +0.30 after).
5. Valid-pairs-only BCE; add entry -> exit junction pairs as positives.
6. Multiply messages by the keep weight, so pruned tokens do not feed round 2 (CGNet feeds unsupervised rows).
7. Bezier of a connected pair from both embeddings, training only, weight 0.01, 4 control points.
8. Junction map only as feature aid, never decoded; skip the 25 GFLOP branch.
9. Do not copy Hungarian one-to-one or the 0.9 threshold with endpoint averaging.
10. At 250 tokens the pair head dominates: report ms, not GFLOPs.
