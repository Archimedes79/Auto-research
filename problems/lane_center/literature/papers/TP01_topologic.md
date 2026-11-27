# TP01 - TopoLogic (geometric endpoint distance + query similarity)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked (paper and the two code files read; paper-versus-code differences listed above). Pipeline stage: S6 (L).*

## Papers

TopoLogic: An Interpretable Pipeline for Lane Topology Reasoning on Driving Scenes (Fu et al., NeurIPS 2024) https://arxiv.org/abs/2405.14747

## Idea

The link score between lane pieces is the sum of two terms: a geometric term that maps the distance from the END of piece i to the START of piece j through a learned monotone decay, and a semantic term (sigmoid of a dot product of two MLP-embedded queries). The fused N x N matrix also drives a GNN that updates the queries after every decoder layer. The geometric term can be added to an already trained detector without retraining.

## Formulation

d_ij = |end_i - start_j| (end point of lane i, start point of lane j, each from 11 predicted 3D points; unit not stated, metres presumed); D = {d_ij}; G_dis(i,j) = exp(-d_ij^alpha / (lambda * sigma)), sigma = std(D), alpha and lambda learnable scalars (1 x 1), init alpha 0.2, lambda 2 (TopoPoint's text writes the two inits the other way round). G_sim = sigmoid(MLP1(Q) MLP2(Q)^T), Q = 200 x 256 lane queries, 3-layer MLPs. G = l1 G_dis + l2 G_sim (l1 = l2 = 1, learnable). GNN: aggregates neighbour queries with G (layer count, normalisation, skip not stated). L = L_det + L_top; L_det = focal (weight 1.5) + L1 (0.025); L_top acts only on G_sim (loss type and weight not named); G_dis has no topology loss, its two scalars learn through the GNN / detection path. Positive (i,j) iff end of i meets start of j (directed). Class imbalance not discussed. Head: Deformable-DETR, 200 queries, 11 x 3 points, supervision on every decoder layer. Alternatives tested for the mapping f: Gaussian exp(-x^2/2), 2/(1+e^x), tanh form. CODE AS READ (sgnn_decoder.py, topologic_head.py) differs from the paper text: d = sum over x, y, z of |end_i - start_j| (L1, not Euclidean) on DENORMALISED coordinates (metres) of the current decoder layer, point coordinates DETACHED (no gradient from topology into the points); mapping exp(-d^P / w) with learnable P (init 2) and w (init 10) and no alpha; sigma = std(D) is computed but never used; adjacency for layer l+1 = lamda_1 * sigmoid(relation branch).detach() + lamda_2 * topo (lamda init 1, learnable), diagonal of the distance term zeroed, no row normalisation or A + A^T; GNN layer: forward messages with adj and backward messages with adj^T (separate weights), edge weight 0.5, concat with the lane-traffic GNN output, Linear 2C -> C, residual; layer 0 gets zeros. The inference score in topologic_head.py hard-codes P = 2, w = 11.5275, lamda = 1: a Gaussian-type kernel with half value at an L1 distance of 2.8 m (d = 1 m -> 0.92, 3 m -> 0.46, 5 m -> 0.11), i.e. a wide tolerance for noisy query endpoints. Targets (topologic_head.py + relationship_head.py): GT adjacency indexed by the Hungarian assignment (pos_inds, pos_assigned_gt_inds) into a Q x Q zero matrix; ALL other pairs, including every pair with an unmatched query, are negatives (full supervision, no sampling, no pos_weight); the relation head embeds both query sets with 3-layer MLPs to 256-d and scores a pair by the DOT PRODUCT (no pair-concatenation classifier; symmetric when the weights are shared); the loss is mmdet FocalLoss (sigmoid, gamma 2, alpha 0.25) on labels passed as 1 - target because mmdet treats label == num_classes as background (so the inversion is a convention, not a semantic flip). Imbalance is handled only by the focal terms. The same relation-head file family is used by TopoPoint, TopoFG and UniTopo (TopoNet lineage).

## Input

lane queries 200 x 256 and predicted points 200 x 11 x 3 of each decoder layer; BEV 200 x 100 (BEVFormer).

## Output

directed N x N link probabilities; lane-traffic topology via the TopoNet path.

## Training

DETR-style set loss with Hungarian matching (from memory, as TopoNet; not restated on the page); 24 epochs, AdamW 2e-4 cosine, R50 + FPN, images 1550 x 2048, batch 2 on 8 RTX 3090.

## Differentiable?

partial - the scores are differentiable in queries and points; the label assignment between queries and ground truth is a matching step.

## Sensors and data

camera; OpenLane-V2 subset A (Argoverse 2) and B (nuScenes); BEV +-50 x +-25 m in a 200 x 100 grid.

## Reported results

subset A v2.1: TOP_ll 23.9 vs TopoNet 10.9, OLS 44.1 vs 39.8, DET_l 29.9 (Table 1). Topology approach (Table 4, TOP_ll): MLP 10.8, similarity 12.9, GeoDist 20.1, both 23.9. Mapping function (Table 3, TOP_ll): tanh 15.1, sigmoid 19.1, Gaussian 21.7, learned exp 23.9. Post-hoc GeoDist (Table 5): TopoNet TOP_ll 10.9 -> 22.3, SMERF 15.4 -> 26.2, LaneSegNet TOP_lsls 25.4 -> 29.6. Subset B: TOP_ll 21.6, OLS 42.3 (no ablation on B). With SD map OLS 47.5.

## Compute

not stated.

## Code

https://github.com/Franpin/TopoLogic (README: Python 3.8, CUDA 11.1, PyTorch 1.9.1; no licence line; ResNet-50 checkpoint on Hugging Face)

## Fit to our project

reimplement - effort S - the mapping needs piece end / start points; in our vertex graph edges share vertices by construction, so it applies only where a link or merge is a decision (tip to existing vertex, DAG clustering).

## Needs

directed pairs with a defined "end" and "start" position; labels defined by endpoint coincidence.

## Conflicts

relies on accurate endpoints (authors: geometry depends heavily on detection accuracy); the tolerance learned for noisy query endpoints (about 3 m) is far wider than our vertex error (0.43 - 0.54 m), so the form transfers but not the bandwidth; original code is a PyTorch 1.9.1 / mm-series code base (the SGNN decoder builds on mmcv layers, from memory) and cannot run in our environment as is.

## Tried in our experiments

E45 (TopoLogic-style scoring) and E42 / E46 / E51 / E56 (learned link heads): within 0.01 of the rules, dropped.
