# TP13 - CoPo (geometry-biased attention and relational contrastive loss)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: partly (appendix read; hard-negative count n, temperature, Angle formula and K / M / N of the deformable attention are not in the text read). Pipeline stage: S6 L S2.*

## Papers

Geometry-Guided Representations for Coherent Lane and Traffic Topology Reasoning in Driving Scenes (CoPo, KDD 2026, arXiv 2506.13553 v4) https://arxiv.org/abs/2506.13553

## Idea

Geometry enters three places: an attention bias from endpoint distance and relative angle between lane pairs, deformable reference points sampled along each lane's Bezier curve, and a distance embedding in the lane-lane head. A contrastive loss with hard negatives is added to the focal topology loss.

## Formulation

Dist(l_i, l_j) = min over x, y in {start, end} of ||p_i^x - p_j^y|| (symmetric); Angle(l_i, l_j) (formula not given); Geometry = GE(Dist || Angle) (sinusoidal encoding + linear); Attention = softmax(QK^T / sqrt(d) + Geometry) V. Curve-guided cross-attention: K = 11 points sampled on the cubic Bezier as reference points, one shared query predicts offsets and weights. L2L head: G = (MLP1(Q) + PE_pre) * (MLP2(Q) + PE_suc), DistEmbed(i,j) = MLP(P_i^end - P_j^start) (directed), T = MLP(G + DistEmbed). Relational contrastive loss L_con = log(1 + sum_{v+} sum_{v-} exp(v- - v+)); connected pairs positive, hard negatives = top-n by logit (n not stated), groups of one positive and three negatives; L_relation = 5 L_focal + 0.1 L_con (training only).

## Input

BEVFormer BEV 100 x 200, 300 lane queries, 6 layers, R50, 1024 x 800.

## Output

lane Beziers, N x N lane-lane and lane-traffic matrices.

## Training

24 epochs, AdamW 2e-4, batch 8, 8 RTX 4090; matching not described (annotated connected lanes share endpoints).

## Differentiable?

partial - as host.

## Sensors and data

camera; OpenLane-V2 A and B.

## Reported results

ablation subset A (DET_l / DET_t / TOP_ll / TOP_lt / OLS): baseline 27.7 / 50.1 / 21.8 / 28.4 / 44.5; +geometry self-attention 28.1 / 48.8 / 24.7 / 28.3 / 44.9; +curve-guided cross-attention 32.7 / 50.1 / 26.8 / 29.9 / 47.3; +L2L head 33.7 / 48.0 / 28.5 / 29.7 / 47.4; +L2L +L2T 33.9 / 50.3 / 28.6 / 30.6 / 48.2; +contrastive 33.8 / 50.9 / 29.2 / 32.2 / 48.9. TopoLogic without its GeoDist L2L term: TOP_ll 11.6, OLS 40.4 (vs 23.9 / 44.1). Adding the fixed TopoLogic mapping on top of CoPo's own geometry encoding does not help: + GDT_L2L TOP_ll 28.7 (-0.5), + GDT_lane 28.4. Main A: OLS 48.9, B: 49.7 (TOP_ll 29.2 / 31.8). Bezier comparison: BeMapNet 46.2, BezierFormer 46.5, CoPo 48.9 OLS. Breakdown by distance: DET_l near (< 25 m, 45 % of GT) 34.4, far 30.2. Authors' future work: temporal continuity.

## Compute

latency 159.7 -> 162.2 ms, model 218.7 -> 224.8 MB.

## Code

https://doi.org/10.5281/zenodo.20455469 (Zenodo release as stated in the paper)

## Fit to our project

reimplement - effort S - symmetric endpoint-distance + angle bias and a hard-negative contrastive loss are both matching-free once positives are defined by geometry.

## Needs

link candidates with positions and headings; a labelled positive set.

## Conflicts

hard-negative count n unspecified; training-only loss tuned for 300-query sets.

## Tried in our experiments

E45 (TopoLogic-style scoring) and E42 .. E56: no gain over rules.
