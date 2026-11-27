# TP08 - TopoMLP (MLP link head; oracle-topology experiment; metric loophole)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S6 E.*

## Papers

TopoMLP: A Simple yet Strong Pipeline for Driving Topology Reasoning (Wu et al., ICLR 2024) https://arxiv.org/abs/2310.06753 ; The 1st-place Solution for CVPR 2023 OpenLane Topology in Autonomous Driving Challenge https://arxiv.org/abs/2306.09590

## Idea

A strong 3D lane detector (PETR style, learnable 3D anchors, each a 4-control-point Bezier) and two plain MLP topology heads on the last-layer queries. The paper's main message is that topology quality is bounded by detection quality: giving the true topology to the predicted lanes barely changes the score.

## Formulation

lane queries Q^L = 300 learnable 3D anchor points -> 256-d by position encoding + linear; 6 decoder layers; offsets and class by MLPs; 11 points per lane for training. LL head: an MLP embeds the predicted lane coordinates and adds them to the query features; features are repeated along two axes and concatenated to N x N x 2C; three linear layers with ReLU give N x N logits G_LL. LT head: lane query + MLP of the 3 x 3 view-transform matrix, concatenated with traffic queries, MLP. Loss: L = L_detl + L_dett + L_topll + L_toplt; lane focal 1.5 + L1 0.2; LL topology: focal weight 5 plus an extra L1 term (0.1) on matched lane points; LT focal 0.5. Matching only for detection, reused for topology. Metric loophole (old metric): unmatched false positives get default confidence 1.0 and rank above true positives; pushing scores to 0 / 1 raised TopoMLP's TOP_ll from 7.2 to 19.0 (TopoNet 4.0 to 11.5); the paper proposes weighting each precision term by N_TP / (N_TP + N_FP). The devkit fix (issue #76, .1 releases) was triggered by this paper but is implemented differently as far as the code digest shows (unmatched predicted lanes are dropped from the matrix, see TP29).

## Input

multi-view images -> backbone -> PETR-style 3D position-embedded features (no BEV encoder); lane range x +-51.2, y +-25.6, z [-8, 4] m.

## Output

up to 300 lanes + N x N topology matrix.

## Training

Hungarian; 24 epochs (48 for one run), AdamW 2e-4 cosine, batch 8, 8 A100, 1550 x 2048 downsampled x0.5; HSV and GridMask augmentation; optional YOLOv8 proposals for traffic elements.

## Differentiable?

partial - Hungarian assignment.

## Sensors and data

camera; OpenLane-V2 A and B.

## Reported results

oracle-topology table (v1 metric, subset A, TOP_ll predicted -> with GT topology): R50 7.2 -> 7.5, VoV 7.9 -> 8.5, Swin-B 9.5 -> 10.0; OLS 38.2 -> 38.3 / 40.1 -> 40.8 / 42.2 -> 43.1. Ablations (R50, DET_l / TOP_ll / OLS): queries 200 / 300 / 500: 28.2 / 6.1 / 36.9, 28.3 / 7.2 / 38.2, 27.9 / 7.3 / 38.0; control points 3 / 4 / 5: 26.6 / 7.0 / 37.3, 28.3 / 7.2 / 38.2, 27.8 / 6.6 / 37.1; LL head without position input 6.9 (OLS 37.9), without the L1 term 6.5 (OLS 37.5). Subset A OLS: R50 38.2, with YOLOv8 41.2, Swin-B 42.2 (24 ep) / 43.7 (48 ep); subset B R50 38.7. In the v2.1 table of other papers TopoMLP has TOP_ll 21.7, OLS 44.1.

## Compute

not stated.

## Code

https://github.com/wudongming97/TopoMLP (not opened)

## Fit to our project

baseline - effort S - the head is a plain pair MLP; the oracle experiment is the argument why the link head stays small.

## Needs

query embeddings and predicted coordinates per lane; Hungarian matching for labels.

## Conflicts

PETR-style 3D position embedding needs perspective features (not available on a cached BEV raster).

## Tried in our experiments

E42 (learned link head) within 0.01 of rules, dropped.
