# TP05 - TopoFG (keypoint queries, boundary-point topology, matching-free denoising target)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M+C** (A abstract, M method, C code page). Verified: partly (paper + head file read; loss types of topology / mask / DN, noise scale, mask GT rasterisation width and the reference-point sampling rule are in files or configs not seen; Table 4 checkmarks ambiguous). Pipeline stage: S1 S6 L.*

## Papers

Fine-Grained Representation for Lane Topology Reasoning (Xu et al., AAAI 2026) https://arxiv.org/abs/2511.12590

## Idea

One query per lane KEYPOINT (11 per lane) instead of one per lane, built from a predicted BEV lane mask (where) and an ordinal in-lane embedding (which keypoint). Links are scored only between the last keypoint of lane i and the first of lane j. A training-only denoising branch gives an adjacency target that does not depend on the matching.

## Formulation

HPE: learnable queries + Mask2Former-style module -> BEV mask M; weights A = M * I[M <= tau] + alpha * I[M > tau] (tau 0.3, alpha 1.0); Q_pos = norm(P^T A) + Q_R (P = sine-cosine grid encoding); Q_seq = MLP(PE(index)) + Q'; fine-grained query Q_F(i,t) = Q_pos(i) + F(Q_seq(t)). RFD: 6 layers; inter-instance then intra-instance self-attention; deformable cross-attention with reference points sampled from the predicted mask (rule not given). RBTR: r_{i->j} = concat(f_i^end, f_j^start); S = sigmoid(MLP(r)); geometric topology = TopoLogic mapping of boundary-point distances; adjacency = S + geometric. Hungarian one-to-one matching places the GT adjacency into the query-index matrix (target moves with the matching); denoising: G = 5 groups of noisy GT-derived queries with the fixed block-diagonal GT adjacency; inference uses the vanilla queries only. Individual loss terms and weights are not reported. CODE AS READ (topofg_head.py): 11 keypoints per lane via nn.Embedding(11) + a sequence positional encoding, fine-grained queries = instance query (detached) + per-keypoint embedding, flattened to lanes x 11; positional map = sine grid encoding weighted by the mask probability where it is below 0.3, normalised by the maximum; reference points sampled from the mask thresholded at 0.3 (sampling rule in a file not shown); mask loss = Dice + a per-pixel term with its own matching of queries to GT masks (types set in the config); lane losses = softmax CE (class weight 1.0, background 0.1) + L1 (5.0) with the same Hungarian costs as TopoLogic (1.0 / 5.0 / GIoU 2.0, GIoU never used in the loss); DN losses weighted 1.0, topology DN target = the same GT adjacency repeated on the diagonal blocks of the G groups; at inference the lane-lane score is the TopoLogic form: L1 distance last keypoint of i -> first keypoint of j, exp(-d^2 / 11.5275), lamda 1 + 1, sigma unused. So the released method is TopoLogic plus keypoint queries, mask prior and denoising; the boundary-point MLP and DN loss type sit in config files not seen.

## Input

multi-scale BEV (grid not stated), 200 lanes x 11 keypoints.

## Output

200 x 11 keypoints (3D lanes) and an N x N adjacency.

## Training

Hungarian for vanilla queries, fixed target for denoising queries; 24 epochs, AdamW 2e-4 cosine, 500 warm-up steps, 8 A100, R50 + FPN; input 800 x 1024 (A, 7 views) / 480 x 800 (B, 6 views).

## Differentiable?

partial - Hungarian for vanilla queries; the denoising branch needs no assignment.

## Sensors and data

camera; OpenLane-V2 A and B; lane-segment table also reported.

## Reported results

ablation subset A (OLS / DET_l / TOP_ll / TOP_lt): TopoLogic 44.1 / 29.9 / 23.9 / 25.4; +HPE 45.4 / 31.3 / 26.1 / 26.6; +RFD 45.8 / 31.8 / 26.8 / 27.7; +RBTR 48.0 / 33.8 / 30.8 / 30.9. RBTR parts (TOP_ll): neither 26.8, boundary topology 28.3, denoising 29.0, both 30.8. Main: A 48.0 / 33.8 / 47.2 / 30.8 / 30.9; B 45.4 / 30.0 / 53.0 / 27.2 / 21.7 (OLS / DET_l / DET_t / TOP_ll / TOP_lt). Lane segment: mAP 34.4, TOP_ls 31.3.

## Compute

not reported.

## Code

https://github.com/GXmmm18/TopoFG (README: PyTorch 1.9.1, CUDA 11.1; no licence line; subset-A checkpoint on Google Drive; builds on TopoLogic / TopoNet / MapTR per acknowledgements)

## Fit to our project

reuse_idea - effort M - a keypoint-per-token query set resembles our vertex tokens; the ordinal embedding, inter / intra attention and the fixed denoising adjacency are small, matching-free ideas.

## Needs

ordered keypoints along every lane; a mask prior; deformable attention over BEV.

## Conflicts

vanilla branch needs Hungarian; mask supervision unspecified; mm-lineage code.

## Tried in our experiments

E44 / E53 (key points, connectors) kept; the dense heat map as spatial prior already exists in the dense head.
