# TP16 - Topo2Seq (training-only topology sequence decoder)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: checked (digest: Eq. 6 garbled, schedule description inconsistent, epochs unclear). Pipeline stage: T L.*

## Papers

Topo2Seq: Enhanced Topology Reasoning via Topology Sequence Learning (Yang et al., AAAI 2025) https://arxiv.org/abs/2502.08974

## Idea

A Pix2Seq-style autoregressive decoder reads the shared BEV features and predicts the lane graph as a token sequence from a prompt of unordered keypoints; it is discarded at inference. Its loss is meant to improve the shared BEV representation, so the lane-segment decoder gains accuracy at no inference cost.

## Formulation

lane-segment decoder = LaneSegNet (6 layers, 200 queries, lane attention); topology A = sigmoid(MLP(concat(MLP(Q), MLP(Q)))). Sequence decoder: causal self-attention + cross-attention to BEV; each edge = 6 integers [x, y, cls (ancestor / lineal / offshoot / clone), con (parent index), bx, by (intermediate cubic-Bezier control point)]; sequence length 802 = 201 keypoint prompt + 601 edge tokens; keypoint prompt in RANDOM order (not the edge order), ordering of edges by depth-first search from the right-front; noise edges with a <ncls> token; max-likelihood loss with class weights. Vocabulary size, bins and depth not stated. L = 0.025 L1 + 1.5 L_cls + 3.0 L_seg + 0.1 L_lt + 5.0 L_top + 1.0 L_seq.

## Input

BEV 200 x 100 (+-50 x +-25 m); 10 points per centerline.

## Output

unchanged lane segments + adjacency at inference.

## Training

matching not stated; 24 + 24 style schedules (inconsistent in the text), 4 A100, batch 4; subset A only.

## Differentiable?

partial - decoder training uses teacher forcing; the benefit passes only through the shared BEV encoder.

## Sensors and data

camera; OpenLane-V2 subset A (27 k train / 4.8 k val frames).

## Reported results

ablation (mAP / AP_ls / AP_ped / TOP_lsls): ordered GT keypoint prompts 32.5 / 31.5 / 33.5 / 25.1; random order 34.9 / 33.4 / 36.3 / 27.8; early interaction 33.8 / 32.3 / 35.4 / 26.5; full (predicted keypoints, extra 24 epochs) 37.7 / 36.9 / 38.5 / 29.9. Versus LaneSegNet at 24 ep: +1.8 AP_ls, +1.5 TOP_lsls, centerline OLS* 42.7 vs 40.7, TOP_ll 27.0 vs 25.3 (the earlier review's "+2.0 / +2.7" came from a third-party summary and differs).

## Compute

14.7 FPS (unchanged).

## Code

none found in the paper

## Fit to our project

not_suitable - the claimed mechanism is an auxiliary gradient into a TRAINABLE BEV encoder, which is frozen in our pipeline; effort would be L for no expected gain.

## Needs

trainable shared encoder; sequence tokenisation.

## Conflicts

frozen foundation; argmax-free token decoding not needed.

## Tried in our experiments

none
