# TP10 - LaneSegNet (lane segments, lane attention, identical initialisation)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2 S6 D.*

## Papers

LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving (Li et al., ICLR 2024) https://arxiv.org/abs/2312.16108

## Idea

The output unit is a LANE SEGMENT (centerline + left and right boundary + boundary types + successor graph), which unifies centerline and map-element tasks. Cross-attention is replaced by "lane attention": the reference points are spread along the predicted segment and each head owns one region; the first decoder layer starts all heads from one shared reference point.

## Formulation

segment = 3 lines x 10 3D points; boundaries V_left = V_center + V_offset, V_right = V_center - V_offset; types non-visible / solid / dashed. Lane attention: LaneAttn(q, p, B) = sum_m W_m [ sum_k a_{m,k} W'_m Bilinear(B, p_m + dp_{m,k}) ], M = 8 heads, K = 32 offsets, reference points 4 along each boundary (n_ref = 8); offsets and weights by linear layers from q. Identical initialisation: in layer 1 every head uses the same reference point from the learnable positional query; later layers use the previous prediction. Topology: q'_i = MLP_pre(q_i), q'_j = MLP_suc(q_j), A_ij = sigmoid(MLP_top(concat)) from query features only. Mask branch: S = sigmoid(E_mask (x) B). Loss L = 0.025 L_vec + 3.0 L_seg(CE + dice) + 1.5 L_cls + 0.01 L_type + 5.0 L_top; Hungarian (or Sinkhorn); topology focal. CODE AS READ (lane_attention.py): the kernel is the ordinary multi-scale deformable-attention op (BEVFormer's fp32 CUDA wrapper on GPU, fallback multi_scale_deformable_attn_pytorch from mmcv.ops, which is grid_sample based); reference points are an input that is broadcast over the 32 sampling points of a head, so the "lane attention" novelty sits in the decoder that builds per-head reference points from the predicted segment (laneseg_decoder.py: per layer a centerline regression and an offset regression, boundaries = centerline +- offset, each boundary sub-sampled at indices [0, 3, 6, 9] of its 10 points, giving 4 left + 4 right = 8 reference points (x, y only), detached between layers, refinement additive in logit space; the assignment of the 8 points to the 8 heads happens inside the attention module, not seen); offsets are linear outputs divided by the feature-map size, initial offset bias = 8 radial directions scaled by point index, zero weights; attention weights softmax jointly over levels and points; value projection, output projection, dropout + residual.

## Input

R50 + FPN, 3 BEVFormer layers, BEV 200 x 100 (+-50 x +-25 m); 200 queries; 6 decoder layers.

## Output

200 lane segments + adjacency.

## Training

Hungarian; 24 epochs, AdamW 2e-4 cosine, batch 8, 8 V100; 14.7 FPS on A100.

## Differentiable?

partial - Hungarian.

## Sensors and data

camera; OpenLane-V2 subset A only (lane-segment labels exist only there; nuScenes and Waymo lack them, per the paper).

## Reported results

formulation ablation (old metric): centerline only DET_l 22.9 / TOP_ll 3.7, lane segment 27.2 / 6.6. Cross-attention designs (mAP / TOP_lsls): deformable (instance) 28.7 / 6.9, deformable (hierarchical) 29.1 / 5.2, multi-point 24.6 / 6.9, lane attention 32.6 / 8.1. Decoder rows: heads-to-regions + identical init 32.6 / 8.1 vs 28.8 / 5.6 without either (the text credits identical init with +3.5 mAP). Mask supervision +2.0 AP_ls. Paper TOP_lsls 8.1 is v2.0 metric; v2.1 value 25.4 (README: mAP 33.5, TOP_lsls 25.4).

## Compute

14.7 FPS (A100); tiny variant mAP 28.5 at 16.2 FPS; 9.4 G memory (README).

## Code

https://github.com/OpenDriveLab/LaneSegNet (README: Apache-2.0; Python 3.8, CUDA 11.1, PyTorch 1.9.1, mmcv-full 1.5.2, mmdet 2.26.0, mmsegmentation 0.29.1, mmdet3d 1.0.0rc6; subset A configs and checkpoint)

## Fit to our project

reuse_idea - effort M - sampling reference points along the element and one region per head matches our ray / ring sampling; the attention op itself is plain deformable attention and has a pure-PyTorch form (grid_sample), so only the decoder logic that places per-head reference points needs re-writing; the repository as a whole cannot be used (exact mm-series pins incompatible with PyTorch 2.8).

## Needs

predicted element geometry per layer to place reference points; a query decoder.

## Conflicts

Hungarian; boundaries only exist in subset A; adapting needs mmcv ops (deformable attention) or a plain-PyTorch grid_sample version.

## Tried in our experiments

E31 / E32 (sparse DETR-style decoder with matching vs dense grid): dropped.
