# VS26 - CLRNet: lane priors refined over three pyramid levels, ROIGather, Line-IoU (+ code facts)

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2, S5, L.*

## Papers

CLRNet (Zheng et al., CVPR 2022) https://arxiv.org/abs/2203.10350 ; code file read https://github.com/Turoad/CLRNet

## Idea

Lane priors (start point, angle, length, N = 72 offsets) start uniformly over the image and are refined in three stages that read feature levels from high-level (semantic) to low-level (detailed); each stage pools N_p = 36 points along the current prior and adds a global attention context; a Line-IoU loss scores the whole lane.

## Formulation

P_t = P_{t-1} o R_t(L_{t-1}, P_{t-1}), T = 3 over FPN levels L0..L2 (strides not stated; the code takes the last three feature maps reversed, stage 0 = deepest); learnable priors (as Sparse R-CNN); ROI features: bilinear samples along the prior, resized map 10 x 25 with C = 64, earlier-level ROI features concatenated, conv + FC to a C x 1 vector per prior; ROIGather W = softmax(X_p^T X_f / sqrt(C)), G = W X_f^T added back; two FC layers give the refinement. Line-IoU: each x_i extended by radius e = 15 px, d_O = min(x_p + e, x_g + e) - max(x_p - e, x_g - e), d_U = max(...) - min(...), LIoU = sum d_O / sum d_U in [-1, 1], loss 1 - LIoU. Assignment: dynamic top-k (YOLOX-style), cost w_sim C_sim + w_cls C_cls, C_sim = (C_dis C_xy C_theta)^2, w_cls = 1, w_sim = 3; top-1 gives the NMS-free variant. Loss = focal (cls) + smooth-L1 (start, angle, length) + LIoU (weights 2 / 4 / 6 tried, 2 best). CODE: 192 priors, 36 sampled points, 64 channels; priors are DETACHED before pooling in the next stage (geometric lines only, offset residuals not carried); EVERY stage gets classification, regression and IoU losses (averaged over stages); defaults cls 2.0, xyt 0.5, IoU 2.0, segmentation 1.0 (training only); grid_sample bilinear, align_corners True, zero padding outside the image; points after the predicted end are invalidated at inference. Input 320 x 800, AdamW 1e-3, 15 epochs (CULane).

## Input

image_feat / bev_raster levels + point_set of priors. output: polyline (lane) with length.

## Training

dynamic top-k assignment (matching_free in our sense), no Hungarian.

## Differentiable?

partial (top-k assignment, detached priors).

## Sensors and data

images (CULane, TuSimple, LLAMAS).

## Reported results

CULane F1@50 / mF1 / F1@75: R18 79.58 / 55.23 / 62.21 (119 FPS), DLA34 80.47 / 55.64 / 62.78 (94 FPS); ablation mF1: baseline 51.90, + LIoU 52.80, + refinement 54.74, + ROIGather 55.23; refinement order R0 -> R1 -> R2 54.74 vs reverse 53.58 vs single stage 52.80; smooth-L1 54.22 vs LIoU 55.23.

## Compute

11.9 GFLOPs (R18).

## Code

checked.

## Fit to our project

reuse_idea / reimplement - S / M - (a) pool N_p points along the current polyline every stage; (b) coarse-to-fine, deepest level first, beat the reverse by 1.2 mF1 - supports f2 first, f0 last; (c) LineIoU-style overlap is a matching-free whole-lane loss (polyline version: overlap of buffered segments); (d) the code is the clearest instance of "detach the geometry, supervise every stage".

## Needs

priors as lines with a start point; any raster features.

## Conflicts

row-parametrised offsets (one x per row) are invalid for BEV graphs of arbitrary heading.

## Tried in our experiments

E39 cascade refiner small gain; E94 reads at fixed bearings.
