# VS20 - Deformable DETR: reference points, learned offsets, multi-scale, iterative box refinement (+ code facts)

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M+C** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S2, S5.*

## Papers

Deformable DETR (Zhu et al., ICLR 2021) https://arxiv.org/abs/2010.04159 ; code files read https://github.com/fundamentalvision/Deformable-DETR

## Idea

Each query reads only K points around a reference point (position = reference + learned offset) with attention weights from the query itself; cost does not depend on map size. The decoder refines the box layer by layer.

## Formulation

DeformAttn(z_q, p_q, x) = sum_m W_m [sum_k A_mqk W'_m x(p_q + dp_mqk)]; one linear layer of 3MK channels: the first 2MK are offsets (unconstrained range), the last MK a softmax over k (multi-scale: over l and k jointly); multi-scale M = 8, K = 4, L = 4 (C3 - C5 + C6). INIT (appendix + code): offset-layer weights 0, bias = 8 fixed directions (cos / sin of the head angle, scaled so the larger component is +-1) times (k + 1) for the k-th point; attention layer weights and bias 0 (uniform weights); the paper says decoder offset biases are further multiplied by 1/(2K). FORWARD (code): location = reference + offset / [W_l, H_l], i.e. offsets are in pixels of each level; for 4-D reference boxes offset / n_points * (w, h) * 0.5; no clamp, tanh or bound anywhere in the file. OPTIMISER (`main.py`): parameters named sampling_offsets and reference_points learn at lr x 0.1 (2e-4 -> 2e-5), backbone 2e-5, gradient clip 0.1. Decoder: reference = linear + sigmoid of the query; iterative refinement b^d = sigmoid(db^d + sigmoid^-1(b^{d-1})) with the gradient blocked at the previous box ("stabilisation", cites RAFT); two-stage: each encoder pixel proposes a box (scale 0.05), top boxes (no NMS) become decoder boxes. Defaults: 300 queries, 6 + 6 layers, 4 levels, 4 + 4 points; costs 2 / 5 / 2 (class / L1 / GIoU). `ms_deform_attn_core_pytorch` is a grid_sample reference (locations 2 * loc - 1, bilinear, zero padding, align_corners False; comment "for debug and test only"), but the module imports the compiled `MultiScaleDeformableAttention` at top level, so the function must be copied out to run without the CUDA build.

## Input

bev_raster / image_feat pyramids + queries with reference points. output: boxes + classes.

## Training

hungarian_set, focal + L1 + GIoU, auxiliary losses on every decoder layer.

## Differentiable?

partial (hungarian; detached reference).

## Sensors and data

images (COCO).

## Reported results

COCO AP / FPS: DETR 42.0 (500 ep) 28 FPS; Deformable DETR 43.8 (50 ep) 19 FPS; + iterative refinement 45.4; + two-stage 46.2; test-dev 52.3 with ResNeXt-101 + DCN + TTA.

## Compute

O(2 N_q C^2 + min(HW C^2, N_q K C^2)); 1.6x faster than DETR-DC5, about 25 % slower than Faster R-CNN + FPN.

## Code

checked (three files above).

## Fit to our project

reuse_idea / reimplement - S - 4 levels x 4 points around a tip is one grid_sample call; the init (a fixed fan of directions scaled by k, zero weights) is the precedent for "hand-built teardrop + zero-initialised learned offsets"; lr x 0.1 for offsets is a cheap stabiliser.

## Needs

Hungarian for the decoder as published; queries with reference points.

## Conflicts

Hungarian (matching-free rule); unbounded offsets can leave the BEV range (zero padding then reads "no lane").

## Tried in our experiments

E31 / E32 matched sparse decoder dropped; E17 / E18 transformer decoder -0.11.
