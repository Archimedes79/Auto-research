# TP15 - TopoBDA (Bezier deformable attention, instance-mask loss, radar and SD fusion)

*Pass: loop 16 (2026-10-23), cluster `topology` (lane topology heads, link scoring, graph priors). Reading depth: **M** (A abstract, M method, C code page). Verified: partly (page read in two parts; epochs of the main run and Table S7 not seen; control points 4 (S.5) versus 8 (conclusion) inconsistent in the paper). Pipeline stage: S2 L B.*

## Papers

TopoBDA: Towards Bezier Deformable Attention for Road Topology Understanding (Kalfaoglu et al., arXiv 2412.18951; Neurocomputing 2026 per the earlier review) https://arxiv.org/abs/2412.18951

## Idea

Each Bezier control point of a centerline query acts as its own attention head and reference point, so sampling follows the curve; an auxiliary instance-mask head supervised by rasterised centerlines (and a mixed mask / L1 matcher) improves thin-curve detection; a one-to-many auxiliary decoder repeats the GT R times in training only.

## Formulation

S(t) = sum_n B_{n,N}(t) c_n with N = 3 (4 control points; 8 gave marginal gain); BDA(q, V, C) = sum_n sum_k A_{n,k} W_n V(c_n + dp_{n,k}) (K per point not stated); control points refined per layer in inverse-sigmoid space. Mask: E_mask = MLP(query), P_mask = F_BEV . E_mask accumulated across layers, GT = centerline raster of width W (value not given), training only. Matching cost lambda_reg L_reg + lambda_mask L_mask + lambda_cls L_cls (reg 3, mask BCE 5, dice 5, cls 2; matcher reg 5); classification alpha 0.1 for matched. Topology: embeddings of lane and traffic queries projected by MLPs, concatenated, MLP (as TopoNet), L_total = L_l + L_t + L_ll + L_lt. Radar: points (x, y, z, features) assigned to the nearest voxel and concatenated with camera voxel features, 2D conv to BEV.

## Input

BEV 200 x 104 cells of 0.5 m, Z = 20 voxels over -10 .. 10 m; 200 queries, 10 decoder layers, 6 encoder layers.

## Output

Bezier centerlines + topology.

## Training

AdamW 3e-4, batch 8, polynomial decay, warm-up 1000 iterations, grad clip 35; 24 epochs for R50 comparisons.

## Differentiable?

partial - mask-L1 mix matcher (assignment).

## Sensors and data

camera, + LiDAR, + SD map (subset A), + radar (subset B nuScenes); OpenLane-V2.

## Reported results

subset A (Swin-B, DET_l / TOP_ll / OLS): baseline 37.0 / 29.0 / 43.6; +instance-mask loss 40.7 / 32.4 / 46.6; +mask-L1 matcher 40.8 / 32.9 / 48.0. Attention: SA 41.0, MA 42.6, single-point deformable 44.1, multi-point (4) 47.4, multi-point (16) 47.5, Bezier 48.0 OLS. Sensors (R50, DET_l / TOP_ll / OLS), subset A: camera 38.9 / 29.4 / 44.1; +SD map 42.7 / 35.7 / 50.1; +LiDAR 46.5 / 36.7 / 52.0; +LiDAR + SD 52.0 / 40.0 / 56.0. Subset B: camera 45.1 / 35.6 / 49.9; +RADAR 49.4 / 38.6 / 54.8; +LiDAR 57.5 / 46.0 / 61.6; radar on top of LiDAR no gain. Main (C): A OLS 51.7, B 54.3.

## Compute

decoder 10 layers: Bezier 18.47 ms (torch) / 8.07 ms (ONNX), RTX A6000.

## Code

project page https://artest08.github.io/TopoBDA.github.io/ (no repository found)

## Fit to our project

reuse_idea - effort M - radar evidence is directly relevant (camera + radar: DET_l +4.3, TOP_ll +3.0, OLS +4.9 on nuScenes); a mask head on rasterised centrelines is a matching-free dense auxiliary.

## Needs

Bezier or ordered control points as reference points; centreline raster for the mask loss.

## Conflicts

control-point count tied to head count of attention (changing it changes channel layout); Hungarian-type matcher.

## Tried in our experiments

E31 / E32 (sparse decoder with matching): dropped.
