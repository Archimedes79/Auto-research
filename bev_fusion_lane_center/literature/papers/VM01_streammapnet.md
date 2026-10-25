# VM01 - StreamMapNet (top-k query propagation + warped BEV memory)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly (fetched text; k and the loss weight needed a second read).. Pipeline stage: S8, S4.*

## Papers

Streaming Mapping Network for Vectorized Online HD Map Construction (Yuan et al., WACV 2024) https://arxiv.org/abs/2308.12570

## Idea

Carries the k most confident decoder queries of frame t-1 to frame t. Each carried query is updated by a residual MLP over [query, flattened 4x4 ego transform T]; its previous predicted polyline is mapped by T and becomes the initial reference points. In parallel a dense BEV memory: the previous BEV feature is warped by T and fused with the current BEV by a GRU (+ LayerNorm). "Multi-point attention" uses all N_p predicted points of the previous layer as reference points (not a single centre).

## Formulation

k = 33 of Nq = 100 (rank-based, no threshold); the first decoder layer runs on the learned queries, after it the top Nq - k = 67 of them are kept and the 33 carried queries are concatenated. Reference points p' = (T [x y 0 1])_xy per point. Transformation loss: smooth-L1 between Reg(carried query) and the previous PREDICTED polyline mapped by T, summed over 20 points, no permutation minimum (identity-based, needs no GT); weight 0.1 (value from the follow-up papers MemFusionMap and MapTCL; the StreamMapNet text is ambiguous). BEV: B = LN(GRU(B_t, warp(B_{t-1}, T))). Main loss: focal (class, weight 5) + smooth-L1 line loss (weight 50) minimised over MapTR's permutation group. N_p = 20. Training: sequences randomly cut in 2 parts per epoch, no gradient into previous-frame memory, first 4 epochs single-frame, inference over full sequences; AV2 subsampled to 2 Hz; 24 epochs nuScenes (30 AV2), AdamW 5e-4, batch 32, 8 GPUs.

## Input

surround images -> ResNet-50 + 1-layer BEVFormer; previous queries (Nq x C), previous polylines (20 x 2 each), previous BEV feature, pose T.

## Output

per query a class score and 20 ordered points (absolute coordinates in the ego frame); fixed query set.

## Training

bipartite matching (algorithm not named on the page) + focal + smooth-L1 + transformation loss; labels = vector map GT; regime hungarian_set plus an identity-based auxiliary loss.

## Differentiable?

partial - rank-based top-k (gradient into the selected queries, none into the previous frame because memory is detached), bipartite matching in the main loss.

## Sensors and data

camera only; nuScenes and Argoverse 2, original and NEW location-disjoint splits (VM46); ranges 60 x 30 m and 100 x 50 m; BEV grid size not stated.

## Reported results

nuScenes NEW split mAP, 60 x 30 m: StreamMapNet 33.9, MapTR 20.9, VectorMapNet 18.0; 100 x 50 m: 23.0 / 14.8 / 8.8 (paper Table 2). AV2 new split 60 x 30 m: 58.1 vs MapTR 51.1 (Table 1). nuScenes original split, 30 m range: 62.9 vs MapTR 48.7, BeMapNet 59.8 (Table 4). 13.2 - 14.2 FPS.

## Compute

FPS only.

## Code

https://github.com/yuantianyuan01/StreamMapNet

## Fit to our project

reuse_idea - S/M - the ego-warped previous set plus a pose-conditioned token update plus an identity-based transformation loss is exactly the primitive of our time recursion; the BEV-GRU is our already tested feature fusion.

## Needs

ego transform between consecutive frames; a persistent per-element token (our 128-d step token); a confidence for the rank rule; fixed or padded set size. Cached-BEV: the memory part works on any BEV raster; the query part needs our own token set.

## Conflicts

the main loss assumes a fixed query set + bipartite matching (we have none); carried queries are concatenated into a DETR decoder (we have a tracer / refiner); frame 0 has no prior (needs a single-frame mode).

## Tried in our experiments

E39 / E41 temporal feature fusion (+0.008 F1, small, kept) = BEV-GRU analogue; E31 / E32 matched sparse decoder dropped; System B input memory (ego-warped previous input, PROJECT_CONTEXT) = raster-side idea; set / token propagation with a transformation loss: none.

## Correction from the level-2 extraction (2026-10-24)

The transformation loss above is described as identity-based and needing no GT. In the released code the target is the previous frame's matched ground-truth polyline (stored per query slot, zero rows weighted 0), so the loss needs labels and inherits the Hungarian matching. Weight in the config: 0.1 x loss_reg 50 = 5.0. The nuScenes "new" split is the Roddick and Cipolla scene list, adopted, not newly made. Details: [../deep/VM01_streammapnet.md](../deep/VM01_streammapnet.md), [../code/streammapnet.md](../code/streammapnet.md).
