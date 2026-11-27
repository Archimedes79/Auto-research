# VM18 - MapExpert (recurrent single-state BEV fusion, sparse expert block)

*Pass: loop 16 (2026-10-23), cluster `vectormap_temporal` (online vector maps and temporal / streaming fusion). Reading depth: **M** (A abstract, M method, C code page). Verified: partly.. Pipeline stage: S8, B.*

## Papers

MapExpert: Online HD Map Construction with Simple and Efficient Sparse Map Element Expert (arXiv 2412.12704; authors not read) https://arxiv.org/abs/2412.12704

## Idea

Argues that stacking S frames gives the current frame weight 1/S so history dominates as S grows, and that map elements fill under 5 % of a raster so extra history mostly adds noise. Replaces stacking by a recurrent "learnable weighted moving descent" with ONE previous fused BEV feature, and adds a sparse mixture-of-experts block (8 experts, top-2 routing, intended one expert per element type) in the tracking decoder.

## Formulation

Eq. 7: C_t = beta F_t + f_t(F_t, C_{t-1}), beta a learnable parameter (shape not stated), f_t an unspecified network; router R(x) = Softmax(TopK(x W_r)), output = weighted sum of top-K SwiGLU experts; balance loss alpha N sum f_i P_i (alpha not given); tracking query TQ(t) = [TQ_element(t-1), TQ_init]; ego-pose alignment not specified; ResNet-50 + FPN, BEVFormer-style BEV as in MapTracker; 100 epochs, AdamW, cosine, 500 warm-up steps, 6 x A800.

## Input

BEV features + previous fused BEV + tracking queries.

## Output

polylines with class.

## Training

as MapTracker; hungarian_set.

## Differentiable?

partial.

## Sensors and data

camera; nuScenes, AV2; ranges 30 m and 50 m as printed, BEV grid not stated.

## Reported results

nuScenes original split 76.5 mAP (easy set) vs MapTracker 74.7 (+1.8; hard set 60.4 vs 58.13); new split 42.0 vs 40.3; AV2 original 30 m 72.8 vs 71.4, new split 71.8 vs 71.3; ablation 73.9 -> 76.5; FPS not reported.

## Compute

not reported.

## Code

none on the page.

## Fit to our project

reuse_idea - S - a single-state recurrence with a learned weight on the current observation is the minimal feature memory; the 1/S argument is a reason to keep the memory weight below the current-frame weight.

## Needs

one stored fused state. Cached-BEV: yes.

## Conflicts

the MoE block and tracking queries are host-specific.

## Tried in our experiments

E41 (GRU fusion of one previous memory).
