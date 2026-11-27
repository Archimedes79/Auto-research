# C24 - RAVE: correlation-volume flow alignment + decoupled pose solver + transformer refinement

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: RAVE: End-to-end Hierarchical Visual Localization with Rasterized and Vectorized HD map (2025) https://arxiv.org/abs/2503.00862
- depth: M (first 100k of 117k characters)
- stage: S9 (localisation), S5
- idea: FLORA estimates a BEV flow between misaligned map features and BEV features with a correlation volume and iterative refinement, projects the flow on a low-rank basis (rank 16) to suppress noise and warps the map features before fusion. DEMA solves yaw (polar / FFT features), then longitudinal and lateral offsets separately (GeM-pooled axis features, ZNCC), turning a product of hypotheses into a sum (96 % less pose-solver memory); POET (6 + 6 layer transformer) refines using vector-map queries and runs only when the DEMA peak probability is below 0.4.
- formulation: search +-2 m, +-1 m, +-2 deg with steps 0.4 m, 0.2 m, 0.4 deg (11 hypotheses per DoF); BEV +-15 x +-30 m at 0.15 m; NLL loss weight 0.1 per DoF + smooth L1 on POET; AdamW lr 1e-4, batch 8, 10 + 15 epochs.
- results: nuScenes MAE 0.132 m longitudinal, 0.128 m lateral, 0.392 deg; about 150 ms per frame on an RTX 3090. Code link: not found.
- differentiable: yes end to end (stage 2). sensors: six cameras, HD map.
- fit_to_us: reuse_idea - effort M - decoupled (yaw, then along, then across) 1-D scoring is a cheap alternative to a full 3-DoF volume; confidence-gated second stage is a template for "refine only if uncertain".
- compat_conflicts: needs a rasterised map and a BEV semantic mask of comparable resolution.
- tried_before: correlation-volume localisation (E103) dropped in favour of cross-attention; registration front end E74 kept.
- verified: partly
