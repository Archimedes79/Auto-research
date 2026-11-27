# C10 - P-MapNet: SD raster tokens with BEV-query cross-attention against weak alignment

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: P-MapNet: Far-seeing Map Generator Enhanced by both SDMap and HDMap Priors (Jiang et al., RA-L 2024) https://arxiv.org/abs/2403.10521
- depth: M
- stage: S9, B
- idea: Rasterised OSM is encoded by a CNN with sine position embedding into tokens; BEV features are down-sampled by d = 4 (50 x 25 grid) to queries; two cross-attention layers let each BEV query find the best-aligned road token, which is the paper's answer to weak GPS alignment. An optional MAE-pretrained HD-map refiner (mask ratio 50 %) is attached.
- results: nuScenes 120 x 60 m C+L mIoU: HDMapNet 49.50, P-MapNet SD 60.20, SD+HD 61.77; cross-attention 60.20 vs concatenation 54.73 / CNN concat 56.27; excluding OSM "service" roads +about 2 mIoU; at 240 x 60 m (camera) 26.77 -> 45.50. FPS 21.2 (HDMapNet), 18.7 (SD), 9.6 (SD+HD) on RTX-class GPU. No pose-noise experiment; paper only discusses misalignment qualitatively.
- differentiable: yes. code: project page https://jike5.github.io/P-MapNet/ (from abstract, not opened).
- fit_to_us: reuse_idea - effort S - same pattern as ours; useful numbers: two cross-attention layers best, four overfit; drop low-value road classes from the SD raster.
- tried_before: E104b cross-attention kept.
- verified: partly
