# C25 - U-ViLAR: differentiable perception-to-map association with uncertainty

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: U-ViLAR: Uncertainty-Aware Visual Localization for Autonomous Driving via Differentiable Association and Registration (Li et al., ICCV 2025) https://arxiv.org/abs/2507.04503
- depth: M
- stage: S9
- idea: Similarity matrix between visual BEV and rasterised-map BEV features -> softmax association distribution (Gaussian soft target); a pixel-wise perceptual uncertainty map is concatenated so uncertain regions (occlusion borders) are relaxed; a decoder outputs independent x, y, heading bin distributions (Shannon entropy per DoF as uncertainty), a cost volume around the coarse pose is fused with the joint prior by a learnable weight, refined pose = softmax-weighted average.
- training: uncertainty-weighted reconstruction, CE to the Gaussian target, symmetric contrastive loss in local windows (weights not stated).
- results: nuScenes HD map multi-view MAE lateral 0.040 m, longitudinal 0.140 m, orientation 0.075 deg (BEV-Locator 0.076 / 0.178 / 0.510); OSM, nuScenes multi-view position recall@1m 34.53 % vs MapLocNet 20.10 %; 28 FPS on a V100, 15 FPS on Orin with TensorRT INT8. GPS noise simulated as +-2 m / +-2 deg (fine) or +-30 m / +-30 deg (relocalisation); no dedicated map-error study.
- differentiable: yes (softmax-weighted pose). code: none found in content.
- fit_to_us: reuse_idea - effort M - soft-argmax pose (weighted average over a cost volume) with an entropy-based uncertainty is a gradient-friendly pose block and gives a confidence output for abstention.
- tried_before: correlation volume localisation (E103) dropped. verified: partly
