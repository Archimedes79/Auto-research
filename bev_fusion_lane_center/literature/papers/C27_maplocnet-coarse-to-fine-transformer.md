# C27 - MapLocNet: coarse-to-fine transformer registration to a navigation map (regression head)

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: MapLocNet: Coarse-to-Fine Feature Registration for Visual Re-Localization in Navigation Maps (Wu et al., 2024) https://arxiv.org/abs/2407.08561
- depth: M
- stage: S9
- idea: Rasterised OSM patch (128 x 128 m, 0.5 m) -> U-Net / VGG-16 map features; BEV (LSS) and map features are fused (7 x 7 conv, 4 x down-sampled), 3-layer transformer encoder + 3-layer MLP regress a 3-DoF offset (coarse), then the same on fine features; final pose = composition T2 T1. Smooth L1 on both stages + BCE segmentation, weights 1; training noise +-30 m and +-30 deg.
- results: nuScenes six cameras position recall @1 / 2 / 5 / 10 m 20.10 / 45.54 / 77.70 / 91.89; orientation @1 deg 58.61; single camera 8.96 / 27.05 / 64.57 / 85.86 vs OrienterNet 5.83 / 18.92 / 52.83; 22.9 M params, 56.15 GFLOPs, 18.2 FPS (V100) for six cameras, 24.4 FPS one-stage; OrienterNet 54.9 M and 8.1 FPS single camera. 200 epochs, about 48 h on 8 V100.
- differentiable: yes (regression head, no differentiable optimiser).
- fit_to_us: reuse_idea - effort S/M - plain regression of (dx, dy, dtheta) from fused tokens is the simplest differentiable registration head; accuracy (20 % within 1 m) shows regression on weak OSM is crude.
- tried_before: registration front end E74 / E48 kept. verified: partly
