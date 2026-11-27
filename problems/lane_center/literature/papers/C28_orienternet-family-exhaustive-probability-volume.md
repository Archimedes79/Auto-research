# C28 - OrienterNet family (OrienterNet, OSMLoc, hierarchical re-localisation 2603.01613, DiffVL): exhaustive probability volume over pose

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: OrienterNet (Sarlin et al., CVPR 2023, older anchor) https://arxiv.org/abs/2304.02009 ; OSMLoc (2024) https://arxiv.org/abs/2411.08665 ; Uncertainty-Aware Hierarchical Re-Localization in OpenStreetMap (arXiv 2603.01613, v2 June 2026) https://arxiv.org/abs/2603.01613 ; DiffVL (2025) https://arxiv.org/abs/2509.14565
- depth: M
- stage: S9
- idea: Neural map F (U-Net on rasterised OSM, 50 cm/px, N = 8 channels) is correlated with a confidence-masked neural BEV T (32 x 32 m at 0.5 m) for every (x, y, theta); posterior P = softmax(M + Omega) with a learned per-cell location prior; rotations by batched multiplication in the Fourier domain (K = 64 train, 512 test); loss = NLL of the GT pose; multi-frame fusion by warping and multiplying per-frame volumes.
- results: OrienterNet MGL position recall @1 / 3 / 5 m 15.78 / 47.75 / 58.98; KITTI 20 s sequences lateral @1 m 79.71 %; about 94 ms per localisation (37 ms BEV + 51 ms matching, RTX 2080); multi-frame fusion roughly doubles recall; fails in repetitive scenes and where the map lacks distinctive elements. OSMLoc: MGL 15.30 / 46.05 / 59.40 vs 10.69 / 42.12 / 54.01, 9.27 FPS (4090). 2603.01613: coarse 1 m / 6 deg grid by FFT correlation, fine 0.5 m / 2 deg; uncertainty U = 1 - (peak mass in a neighbourhood) only sets the fine search window, NOT an abstain; 25.80 FPS vs OrienterNet 7.51. DiffVL: refinement ablation 0.1554 vs 0.0978 recall@1 m (KITTI); runtime not reported.
- differentiable: yes (volume + softmax; argmax only at readout).
- sensors_data: gravity-aligned camera, OSM; training prior offsets up to +-20 m.
- fit_to_us: reuse_idea - effort M - the volume is too heavy for 31 ms with our 200 x 200 grid, but the loss (NLL of the true pose in a small volume, e.g. across-road offsets only, 1-D) and temporal product of volumes are exactly the missing tools for per-road lateral bias.
- tried_before: correlation-volume localisation E103 dropped (accuracy near cross-attention, more cost).
- verified: partly
