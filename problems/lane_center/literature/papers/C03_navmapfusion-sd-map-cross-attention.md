# C03 - NavMapFusion: SD-map cross-attention with noise / wrong-map training, discrepancy = diffusion noise

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: NavMapFusion: Diffusion-based Fusion of Navigation Maps for Online Vectorized HD Map Construction (Monninger et al., WACV 2026) https://arxiv.org/abs/2512.03317 ; code https://github.com/tmonnin/navmapfusion
- depth: M
- stage: S9, T
- idea: A diffusion decoder denoises queries into map elements; the SD map enters via a Map Cross-Attention before the sensor cross-attention, and the paper views the mismatch between map and sensor as the noise the diffusion is supposed to remove. The practical value is its systematic robustness study: translation noise, mis-aligned maps, element dropout, wrong location, wrong frame.
- formulation: 6 decoder layers; BEV grid 100 x 50; BEV dropout d = 0.30 (zeroed BEV grid, to avoid over-reliance on sensors); AdamW 2e-4, cosine, 24 epochs; k denoising steps (k = 1 / 3 / 5), final k = 5, eta 0.5, tau 0.5.
- input: BEV features + SD map (OSM) features; random queries.
- output: vectorised HD map elements.
- training: standard detection loss with perturbed SD map; training perturbations with probability p (e.g. 10 %).
- differentiable: yes (no matching on the map side).
- sensors_data: camera; nuScenes (A6000, 100 x 50 BEV).
- results: mAP 26.6 aligned; Gaussian translation noise hurts beyond sigma = 1 m, training with sigma = 1 m keeps performance up to test sigma = 3 m (values only in a figure). Aligned- vs unaligned-trained: 26.6 / 26.0 vs 26.8 / 26.5. Wrong map: random location (p = 1) 26.6 -> 20.1; other frame's map (p = 1) -> 17.8; training with p = 0.1 limited this to 24.4 / 24.0 (still above the 22.4 MapDiffusion baseline, no-map reference); element dropout above 0.7 falls below the no-SD-map reference, 10 % dropout in training most robust. FPS 14.7 (k = 1), 10.3 (k = 3), 8.1 (k = 5).
- compute: A6000 FPS above; params not stated.
- code: https://github.com/tmonnin/navmapfusion (not opened)
- fit_to_us: reuse_idea - effort S - take the training recipe: sigma = 1 m shift noise, 10 % element dropout, p = 0.1 "wrong map" (other frame / random place) samples; and report the same sweep for route success.
- compat_needs: SD map available as random-access tokens, ability to swap in another frame's map in the data loader.
- compat_conflicts: diffusion decoder itself is not needed.
- tried_before: map offsets 0 - 3 m in route-head training and evaluation (yes); explicit wrong-map and element-dropout training samples: not recorded as run.
- verified: partly

## Correction from the level-2 extraction (2026-10-24)

- The paper has no map-trust, gate or abstain output; robustness comes only from train-time corruptions (wrong map in 10 % of samples, polyline dropout 0.1 - 0.3, noise sigma 1 m). "Discrepancy = diffusion noise" is a framing: the forward noise acts on the ground-truth query polylines and the noise level is the diffusion timestep, tied to no measured mismatch.
- The official repository holds a README and a GPL-3.0 LICENSE and no code (host model: tmonnin/mapdiffusion, GPL-3.0).
- "Tried before" is out of date: our map-fault training (E87 and earlier, `degrade_sd` in `training/data/sd_lib.py`) covers shape sparsification, lane-count errors and missing side roads; it never drops the route polyline itself (checked in the code).

Details: [../deep/C03_navmapfusion.md](../deep/C03_navmapfusion.md), [../code/navmapfusion.md](../code/navmapfusion.md).
