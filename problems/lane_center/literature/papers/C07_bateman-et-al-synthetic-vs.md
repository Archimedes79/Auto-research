# C07 - Bateman et al.: synthetic vs real map change gap

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: Exploring Real World Map Change Generalization of Prior-Informed HD Map Prediction Models (Bateman et al., CVPRW 2024) https://arxiv.org/abs/2406.01961
- depth: M
- stage: T, E
- idea: Prior tokens replace queries in a MapTR-style model; trained with synthetic prior perturbations, then tested against a real 3-year-old map version (2020 prior, 2023 truth). Reports which synthetic perturbation best predicts real robustness and shows the gap remains.
- formulation: discrete perturbations (feature dropout, duplication, wrong class), continuous (control point perturbation, location perturbation, global rotation / shift, Perlin warp). Training 75 k steps on 32 A100.
- results: mAP no-noise baseline 0.8980 synthetic / 0.8239 real; "Low All Noise" 0.9934 / 0.8571 (gap ~0.136). Best single settings on real data: localisation noise 0.5 m / 0.5 deg 0.8648, control-point 0.5 m 0.8640, Perlin warp 1.0 m 0.8623, duplicate p = 0.5 0.8604. Heavily corrupted priors can be worse than no prior; for complex real changes (new medians, construction) the model tends to copy the prior.
- data: Houston + Mountain View, 13.7 k training scenes, 1,240 real-change holdout scenes; private, so not reproducible by us.
- compute / code: not stated / none found.
- fit_to_us: reuse_idea - effort S - a warning for the metric: synthetic map offsets overstate robustness; keep a "copy-the-map" check (does the route head follow the map when the lanes disagree?).
- compat_conflicts: none. tried_before: map-agreement monitor (label-free) is related.
- verified: partly
