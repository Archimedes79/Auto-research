# C34 - Online map uncertainty handed to a downstream consumer (Laplace per vertex)

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: Producing and Leveraging Online Map Uncertainty in Trajectory Prediction (Gu et al., CVPR 2024) https://arxiv.org/abs/2403.16439 ; code https://github.com/alfredgu001324/MapUncertaintyPrediction ; related: Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention https://arxiv.org/abs/2407.06683
- depth: M
- stage: S10, L
- idea: The map head outputs, per vertex, a Laplace location mu and scale b per coordinate (NLL: log 2b + |v - mu| / b) plus class probabilities; mu, b and class probabilities are concatenated into the predictor's polyline encoder (sigma = sqrt(2) b when rotating). Related paper: skip the vector map and let agents attend to BEV patches (patch 20 x 20, about 6 m).
- results: HiVT + MapTR minADE 0.4015 -> 0.3854, minFDE 0.8418 -> 0.7909, MR 0.0981 -> 0.0834; DenseTNT + MapTR convergence 88 -> 44 epochs; not every pair improves (DenseTNT + MapTRv2-centerline MR 0.1520 -> 0.1593); mapping mAP within about 2 %. Direct BEV attention: up to 29 % better minFDE and up to 73 % faster inference (HiVT + MapTR 22.4 -> 9.1 ms, RTX 4090), trained not jointly.
- differentiable: yes (the uncertainty is a plain regression output).
- fit_to_us: reuse_idea - effort S - add per-vertex log-scale and the route probability / entropy to the consumer input; E117 already receives edges + route probability, so only the scale is new.
- tried_before: none for scale. verified: partly
