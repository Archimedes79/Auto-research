# C18 - Hidden Biases: the target-point shortcut

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: Hidden Biases of End-to-End Driving Models (Jaeger, Chitta, Geiger, ICCV 2023) https://arxiv.org/abs/2306.07957 (older anchor)
- depth: M
- stage: S10, T
- idea: Planners that get target points (GNSS points about 30 m apart on lane centres) steer towards the nearest point and so recover from errors, but can cut turns when the point is far; the decoder puts about 25 % of its attention on the target-point token (16.5 x uniform).
- results (CARLA): TransFuser with target points 39 DS / 84 RC with no route deviations, with discrete commands 32 DS / 56 RC; AIM 86 RC (target points) vs 50 RC (commands). Mitigations: transformer-decoder pooling instead of global average pooling (RC 84 -> 93, static collisions halved), shift / rotation augmentation (about 5 x fewer static collisions), predicting path + classified target speed.
- fit_to_us: reuse_idea - effort S - risk for our consumer: it may rely on the route probability instead of the geometry; use route-attribute dropout and shifted-map augmentation.
- differentiable: yes. tried_before: none. verified: partly
