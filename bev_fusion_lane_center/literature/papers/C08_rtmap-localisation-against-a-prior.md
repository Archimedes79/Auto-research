# C08 - RTMap: localisation against a prior map + change detection inside one network

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: RTMap: Real-Time Recursive Mapping with Change Detection and Localization (Du et al., ICCV 2025) https://arxiv.org/abs/2507.00980 ; code announced https://github.com/CN-ADLab/RTMap (availability not verified)
- depth: M
- stage: S9, S8
- idea: Prior-map elements are queries that are either "map" (still valid) or "fake" (outdated) at training time using ground-truth change labels; at inference outdated elements show up as low category confidence. Pose against the prior map is obtained either by an end-to-end delta-pose MLP (smooth L1) or by an LM least-squares step on matched vertices with a Mahalanobis residual.
- formulation: per-vertex Laplace density with NLL + Manhattan loss, weights 0.03 and 5.0 (lambda1, lambda2) over 20 vertices per element; Hungarian matching for new elements.
- differentiable: partial - MLP pose yes; LM pose reported as better but gradient through the solver not stated; Hungarian in the detection loss.
- sensors_data: camera, nuScenes + TbV-style data.
- results: TbV lateral error mean / 90th 0.125 / 0.256 m (with map queries) vs 0.163 / 0.318; nuScenes (LM pose) lateral 0.121 m, longitudinal 0.586 m, yaw 0.368 deg; change detection mAcc 57.4 vs 54.1 (changed class 40.0 -> 48.9). FPS: not reported in the content.
- fit_to_us: reuse_idea - effort M - the "map query confidence = is this map element still true" idea fits as a per-road validity output of the route head; the LM / MAP pose step is a template for per-road lateral bias with covariance.
- compat_conflicts: needs Hungarian for new elements (we avoid it).
- tried_before: label-free map-agreement monitor (project) is the same goal without labels.
- verified: partly
