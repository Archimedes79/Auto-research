# C02 - UMPE: unified map-prior encoder with SE(2) micro-alignment, confidence and gating

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: Unified Map Prior Encoder for Mapping and Planning (Zhang et al., ICRA 2026 accepted, arXiv May 2026) https://arxiv.org/abs/2605.02762 ; code https://github.com/Ethan-Zheng136/UMPE (link from paper)
- depth: M
- stage: S9, B (prior fusion), S10
- idea: Fuse any subset of HD vector, SD vector, SD raster and satellite priors into BEV features. Vector priors get a per-frame rigid correction, polyline tokens get a sigmoid confidence that enters cross-attention as a log bias, raster priors get a regressed pose that is applied with grid_sample, and a per-channel softmax gate with a presence mask mixes the sources; missing sources are trained by source dropout.
- formulation: vector branch: frame-wise (dx,dy,dtheta) applied to all polyline points, regulariser lambda_t * ||T||^2 + lambda_r * dtheta^2 (lambdas not stated; the predictor of (dx,dy,dtheta) not stated). Cross-attention logit bias log(clamp(U, eps, 1)). Gate: softmax over sources of (L + log(pi + eps)), pi = presence mask. Raster branch: ResNet-18 + FiLM, pose regressed from raster + BEV reference, converted to a normalised affine (metres per pixel) and applied with grid_sample; zero-initialised residual fusion. SourceDropout p = 0.3 per encoder during training. 60 epochs, 8 x A800.
- input: BEV features of the host mapper (MapTRv2 / MapQR / VAD) + prior polylines / rasters.
- output: prior-augmented BEV features (+ polyline tokens with confidence).
- training: end-to-end with the host loss (mapping or VAD planning loss); no extra labels.
- differentiable: yes in form (grid_sample, softmax gate); the paper does not claim a gradient path for the vector correction.
- sensors_data: camera BEV; nuScenes, Argoverse2.
- results: nuScenes mapping MapTRv2 61.5 -> 67.4 mAP, MapQR 66.4 -> 71.7; Argoverse2 +4.1 mAP. Planning with VAD backbone (nuScenes val, average): L2 0.72 -> 0.42 m, collision 0.22 -> 0.12 %; vector encoder only 0.48 / 0.17, raster only 0.55 / 0.19; other priors in the same table: P-MapNet 0.68 / 0.66, PriorDrive 0.73 / 0.61. Test-time subsets (no retraining): SD raster only 64.5 mAP vs 63.5 for a single-prior model. No injected-noise experiment, no wrong-prior experiment (stated in the paper content I read).
- compute: MapTRv2 40.3 M params / 11.57 FPS -> 54.8 M / 10.98 FPS with UMPE on an RTX 3090; planning model latency not stated.
- code: https://github.com/Ethan-Zheng136/UMPE (not opened)
- fit_to_us: reuse_idea - effort S/M - take (1) presence-mask gate + source dropout for "missing map", (2) zero-init residual fusion, (3) a small SE(2) micro-alignment with grid_sample on the SD raster; the host BEV encoder is not ours.
- compat_needs: map tokens or raster with metric size, a place to add a residual (our SD raster tokens before the cross-attention).
- compat_conflicts: SE(2) correction is global per frame - cannot fix per-road lateral bias.
- tried_before: learned registration of the SD raster (E40 / E48) kept; "no map" training mode: not run as an explicit trained mode.
- verified: partly (paper vague about the correction network; gradient path unconfirmed)
