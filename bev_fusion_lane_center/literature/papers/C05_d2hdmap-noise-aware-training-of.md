# C05 - D2HDMap: noise-aware training of a map-prior model with translation and rotation errors

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: D2HDMap: Non-visible Driveline Map Prior for Online Vectorized HD Map Prediction (2026) https://arxiv.org/abs/2606.20725 (authors not extracted)
- depth: M
- stage: T, S9
- idea: Lane drivelines (centre-lines) as a non-visible prior rasterised and used to initialise map queries; the useful part is the clean table of pose-noise training vs test.
- formulation: noise = zero-mean Gaussian translation (sigma 0 / 0.5 / 1 / 2 m) and rotation (0 / 2.5 / 5 deg), the same perturbed pose used for prior retrieval and for accumulating the temporal map. Whether the training noise set equals the test set is unclear in the text.
- input: BEV + rasterised driveline prior; output: vector map. Training: detection loss with noisy prior; regime matching_free is not the case (host uses DETR matching).
- differentiable: yes.
- sensors_data: nuScenes geosplit, Argoverse 2; clean mAP 44.8.
- results: robust model test mAP: 0.5 m 38.8; 1.0 m 36.0; 2.0 m 31.8; 2.5 deg 37.3; 5 deg 34.7; 2 m + 5 deg 32.3. Robust vs non-robust: +3.86 mAP average, +6.9 at 2 m / 5 deg (32.3 vs 25.4); mean loss 14.2 -> 10.3, worst case -19.4 -> -13.1. No misalignment estimation.
- compute: not stated. code: none found (no link in the content).
- fit_to_us: baseline/reuse_idea - effort S - note that mAP falls roughly linearly with shift even with noise training: shifts are absorbed only partly; our route success loses 0.03 per metre which is a much flatter curve.
- compat_needs: none. compat_conflicts: none.
- tried_before: map offsets 0 - 3 m: yes.
- verified: partly
