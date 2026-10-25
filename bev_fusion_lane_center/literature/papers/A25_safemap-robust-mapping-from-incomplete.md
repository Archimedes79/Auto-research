# A25 - SafeMap: robust mapping from incomplete camera sets

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: SafeMap: Robust HD Map Construction from Incomplete Observations (2025) https://arxiv.org/abs/2507.00861
- depth: M
- stage: B, T
- idea: G-PVR reconstructs the feature of a missing view with a learnable query attending to available views (Gaussian-distributed reference points on a panorama tiled by spatial distance, MAE-like decoder, reconstruction loss). D-BEVC pulls the BEV features of the incomplete input toward those of the complete input by MSE (distillation).
- formulation: loss = map + 0.05 L_rec + 5 L_corr; Gaussian sigma = 3; views masked / dropped randomly in training. Temporal use not stated in the paper digest.
- input: multi-view images; output: vector map (MapTR / HIMap host).
- training: fine-tuning 8 epochs nuScenes, 4 x 3090.
- differentiable: yes.
- sensors_data: camera, nuScenes + AV2.
- results: MapTR mAP all views 50.3 -> 52.5; front missing 31.3 -> 42.4; back missing 28.5 -> 40.5; 3 missing views 19.5 -> 22.8; 5 missing 4.0 -> 6.0; HIMap front missing 38.0 -> 54.3.
- compute: MapTR 39.1 M params, 21.5 FPS -> 39.5 M, 21.4 FPS; HIMap 68.1 M, 9.7 FPS -> 71.7 M, 9.2 FPS.
- code: not checked
- fit_to_us: reuse_idea - effort S - the distillation loss (incomplete-input BEV pulled toward complete-input BEV) is a feature-level teacher that can be applied to our 42-channel input: teacher = clean input, student = failed input; no inference cost.
- compat_needs: paired clean / failed samples (we synthesise them).
- compat_conflicts: view reconstruction needs image features (we have none).
- tried_before: failure-frame fine-tuning (label loss only); feature-level distillation not tried.
- verified: partly
