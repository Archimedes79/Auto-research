# A23 - Post Fusion Stabilizer: a 3.3 M adapter between frozen fused BEV and frozen head

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: Post Fusion Bird's Eye View Feature Stabilization for Robust Multimodal 3D Detection (Dong, Thakkar, Sargolzaei, Lin, 2026) https://arxiv.org/abs/2603.05623
- depth: M
- stage: B, T
- idea: A near-identity module between the fused BEV map and the frozen detection head, trained on corrupted data only (host frozen). Block 1: global-context shift normalisation, gate sigmoid(alpha), alpha init -5 (about 0.0067). Block 2: per-pixel reliability map R in [0,1] from shifted features + raw LiDAR BEV (bias init +4), F_clean = R * F_shift; BCE to a point-density ratio target on corrupted samples plus an anchor pushing R to 1 on clean samples (weight 0.2). Block 3: semantic and geometric experts give correction dF, spatial gate G (bias -4), output F_shift + G * dF.
- formulation: three-stage curriculum (6 / 4 / 4 epochs; AdamW 2e-4 / 1e-4 / 2e-4); corruptions per modality with probability 0.30 each: 49 % clean, 21 % camera-only, 21 % LiDAR-only, 9 % both.
- input: fused BEV feature (+ raw LiDAR BEV); output: corrected BEV feature.
- training: host detection losses + reliability BCE; host frozen.
- differentiable: yes.
- sensors_data: camera-LiDAR, nuScenes val.
- results: BEVFusion mAP clean 69.2 -> 69.5; camera dropout (6 views) 62.6 -> 63.8; low light 58.8 -> 63.2; rain lens 60.0 -> 63.6; 8-beam LiDAR 12.1 -> 17.5. No temporal element.
- compute: BEVFusion 40.8 M -> 44.1 M params, 70.9 -> 76.6 ms (+8.1 %), 14.1 -> 13.1 FPS on RTX 4080 Super.
- code: not checked
- fit_to_us: reimplement - effort S/M - this is the cleanest match to our situation (frozen foundation, cached BEV, small head trained for failures): a near-identity residual adapter with a supervised reliability map, trained by a curriculum that leaves clean performance untouched. We know the corruption mask exactly in training (cameras off, night, blur).
- compat_needs: a reliability target (density of valid input per cell); our radar and camera are already fused, so the target must come from known camera coverage.
- compat_conflicts: our features are PCA channels + logits (not a raw fused map); the adapter would act on the 42-ch trace input.
- tried_before: failure-frame fine-tuning (project memory); a supervised reliability map and near-identity gating not tried.
- verified: partly (digest notes text vs table inconsistencies in the real-world test)
