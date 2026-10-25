# RB03 - SafeMap: robust online HD map with missing cameras (view reconstruction + BEV distillation)

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: SafeMap: Robust HD Map Construction from Incomplete Observations (Hao, Kong et al., ICML 2025) https://arxiv.org/abs/2507.00861 (opened HTML https://arxiv.org/html/2507.00861v1); builds on MapBench (RB18)
- depth: M
- stage: B, L, T
- idea: Two plug-ins for MapTR-type camera mappers. G-PVR rebuilds the features of a missing perspective view from the remaining views with a deformable-attention decoder whose reference points are Gaussian-distributed toward the neighbouring views. D-BEVC distils the BEV feature from the complete-view model into the BEV feature built from incomplete views with an MSE loss.
- formulation: L = L_map + 0.05 L_Rec + 5 L_Cor; sigma = 3 for the Gaussian reference sampling; MSE beat L1 (47.1) and KL (46.8) at 47.3 mAP; views randomly dropped in training; AdamW lr 4.2e-4, 8 epochs on nuScenes.
- input: multi-view image features (G-PVR) and BEV features (D-BEVC).
- output: reconstructed view features; corrected BEV features.
- training: Hungarian matching inherited from MapTR (not ours); the distillation term itself is matching-free.
- differentiable: yes (the plug-ins); the host loss uses matching.
- sensors_data: camera only; nuScenes (also Argoverse2); MapTR, HIMap.
- results: nuScenes mAP missing front: MapTR 31.3 -> 42.4, HIMap 38.0 -> 54.3; missing back: MapTR 28.5 -> 40.5, HIMap 33.1 -> 51.3; complete input 50.3 -> 52.5 (MapTR). 1 / 3 / 5 missing views average 39.9 -> 47.3 / 19.5 -> 22.8 / 4.0 -> 6.0. MapBench corruptions: HIMap mRR 56.6 -> 62.8, mCE 100 -> 83.2.
- compute: MapTR 39.1 M -> 39.5 M params, 21.5 -> 21.4 FPS, 2298 -> 2300 MB; HIMap 68.1 M -> 71.7 M, 9.7 -> 9.2 FPS (Table 8).
- code: none found on the page.
- fit_to_us: reuse_idea - effort S for D-BEVC - the clean-teacher MSE on the BEV feature can be applied to cached features (clean feature as target for a corrupted-input repair module); G-PVR needs image features, which we do not have.
- compat_needs: paired clean / incomplete features.
- compat_conflicts: view reconstruction is impossible on our cached fused raster.
- tried_before: none (failure fine-tuning uses the task loss only).
- verified: checked
