# RB04 - RoboMap: what matters for robust multi-sensor HD map construction (augmentation + fusion + modality dropout)

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: What Really Matters for Robust Multi-Sensor HD Map Construction? (arXiv 2507.01484, 2025) https://arxiv.org/abs/2507.01484 (opened HTML https://arxiv.org/html/2507.01484); project https://robomap-123.github.io/
- depth: M
- stage: B, T
- idea: A recipe study on MapTR / HIMap with camera + LiDAR: camera GridMask and LiDAR point dropout augmentation, a cross-modal interaction transformer (CIT, self-attention over concatenated flattened BEV tokens) with a dynamic fusion, and modality dropout of whole BEV feature maps.
- formulation: drop probability p_md; if dropped, LiDAR kept with p_L, camera with 1 - p_L (values not reported). ResNet50 + GKT camera, SECOND-style LiDAR, AdamW lr 4.2e-4.
- input: camera BEV and LiDAR BEV features.
- output: fused BEV feature for the map head.
- training: end-to-end; Hungarian loss of the host; no frozen foundation.
- differentiable: yes.
- sensors_data: camera + LiDAR; nuScenes val with the MapBench corruption suite.
- results: clean mAP MapTR (C+L) 62.5 -> 71.5; HIMap 74.3 -> 77.0 (Table I); mRS MapTR 69.51 -> 79.06, HIMap 67.69 -> 81.99 (Tables II / III). Ablation gains on MapTR (fusion / augmentation / dropout) +3.3 / +4.5 / +1.6 mAP; components help only in combination. Snow stays weak (19.55 -> 23.78); motion blur 70.00 -> 89.88.
- compute: not reported.
- code: none found (project page only).
- fit_to_us: reuse_idea - effort S - confirms that augmentation gives the largest single gain and that modality dropout is a small add-on; with a frozen camera-radar foundation, dropout can only be emulated by caching foundation features for each failure condition.
- compat_needs: per-condition feature caches.
- compat_conflicts: needs separable modality BEV maps; ours are already fused.
- tried_before: failure-frame training (corresponds to the augmentation part).
- verified: checked
