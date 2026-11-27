# RB06 - Grace-BEV: trust gate + failsafe fusion trained with modality dropout over a frozen vision backbone

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: Can BEV Perception Gracefully Degrade under Sensor Failures? (Zhang, Wang, Wang, Li, Zuo, arXiv 2605.30983, 29 May 2026) https://arxiv.org/abs/2605.30983 (opened HTML)
- depth: M
- stage: B, T
- idea: A plug-in for LSS-style BEV fusion. TrustGate pools the LiDAR BEV into a descriptor, a small MLP outputs a scalar trust alpha, and blends a LiDAR-guided expert with a vision-only expert; a FailSafe block applies a learned element-wise gate. Three-phase training: vision pre-training, frozen backbone with modality dropout (both 0.5, LiDAR-drop 0.25, camera-drop 0.25), then low-learning-rate end-to-end.
- formulation: bottleneck MLP 64 hidden units, ReLU, sigmoid; gate reduction 4 with dilated 3 x 3 conv, last bias init 0; loss not named.
- input: LiDAR BEV + image BEV features. output: fused BEV feature.
- training: staged, frozen backbone in stage 2.
- differentiable: yes.
- sensors_data: camera + LiDAR; nuScenes-R (LiDAR beam reduction / drop / limited FOV, camera view drop, object reflection noise), nuScenes-C weather.
- results: LiDAR drop BEVFusion-AD 0.0 -> 34.7 mAP, BEVFusion-MIT 0.0 -> 32.8; clean +1.1 / +1.4 mAP; ablation router alone 28.5, block alone 15.5, both 32.8 (LiDAR drop, MIT). Detection only; no map segmentation.
- compute: +1.2 M params; BEVFusion-MIT latency 40.76 -> 40.81 ms (A100, batch 1).
- code: none found.
- fit_to_us: reuse_idea - effort S - proves a scalar trust gate is almost free in ms (0.05 ms) and shows the three-phase frozen-backbone training recipe.
- compat_needs: a sensor-separable BEV pair.
- compat_conflicts: our fused features offer no pair; a gate would blend "repaired" and "raw" features instead.
- tried_before: none.
- verified: checked
