# RB09 - SB-BEVFusion: identity fusion when one sensor is absent, single-branch training

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: SB-BEVFusion: Enhancing the Robustness against Sensor Malfunction and Corruptions (arXiv 2605.11799, May 2026) https://arxiv.org/abs/2605.11799 (opened alphaxiv overview https://www.alphaxiv.org/abs/2605.11799)
- depth: A
- stage: B, T
- idea: Equalise channel counts of camera and LiDAR BEV features, then use an operator (average, max, cross-attention, progressive modality decay) that reduces to identity when only one sensor is active; each training scene is shown as L+C, L-only and C-only in shuffled mini-batches through one shared encoder and head.
- formulation: not read beyond the overview.
- input/output: BEV features -> fused BEV feature.
- training: mixed-availability batches.
- differentiable: yes.
- sensors_data: camera + LiDAR, MultiCorrupt (RB19), nuScenes.
- results: camera only: BEVFusion NDS about 0.70 -> 0.10; LiDAR only SB-BEVFusion 0.6959 vs UniBEV 0.653; average fusion best on mean resistance, attention variants better on spatial misalignment. No latency.
- code: none found.
- fit_to_us: reuse_idea - effort S - the "all availability subsets in every batch" trick, which we can do with cached feature sets per failure type.
- compat_needs/conflicts: as RB07.
- tried_before: none.
- verified: partly
