# RB18 - MapBench (and MSC-Bench): robustness benchmark for online HD map construction

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: Is Your HD Map Constructor Reliable under Sensor Corruptions? (Hao et al., NeurIPS 2024 D&B) https://arxiv.org/abs/2406.12214 (opened HTML v3); MSC-Bench: Benchmarking and Analyzing Multi-Sensor Corruption for Driving Perception (arXiv 2501.01037, Jan 2025) https://arxiv.org/abs/2501.01037 (abstract page opened)
- depth: M (MapBench) / A (MSC-Bench)
- stage: E, D
- idea: 29 corruptions (8 camera, 8 LiDAR, 13 multi-sensor combinations) at 3 severities on nuScenes val, 31 map constructors, metrics mAP, mCE (vs MapTR = 100), mRR (corrupted / clean).
- results: MapTR clean 50.3 mAP -> Camera Crash 15.0, Frame Lost 14.2; HIMap 65.5 -> 19.4 / 19.0. Snow worst (MapTR mRR 7.7 %). Fusion not clearly more robust (HIMap fused mRR 41.7 vs camera-only 56.6). What helps (all camera-only): Swin backbone (mCE -22.2 on PivotNet), temporal fusion (StreamMapNet mRR 47.1 -> 55.5, mCE 100 -> 85.9, Table 3), PhotoMetric augmentation (mRR 41.1 -> 49.3, mCE 100 -> 84.5), longer training; rotate / flip augmentation hurt. Combined camera + LiDAR failures worse than either alone. MSC-Bench: 16 combinations of camera / LiDAR corruption, 6 detectors and 4 map models, no radar (no numbers on the page).
- compute: not applicable.
- fit_to_us: baseline - effort S - the metric definitions (mRR, mCE) and the finding that temporal fusion and photometric augmentation are the two cheap robustness levers for map output.
- tried_before: temporal hold, failure fine-tuning (consistent with the two levers).
- verified: checked
