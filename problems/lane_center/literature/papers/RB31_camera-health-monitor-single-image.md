# RB31 - Camera health monitor: single-image degradation index with early warning

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: Safety-Critical Camera Reliability Monitoring for ADAS via Degradation-Aware Uncertainty Pattern Analysis (arXiv 2605.05439, May 2026) https://arxiv.org/abs/2605.05439 (opened HTML)
- depth: M
- stage: E, X
- idea: An EfficientNet-B2 with light heads predicts degradation presence, per-mode severity (12 modes), a spatial degradation map and a multiplicative health index (GSHI); regimes Healthy / Degraded / Critical at 0.9 / 0.6.
- results: health MAE 0.064; lead time 0.47 severity units before a YOLOv8n failure; GSHI correlates 0.95 with detector mAP on failure subset vs 0.90 for YOLO confidence and 0.60 for clean-feature OOD; DAWN zero-shot balanced accuracy 84.2 % (fog / rain / snow). Synthetic KITTI degradations only.
- compute: 7.91 M params, 0.72 GFLOPs, 2.27 ms at 224 x 224 on an RTX 5090.
- fit_to_us: reuse_idea - effort M - needs RGB, not BEV features; but the evaluation idea (correlation with downstream failure, lead time) is directly usable for our heat monitor.
- tried_before: label-free heat monitor.
- verified: checked
