# RB16 - ReliFusion: contrastive cross-modal reliability scores for confidence-weighted fusion

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: Reliability-Driven LiDAR-Camera Fusion for Robust 3D Object Detection (Sadeghian et al., arXiv 2502.01856, Feb 2025) https://arxiv.org/abs/2502.01856 (opened HTML v1)
- depth: M
- stage: B
- idea: Two MLPs map LiDAR and camera BEV features into a shared 128-d space; a contrastive loss (tau 0.07) separates matched from corrupted pairs; a sigmoid head gives one confidence per modality; confidence-weighted mutual cross-attention multiplies each modality's attended features by its confidence.
- training: contrastive + confidence regression pretraining (target not defined in what I read), confidence loss weight 0.05 in fine-tuning.
- results: clean 70.6 mAP / 73.2 NDS; LiDAR FOV +-pi/2: 52.4 / 59.6; zero FOV 24.6 / 28.7 (tables inconsistent between Table 2 and 4). No latency.
- code: none found.
- fit_to_us: reuse_idea - effort M - reliability from contrastive agreement between two streams; same limit as RB05.
- tried_before: none.
- verified: checked
