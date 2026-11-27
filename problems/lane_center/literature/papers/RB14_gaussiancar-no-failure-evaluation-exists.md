# RB14 - GaussianCaR (our foundation): no failure evaluation exists

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: GaussianCaR: Gaussian Splatting for Efficient Camera-Radar Fusion (Montiel-Marin et al., ICRA 2026) https://arxiv.org/abs/2602.08784 (opened HTML https://arxiv.org/html/2602.08784v1); code https://www.github.com/santimontiel/gaussiancar; project https://www.santimontiel.eu/projects/gaussiancar
- depth: M
- stage: B
- idea: Camera pixels (depth distribution + offset) and radar points (Point Transformer v3) become 3D Gaussians with 128-d features, alpha-blended into BEV; a four-stage multi-scale fusion (CMX-like modules) and a DPT decoder produce segmentation.
- results: vehicle 57.3, drivable area 82.9, lane divider 50.1 IoU; GaussianLSS camera baseline 46.1; radar sweeps 1 / 4 / 7 give 54.9 / 56.2 / 57.3.
- compute: 75.6 ms vehicle / 81.1 ms map on RTX 4090 at 448 x 800 (Table IV); BEVCar 245.6 ms; params not reported.
- robustness: the paper has NO sensor-failure, night, rain or modality-removal experiment (only qualitative figures); so the robustness of our frozen foundation is unmeasured in the literature and is purely our own finding.
- fit_to_us: baseline - effort S - any "radar saves the day" or "camera needed" claim must be measured on this model; the Gaussian representation might make a per-Gaussian opacity gate possible, but we cannot touch the frozen weights.
- tried_before: our own failure tests.
- verified: checked
