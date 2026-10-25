# RB33 - LIAR: illumination-guided sampling and projection for nighttime occupancy

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: See through the Dark: Learning Illumination-affined Representations for Nighttime Occupancy Prediction (arXiv 2505.20641) https://arxiv.org/abs/2505.20641 (opened HTML v3); code https://github.com/yanzq95/LIAR
- depth: M
- stage: B
- idea: A frozen Retinex-style illumination estimator (pretrained on night images, since joint training collapses on day-dominated data) gates enhancement, scales deformable offsets in 2D sampling and weights BEV residuals along vertical columns.
- results: Occ3D-nuScenes night mIoU 19.27 (1 frame) / 22.09 (2 frames) vs 20.01 second best; day+night training 23.67 / 27.33; nuScenes-C darkening easy / moderate / hard 30.66 / 25.67 / 17.31. Generic low-light preprocessors on other models gave only +0.08 to +0.37 mIoU.
- compute: 4.7 FPS, 64.39 M params, 5.05 GB (RTX 4090).
- fit_to_us: not_suitable - effort L - image-side; the lesson is that off-the-shelf enhancement barely helps and illumination priors must be integrated into sampling.
- verified: checked
