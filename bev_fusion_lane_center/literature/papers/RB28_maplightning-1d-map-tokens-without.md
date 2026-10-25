# RB28 - MapLightning: 1D map tokens without camera projection

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: MapLightning: Online Vectorized HD Map Construction with 1D Map Tokens (Zheng, Ghosh, Ramanagopal, Narasimhan, arXiv 2610.01905, 1 Oct 2026) https://arxiv.org/abs/2610.01905 (abstract page opened)
- depth: A
- stage: B
- idea: Learnable 1D map tokens plus image tokens in one self-attention transformer; image tokens are discarded and map tokens decoded. Claim: because camera projection parameters are not used, it is robust to extrinsic perturbation (no experiment on the page).
- results: lightweight variant +10.1 mAP vs MapTRv2 on nuScenes, +16.2 on Argoverse 2, 1.73x faster (40+ FPS), 53 % less memory, 16.7x fewer tokens (abstract).
- code: announced, no link.
- fit_to_us: reuse_idea - effort L - same calibration-free argument as CFT; vectorized map decoders were already compared (E31, E32).
- tried_before: sparse DETR-style decoder dropped (E31, E32).
- verified: checked (abstract)
