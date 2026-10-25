# RB15 - M-BEV: masked view reconstruction for missing cameras

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: M-BEV: Masked BEV Perception for Robust Autonomous Driving (Chen et al., AAAI 2024; arXiv 2312.12144) https://arxiv.org/abs/2312.12144 (abstract page opened); code https://github.com/Sranc3/M-BEV
- depth: A
- stage: B, T
- idea: During training random camera views are masked at the feature level; a masked-view reconstruction module (global or local, learned mask tokens, 3D position embedding, cross-view context) rebuilds them, with the original features as self-supervision.
- results: back camera missing: +10.3 % mAP on PETRv2 (abstract); detection only.
- compute: not stated.
- fit_to_us: reuse_idea - effort M - same family as SafeMap G-PVR; needs per-view features.
- tried_before: none.
- verified: partly
