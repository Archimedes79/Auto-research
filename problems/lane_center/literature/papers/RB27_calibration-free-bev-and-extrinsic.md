# RB27 - Calibration-free BEV (CFT) and extrinsic-sensitivity evidence

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: Multi-Camera Calibration Free BEV Representation for 3D Object Detection (arXiv 2210.17252, Oct 2022; outside the date window, included because it is the standard calibration-free reference) https://arxiv.org/abs/2210.17252 (abstract page opened); noisy-extrinsic numbers for BEVFormer / PETRv2 come from a search snippet that cited arXiv 2203.17270 / 2206.01256 (not opened)
- depth: A
- stage: B
- idea: Learn the image-to-BEV mapping with position-aware enhancement and view-aware attention without any intrinsics / extrinsics, so noisy calibration cannot hurt.
- results: 49.7 NDS on nuScenes leaderboard; no noise experiment on the page I opened (snippet only: BEVFormer loses 0.6 - 6.8 % and PETR 4.7 - 7.6 % NDS under small extrinsic noise while CFT stays stable - not verified).
- fit_to_us: not_suitable - effort L - the projection lives inside the frozen foundation.
- verified: partly
