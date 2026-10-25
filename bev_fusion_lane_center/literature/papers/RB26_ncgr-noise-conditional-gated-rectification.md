# RB26 - NCGR: noise-conditional gated rectification for camera extrinsic perturbations

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: NCGR: Noise-Conditional Gated Rectification for Camera Extrinsic Perturbations in BEV 3D Object Detection (Pan et al., arXiv 2608.03895, 4 Aug 2026) https://arxiv.org/abs/2608.03895 (abstract page opened)
- depth: A
- stage: B, T
- idea: For spatial-cross-attention BEV detectors, predict a 2D offset per query-camera pair, scaled by a per-camera gate, to correct wrong projections instead of estimating a 6-DoF extrinsic; clean-teacher / perturbed-student training with a BEV-consistency loss; no perturbation metadata at inference.
- results: five-camera dynamic stress test 39.69 % NDS vs BEVFormer 28.00 and CAPE 33.23; clean performance comparable to BEVFormer.
- compute/code: not reported / none found.
- fit_to_us: reuse_idea - effort L - the repair sits inside the image-to-BEV transform, which we cannot change; the teacher-student BEV-consistency loss is portable to RB01-type modules.
- tried_before: none (failure fine-tuning did not help extrinsics).
- verified: checked (abstract)
