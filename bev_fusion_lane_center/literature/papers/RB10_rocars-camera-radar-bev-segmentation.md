# RB10 - RoCaRS: camera-radar BEV segmentation for sensor failure (link not verified)

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: RoCaRS: Robust Camera-Radar BEV Segmentation for Sensor Failure Scenarios (Park, Kim, Cho, Hwang, 2025) - link not verified (no arXiv or publisher page opened; abstract text seen only in search snippets that came from a lab database listing)
- depth: A (snippet level)
- stage: B, T
- idea: A camera-radar BEV segmentation model with a radar-aware backbone and "feature spreading", a Dynamic Input Dropout Strategy (DIDS) for training with missing inputs, and a Bidirectional Feature Refinement (BFR) module that fills the missing branch from the present one.
- formulation: not read.
- input/output: camera + radar -> BEV map and vehicle segmentation.
- results (snippet, unverified): with all cameras removed +23.2 mIoU (map) and +30.0 IoU (vehicle) over baseline; keeps 99 % of radar-only and 103 % of camera-only performance when all cameras or all radars are disabled, without retraining. nuScenes.
- compute/code: not stated / none found.
- fit_to_us: reuse_idea - effort unknown - the only camera-radar map-segmentation failure paper found, but I could not read it; worth locating the real paper before relying on any detail.
- compat_needs/conflicts: separable camera / radar branches.
- tried_before: none.
- verified: from memory (snippets only)
