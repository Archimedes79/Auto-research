# A26 - ReSMap and other reliability-weighted fusions (ReSMap, ReliFusion, RoCaRS)

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: ReSMap: Recasting Satellite Priors for Robust and Accurate Online HD Map Construction (Kim, Lee, Jeong, Lim, Park, Hwang, NeurIPS 2026 poster) https://neurips.cc/virtual/2026/poster/149564 ; ReliFusion: Reliability-driven LiDAR-camera fusion (2025) https://arxiv.org/abs/2502.01856 (listing seen in search, abstract not opened) ; RoCaRS: Robust Camera-Radar BEV Segmentation for Sensor Failure Scenarios (Park, Kim, Cho, Hwang, 2025) - no arXiv id found, link not verified (only a database page seen)
- depth: A (ReSMap page abstract; others from search snippets)
- stage: B, S8
- idea: ReSMap: Predict-then-Fuse weights camera and satellite BEV features by each branch's own prediction confidence as a per-pixel reliability weight (no router); decoder with per-modality attention and a learned per-token gate; satellite as redundancy when cameras fail. ReliFusion: per-modality confidence scores feed cross-attention, plus a spatio-temporal aggregation module (not read). RoCaRS: dynamic input dropout + bidirectional feature refinement for camera-radar BEV segmentation.
- formulation: not read.
- results: ReSMap "state of the art under camera failure", no numbers on the page. RoCaRS (abstract summary, unverified): keeps 99 % of the radar-only model when all cameras are off, +23.2 mIoU for map without camera over the baseline.
- differentiable: yes (weights are products of predictions).
- compute: not stated.
- code: ReSMap project page dehsurc.github.io/resmap-page (named, not opened)
- fit_to_us: reuse_idea - effort S - "use the model's own confidence heat map as per-pixel weight between present and held input" is what our hold does without learning; ReSMap gives a published template for making it smooth rather than a hard max.
- compat_needs: two branches that both produce confidence (current frame, held frame).
- compat_conflicts: none.
- tried_before: hold = hard selection where previous heat is higher.
- verified: partly (A only)
