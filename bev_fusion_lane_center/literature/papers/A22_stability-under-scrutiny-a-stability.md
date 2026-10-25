# A22 - Stability under scrutiny: a stability metric for temporal map models

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: Stability Under Scrutiny: Benchmarking Representation Paradigms for Online HD Mapping (2025) https://arxiv.org/abs/2510.10660
- depth: M (first 100 k characters)
- stage: E
- idea: mAS = mean average stability, per element presence x weighted mix of localisation jitter and shape (curvature) difference across frames; 42 models evaluated.
- formulation: details of the weights not captured.
- results: StreamMapNet temporal vs not: mAS 83.8 -> 91.9, mAP 51.7 -> 63.3; accuracy and stability are largely independent (HRMapNet with training-map priors mAP 83.6 but mAS 76.7; MapTracker 75.95 / 90.4); camera-only MapTR mAS 71.6 -> 61.9 when the sampling interval grows to 10 frames.
- compute: n/a. code: not checked.
- sensors_data: camera, nuScenes.
- differentiable: n/a.
- fit_to_us: reuse_idea - effort S - our "temporal consistency" metric is undefined in the context; presence x jitter x shape on matched pieces is a ready recipe; also says global raster priors improve accuracy but not stability.
- compat_needs: matched elements across frames (we have a piece graph).
- compat_conflicts: none.
- tried_before: temporal consistency metric exists (definition not given in context).
- verified: partly
