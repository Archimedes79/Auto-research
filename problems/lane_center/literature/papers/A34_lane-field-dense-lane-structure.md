# A34 - Lane Field: dense lane-structure field as conditioning, with uncertainty gates

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: Learning Lane-Structured Conditioning Fields for Online HD Mapping (Bertogalli et al., 2026) https://arxiv.org/abs/2610.25284
- depth: M
- stage: S2, S9
- idea: A 5-channel BEV field (3 for unoriented lane direction, proximity, direction confidence) built analytically from centreline polylines (closed-form Gaussian line integral), predicted by an auxiliary head from BEV features or built from SD-map polylines; used for SPADE-like feature modulation and to stretch deformable-attention sampling along the local lane direction. Guidance is gated by direction confidence, a predicted-uncertainty gate (Laplacian log-scale) and a learnable per-head gate.
- formulation: gradient-flow ablation: fully attached gradients 36.1 mAP vs stop-gradient variants 34.1 - 34.8. No temporal component, no sensor-failure test, no code (project page https://andberto.github.io/LaneField/).
- results: LaneSegNet OpenLane-V2 subset A mAP 32.1 -> 36.1, TOP_lsls 25.5 -> 28.4; SD-map variant 36.8.
- compute: LaneSegNet 14.6 -> 13.9 FPS (RTX 4000 Ada), params 45.4 M -> 52.6 M.
- code: none found
- training: end-to-end, closed-form target.
- differentiable: yes.
- sensors_data: camera + optional SD map.
- fit_to_us: reuse_idea - effort S/M - our heat + 2 direction channels are the same kind of field; the sampling-direction idea matches our fan sampling; the uncertainty gate shows how to switch the guidance off when unsure.
- compat_needs: dense field head (we have).
- compat_conflicts: none.
- tried_before: dense head heat + direction (ours).
- verified: partly
