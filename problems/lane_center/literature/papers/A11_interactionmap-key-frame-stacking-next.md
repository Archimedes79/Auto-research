# A11 - InteractionMap: key-frame stacking next to streaming

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: InteractionMap: Improving Online Vectorized HDMap Construction with Interaction (CVPR 2025) https://arxiv.org/abs/2503.21659
- depth: M (temporal part only)
- stage: S8
- idea: Local fusion = warped previous BEV + GRU + LayerNorm + residual block (streaming). Global fusion = key-frame selection by distance stride (d_stride = 5 m): either KFS-streaming (previous global feature warped, GRU) or KFS-stacking (up to N_pre = 4 warped key-frame submaps concatenated, residual conv). Argument: pure streaming forgets long-range history.
- formulation: training samples previous submaps randomly within N_pre x d_stride; first half of training single-frame. Ablation (nuScenes mAP): baseline 50.2, plain streaming 54.0, KFS-stacking 57.6, KFS-streaming 57.4.
- input: BEV features, 4x4 relative pose; output: BEV feature for the decoder.
- training: first N_total/2 epochs single-frame.
- differentiable: yes in the fusion.
- sensors_data: camera, nuScenes / AV2.
- results: nuScenes val 71.6 (R, 24 ep) / 75.5 (110 ep); AV2 73.7.
- compute: 12.1 FPS, 49.9 M params (R); 11.0 FPS, 116.7 M (C) (units as written in the paper: "MB").
- code: not checked
- fit_to_us: reuse_idea - effort M - a stack of 2 - 4 distance-spaced warped inputs gives long memory without a recurrent state that can be poisoned; cost is linear in the stack depth on our 200x200x42 raster (conv only).
- compat_needs: distance-based buffer.
- compat_conflicts: more channels into the trace-network input (42 x K).
- tried_before: E41 temporal fusion (+0.008 F1).
- verified: partly
