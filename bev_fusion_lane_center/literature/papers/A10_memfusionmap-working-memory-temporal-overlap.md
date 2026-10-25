# A10 - MemFusionMap: working memory + temporal overlap heatmap

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: MemFusionMap: Working Memory Fusion for Online Vectorized HD Map Construction (Song et al., WACV 2025) https://arxiv.org/abs/2409.18737
- depth: M
- stage: S8
- idea: The last T_WM = 4 BEV features (warped by ego motion) are concatenated with the current BEV and a "temporal overlap heatmap" feature, then passed through three dilated (2,2) 3x3 conv layers + LayerNorm; a GRU still carries the long state. The heatmap starts as ones, is warped by ego motion with zero padding and incremented by 1 per step, so it counts how often a cell has been seen; a small conv stack with sigmoid turns it into a 32-channel feature that tells the network how much to trust the present vs the past.
- formulation: T_WM best 4 (mAP 24.3 / 25.9 / 27.6 / 26.2 / 27.0 for 1 / 2 / 4 / 6 / 8, 50 m nuScenes new split); ablation 100x50 m: no GRU 19.5, + working memory 25.5, + dilation 25.9, + heatmap 27.6; sigmoid vs ReLU on heatmap 27.6 vs 26.5. No gating; memory tensor detached. The heatmap is unbounded over long sequences and not occlusion-aware (paper's own limitation).
- input: BEV features, ego motion; output: BEV feature for a MapTR-style decoder.
- training: two-stage (single frame, then temporal), 8 x V100.
- differentiable: yes in the fusion; detached across frames.
- sensors_data: camera, nuScenes / AV2 new splits.
- results: nuScenes 30 m 38.0 vs StreamMapNet 34.1; 100x50 m 27.6 vs 22.2; AV2 60.6 vs 58.3; max gain 5.4 mAP. About 15.1 FPS (StreamMapNet 15.3, MapTracker 11.6, A6000). No pose-error or missing-frame test.
- compute: 15.1 FPS.
- code: not checked
- fit_to_us: reimplement - effort S - an "age / coverage" channel next to the held input costs one channel and tells the tracer how stale each cell is; it is the simplest published trust cue. Can be extended with a "this frame failed" channel.
- compat_needs: warped history stack (we have the hold).
- compat_conflicts: none.
- tried_before: E41 temporal feature fusion (small gain); hold with decay (loop 16). Age channel not tried.
- verified: partly
