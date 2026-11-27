# A29 - StreamPETR: object-centric memory queue with motion-aware layer norm

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection (Wang et al., ICCV 2023) https://arxiv.org/abs/2303.11926
- depth: M
- stage: S8
- idea: Memory queue N x K = 4 x 256 top-scoring queries (FIFO). Propagation transformer with motion-aware layer norm: affine parameters (gamma, beta) are produced from the ego transform, velocity and time interval; hybrid attention concatenates stored and current queries. Gradients detached for the first 6 of 8 training frames, loss on the last 2.
- formulation: E = E_t^-1 E_{t-1}; Q~ = E Q (Eq. 8-9); gamma, beta = linear(ego transform, v, dt) (Eq. 10). Explicit motion compensation alone did not help (0.380 vs 0.378 mAP), which they attribute to error propagation early in training. Memory length: mAP 39.4 / 40.1 / 40.2 for 1 / 2 / 4 frames (saturates at about 2 frames, about 1 s).
- results: nuScenes val R50: 43.2 mAP / 54.0 NDS at 27.1 FPS; light version 45.0 / 55.0 at 31.7 FPS; MLN ablation: pose encoding +2.0 mAP, +time and velocity to 40.2 mAP.
- compute: 27.1 - 31.7 FPS.
- code: not checked
- training: AdamW 4e-4, batch 16; random frame skip as augmentation.
- differentiable: partial (detach).
- sensors_data: camera, nuScenes.
- fit_to_us: reuse_idea - effort S - (a) condition a normalisation on ego motion and dt instead of (or beside) a hard warp; (b) truncated-gradient training: detach most of the history, backprop through the last frames only; (c) random frame skip.
- compat_needs: token memory.
- compat_conflicts: detection-specific (velocity).
- tried_before: none.
- verified: partly
