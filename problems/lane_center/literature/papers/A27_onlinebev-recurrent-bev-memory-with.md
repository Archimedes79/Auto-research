# A27 - OnlineBEV: recurrent BEV memory with motion-guided deformable alignment and a consistency loss

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: OnlineBEV: Recurrent Temporal Fusion in Bird's Eye View Representations for Multi-Camera 3D Perception (2025) https://arxiv.org/abs/2507.08644
- depth: M
- stage: S8, L
- idea: One recurrent BEV state H_{t-1}, updated per layer (L = 3) by a motion-guided fusion: a motion feature extractor takes the difference between target and history queries (FC + channel attention) and predicts deformable-attention offsets and weights that align the ego-compensated history to the current frame. HTC-loss: shared heads produce heatmaps from H_t and from the aligned history; L2 between them with stop-gradient on the target branch, weight 2.
- formulation: loss = focal (1) + L1 (0.25) + 2 HTC. Whether BPTT or detach: not stated; memory resets at scene start. Two phases: 6 epochs single-frame then 90 epochs recurrent.
- input: BEV queries; output: detection heatmaps.
- training: 4 x 3090, batch 32.
- differentiable: yes in module.
- sensors_data: camera, nuScenes.
- results: val R50 44.4 mAP / 54.5 NDS; ablation (NDS): single frame 41.9, + recurrent fusion +8.5, + motion-guided +1.0, + HTC +0.5.
- compute: 79.3 ms (SOLOFusion 72.8), 205.7 GFLOPs, 65.0 M params, 3.4 GB, RTX 3090.
- code: not checked
- fit_to_us: reuse_idea - effort M - learned residual alignment after the ego warp (offsets from a motion feature) is the only concrete answer to "ego motion error" at feature level; plus a training-only heat-map consistency loss with stop-gradient on the target branch (cheap, A09-like).
- compat_needs: dense BEV (ours); deformable attention over 200x200 is expensive; a plain predicted flow field + grid_sample would be the cheap version.
- compat_conflicts: designed for moving objects (flow), our lanes are static, so the offsets would mainly absorb pose error.
- tried_before: learned registration for map priors (E48).
- verified: partly
