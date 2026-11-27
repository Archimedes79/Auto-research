# A02 - SIGMA-Lane: gated Mamba2 temporal state that is protected against contaminated frames

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: SIGMA-Lane: Scale-pyramId Gated MAmba for Temporally Consistent Video Lane Detection (Zhang, Wang, Gao, Kong, Shen, Du, ECCV 2026) https://arxiv.org/abs/2608.16338
- depth: M
- stage: S8
- idea: Video lane detection with an SSM temporal core (Mamba2). The authors name the failure mode "state contamination": an occluded or corrupted observation written into the recurrent state damages later frames. They gate the write path (input gate: x' = x * (1 - m), m = token-wise occlusion probability) and the residual-fusion path (scalar retention ratio r from the mean mask via small MLP + sigmoid, initialised near 0.2). A lane-guided affine warp aligns the previous features before the SSM, and a cross-attention "structural spatial retrieval" recovers lane structure from the aligned past that the gates cannot restore.
- formulation: SSM update h_t = A_t h_{t-1} + B_t x_t, gate scales the write term; memory bank = past 8 frames' features + masks + previous lane mask; learnable start token; affine residual (2x3, tanh, zero-initialised final layer) from lane-masked pooled previous features + current features. Gate signal = obstacle mask from a frozen segmenter; gate placement matters more than mask precision (removing both gates drops F1 by 0.018 on OpenLane-V).
- input: aggregated frame features + obstacle masks (image space, not BEV).
- output: lane masks / coefficients (per-frame lane detection).
- training: base detector frozen, only the temporal aggregation modules trained (0.35 M params) with the base detector's losses. Hyper-parameters deferred to the supplement (not read).
- differentiable: yes in the module (gates, SSM, affine); gradient across frames not discussed.
- sensors_data: camera video, VIL-100, OpenLane-V; 384x640.
- results: VIL-100 mIoU 0.801, F1@0.5 0.940; ablation (VIL-100) baseline Mamba2 swap 0.775 mIoU / flicker R_F@0.5 0.052, both gates 0.787 / 0.024, full model 0.801 / 0.023. OpenLane-V mIoU 0.761, F1@0.5 0.838.
- compute: OpenLane-V, RTX 4090: 27.1 GFLOPs, 118.8 model FPS (96.7 end-to-end), 3.96 M main params, 0.35 M aggregation params.
- code: not checked
- fit_to_us: reuse_idea / reimplement - effort S/M - the gated write is a one-line change to our spatial hold (our hold already prefers the previous input where its lane heat is higher; this makes the write depend on an explicit per-pixel corruption probability). The only paper in the set that models contamination of the state explicitly.
- compat_needs: a per-pixel corruption / reliability map m (we know which cameras we switch off in training, so m can be supervised; at test time it must be predicted).
- compat_conflicts: image-space affine warp is a short-range approximation, not ego-motion; our BEV uses exact ego warp.
- tried_before: none (spatial hold is the training-free analogue).
- verified: partly
