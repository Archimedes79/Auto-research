# A31 - TempCoBEV: a pluggable temporal module that is trained on cached embeddings and survives lost inputs

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: Unlocking Past Information: Temporal Embeddings in Cooperative Bird's Eye View Prediction (TempCoBEV, 2024) https://arxiv.org/abs/2401.14325
- depth: M
- stage: S8, T
- idea: Module between a fused BEV embedding and a frozen decoder, trained on pre-computed stored embeddings (backbone removed): an importance map per embedding (channel-max + sigmoid, softmax across embeddings), importance-weighted sum as query for deformable cross-attention (4 keypoints, 3 stacked blocks), final importance-weighted blend of current and temporal output so the current frame keeps priority. No explicit ego warp: alignment only by learned deformable offsets.
- formulation: more than one historical frame reduced IoU in the ablation; removing importance components lowered performance.
- training: IoU loss through frozen decoder; per-epoch training time about 50 min -> 3 min.
- differentiable: yes inside; not end-to-end with backbone.
- sensors_data: camera cooperative BEV, simulated OPV2V.
- results: when communication is lost, IoU gain over base models for t+1..t+4 of +19, +13.1, +7.9, +3 (CoBEVT), up to 2 % for the current frame.
- compute: training-time saving; inference latency not stated.
- code: link in paper (not opened)
- fit_to_us: reuse_idea - effort S - same recipe as ours (cached embeddings, frozen decoder, small trained module) and the only paper that evaluates memory as a replacement for lost inputs, but on simulated data; shows that the importance weighting keeps the present dominant.
- compat_needs: stored embedding sequences.
- compat_conflicts: simulated cooperative setting.
- tried_before: E41 temporal feature fusion on cached features (+0.008 F1).
- verified: partly
