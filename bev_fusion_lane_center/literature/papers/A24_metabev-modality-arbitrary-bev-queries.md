# A24 - MetaBEV: modality-arbitrary BEV queries with switched-modality training

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: MetaBEV: Solving Sensor Failures for BEV Detection and Map Segmentation (Ge et al., ICCV 2023) https://arxiv.org/abs/2304.09801
- depth: M
- stage: B, T
- idea: Dense learnable BEV queries are refined by a decoder with cross-modal deformable attention over camera BEV, LiDAR BEV, or both (modality-specific MLPs give offsets / weights). A missing modality's features are simply dropped (baselines zero-fill). Switched-modality training samples camera-only, LiDAR-only, both with probability 1/3 each, one set of weights.
- input: camera and LiDAR BEV; output: detection + map segmentation.
- training: starting from task-specific pretrained models; mixture-of-experts layers (M2oE) to reduce multi-task conflict.
- differentiable: yes.
- sensors_data: camera, LiDAR, nuScenes.
- results: map segmentation mIoU 70.4 full, 53.7 camera missing, 54.4 LiDAR missing; no temporal component; params / latency not stated.
- code: not checked
- fit_to_us: reuse_idea - effort S - modality dropout with equal-probability modes; for us "radar off" and "cameras off" modes already exist in the failure set.
- compat_needs: per-modality BEV (we only have fused rasters).
- compat_conflicts: foundation is fused; modality-wise dropping needs the second (fused) feature or retraining.
- tried_before: failure-frame fine-tuning.
- verified: partly (2023, base of the line)
