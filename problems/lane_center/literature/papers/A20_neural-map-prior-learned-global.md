# A20 - Neural Map Prior: learned global BEV feature memory with gated ConvGRU update

*Pass: loop 17 (2026-10-24), file `A_streaming_memory.md`, cluster `streaming_memory`. Depth M here means: method sections queried through a page digest (see sources/lit17/A_streaming_memory.md, section 1).*

- papers: Neural Map Prior for Autonomous Driving (Xiong et al., CVPR 2023) https://arxiv.org/abs/2304.08481
- depth: M
- stage: S8
- idea: Sparse tile-based global neural map (0.3 m default). Current BEV queries cross-attend to the prior (current-to-prior attention, 10x10 patches); a 2D ConvGRU (update + reset gate) writes fused features back to the tiles. Pose transforms each BEV cell to the global frame.
- formulation: pose noise not discussed in the digest.
- input: BEV features + global neural map; output: refined BEV (segmentation or vector heads).
- training: end-to-end with gated update.
- differentiable: yes (also through the memory update).
- sensors_data: camera + GPS / IMU, nuScenes.
- results: mIoU +4.32 (HDMapNet), +5.02 (LSS), +5.50 (BEVFormer); VectorMapNet +3.90 mAP; larger gains in rain (+5.92) and night (+7.73) than normal (+5.20); night-rain +2.90.
- compute: inference "comparable to previous methods"; the HRMapNet paper says NMP's BEV-feature memory needs over 11 GB for Boston vs about 120 MB for a raster map (from HRMapNet digest).
- code: not checked
- fit_to_us: reuse_idea - effort - - the gated ConvGRU write is the template for a learned gate; the large night gains show a prior helps most when the sensor is weak. Global memory itself needs repeat drives and consistent pose (not our setting: 255 scenes, scene-disjoint).
- compat_needs: global localisation.
- compat_conflicts: scene-disjoint split makes the global prior useless in our protocol.
- tried_before: E40 / E48.
- verified: partly (2023 paper, included as base of the 2024+ line)
