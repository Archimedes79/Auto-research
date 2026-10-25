# RB08 - UniBEV: uniform BEV encoders, modality dropout and channel-normalised fusion

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: UniBEV: Multi-modal 3D Object Detection with Uniform BEV Encoders for Robustness against Missing Sensor Modalities (Wang, Caesar, Nan, Kooij, IEEE IV 2024) https://arxiv.org/abs/2309.14516 (opened HTML v3)
- depth: M
- stage: B, T
- idea: Both modalities are mapped to BEV with the same deformable-attention scheme; features are fused by channel-normalised weights (CNW) over the available modalities only (missing ones are skipped, not zero-filled). Training with modality dropout p_md = 0.5, LiDAR kept with p = 0.5.
- formulation: fusion options concat 51.9, average 52.3, CNW 52.5 mAP averaged over input combinations; roughly 50 % full, 25 % LiDAR-only, 25 % camera-only.
- input: camera + LiDAR BEV. output: fused BEV -> detection head.
- training: modality dropout.
- differentiable: yes.
- sensors_data: nuScenes val, R101-FPN + VoxelNet.
- results: camera-only NDS/mAP 42.4 / 35.0 vs BEVFusion 29.6 / 22.6 and MetaBEV 33.6 / 25.9; LiDAR-only 65.3 / 58.2; all combos average mAP 52.5 vs 43.5 (BEVFusion) and 48.7 (MetaBEV) (Table I, baselines retrained without augmentation).
- compute: about 1.6 FPS fused on a V100, batch 1 (BEVFusion 0.7, MetaBEV 1.4).
- code: github.com/tudelft-iv/UniBEV (from search listing, not opened).
- fit_to_us: reuse_idea - effort S - skip-missing averaging as the safe fusion operator; map segmentation explicitly left as future work.
- compat_needs/conflicts: as RB07.
- tried_before: none.
- verified: checked
