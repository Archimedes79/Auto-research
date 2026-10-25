# RB07 - MetaBEV: modality-arbitrary decoder with switched-modality training

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: MetaBEV: Solving Sensor Failures for 3D Detection and Map Segmentation (Ge et al., ICCV 2023) https://arxiv.org/abs/2304.09801 (opened ar5iv https://ar5iv.labs.arxiv.org/html/2304.09801); CVF page https://openaccess.thecvf.com/content/ICCV2023/html/Ge_MetaBEV_Solving_Sensor_Failures_for_3D_Detection_and_Map_Segmentation_ICCV_2023_paper.html
- depth: M
- stage: B, T
- idea: Learnable dense BEV queries (180 x 180) cross-attend to camera and LiDAR BEV features through four deformable cross-modal layers plus two self-attention layers, with modality-specific MLPs producing offsets and weights, so any subset of sensors can be used. Mixture-of-expert FFNs (M2oE) serve detection and segmentation. Training randomly feeds camera-only, LiDAR-only or both at 1/3 each (switched-modality training) starting from a pretrained task model.
- formulation: detection and segmentation losses weighted 10 : 1 (as BEVFusion); AdamW wd 0.05, cyclic lr, 26 epochs (detection) / 20 (segmentation); Swin-T + VoxelNet encoders. Frozen backbones: not stated.
- input: camera BEV + LiDAR BEV. output: dense BEV queries -> heads (map segmentation head included).
- training: sensor-switching, no matching for the map head.
- differentiable: yes.
- sensors_data: camera + LiDAR, nuScenes val.
- results: map segmentation mIoU, missing camera: BEVFusion 4.1 vs MetaBEV 53.7; missing LiDAR: 36.7 vs 54.4; full 70.4 (Table 2; baselines run with the missing input zeroed). Detection NDS missing LiDAR 42.6.
- compute: latency, FPS, parameters not reported (limitations mention a slight increase).
- code: not checked.
- fit_to_us: reuse_idea - effort M - one of the few papers with BEV MAP results under sensor loss; shows zero-filling is a bad baseline and a learned decoder over "available subset" fixes it.
- compat_needs: separable modality BEV maps.
- compat_conflicts: conflicts with a pre-fused frozen raster.
- tried_before: none.
- verified: checked
