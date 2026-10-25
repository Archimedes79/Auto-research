# RB11 - RCBEVDet / RCBEVDet++: radar-camera BEV fusion with explicit view / radar dropout tests

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: RCBEVDet: Radar-camera Fusion in Bird's Eye View for 3D Object Detection (Lin et al., CVPR 2024) https://arxiv.org/abs/2403.16440 (opened HTML); RCBEVDet++ (arXiv 2409.04979, abstract page opened: no robustness claim there) https://arxiv.org/abs/2409.04979
- depth: M
- stage: B
- idea: RadarBEVNet encodes radar with a point branch + a distance-modulated transformer branch and scatters points into BEV with an RCS-dependent footprint; a cross-attention multi-layer fusion (deformable, both directions) corrects azimuth misalignment before a conv fusion.
- formulation: deformable cross-attention cost O(HWC^2K); dropout test drops camera views (0 / 1 / 3 / all) or radar input.
- input/output: radar points + camera BEV -> fused BEV for detection.
- training: standard detection training.
- differentiable: yes.
- sensors_data: camera + radar, nuScenes, VoD.
- results (Table 7, car mAP): RCBEVDet drops camera views 72.5 / 66.9 / 53.5 / 16.5 (0 / 1 / 3 / all); CRN 68.8 / 62.4 / 48.9 / 12.8; BEVFusion 63.9 / 58.5 / 45.7 / 14.3. Radar-drop rows RCBEVDet 71.6 / 66.1 / 62.1 against CRN 64.3 / 57.0 / 43.8 (the summary's quoted drop percentages were inconsistent with these cells; use the cells). Camera loss costs far more than radar loss.
- compute: 21.3 FPS (R50, 256 x 704) vs CRN 20.4; RTX 3090, batch 1, FP16; no ms.
- code: github.com/VDIGPKU/RCBEVDet (announced on the abstract page).
- fit_to_us: baseline - effort S - the best available numbers on how much a camera-radar network keeps when radar or cameras go; supports "radar is the fallback, cameras the single point of failure".
- compat_needs/conflicts: detection only; no map output.
- tried_before: none.
- verified: checked
