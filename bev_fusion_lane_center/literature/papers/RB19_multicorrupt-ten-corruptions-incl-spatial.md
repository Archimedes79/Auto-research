# RB19 - MultiCorrupt: ten corruptions incl. spatial misalignment (miscalibration) and missing camera

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: MultiCorrupt: A Multi-Modal Robustness Dataset and Benchmark of LiDAR-Camera Fusion for 3D Object Detection (Beemelmanns, Zhang, Eckstein, IEEE IV 2024) https://arxiv.org/abs/2402.11677 (opened HTML); code github.com/ika-rwth-aachen/MultiCorrupt (listing)
- depth: M
- stage: E, D
- idea: Darkness, brightness, points reducing, temporal misalignment, spatial misalignment (extrinsic rotation / translation offsets), motion blur, missing camera (random frame drops), beams reducing, fog, snow; resistance ability RA = corrupted / clean NDS, mRA over severities.
- results: mRA CMT 0.865, SparseFusion 0.834, BEVFusion 0.830, TransFusion 0.824, DeepInteraction 0.795; lowest single cells DeepInteraction fog 0.583, TransFusion spatial misalignment 0.685. Findings: masked-modal training (CMT) and independent modality handling help; LiDAR-only query initialisation and early deep coupling hurt.
- fit_to_us: baseline - effort S - gives a ready recipe for spatial misalignment and temporal misalignment synthesis; detection only, LiDAR-camera.
- tried_before: none (we lack an extrinsics test).
- verified: checked
