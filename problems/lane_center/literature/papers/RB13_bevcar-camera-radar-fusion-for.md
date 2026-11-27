# RB13 - BEVCar: camera-radar fusion for BEV map and vehicle segmentation, day / rain / night split

*Pass: loop 17 (2026-10-24), file `B_failure_robustness.md`, cluster `failure_robustness`. Depth M here means: method sections queried through a page digest (see sources/lit17/B_failure_robustness.md, section 1).*

- papers: BEVCar: Camera-Radar Fusion for BEV Map and Object Segmentation (Schramm, Vödisch et al., IROS 2024) https://arxiv.org/abs/2403.11761 (opened HTML); project https://bevcar.cs.uni-freiburg.de/
- depth: M
- stage: B
- idea: Frozen DINOv2 ViT-B/14 + adapter on six cameras; radar point encoding guides deformable-attention lifting of image features to BEV; radar BEV queries cross-attend to the lifted image features; one head outputs vehicle and map channels.
- formulation: 200 x 200 grid over 100 m x 100 m (0.5 m), 5 radar sweeps; vehicle BCE, map alpha-balanced focal loss (alpha 0.25, gamma 3).
- input: six images 448 x 896 + radar points. output: BEV segmentation (map + vehicle).
- training: end-to-end except the frozen DINOv2.
- differentiable: yes.
- sensors_data: camera + radar, nuScenes (28,130 train / 6,019 val; val split day 4,449 / rain 968 / night 602).
- results: mIoU 70.9 (camera + radar) vs 65.0 camera-only; night vehicle / map IoU 57.4 / 42.4 (BEVCar) vs 42.1 / 39.2 (camera-only BEVCar) vs 52.4 / 37.8 (Simple-BEV++); day map 57.3 vs night 42.4. Radar helps vehicles at night (+15 IoU) but map classes only a little (+3).
- compute: not reported here; GaussianCaR's paper lists 245.6 ms.
- code: github.com/robot-learning-freiburg/bevcar (from search listing).
- fit_to_us: baseline - effort S - the number to remember: radar does not repair night for planar map classes (lane-like layers); this is exactly our failure.
- tried_before: night / blur failure fine-tuning.
- verified: checked
