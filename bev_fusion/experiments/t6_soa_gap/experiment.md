# t6 maps-only test - does detection competition hold the maps back? (card written 2026-10-30, before the run; Mirko: "Yes, maps-only")

Context: `report.md` (same folder). Our maps reach 17 - 85 % of GaussianCaR's IoU on the same frames and underfit (train ~ val). Two
candidates remain that the earlier arms cannot separate: (M6) the frozen, detection-trained stride-16 image features hold too little map
information; (M7) the detection loss dominates the shared trunk (gradient at the BEV feature: objects 0.111, lines 0.012, areas 0.005).

## 1 Card

- **Question.** With everything else identical, how much do the maps gain when the objects and the motion head no longer take part in the
  training? (Task competition alone, at an unchanged feature source.)
- **Hypothesis.** If task competition limits the maps (M7), removing it lifts the map IoU clearly above the 3000-step reference; if the
  features limit them (M6), the maps stay where they are, because the same frozen features and the same trunk capacity are used.
- **Design.** `training/configs/t6_maps_only.json` = `t2_ref_s0.json` with three changes: `balancing "fixed"` with
  `fixed_weights [1, 1, 0, 0]` (areas, lines, objects, motion: the objects and motion groups carry weight 0, so no gradient from them
  reaches any parameter), `w_radar_aux 0` (the radar auxiliary head only supervises objects and the stationary proxy). Unchanged: code
  (pinned commit 0a3d73c = the code of `t2_ref_s0`, queued as `cfg:t6_maps_only:0a3d73c`), data and scenes (428 train, 82 val), 3000
  steps, seed 0, lr, clips of 3, flip augmentation, dt weights and warm-up, camera aux head (weight 0.3: it still carries a dense object
  term on the camera BEV, which does not reach the shared trunk), EMA. Evaluation: the lane's `bev_suite eval --n 1000` (cold VAL, 1000
  keyframes of 82 scenes), the table of `t2_block_swaps/results/arms.md`.
- **Controls (same frames, same evaluation).** `t2_ref_s0` (3000 steps, Kendall, all tasks), its seed twin `t2_ref_s1`, and `t3_ref_long`
  (6000 steps, different commit 1a6b8d8) for the value of doubling the steps.
  | layer | ref_s0 | ref_s1 | ref_long (6000) | GaussianCaR (400 frames, other subset) |
  |---|---|---|---|---|
  | drivable | 0.666 | 0.663 | 0.689 | 0.845 |
  | walkway | 0.191 | 0.164 | 0.225 | 0.594 |
  | carpark | 0.106 | 0.113 | 0.155 | 0.659 |
  | pedestrian crossing | 0.002 | 0.000 | 0.057 | 0.536 |
  | lane divider (0.5 m) | 0.181 | 0.179 | 0.234 | 0.691 |
  | road divider (0.5 m) | 0.220 | 0.230 | 0.297 | 0.634 |
- **Prediction (before the run).** The maps gain about what a doubling of the steps gives: walkway 0.22 +- 0.04, lane divider 0.23 +- 0.04,
  road divider 0.28 +- 0.05, carpark 0.14, crossing 0.03, drivable 0.68. Maps near the SOA (walkway >= 0.5) would be a surprise (< 10 %).
- **Decision rule (written now).** Noise band 0.054 on the map layers (2 x the seed spread of `t3`; walkway sets it). Gain = maps-only minus `ref_s0`.
  - R1 competition is a main limiter: walkway, lane divider and road divider each gain > 0.054 AND the mean gain of the five thin / rare
    layers (walkway, carpark, crossing, two dividers) is >= the gain of `ref_long`. Next: decouple the tasks (maps-first schedule, gradient
    scaling of the objects at the trunk, or a separate map branch).
  - R2 the features limit the maps: all layers within the band of `ref_s0` (and below `ref_long`). Next: map-supervised, higher-resolution image
    features (plan 2.7) and an unfreezing arm; task decoupling is not worth building.
  - R3 mixed: one or two layers outside the band -> add the seed twin before deciding.
- **Gate (done, `temp/soa_gap/check_maps_only_cfg.py`, 2026-10-30).** On the pinned code the config builds, the task weights are (1, 1, 0, 0), the
  radar aux weight is 0, and the gradient of the objects and motion groups with respect to a shared parameter is exactly 0.
- **Kill / validity criterion.** The vehicle AP of the run must collapse (the object head is not trained: AP@2 about 0); if it does not,
  the weights did not apply and the run is void. A run that diverges or whose loss is NaN is repeated once at the same settings.
- **Cost.** One GPU job, about 1.7 - 2 h for 3000 steps plus about 10 min evaluation; it waits behind `lift_s` and `trunk_small` in the
  lane's queue (`temp/study/queue3.txt`).

## 2 Log

- 2026-10-30: config and gate done; queued at the end of the lane's queue (see the commit of this card).
