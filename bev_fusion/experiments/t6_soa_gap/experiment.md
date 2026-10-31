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

## 3 Result (run overnight by the lane: trained 02:07 - 03:48, evaluated 03:48 - 03:57 on 2026-10-31; `runs/t6_maps_only/eval/`)

Cold VAL, 1000 keyframes of 82 scenes, dt 0, camera cells (`temp/soa_gap/steps_curve.py`, `compare_maps_only.py`; paired scene bootstrap 1000
draws, b - a, `evaluation/bev_legacy.compare`):

| layer | ref_s0 | ref_s1 | **maps-only** | gain vs ref_s0 [95 %] | gain vs ref_s1 [95 %] | ref_long (6000 steps) | GaussianCaR (other subset) |
|---|---|---|---|---|---|---|---|
| drivable | 0.666 | 0.663 | **0.681** | +0.015 [+0.006, +0.024] | +0.018 [+0.007, +0.028] | 0.689 | 0.845 |
| walkway | 0.191 | 0.164 | **0.228** | +0.037 [+0.028, +0.046] | +0.064 [+0.053, +0.075] | 0.225 | 0.594 |
| carpark | 0.106 | 0.113 | **0.146** | +0.040 (no interval: not in the legacy records) | +0.033 | 0.155 | 0.659 |
| pedestrian crossing | 0.002 | 0.000 | **0.042** | +0.040 (same) | +0.042 | 0.057 | 0.536 |
| lane divider (0.5 m) | 0.181 | 0.179 | **0.227** | +0.047 [+0.035, +0.059] | +0.049 [+0.035, +0.062] | 0.234 | 0.691 |
| road divider (0.5 m) | 0.220 | 0.230 | **0.271** | +0.051 (same) | +0.041 | 0.297 | 0.634 |

Validity check passed: vehicle AP@2 0.6061 -> 0.0001 and pedestrian AP@2 0.2039 -> 0.0000 (the objects are not trained), so the weights applied.
Seed spread for scale (ref_s0 -> ref_s1): walkway -0.027 [-0.036, -0.018], lane divider -0.002, drivable -0.003.

**Verdict by the written rule: R2 - the features, not the task competition, are the main limit.** No thin layer gains more than the band
0.054 over `ref_s0` (walkway +0.037, lane divider +0.047, road divider +0.051; carpark +0.040, crossing +0.040), and the mean gain of the five
thin / rare layers (+0.043) is below the gain of doubling the steps (`ref_long` +0.054). R1 is not met. Caveats: (a) the effect is real and
one-signed - all six layers rise and the three layers with intervals exclude 0 - so competition costs the maps something; (b) against the seed
twin `ref_s1` the walkway gain is +0.064, above the band; the pre-registered control `ref_s0` decides, and the two seeds differ by 0.027
there; (c) one seed, and the run also switched the radar aux head off (the camera aux head stays).

How much of the gap to GaussianCaR does the removal of the competition close? gain / (SOA - ref_s0): drivable 8 %, walkway 9 %, carpark 7 %,
crossing 8 %, lane divider 9 %, road divider 12 % (the SOA numbers are from a different 400-frame subset, so the shares are approximate):
**about a tenth**. The remaining nine tenths are upstream of the task balance (features, budget, data).

Next, by the rule: map-supervised, higher-resolution image features (plan 2.7 / 2.8) and an unfreezing arm; do not build task decoupling.
Related run of another chat (t7, one seed, band 0.02 on the lane divider): a frozen stride-4 ResNet branch lifted into the BEV gives lane
divider 0.246 (+0.065 vs ref_s0), road divider 0.345 (+0.125), drivable 0.677, walkway 0.200, carpark 0.130, crossing 0.000: finer
image content helps the thin LINES more than removing the competition does, and does nothing for the classes that need semantics
(walkway, carpark, crossing). The two effects are different and may add; their combination was not run.

## 2 Log

- 2026-10-30: config and gate done (commit bb745cf); queued as the last line of `temp/study/queue3.txt` (`cfg:t6_maps_only:0a3d73c`, after
  `train:lift_s` and `train:trunk_small`; `temp/soa_gap/queue_maps_only.sh`); the owner of the lane (session "BEV system") was told. The lane
  was evaluating `trunk_second` at the time; the run starts after the two jobs in front of it.
- 2026-10-31: finished (exit 0, evaluated); result and verdict in section 3 (the section sits above this log).
