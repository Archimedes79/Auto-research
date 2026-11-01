# t11 motion field - is the dead motion head of the time mover revived by the same hurdle, and does it lift detection at dt +-0.5 s? (card written 2026-10-31 23:50, before any code or run)

Context (system review of the val loss group "motion", 2026-10-31 night; numbers from stored files, nothing re-run):
- Finding: the val group "motion" is 1.0945 at steps 1000 / 2000 / 3000 in `t10_dual_s0`, `t8_bevfeat_s0` and `t8_ctl_s0` (`runs/<run>/log.jsonl` "val" rows: 1.094472 ->
  1.094471 -> 1.094470 in t10, the same to five digits in the other two; t1 showed the same at 1.126). The term is computed correctly - `losses.motion_term`
  (`training/losses.py:354-361`, added to group "motion" at dt = 0 only, `losses.py:476-478`) evaluates `MotionHead.loss` (`training/timed_head.py:46-53`: L1 to the GT velocity on object
  cells, GT below 1 m/s set to 0, 0.1 x L1 to 0 elsewhere, each part normalised by its own cell count) on `v_hat = self.mover.motion(S)` (`training/model.py:365`) under the
  EMA weights (`train.py` `evaluate`). What is flat is the head: its last layer starts at zero (`training/splat.py:96-97`) and is still at zero after 3000 steps -
  `runs/t10_dual_s0/ckpt_00003000.pt`: `mover.motion.f.net.2.weight` max |w| 8.8e-4 (mean 4e-5), bias 1.7e-7, the EMA copy the same; `t8_ctl_s0` 1.2e-3. So `v_hat` is about 1e-3 m/s
  everywhere and the val term is the loss of the zero field (on 400 random train keyframes the zero field gives 0.65 per component; the val subset's 1.0945 is the same quantity on other frames).
- Why it does not move: the same zero-inflated L1 as the object head (about 80 % of the supervised object cells are slower than 1 m/s and set to 0; 19.6 % move, measured on the same
  400 keyframes, plus the background term to 0), with an extra trap: at an exactly-zero output the L1 subgradient of every zero-target cell is 0, and the only other teacher, the
  train-only cross-time consistency that the trainer's selftest names as the motion head's gradient source (`train.py:810`), is off in every run (`"consistency": 0.0`), and the
  gradient through the splat from the dt +-0.5 losses is about 1e-6 at the BEV feature (t1 review, `../t1_full_chain/experiment.md`).
- What it means. For AVE: nothing directly - AVE reads the object head's `vel` (`object_head.py:180`), see `../t11_vel_hurdle/experiment.md`. For the decoding at dt != 0: the time mover
  moves the state by the ego motion only, so a moving vehicle (6.6 m/s mean over the matched moving vehicles) lands about 3.3 m from where it is 0.5 s later, beyond the 2 m
  match threshold. t10_dual_s0 (`eval/table.txt`): vehicle AP@2 0.614 at dt 0, 0.319 at +0.5 s, 0.365 at -0.5 s; vehicle footprint IoU 0.413 -> 0.318 at +0.5 s. 36 % of the
  matched vehicles move (1107 / 3087), so the dead field explains a good part, not all, of the halving.

## 1 Card

| | |
|---|---|
| lens | system review (a dead head with a flat val curve nobody flagged) / training and losses |
| group | motion (time decoding at dt +-0.5 s) |
| sources | val rows of t1 / t8 / t10, t10 checkpoint, t1 motion review, plan 8.2 / 8.4 motion head spec, `../t11_vel_hurdle/experiment.md` |
| status | proposed (runs after t11.vel_hurdle; stacked on its arm B) |

- **Question.** Does a moving-cell logit next to the velocity field (hurdle: the field is trained on moving cells only, gated by the logit) make the motion head learn in 1000 steps, and does that lift detection at dt +-0.5 s?
- **Hypothesis (falsifiable).** The motion head is dead for the same reason as the object velocity (zero-inflated L1 plus a zero start, no other teacher); with the hurdle the
  effective field `v_hat = v_raw * sigmoid(m)` follows the moving objects, the splat moves them, and vehicle AP@2 at dt +0.5 rises by >= 0.054. Refuted if the comparable motion loss stays at the zero level (>= 1.09).
- **The change (small, default off, bit-identical defaults; the coder builds it with the test below).**
  1. `ModelConfig.motion_moving: bool = False` -> `MotionHead(cin, hidden, v_max, moving=True)` (`training/timed_head.py`): a second 1 x 1 MLP `m` (cin -> hidden -> 1, bias of the
     last layer -4.0); `forward(state)` returns `v_raw * sigmoid(m)` (what the splat uses, differentiable), `parts(state)` returns `(v_raw, logit)`; `v_raw` keeps the zero start.
  2. `model.py:365`: at dt = 0 also `p["v_parts"] = self.mover.motion.parts(S)` when `motion_moving`.
  3. `losses.motion_term`: with `v_parts`: moving cell = object cell with |v_gt| >= `motion_min_speed` (1 m/s); loss = BCE(logit, moving) on object cells (normalised by their count) +
     0.1 x BCE(logit, 0) on the other valid cells (normalised by theirs) + L1(v_raw, v_gt) on moving cells only; and always `terms["motion_l1"]` = today's `MotionHead.loss` of the
     effective field, logged and not trained, so every arm has the same comparable number (zero field: 1.0945 on the val subset).
  4. Logging: `motion_l1@+0.0` joins the logged keys of `../t11_vel_hurdle/experiment.md` change 5.
- **Gate (layer 0, CPU).** `tests/test_motion_hurdle.py`: (G1) `motion_moving` False: `MotionHead` output and `motion_term` unchanged (same tensors for a fixed seed), the
  `timed_head.py` selftest passes; (G2) zero start: at init `v_hat` = 0 exactly, so the splat is the ego move (selftest cases 1-2 hold with `moving=True`); (G3) toy: a blob moving at
  6 m/s on a static field, Adam 300 steps: `v_hat` at the blob within 0.5 m/s of 6, background |v_hat| < 0.1 m/s; (G4) `motion_l1` equals `MotionHead.loss(v_hat_effective, ...)`;
  (G5) a loss at dt +0.5 gives gradient to `m` and `v_raw` through the splat.
- **Design.** Same warm start and schedule as t11.vel_hurdle (`runs/t10_dual_s0/ckpt_00003000.pt`, 1000 steps, lr 1.5e-4, warm-up 50, `loss.warmup_hold` 0, `warmup_ramp` 0.01, seed 0, eval_every 500).
  - arm C `t11.motion_hurdle` = arm B (`object_moving` true) + `motion_moving` true. Fresh keys at the warm start: `obj_head.head.mov.*` and `mover.motion.m.*` (three aliases of the shared mover).
  - matched reference: arm B `t11.vel_hurdle` (same start, steps, seed); zero-field reference: arm A `t11.vel_ctl` (its `motion_l1` must read 1.0945 +- 0.001, a layer-1 reproduction of the stored number).
  - seeds: one tonight; the twin only on the rule.
  - evaluation: cold VAL 1000, `bev_suite eval --n 1000`; det rows at dt +-0.5 and dt 0 of `table.txt`, footprint IoU at dt +0.5, front mAP2 at dt 0.
  - GPU: ~52 min (38 training + 14 evaluation), whole GPU (one t10-size job holds 11.2 of 12.3 GB), after arms A and B.
- **Prediction (cold VAL 1000).**

  | metric | t10_dual_s0 | B (predicted) | C predicted (range) |
  |---|---|---|---|
  | val motion_l1 (step 1000, zero level 1.0945) | 1.0945 | 1.0945 | 0.80 (0.65 - 0.95) |
  | vehicle AP@2, dt +0.5 | 0.319 | 0.32 | 0.38 (0.33 - 0.45) |
  | vehicle AP@2, dt -0.5 | 0.365 | 0.37 | 0.41 (0.37 - 0.47) |
  | vehicle footprint IoU, dt +0.5 | 0.318 | 0.32 | 0.34 |
  | dt 0: front mAP2, maps, vehicle ave_skill_moving | see t11_vel_hurdle | = B | = B within the bands (the field is not on the dt 0 path) |

- **Decision rule (written now).** Bands: vehicle AP@2 at dt +-0.5 0.054 (the t-series mAP2 band) with the paired scene bootstrap of the scene records; maps and dt 0 as in t11_vel_hurdle.
  - M1 revived and useful: `motion_l1` of C <= 0.95 and vehicle AP@2 at dt +0.5 >= B + 0.054, dt 0 metrics >= B - band -> seed twin; with it, `motion_moving` becomes the default
    and C (if t11_vel_hurdle reached V1) is the prototype candidate (t10 + velocity + motion field).
  - M2 revived, no detection gain: `motion_l1` <= 0.95 but AP@2 dt +0.5 within the band -> the field learns but the dt decoding is limited elsewhere (unknown cells, the dt heads'
    own refine block): keep the option off, next = per-object propagation with the object head's velocity (plan 8.3 S5b, `propagate_box_tensors`) as the dt route.
  - M3 still dead: `motion_l1` >= 1.09 -> the cells carry no motion information either: same conclusion as t11_vel_hurdle V3 (input route: radar Doppler skip / two-frame flow).
  - Between (0.95 < motion_l1 < 1.09, or AP gain between 0.03 and 0.054) -> INCONCLUSIVE: twin, do not adopt.
  - dt 0 maps or mAP2 of C below B by more than the band -> CUT.
- **Kill / validity.** (1) G1 / G2 fail -> no run. (2) val at step 500: `motion_l1` of C above 1.0945 (the gate makes it worse than zero: background moved) -> stop and check the background term.
  (3) If t11_vel_hurdle ends in V3 before C starts, C is not run (same mechanism, same missing information); the slot goes to the input-route card instead. (4) NaN / CUDA error -> resume once.
- **What it does not test.** The cross-time consistency arm (`consistency` > 0, the designed teacher of the field; config-only, a cheaper fallback if the hurdle fails: then a
  separate arm, not tonight); from-scratch training; the warm protocol (memory over whole scenes); latency (a 1 x 1 MLP on the head state: negligible).
- **Cost.** ~0.9 GPU-hours (one arm, whole GPU), ~30 lines in timed_head.py, model.py, losses.py + the test (coder, ~45 min, can be built while A and B train); risk: the gated
  field moves background in the splat if the logit leaks (kill 2 and G3 watch it).
- **Inputs.** Ready: the t10 checkpoint, arms A and B of `../t11_vel_hurdle/experiment.md` as references. To build: the four changes and the test; config
  `training/configs/t11_motion_hurdle.json` = `t11_vel_hurdle.json` + `model.motion_moving` true.
- **Referee.** -
