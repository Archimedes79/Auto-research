# t11 vel hurdle - does the object head's velocity leave the zero predictor when moving / static is split from the regression? (card written 2026-10-31 23:45, before any code or run)

Context (system review of the motion term, 2026-10-31 night; numbers from the stored files named here, nothing re-run):
- The object head's velocity, which is what AVE reads (`training/object_head.py:180`, decode -> `vx, vy`), is exactly the zero predictor in the best open arm:
  `runs/t10_dual_s0/eval/table.txt`, det dt+0.00: vehicle `ave_moving` 6.5995 = `ave0_moving` 6.5995, `ave_skill_moving` -0.0000 (1107 moving TPs),
  `ave_static` 0.0289 = `ave0_static` (1980 static TPs); pedestrian `ave_skill_moving` 0.0000 (568 TPs). The vehicle AVE 2.385 of t10 (and 2.35 - 2.60 of every
  3000-step cached run t2 - t10) is the zero-velocity baseline of the matched boxes: AVE differences between those arms are differences of the matched sets, not of
  velocity. t3 found this on 2026-10-30 13:40 (`../t3_improve_loop/experiment.md`); `t3_vel10` (velocity weight x10) stayed at the zero level (2.38). With Adam, a
  constant weight on the only loss of a branch hardly changes that branch's step (Adam normalises per parameter), so x10 did not test the cause.
- The GT is learnable: on 300 random train keyframes (GT signature 15efbde2d1) the box velocity is in ego axes and lies along the box yaw for moving vehicles
  (median |cos(angle(vel) - yaw)| 0.9999 over 883 boxes faster than 2 m/s); unknown-velocity boxes (`vel_known` False, vel [0, 0]) are 52 of about 11,500 boxes (< 0.5 %).
- Mechanism of the zero (the competing explanation is "no motion information at the centre features", see the rule): the velocity loss is L1 (`object_head.py:152`),
  whose minimiser is the per-component conditional median. Of the matched vehicles 64 % are static (1980 / 3087), of the vehicle boxes of 300 random train keyframes 84 % are slower than 1 m/s,
  and the moving ones point both ways in vx and vy, so the median is 0 wherever the features do not make "moving AND this sign" more likely than not. For the AVE (a
  norm error) 0 is also the right answer under that ignorance, so nothing pushes the head off it until the features separate moving from static. The long-trained
  `t1_r4_soa` (7000 x 2 steps) left it partly (`runs/_golden_t1_r4_soa/eval/table.txt`: vehicle AVE 1.52, `ave_skill_moving` 0.286).

## 1 Card

| | |
|---|---|
| lens | system review (a silent failure in the shipped chain) / training and losses |
| group | objects + motion (requirements: velocity of moving objects) |
| sources | t3 velocity finding (2026-10-30 13:40), `t3_vel10`, `t1_r4_soa` golden card, t10 card and eval, CenterPoint (L1 velocity on lidar sweeps, where the input shows motion directly) |
| status | proposed |

- **Question.** Does the object head learn velocity in 1000 steps when the regression is trained on moving objects only and a separate moving / static logit decides whether the velocity is used (a hurdle model), starting from the t10 checkpoint?
- **Hypothesis (falsifiable).** The zero velocity is a loss attractor (L1 median of a zero-inflated, sign-ambiguous target), not missing information: with the hurdle the
  vehicle `ave_skill_moving` rises from 0.00 to >= 0.10 on cold VAL 1000 while the static error stays below 0.3 m/s. Refuted if the moving logit does not beat its prior
  (val BCE not below the base-rate entropy) - then the centre features carry no usable motion information and the lever is the input (radar Doppler path / two-frame flow), not the loss.
- **The change (small, default off, bit-identical defaults; the coder builds it with the test below).**
  1. `ModelConfig.object_moving: bool = False` -> `ObjectHead(..., moving=True)` adds one branch `mov = _branch(hidden, hidden, 1, bias=-1.39)` (prior 0.2) to `names`.
  2. `object_head.head_loss`: when `"mov" in out`: `moving = |t["vel"]| > 0.5 m/s` (the eval's MOVING_SPEED, `evaluation/metrics.py:56`); `losses["mov"] = w_mov * BCE(mov at the centres, moving)` over the centre mask `m`;
     `losses["vel"] = w_vel * L1(vel, t["vel"])` over `m * moving` only (static boxes no longer pull the regression to 0). `w_mov` 1.0 (`LossConfig.w_obj_mov`, passed by `losses.object_head_terms`).
  3. `object_head.decode`: when `"mov" in out` and `sigmoid(mov) < 0.5` at the peak cell: `vx = vy = 0` (the AVE-optimal choice for a two-point mixture; an inference
     threshold, not on the training path; the training path stays differentiable).
  4. `TrainConfig.init_from: str = ""` (train.py): a checkpoint path; used only when the run folder has no checkpoint of its own: the model takes the checkpoint's EMA
     tensors for every key it shares with them (shapes must match, else it raises), the new keys keep their initialisation and are printed; the loss takes the checkpoint's
     `loss.*` (Kendall s); optimiser, step and schedule start fresh.
  5. Logging (train.py `run_clip` filter, line 514 today, and `evaluate` filter, line 541): also keep the terms `objhead.vel@+0.0`, `objhead.mov@+0.0`, `motion@+0.0`, so the
     velocity terms are visible in `log.jsonl` (today they hide inside group.objects / group.motion).
- **Gate (layer 0, CPU, before the run).** `tests/test_velocity_hurdle.py`: (G1) defaults: `ObjectHead()` outputs and `head_loss` unchanged (no `mov` key, same terms);
  (G2) toy: 80 % static targets, 20 % moving with velocity readable from a feature, Adam 300 steps: the hurdle head's error on moving samples < 0.3 m/s and on static < 0.1 m/s;
  gradients reach `mov` and `vel`; (G3) decode zeroes the velocity where `sigmoid(mov) < 0.5` and keeps it otherwise; (G4) `init_from` on the real
  `runs/t10_dual_s0/ckpt_00003000.pt` (CPU): every shared tensor equals the checkpoint's EMA tensor bit for bit, the fresh keys are exactly `obj_head.head.mov.*`, a
  shape mismatch raises; (G5) the existing selftests of object_head.py, losses.py, train.py pass.
- **Design.** Warm start from `runs/t10_dual_s0/ckpt_00003000.pt` (the evaluated t10 model), 1000 steps, everything else as `t10_dual_s0.json` (dual features, 3 frames,
  seq_bptt, batch 3, Kendall, EMA, flip 0.5, GT 15efbde2d1), lr 1.5e-4, warm-up 50, cosine; `loss.warmup_hold` 0, `loss.warmup_ramp` 0.01 (the dt != 0 terms are already
  trained; no 400-step hold of the restart); `eval_every` 500.
  - arm A `t11.vel_ctl` (matched control, equal budget, same start, same seed 0): `init_from` + 1000 steps, unchanged loss. It also measures what 1000 more steps give (the
    "val still falls" question of the 6000-step plan, cheaply).
  - arm B `t11.vel_hurdle`: the same + `model.object_moving` true.
  - null: the zero predictor (`ave0_*` in the same eval, same pairs); oracle-style reference: `t1_r4_soa` (skill 0.286 after 14,000 steps).
  - seeds: one each tonight; the seed twin of B (`seed` 1, same start) only on the rule below.
  - evaluation: cold VAL 1000 frames / 82 scenes, `evaluation/bev_suite.py runs/<run> eval --n 1000` as t10; maps full area, boxes front x < 50 m for mAP2 (`results_x50.md` way), velocity rows of `table.txt`.
  - GPU: about 38 min training (2.3 s/step as t10_dual_s0) + 14 min evaluation = ~52 min per arm, <= 60 with a shared box; 2 arms ~2 h.
    One t10-size training holds 11.2 of 12.3 GB (nvidia-smi 23:38 with `t10_dual_s1` alone), so these arms cannot run next to another t10-size job: they need the whole GPU, in series.
- **Prediction (cold VAL 1000, dt 0).**

  | metric | t10_dual_s0 | A predicted | B predicted (range) |
  |---|---|---|---|
  | vehicle ave_skill_moving | -0.000 | 0.00 (<= 0.02) | 0.25 (0.12 - 0.40) |
  | vehicle ave_static (m/s) | 0.029 | 0.03 | 0.20 (<= 0.30) |
  | vehicle AVE (m/s) | 2.385 | 2.38 | 1.95 (1.70 - 2.20) |
  | pedestrian ave_skill_moving | 0.000 | 0.00 | 0.05 (0.00 - 0.15): walking direction stays ambiguous |
  | front mAP2 / lane divider@0.5 / drivable / walkway | 0.361 / 0.335 / 0.806 / 0.340 | t10 +-0.01 | = A within the bands |
  | val objhead.mov (BCE, logged) | - | - | below the base-rate entropy by >= 20 % at step 1000 (0.50 nats if 20 % of the centres move; the coder computes the real base rate of the val subset in the gate) |

- **Decision rule (written now).** Bands: maps max(0.02, 2 |t10 s0 - s1|) (0.02 until s1 is evaluated), walkway 0.054, front mAP2 0.054 with the paired scene bootstrap
  (`temp/t4/compare_front.py` way); vehicle `ave_skill_moving` band 0.05 (assumed: no seed pair of a head that learns velocity exists; the zero predictors give exactly 0).
  - V1 learns: B - A >= 0.10 on vehicle `ave_skill_moving`, vehicle `ave_static` <= 0.30, and maps / front mAP2 of B >= A - band -> seed twin of B; if it agrees, the hurdle is
    the default and B is the prototype candidate of D28 (t10 + velocity), reported with one-seed caveat until the twin is in.
  - V2 in between: 0.05 <= B - A < 0.10 -> INCONCLUSIVE: seed twin, do not adopt.
  - V3 no gain (< 0.05) and the moving BCE at its prior -> information, not loss: the centre features do not separate moving from static. Next card = the input route
    (a zero-started 1 x 1 skip of the radar branch's features (Doppler code) into the head state, or the fused two-frame flow of plan 7.4); the motion-field card
    (`../t11_motion_field/experiment.md`) is then moot, and every AVE comparison stays void.
  - V4 no gain but the moving BCE well below its prior -> the logit sees motion, the regression does not get the sign: try speed x axis (|v| from the regression, direction from the box axis + dir head) next.
  - maps or mAP2 of B below A by more than the band -> the hurdle costs the shared trunk: CUT even if the velocity gains.
  - A itself: if A's maps or mAP2 beat t10_dual_s0 by more than the band, longer training is a lever on its own (supports the 6000-step run).
- **Kill / validity.** (1) G4 parity fails -> no run. (2) val at step 500: group.lines or group.areas of B above A's by more than 0.05 -> stop B (the head change disturbs the trunk). (3) NaN or CUDA
  unknown error -> resume once. (4) A's vehicle `ave_skill_moving` > 0.05 -> the continuation alone starts to learn velocity; then B is compared to A as written, but the reading "loss attractor" needs the twin.
- **What it does not test.** The dense motion field of the time mover (`../t11_motion_field/experiment.md`); a from-scratch 3000-step run with the hurdle (the warm start is
  a cheaper proxy and is labelled so); pedestrian direction; latency (one 3 x 3 + 1 x 1 branch at 64 channels on the head: negligible, measured with the prototype).
- **Cost.** ~2 GPU-hours (2 arms x ~55 min, whole GPU), ~40 lines in object_head.py, losses.py, train.py, model.py + the test file (coder, ~1 h); risk: the warm start
  hides an interaction with the restart (both arms share it).
- **Inputs.** Ready: checkpoint `runs/t10_dual_s0/ckpt_00003000.pt`, caches, GT 15efbde2d1, eval suite with `ave0` / `ave_skill`. To build: the five changes and the test above;
  configs `training/configs/t11_vel_ctl.json`, `t11_vel_hurdle.json` (= t10_dual_s0.json + `init_from`, `steps` 1000, `lr` 1.5e-4, `warmup` 50, `eval_every` 500,
  `loss.warmup_hold` 0, `loss.warmup_ramp` 0.01; B adds `model.object_moving` true).
- **Referee.** -

## 2. Detail plan (2026-11-01, after the code step)

Changes to the design of section 1 (made before any run; the section 1 text stays as it was written):
- **Arms A' / B' instead of A / B: both start with a fresh `vel` branch.** `../t11_vel_cause/experiment.md` found the `vel` branch's hidden ReLU layer off at every
  supervised centre in the batch-3 checkpoints, `t10_dual_s0` included (its output is its bias, 2.5e-7). Warm-started as written, arm B could move only that bias, and a
  "no gain" would be misread as V3 ("no motion information"); its rule C1 asks for a re-initialised branch. Both configs now carry `"init_reset": ["obj_head.head.vel."]`
  (`TrainConfig.init_reset`, `train.warm_start`: those tensors keep their fresh initialisation, the same in both arms; everything else is the checkpoint's EMA):
  `training/configs/t11_vel_ctl.json` = arm A' (fresh `vel` + today's L1 on all centres; per t11_vel_cause's P4 it should die again),
  `training/configs/t11_vel_hurdle.json` = arm B' (fresh `vel` + the hurdle). File names and run names are unchanged; each file says so in its `_about` line
  (`train.read_config` drops keys starting with "_").
- **Rule thresholds use the measured moving rate.** On the validation subset of these runs (150 clips x 3 keyframes, 4141 supervised centres) 1579 centres move
  (> 0.5 m/s): rate 0.381, not the 0.2 assumed in the prediction table. The prior level of the logged val `objhead.mov@+0.0` (a constant at the base rate) is
  0.664 nats, so "below the base-rate entropy by >= 20 %" means val `objhead.mov@+0.0` < 0.531 (rules V3 / V4 and the refutation). The `mov` bias still starts at
  the card's 0.2 (-1.39).
  - The 0.531 is approximate: the logged value is a mean over validation batches of each batch's mean BCE, not the entropy of the pooled centres (the constant
    at the pooled rate logs 0.6636, H(p) is 0.6647), and the constant level at the start bias -1.39 is 0.752. Read V3 / V4 against the logged value with that
    slack, not to the third digit.
- **Rule "A itself" is confounded now.** A' has a fresh `vel` branch, and its objects-group loss jumps at the start (an untrained velocity regression), so A'
  is no longer a pure continuation of t10_dual_s0: its maps / front mAP2 against t10 measure "1000 more steps + a restarted velocity branch", not longer training
  alone. The 6000-step question needs a continuation without the reset.
- **Addition: re-seed after the warm start.** `train()` seeds torch / numpy with `seed` again after `warm_start`, so A' and B' see the same data order, flips and
  modality drops whatever their extra branch drew at construction.

- **Arms changed again (2026-11-01 01:05, lead, before any arm ran): B' and B2 instead of A' and B'.** The velocity session's table over all 34 bev runs
  (`temp/box_vel/out/alive_table.txt`) shows a dead velocity output in all 24 runs with an effective batch of 3 (t3_ref_long at 6000 steps included) and a live
  one only at an effective batch >= 6 (t1_r4_soa, t3_img_long, t1_r0); an association, two of the live cases confounded with 7000 steps. A' (batch 3, today's L1)
  would only repeat a 24-of-24 result, so its GPU hour goes to **B2 = B' with gradient accumulation 2** (`training/configs/t11_vel_hurdle_acc2.json`, run
  `t11.vel_hurdle_acc2`, about twice B''s step time). Reading: B' against t10_dual_s0 (vel skill 0, the dead reference) answers "does the hurdle alone revive the
  velocity at batch 3"; B2 against B' answers "does the effective batch matter". Maps and front mAP2 of both are guarded against t10_dual_s0 / s1 with the t10
  bands. A null result means "not learned in 1000 steps from a trunk that never received a velocity gradient" (the frozen-trunk probe of the velocity session:
  ridge skill 0.019, moving AUC 0.77), not "the input carries no motion": V3 is read that way.

| step | what exactly | where (file / function) | done when | budget |
|---|---|---|---|---|
| code | `mov` branch, hurdle loss, decode gate, `init_from`, `init_reset`, `LOGGED_TERMS` | `training/object_head.py`, `losses.py`, `model.py`, `train.py` | `tests/test_velocity_hurdle.py` passes (G1 - G5, 2026-11-01, with the reviewer's gaps closed); default-off configs train bit for bit as before (3-step probe) | done |
| train | A' `t11.vel_ctl`, B' `t11.vel_hurdle`, seed 0, in series on the whole GPU | `run.py ... -- bash temp/train/run_ar.sh training/configs/t11_vel_{ctl,hurdle}.json` | done flags | ~2 GPU h |
| evaluate | cold VAL 1000 frames as t10, velocity rows of `table.txt`, front mAP2 | `evaluation/bev_suite.py runs/<run> eval --n 1000` | metric cards | ~30 min |
| decide | rule V1 - V4 of section 1 with the thresholds above | `decide.py` | verdict in the ledger | - |

## 3. Summary (lead, overnight round D28, 2026-11-01 06:10)
- **Runs**: `t11.vel_hurdle` (arm B': hurdle, batch 3) and `t11.vel_hurdle_acc2` (arm B2: hurdle, accumulation 2 = effective batch 6), each 1000 steps from `t10_dual_s0`'s step-3000 EMA weights with a fresh velocity branch (`init_reset obj_head.head.vel.`); code tags `bf/t11.vel_hurdle`, `bf/t11.vel_hurdle_acc2` (training code identical between the two tags; GT signature 0.5_4_15efbde2d1 for both and for t10). Cold VAL 1000 (82 scenes) through `bev_suite.py eval --n 1000`; front mAP2 by `temp/t4/compare_front.py` (100 paired scene draws, `front_compare.md` in this folder). **Decision: V1 (learns) for B2**: velocity skill +0.566 >= +0.10, static error 0.284 <= 0.30, maps and front mAP2 within the bands. B' learns too (skill 0.470) but misses the static clause (0.336 > 0.30).
- **Setup**: object head of t10_dual: a moving / static logit (BCE) next to the velocity regression, the velocity L1 only on moving centres, decode gate (velocity 0 when the logit says static, inference only); the rest of the system unchanged.
- **Numbers** (cold VAL 1000; reference `t10_dual_s0`, seed spread `t10_dual_s1`):

| | t10_dual_s0 | t10_dual_s1 | B' | B2 |
|---|---|---|---|---|
| vehicle ave_skill_moving | 0.000 | 0.000 | 0.470 | **0.566** |
| vehicle AVE (m/s) | 2.385 | 2.376 | 1.445 | 1.189 |
| vehicle ave_moving / zero predictor | 6.60 / 6.60 | 6.64 / 6.64 | 3.49 / 6.59 | 2.86 / 6.58 |
| vehicle ave_static (m/s) | 0.029 | 0.029 | 0.336 | 0.284 |
| vehicle AOE (rad) | 1.289 | 1.336 | 0.876 | 0.871 |
| drivable / walkway / lane divider IoU | 0.756 / 0.340 / 0.335 | 0.757 / 0.354 / 0.350 | 0.759 / 0.345 / 0.343 | 0.757 / 0.345 / 0.343 |
| front mAP2 (0 - 50 m) | 0.361 | 0.364 (+0.003) | 0.346 (-0.015 [-0.020, -0.007]) | 0.347 (-0.014 [-0.020, -0.006]) |
| vehicle AP@0.5 front / pedestrian AP@2 front | 0.254 / 0.199 | 0.265 / 0.202 | 0.204 / 0.211 | 0.207 / 0.215 |

- **What it says**: the velocity output of every batch-3 bev run was dead because the L1 regression on all centres was pulled to zero by the static majority (dead ReLU of the vel branch); the hurdle (moving logit + regression only on moving centres) revives it even at batch 3, and the effective batch adds +0.10 skill and brings the static error under 0.3 m/s. AVE halves (2.39 -> 1.19 m/s). The cost: vehicle localisation at 0.5 m drops (front AP@0.5 0.254 -> 0.207) and front mAP2 -0.014, an interval that excludes 0 but stays inside the band; the static error rises from 0.03 (the zero predictor) to 0.28 m/s.
- **Weaknesses / next**: one seed each (the twin of B2 is the next run); 1000 steps only (the 6000-step run `t10.dual_long_s0` waits for the Architecture merge and should carry the hurdle + accumulation 2); B' and B2 differ in the step count of the effective data seen (B2 saw twice the samples), so "batch" and "more data" are not separated; pedestrian velocity stays near 0 (skill 0.014); the existing fused flow pyramid (plan 7.3, `tests/test_flow_fused.py`: 0.95 - 1.04 m/s on moving vehicles on GT maps) and radar Doppler at boxes (skill up to 0.79 in the velocity session's probe) are the next levers for the moving error.
- **ms per frame** (added 06:26): two alternating passes of `bev_suite.py RUN time` on the idle GPU (batch 1, cached chain, dt -0.5 / 0 / +0.5; files `temp/retime_2026-11-01/`): t10_dual_s0 89.8 / 92.1 ms, B2 90.7 / 89.8 ms (dt 0 alone about 71 ms for both): within 1 - 2 ms (B2 passes idle; both t10 passes flagged not idle at 16 - 18 % GPU: indicative). A first B2 pass at 06:15 gave 111.1 ms (a first-call outlier, not repeated).
