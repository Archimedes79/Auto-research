# t18 motion block, own learning rate - does a smaller step for M take the noise off the static cells without losing the moving ones? (card 2026-11-02, before any run; the M part of the bundle arm `t18.fix` of the BEV system lead chat (D44 / D46: a good working system by the afternoon of 2026-11-03); written as `t17_motion_lr`, renamed because t17 is the data card `../t17_more_data/`; the train.py change of section 3 is the Training chat's, reviewed by this chat)

Context: `../../../experiments/t16_full_network/experiment.md`. On the small set (8 train / 2 val scenes, 400 steps, GT v8, `bev_suite.py --velocity eval --n 80`)
the chain with the motion block M (`t16.small_s0`) beat its control without M (`t16.small_ctl_s0`) on the moving parts and lost on the
static cells:

| VAL n 80 (small set, 400 steps), m/s | M (`t16_small_s0`) | M, background weight 1.0 (`t16_small_bg1_s0`) | control, 1 x 1 head (`t16_small_ctl_s0`) | zero |
|---|---|---|---|---|
| static cells `vel/dt+0.00/cells/static/epe` | 1.260 | 1.039 | 0.509 | 0 |
| moving vehicle cells `cells/vehicle_moving/epe` | 1.758 | 1.688 | 2.934 | - |
| moving box centres `centre/field/moving/epe` | 1.401 | 1.413 | 1.917 (objhead 1.882) | 2.603 |

The lead chat's check (2026-11-02): the field on static cells has mean ~0 (|mean| <= 0.37 m/s at ego 2 - 5 m/s), std 0.6 - 1.6 m/s, so
zero-mean noise, not a frame or alignment error. Cause in the code: every level of M adds `10 x net(...)` (`training/motion_block.py`
`Level.forward`), so under AdamW (scale-invariant, the clip does not shrink its step) one step of the last layer moves the field about
10 x as far as one step of the old 1 x 1 head (`VelocityHead`, bounded 40 tanh(v / 40)); t12 (M alone, lr 1e-3) also needed about 2000 steps
before its motion term settled (0.07 - 1.7 between logs at steps 100 - 500).

## 1 Card

- **Question.** With M's parameters on their own learning rate (1e-4 instead of the chain's 3e-4) and a longer warm-up (1000 steps
  instead of 300), does the field's error on static cells fall, with the moving error kept?
- **Hypothesis.** Yes: the static error is step noise of an over-large effective step (zero-mean, mean ~0); a 3 x smaller step and a
  slower start give less noise on the cells whose target is 0, while the moving cells, which carry the large L1 gradient, still learn
  in 6000 steps.
- **Design.** Run **`t18.fix`** (the lead chat's bundle arm) = the 700-scene data recipe of `t17.data` (`training/configs/t17_data.json`:
  t16's chain and recipe, 6000 steps, batch 3 x accumulation 2, lr 3e-4, warm-up 300, `grad_clip_split` ["motion_block"],
  `motion_w_background` 0.1, the hurdle on, GT v8, seed 0) + the hygiene flags of the Training chat + `lr_groups` {"motion_block":
  {"lr": 1e-4, "warmup": 1000}} (the TrainConfig field of section 3). Control: **`t17.data`** (the same data and recipe, M at the chain's
  lr, without the hygiene flags). One seed each, so the outcome is labelled "by hand: one seed per arm" and becomes the default only
  under the large-effect rule below.
- **Bundle and attribution.** `t18.fix` changes two things against `t17.data`: the hygiene flags and M's lr. The M rule below is read on
  the static EPE, which the hygiene flags are not meant to move; the attribution of a static gain to the lr holds only if none of the
  flags changes M's optimisation or its loss (its clip group, the EMA, the schedule, `motion_term`, the motion group's weight). The
  ledger note lists the flags; if one does, the verdict is labelled "bundle: lr + <flag>" and the lr alone is an ablation for later.
- **Metrics** (VAL, `bev_suite.py RUN --velocity eval --n 1000`, the same frames and GT signature for `t18.fix` and `t17.data`, `compare --velocity`
  for the paired scene bootstrap): headline static EPE `vel/dt+0.00/cells/static/epe`; moving EPE `vel/dt+0.00/cells/vehicle_moving/epe`;
  moving box centres `vel/dt+0.00/centre/field/moving/epe`; guards: maps (mean IoU of `seg_fov`), front mAP2 (`det`), the thresholded
  vehicle IoU at +-0.5 s (`seg_obj/seg/dt+-0.50/vehicle/all/all/iou@0`); `motion@+0.0` of the in-training validation at steps 2000 / 4000 / 6000.
- **Prediction** (relative to `t17.data`, written before either number exists): static EPE <= 0.6 x the control's and <= 0.30 m/s;
  moving vehicle EPE and centre EPE not worse than the control by more than the band; guards inside their bands.
- **Decision rule** (band per metric = max(floor, the 95 % half-width of the paired scene bootstrap of `compare --velocity`); floors:
  static 0.02 m/s, moving and centre 0.10 m/s, maps / mAP2 0.010, IoU 0.02; one seed per arm, so every verdict is labelled "by hand:
  one seed"):
  - ADOPT as the M default: static EPE below the control by more than 3 x its band AND moving / centre EPE not worse than the control
    by more than the band AND the guards hold (a large effect, so one seed is acceptable for a default; the seed twin follows when the
    GPU is free);
  - static better by more than the band but less than 3 x: CANDIDATE, the seed twin of both decides;
  - static inside the band: TIE, keep the chain's lr (no extra config);
  - moving or centre EPE worse by more than the band: CUT (the smaller step costs the moving cells: then the next arm is a lower output
    gain of M, not a smaller lr).
- **Kill.** Non-finite loss in more than 1 % of the steps; `motion@+0.0` at step 2000 above `t17.data`'s at step 2000 by more than 50 %
  (the slower start must not stall M); the default path not bit-identical (section 3, test).
- **Budget.** One run of the length of `t17.data` (the lead's estimate about 6 h; `t16.full_s0` logged 5.4 s per step at step 240, which
  would be 9 h at that rate; the per-group lr costs nothing) + one evaluation (about 25 min). CPU before it: the train.py change and its test (about 1 h).
- **What it can and cannot show.** One seed per arm; the small-set ratios above are 400-step numbers on 80 frames, only the motivation.
  Not tested: a lower output gain of M (the alternative lever, it changes `motion_block.py` and t12's equivalence), a longer run.

## 2 Why the lr and not the background weight or the gain
- Background weight 1.0 (`t16_small_bg1_s0`) took 1.26 -> 1.04 only: the noise is not a missing pull towards 0.
- The output gain 10 is the cause, but changing it changes the block (t12's checkpoints and equivalence); a per-group lr changes only the
  optimiser and is bit-identical when off. If Mlr is CUT for the moving cells, the gain is the next arm.

## 3 What the param-group change needs in `training/train.py` (to be written by the code owner after the control run has started; nothing done here)
- `TrainConfig.lr_groups: dict = {}` - module prefix -> {"lr": float, "warmup": int} (either key optional, default the chain's); the prefix
  rule of `grad_clip_split` (`clip_groups`, train.py:616: a prefix matches the module and its children, never `motion_blockx`; a prefix
  naming no trainable parameter or two overlapping prefixes are errors).
- `param_groups(model, loss, lr, wd)` (train.py:271, today two groups: decay / no decay) -> with `lr_groups`: the parameters of each
  prefix in their own decay / no-decay pair, each group tagged `base_lr` and `warmup`; WITHOUT `lr_groups` exactly today's two groups in
  today's order (so the optimiser state layout, resume and every stored run stay bit-identical).
- The schedule loop (train.py:985-987, today `g["lr"] = cfg.lr * f` with one `f`): `g["lr"] = g.get("base_lr", cfg.lr) *
  lr_factor(step, g.get("warmup", cfg.warmup), cfg.steps, cfg.lr_min_frac)` (lr_factor starts at lr / 3, so M starts at 3.3e-5).
- Resume: add `lr_groups` to the keys whose change mid-run is refused (train.py:411, as `grad_clip_split`): a different group layout
  would not load the optimiser state.
- Test (train.py selftest or a CPU test): default = two groups and the same lr sequence as today; with {"motion_block": {"lr": 1e-4,
  "warmup": 1000}} on a model with the block: four groups, the block's lr at steps 0 / 999 / 1000 / end against lr_factor by hand, the
  rest unchanged; a typo prefix fails at start.

## 4 Results

Run `t18.fix` (bundle: the hygiene flags + `lr_groups` {"motion_block": {"lr": 1e-4, "warmup": 1000}} on the 700-scene recipe of `t17.data`) finished
2026-11-03 03:29 (exit 0, 6000 steps, final validation loss 8.00 against 7.84 for `t17.data`). The first attempt died at 23:18 with a CUDA
"unknown error"; `run.py` retried and the trainer resumed from `ckpt_00002500.pt`, so the run is not bit-identical to an
uninterrupted one (the same happened to `t17.data`). Evaluated by
`bev_suite.py runs/t18_fix --curves --fusion --velocity eval --n 1000` and compared with `t17.data` by the paired scene bootstrap
(`temp/t16/compare_t18.txt`, the rule applied by `temp/t16/rule_t18.py`). VAL 1000 frames of 82 scenes, GT v8, one seed per arm.

| quantity | `t17.data` (control) | `t18.fix` | b - a | band | reading |
|---|---|---|---|---|---|
| static EPE, cells (m/s) | 0.2164 | 0.2413 | +0.025 | 0.020 | not better; prediction (at most 0.6 x = 0.130) missed |
| moving vehicle EPE, cells (m/s) | 1.364 | 1.575 | +0.211 | 0.10 | worse |
| moving box-centre EPE (m/s) | 1.069 | 1.204 | +0.135 [+0.107, +0.163] | 0.10 | worse |
| mean of the six map IoUs | 0.4425 | 0.4518 | +0.009 | 0.010 | guard holds (drivable area +0.007, walkway +0.021, both intervals above 0) |
| front mAP (det) | 0.3423 | 0.3471 | +0.005 | 0.010 | guard holds |
| vehicle IoU at +0.5 s | 0.3987 | 0.4054 | +0.007 | 0.02 | guard holds |

Other changes with an interval that excludes 0: vehicle moving AVE skill 0.860 -> 0.826 (worse), vehicle IoU (dt 0) +0.013, walkway AP +0.014;
the lane divider zero-crossing F1 moved -0.013 (inside the interval). Kill criteria not hit: the motion term of the training log at step
2000 is +3 % over `t17.data`'s (limit 50 %), no non-finite loss.

**Verdict: CUT** (by hand: one seed per arm; label **bundle: lr + hygiene flags** - the flags change the velocity losses that M learns from, so
the lr alone is not isolated). The written rule: moving and centre EPE are worse than their band, static EPE is not better, so the smaller
step costs the moving cells and does not remove the static noise. The static noise of the full run is small already (0.22 m/s, against 1.26 on
the 400-step small set the card was motivated by), so the prediction was built on an early-training number. `t16.full_s0` stays the base.
The hygiene flags are not separated from the lr by this run; their visible effect is the small gain on the maps and the vehicle IoU,
inside the guards' bands. Next arm by the card: a lower output gain of M (it changes `motion_block.py` and t12's equivalence), not a smaller
lr. Not started: Mirko ordered all BEV system trainings stopped after `t18.fix` (2026-11-02).

**Against the baseline `t16.full_s0`** (same frames, paired bootstrap `temp/t16/compare_t18_vs_base.txt`; read by the numbers critic of the report, 2026-11-03):
the bundle is worse on velocity (moving box-centre EPE 1.112 -> 1.204 m/s [+0.061, +0.121], static cells 0.189 -> 0.241 [+0.032, +0.068]) and its
maps are mixed, not level: walkway +0.010 [+0.003, +0.018] and vehicle IoU +0.008 [+0.002, +0.015] are better, drivable area -0.006 and the lane divider
IoU are inside the noise, car park 0.274 -> 0.253 (-0.020) and road divider IoU@1 m 0.572 -> 0.560 (-0.012) are worse than the 0.010 band (no
bootstrap interval was computed for these two), and the thin-line F1 of the road divider falls 0.511 -> 0.307 (precision 0.81 -> 0.73, recall
0.37 -> 0.19). The guard in section 1 was read against `t17.data`, where the mean of the six map IoUs holds; against the baseline the road divider
is a loss the card did not look at. This strengthens the CUT; it is not part of the written rule.
