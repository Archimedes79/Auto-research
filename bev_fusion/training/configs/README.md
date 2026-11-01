# configs - training configurations (`python training/train.py <config.json>`)

A config is the JSON of `train.TrainConfig` (unknown keys are an error); `model` holds `model.ModelConfig` fields, `loss` holds
`losses.LossConfig` fields (the head layout flags are taken from `loss`). One run = one config; an arm is the base config with
the listed fields changed (copy the file, change the name and the fields, never edit the base).

**Presets** (`presets/*.json`, overrides, not configs): `smoke` (20 steps, the end-to-end smoke of a new version), `small` (400 steps, evaluation every 100: a quick look), `onfly` (all caches off: preprocessing on the fly). `python -m trainsuite start training/configs/<base>.json --id <id> --preset small --set seed=3 ...` composes the run's config (written to `temp/compose/`, scratch), runs `startable.py` and prints the `run.py` command; `compose` only shows and writes the config. The full run is the base config as it is; data computed elsewhere is `--set feature_cache=<path>` (and `gt_cache`, `radar_cache`); one module with the rest frozen is `--set 'train_only=["fusion."]'` once `train.py` knows the key (see `freeze.py`).

| file | what | status |
|---|---|---|
| `t1_base.json` | the first full training: cached camera-head features (`data/cache/cam_feats`, built by `temp/setup/build_full_cache.sh`), GT at -0.5 / 0 / +0.5 s with the true keyframe gap (`gap_ego`), Kendall task weights, AdamW 3e-4 with 500 warm-up steps from lr / 3 and cosine, effective batch 16 (micro-batch 4 x 4 accumulation: 2.5 s per optimiser step, 4.4 GB; 2 x 8 takes 5.3 s, 8 x 2 takes 2.1 s with 8.6 GB; idle GPU, 4 workers, 2026-10-26), EMA 0.999, gradient-conflict log every 500 steps, 20000 optimiser steps | not run at full length (the caches are built, the trainer self-test passes on mini and a 12-step smoke ran on the real caches) |
| `t1_r0.json` | experiment t1 round 0 (`../../experiments/t1_full_chain/`): t1_base with 5000 steps (4.9 epochs), warm-up 300, 8 loader workers (1.7 s per step) | ran 2026-10-26 |
| `t1_r1a_dir.json` | round 1 arm a: t1_r0 + `model.object_direction` (object heading as sign-free axis + front / back class, SECOND) | round 1 |
| `t1_r1b_uwso.json` | round 1 arm b: t1_r0 with UW-SO task balancing (T = 5) instead of Kendall | round 1 |
| `t7_res4_s0.json`, `t7_res4_s1.json`, `t7_rgb_s0.json`, `t7_rgb_s1.json` | experiment t7_lane_branch (`../../experiments/t7_lane_branch/`): `t2_ref_s<seed>` + `model.lane_branch` = "res4" (frozen ResNet-50 layer 1) / "rgb" (CNN from scratch) and `lane_images` = `data/cache/img_front`; nothing else changed (made by `../../temp/t7/make_configs.py`) | queued (2026-10-30) |
| `t8_bevfeat_s0.json`, `t8_bevfeat_s1.json` | plan 2.8 step 0, the feature swap: `t2_ref_s<seed>` with `feature_cache` = `data/cache/cam_feats_bevfusion` (the BEVFusion camera-only image side, stride 8, 64.8 GB on D: behind a symlink, built by `training/cache_bevfusion_features.py`) and `model.camera_features` = "bevfusion"; everything else is the reference's (3 frames, window memory, 3000 steps), so comparable with ref_s0 / ref_s1; made by `temp/setup/make_t8_configs.py` | not run (cache not built yet) |
| `t10_dual_s0.json`, `t10_dual_s1.json` | experiment t10_dual_features (`../../experiments/t10_dual_features/`): both image-feature sets in one chain (`model.camera_features` "dual": the cached DualBEV heads' features and the BEVFusion map-trained features, two lifts merged by a 1x1 convolution), 3000 steps, batch 3, accumulation 1, seed 0 / 1 (the twin differs in the seed only) | ran 2026-10-31 (s0) and 2026-11-01 (s1) |
| `t10_dual_long_s0.json` | lever 1 of the case (`../../experiments/t10_dual_features/experiment.md` sections 3b / 3c): `t10_dual_s0` + `steps` 6000, `eval_every` 2000, `accum` 2 (effective batch 6), `model.object_moving` true (the velocity hurdle from step 0, no `init_from`); its `_about` key says so | started 2026-11-01 09:01 as run `t10.dual_long_s0` |
| `t14_dc_ctl_s0.json`, `t14_dc_fusion_s0.json` | experiment t14_dontcare (`../../experiments/t14_dontcare/`): `t11_vel_hurdle_acc2` continued 1000 steps from its own EMA weights (`init_from`, fresh optimiser and schedule); arm A the loss as in every stored run, arm B `loss.fusion_area_only` (D38 (b): no loss on the fused heads outside camera x radar); the two files differ in that one field; made by a throw-away script, checked with `python training/startable.py <config>` | written 2026-11-01, not run |
| `t11_vel_hurdle_acc2.json` | arm B2 of experiment t11_vel_hurdle: `t11_vel_hurdle.json` with gradient accumulation 2 (effective batch 6; every run with effective batch 3 had a dead velocity branch); made by a throw-away script | written 2026-11-01, queued after B' |
| `t11_vel_ctl.json`, `t11_vel_hurdle.json` | experiment t11_vel_hurdle (`../../experiments/t11_vel_hurdle/`): `t10_dual_s0` continued from its `ckpt_00003000.pt` (`init_from`: EMA weights + Kendall s, fresh optimiser and schedule) for 1000 steps, lr 1.5e-4, warm-up 50, `eval_every` 500, `loss.warmup_hold` 0, `loss.warmup_ramp` 0.01, seed 0, `init_reset` `obj_head.head.vel.` (a fresh velocity branch: arms A' / B', rule C1 of `t11_vel_cause`); the hurdle arm adds `model.object_moving` (moving / static logit, velocity on moving centres only); made by a throw-away script from `t10_dual_s0.json`; `_about` keys are comments (`train.read_config`) | written 2026-11-01, not run |
| `t6_maps_only.json` | experiment t6_soa_gap: `t2_ref_s0` with fixed task weights (areas 1, lines 1, objects 0, motion 0) and the radar aux head off, pinned to the code of `t2_ref_s0` (commit 0a3d73c): does detection competition hold the maps back? card `../../experiments/t6_soa_gap/experiment.md` | queued 2026-10-30 behind the lane |

## Trunk settings of the configs (read from the files on 2026-11-01; plan section 6.1 tells how it came about)

| `trunk_kind`, `trunk_width`, `trunk_detail`, start | configs | meaning |
|---|---|---|
| bevtrunk, 1.0, 32, pre-trained (DualBEV encoder) | 40: every config since `t1_r4_soa` except the four rows below (t2 reference and arms, t3, t6, t7, t8, t10, t11, r2) | **what runs**: T0 + detail path, 27 M parameters, 43 of 78 GMACs of the cached chain |
| bevtrunk, 1.0, 32, pre-trained, `trunk_fpn_original` | `t4_fpn_order` | the same with the concatenation order the pre-trained weights expect (no gain) |
| bevtrunk, 0.5, default (0), random | `t1_base`, `t1_r0`, `t1_r1a_dir`, `t1_r1b_uwso`, `t1_r3_combined` (2026-10-26) | the first round; the plan's T1 |
| bevtrunk, 1.0, 32, random | `t2_trunk_rand` | null for the pre-training |
| bevtrunk, 0.5, 32, random | `t2_trunk_small` | half width |
| second, 1.0, 0, random | `t2_trunk_second` | SECOND + SECONDFPN |

The defaults of `model.ModelConfig` (`trunk_width` 0.5, `trunk_detail` 0, `trunk_pretrained` empty) are the third row of the table (the first round), not the first (what runs), and `BEVTrunk(light_up=...)` (the plan's T2) has no `ModelConfig` field, so a config cannot ask for it. A config copied from the last one inherits all of this: state the trunk in the `_about` key when a config means to change it, and when it does not, leave the three fields as they are.

Arms (fields to change against `t1_base.json`; none is run yet, every one is a proposal of plan 8.4):

| arm | fields | question |
|---|---|---|
| balancing | `"balancing": "fixed"` / `"uwso"` with `"uwso_T": 1, 2, 5, 10, 20` | does UW-SO (2408.07985) beat Kendall weights; T searched on quarter-length runs (`steps` / 4) |
| consistency | `"consistency": 30` / `100` (the raw value is about 0.01 in the first steps, a mean over all cells and channels; the diagnostic logs its gradient norm at the BEV feature as group `consistency`: choose the weight so that weight x norm is 10-30 % of the largest other group's norm, and re-check after the hold) | does the bidirectional cross-time consistency help dt = +-0.5 s |
| area overlap | `"loss": {"area_overlap": "semimetric"}` | Dice semimetric instead of 1 - soft IoU on soft coverage |
| line boundary | `"loss": {"line_boundary": true}` | boundary false-positive loss on the lines |
| dt weights | `"loss": {"dt_weights": [[0, 1], [0.5, 0.9], [1, 0.25]]}` | published 0.9 per 0.5 s step against 0.5 |
| ego control | `"gap_ego": false` | the nominal gap and the last step's ego motion: measures what the true gap is worth (0.2-0.35 m label error) |
| no-time control | `"loss": {"dt_weights": [[0, 1], [0.5, 0]]}` | the same chain trained at dt = 0 only (every decoded \|dt\| needs a listed weight: `[[0, 1]]` alone is refused, it used to weigh dt = +-0.5 like dt = 0) |
| sequence | `"seq_len": 3, "batch_size": 2, "accum": 3` | clips of 3 consecutive keyframes (2 clips = 6 samples per micro-batch, 18 per step) with the world memory carried and detached at every keyframe; the validation log gets `total_last_warm` / `total_last_cold` (the same last keyframes with and without the carried memory: does the memory help) |
| sequence + BPTT | `"seq_len": 3, "batch_size": 2, "accum": 3, "seq_bptt": true` | the same with the graph kept through the memory (about 3 x the activation memory of a micro-batch; GPU memory not measured), so the memory write path (static weight, velocity head) is trained by later keyframes |
| micro-batch | `"batch_size": 2, "accum": 8` / `"batch_size": 8, "accum": 2` | BatchNorm statistics from 2 / 8 instead of 4 samples (2 x 8: 5.3 s per step, 2.3 GB; 8 x 2: 2.1 s, 8.6 GB) |
