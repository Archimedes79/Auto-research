# configs - training configurations (`python training/train.py <config.json>`)

A config is the JSON of `train.TrainConfig` (unknown keys are an error); `model` holds `model.ModelConfig` fields, `loss` holds
`losses.LossConfig` fields (the head layout flags are taken from `loss`). One run = one config; an arm is the base config with
the listed fields changed (copy the file, change the name and the fields, never edit the base).

| file | what | status |
|---|---|---|
| `t1_base.json` | the first full training: cached camera-head features (`data/cache/cam_feats`, built by `temp/setup/build_full_cache.sh`), GT at -0.5 / 0 / +0.5 s with the true keyframe gap (`gap_ego`), Kendall task weights, AdamW 3e-4 with 500 warm-up steps from lr / 3 and cosine, effective batch 16 (micro-batch 4 x 4 accumulation: 2.5 s per optimiser step, 4.4 GB; 2 x 8 takes 5.3 s, 8 x 2 takes 2.1 s with 8.6 GB; idle GPU, 4 workers, 2026-10-26), EMA 0.999, gradient-conflict log every 500 steps, 20000 optimiser steps | not run at full length (the caches are built, the trainer self-test passes on mini and a 12-step smoke ran on the real caches) |
| `t1_r0.json` | experiment t1 round 0 (`../../experiments/t1_full_chain/`): t1_base with 5000 steps (4.9 epochs), warm-up 300, 8 loader workers (1.7 s per step) | ran 2026-10-26 |
| `t1_r1a_dir.json` | round 1 arm a: t1_r0 + `model.object_direction` (object heading as sign-free axis + front / back class, SECOND) | round 1 |
| `t1_r1b_uwso.json` | round 1 arm b: t1_r0 with UW-SO task balancing (T = 5) instead of Kendall | round 1 |

Arms (fields to change against `t1_base.json`; none is run yet, every one is a proposal of plan 8.4):

| arm | fields | question |
|---|---|---|
| balancing | `"balancing": "fixed"` / `"uwso"` with `"uwso_T": 1, 2, 5, 10, 20` | does UW-SO (2408.07985) beat Kendall weights; T searched on quarter-length runs (`steps` / 4) |
| consistency | `"consistency": 30` / `100` (the raw value is about 0.01 in the first steps, a mean over all cells and channels; the diagnostic logs its gradient norm at the BEV feature as group `consistency`: choose the weight so that weight x norm is 10-30 % of the largest other group's norm, and re-check after the hold) | does the bidirectional cross-time consistency help dt = +-0.5 s |
| area overlap | `"loss": {"area_overlap": "semimetric"}` | Dice semimetric instead of 1 - soft IoU on soft coverage |
| line boundary | `"loss": {"line_boundary": true}` | boundary false-positive loss on the lines |
| dt weights | `"loss": {"dt_weights": [[0, 1], [0.5, 0.9], [1, 0.25]]}` | published 0.9 per 0.5 s step against 0.5 |
| ego control | `"gap_ego": false` | the nominal gap and the last step's ego motion: measures what the true gap is worth (0.2-0.35 m label error) |
| no-time control | `"loss": {"dt_weights": [[0, 1]]}` | the same chain trained at dt = 0 only |
| micro-batch | `"batch_size": 2, "accum": 8` / `"batch_size": 8, "accum": 2` | BatchNorm statistics from 2 / 8 instead of 4 samples (2 x 8: 5.3 s per step, 2.3 GB; 8 x 2: 2.1 s, 8.6 GB) |
