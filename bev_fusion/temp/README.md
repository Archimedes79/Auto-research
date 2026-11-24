# temp - scratch of the BEV system (not in git)

Output, logs, caches and throw-away scripts go here and nowhere else. Nothing in `training/`, `evaluation/`, `GtGenerator/` imports this folder. Only this README and the file below are tracked (the root `.gitignore` ignores the rest); a script that a card, report or plan names as how a number was made belongs in a code folder: the recipes of the closed experiments t1 - t8 are in `../studies/recipes/` (moved out of here on 2026-11-10, `../studies/recipes/README.md`).

| file | what |
|---|---|
| `train/run_ar.sh` | the redirect to `../tools/run_ar.sh` that the recorded commands of old and held runs still name (58 ledger rows and `run.json` files; `provenance.trainer_python` reads the interpreter from it); the launcher itself has been `../tools/run_ar.sh` since D73. Leave it where it is (`tests/test_run_ar.py` runs it) |
| `compose/` | empty on purpose: `project.json` `training.out` = `temp/compose`, where `python -m Training check` writes the composed config |

## Emptied on 2026-11-23 (D140)

The owner's order D140 (`../../bev_fusion_lane_center/experiment_execution/DECISIONS.md`) deleted all scratch of this folder (135 top-level entries, 14,744 files, 9.5 GB after the moves below) and the folder `_to_delete/` of this problem (230 MB, with its README; D141 added its tag bundle) without a local tag, after a check of what living code, cards and checkpoints still need. Where the rest went:

| what | now |
|---|---|
| checkpoints that are not a copy of a file in `runs/` (4 files) and the 50 probe models of the early velocity studies (`checks/model_*.pt`, 2 GB) | `../runs/_from_scratch_2026-11-23/<old path>` (not in git; the owner may delete them) |
| checkpoint copies: t26/base_run/ckpt_00006000.pt (= `runs/t20_base_s0/ckpt_00006000.pt`, sha256 76b6f443...), t22/dev_runs/t22_A1_s0/ckpt_00000710.pt (= `runs/t22_A1_s0/ckpt_00000710.pt`) | dropped: byte-identical to the file in `runs/` (checked by hash) |
| the EfficientNet-B0 / B2 ImageNet weights (from `_to_delete/pretrained_candidates_2026-11-04/`) | `../models/pretrained/imagenet/` |
| t26 (jobs, data passes, CRN overlays, the by_n store of the base, the per-frame counts) | `../runs/t26_prep/` and `../records/t26_g0/` (base_card_n1000.json); the run guard is `../tools/run_guard.py` |
| the SOA reference tables and counts (t16/soa/) | `../runs/t16_soa_ref/` (the gate of `evaluation/soa_maps.py --check`) |
| the junction reference (`overview/rp_out`, rp_frames.json, intersection_iou_result.txt) | `../runs/junction_ref/` (`evaluation/bev_junction_check.py --reproduce`) and `../records/intersection_iou_result.txt` |

What is gone for good: the composed configs `compose/*.json` and the job scripts that the ledger rows and `records/runs/*/run.json` name as commands (the composed config of every run is `runs/<run>/config.json`; the commands in the records stay as history), the inputs that the closed recipes of `../studies/recipes/` read from here (`t5/frames*.json`, soa_gap/, train/speed_b4.json, the helper modules of checks/: those recipes cannot be run again), and the tensor dumps, review dumps and logs of every loop.

## Size rules (since 2026-11-01: temp/ held 18 GB, C: was 100 % full)

Budget 10 GB for `temp/` (`python training/scratch.py report` shows it); the helpers are in `training/scratch.py`.

- No double dumps: a feature dump is keyed by what it holds (`scratch.dump`); a finished one is reused, a new one passes `scratch.guard` first. Probe only the layers, frames and range the question needs (every 4th frame is usually enough).
- Outdated caches are deleted, not kept: list caches go through `scratch.cached` (a bigger cache serves a smaller n, one older than its inputs or built by another builder is deleted).
- Smoke runs and test outputs go into `scratch.tmpdir(...)`, which is removed on exit; a run that finishes keeps its `keep` complete checkpoints so that it can be resumed later (`slim_at_end` is an opt-in that keeps one slim checkpoint).
- Cleanup in the end: the last step of every card is the line `scratch left behind: <size>, keep / move / delete` (`../experiments/README.md`); what stays goes to the D: archive (`bev_fusion_temp_archive`), not into `temp/`.
