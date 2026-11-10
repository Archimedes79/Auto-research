# temp - scratch of the BEV system (not in git)

Output, logs, caches and throw-away scripts go here and nowhere else. Nothing in `training/`, `evaluation/`, `autogt/` imports this folder. Only this README and the file below are tracked (the root `.gitignore` ignores the rest); a script that a card, report or plan names as how a number was made belongs in a code folder: the recipes of the closed experiments t1 - t8 are in `../studies/recipes/` (moved out of here on 2026-11-10, `../studies/recipes/README.md`).

| file | what |
|---|---|
| `train/run_ar.sh` | the redirect to `../tools/run_ar.sh` that the recorded commands of old and held runs still name (58 ledger rows and `run.json` files); the launcher itself has been `../tools/run_ar.sh` since D73. Leave it where it is |

## Size rules (since 2026-11-01: temp/ held 18 GB, C: was 100 % full)

Budget 10 GB for `temp/` (`python training/scratch.py report` shows it); the helpers are in `training/scratch.py`.

- No double dumps: a feature dump is keyed by what it holds (`scratch.dump`); a finished one is reused, a new one passes `scratch.guard` first. Probe only the layers, frames and range the question needs (every 4th frame is usually enough).
- Outdated caches are deleted, not kept: list caches go through `scratch.cached` (a bigger cache serves a smaller n, one older than its inputs or built by another builder is deleted).
- Smoke runs and test outputs go into `scratch.tmpdir(...)`, which is removed on exit; a run that finishes keeps its `keep` complete checkpoints so that it can be resumed later (`slim_at_end` is an opt-in that keeps one slim checkpoint).
- Cleanup in the end: the last step of every card is the line `scratch left behind: <size>, keep / move / delete` (`../experiments/README.md`); what stays goes to the D: archive (`bev_fusion_temp_archive`), not into `temp/`.
