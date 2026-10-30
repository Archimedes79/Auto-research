# _to_delete - things taken out of the live tree (nothing is deleted; the git history keeps them)

| entry | what it is | why it is here |
|---|---|---|
| `temp_oneoffs/` | 52 finished one-off scripts of `temp/` (names are `<origin folder>__<file>`): 36 from `checks/` (`plan_*` / `edit_*` exact-string patch scripts of plan.md and tests, `dry_*` 2-step dry runs of finished tests, the generated model copy `model_mine`), 11 from `study/` (`patch_*`), 4 from `setup/`, 1 from `radar/` (`_scratch_patches/`) | their effect is in the git history of the file they patched, no document or script refers to them (checked by `git grep` on 2026-10-30); only `temp/setup/move_from_lcr.py` (history of the move out of the lane project) mentions two of them |
| `evaluation_pre_suite/` | `evaluate_run.py` and its helpers before the evaluation suite (2026-10-30) | replaced by `evaluation/bev_suite.py` (+ `../../autoresearch/evalsuite/`); kept until the last lane script that calls the stub has ended |
| `autogt_old_outputs/` | pictures / videos of earlier autoGT runs | superseded by `autogt/examples/` |
| `runs_t1_r1a_dir_aborted_ram/`, `runs_t1_r1a_dir_aborted_slow_loader/`, `runs_t1_r1a_dir_crashed_getitem/`, `runs_t1_r1a_dir_paused_gpu_shared/` | the four aborted first attempts of run t1_r1a | aborted (RAM, slow loader, a crash in `__getitem__`, GPU shared); logs only |
| `collision_stats_scout_unrun_duplicate.py`, `2312.xxxxx_PLACEHOLDER_REMOVE.md` | a duplicate scout script that was never run, a placeholder note | duplicates / placeholders |
