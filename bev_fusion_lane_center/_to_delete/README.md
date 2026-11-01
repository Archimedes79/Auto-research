# _to_delete - files moved out of the live tree, kept until Mirko empties the folder

Not in git (scratch rule). What left the tip of main is still in git history: restore with `git show <commit>:<path>` (the commit before the cleanup commit).

| folder | what, why |
|---|---|
| `cleanup_2026-11-01/evaluation/loops/loop16/` | `p16_diag.py`, `p16_junc_head_eval.py`: no importer, no README row, no card names them (inventory of 2026-11-01) |
| `cleanup_2026-11-01/evaluation/loops/loop17/p17_latency.py` | no caller; its classes are byte-identical to `p18_latency.py` (`--union 1 --fork_prune 0` reproduces it); only a docstring named it (review of 2026-11-01) |
| `cleanup_2026-11-01/tools/maintenance/resolve_links.py` | its input is gone: no `[[ID]]` token is left in `literature/`; no caller (review of 2026-11-01) |
| `temp_loop19/` ... `temp_loop22/` | the scratch of lane loops 19 - 22 (filed 2026-11-01 with `clean_temp.py --move`; the scripts are in the tags `lcr/loop19-scratch` ... `lcr/loop22-scratch`, the other files only here). `run_b456_gt.sh` (b456 ground truth, E235) was moved to `training/data/prep/` first and is in `data/rebuild.json` (id `gt_b456`). Run records and cards name these scripts by their old path `temp/loopNN/...`: restore with `git show lcr/loopNN-scratch:bev_fusion_lane_center/temp/loopNN/<file>` |
| `experiment_json_2026-11-01/E<id>_*/experiment.json` | the machine-readable cards of 26 experiments, moved into the `json card` block of each `experiment.md` by `experiment.py migrate` (fewer files, `autoresearch/fewer_files.md`); the same text is in the block and in git history |
| `cleanup_2026-11-01/evaluation/loops/loop17/p17_{extra,maponly,mapdiag,master,merge}.py` and `cleanup_2026-11-01/training/train/p17_{ft,joint}.py` | loop-17 leaves: nothing outside the group imports them, no ledger row or run record names them (`p17_joint` was the E136 trainer, `p18_joint` is its superset; the results are in stored files; the code is in git history and in `experiment_execution/history/loop17_recovery/`) (review of 2026-11-01) |
| `review_session_scratch_2026-11-01/` | the scratch of the review chat of 2026-11-01 (325 one-off scripts, patches, logs, old bundle copies, a partial worktree): the reusable parts were promoted (commit_paths.py, provenance.py, package.py, plan.py); no tracked file refers to it |
