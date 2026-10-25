# _to_delete - staged, NOT purged (the user decides)

Nothing here is needed by the live system (checked by the smoke tests of 2026-10-24) and nothing is lost: files that carry information were copied into `experiments/` or `literature/` first. Purge a folder by deleting it; restore a file by moving it back (`experiment_execution/history/MIGRATION_LOG.csv` has every old path).

| folder | what | size | why it can go | moved |
|---|---|---|---|---|
| `runs/` | 709 historic run directories of `out/loop2` (dense-head runs of loops 2-9 with prediction arrays and checkpoints, debug / smoke runs, build logs) | 34 GB | not referenced by the live code; their small result files (json / csv / log / png, 18 MB) are in `experiments/E<id>_*/results/`; reproducible by rerunning the experiment | 2026-10-24 |
| `bundles/` | code bundles (zip + folders) mailed in loops 5-14 | 1.2 GB | snapshots of code that now lives in `training/` etc. | 2026-10-24 |
| `report_leftovers/` | brief / mail drafts, trial and test builds of reports (`test7*.html`, `_trial10.html`) | 59 MB | superseded builds | 2026-10-24 |
| `recon/` | recovery backups (`backup_*`), old replay scripts | 57 MB | an earlier recovery; `finish_report6.ps1` no longer uses its Chrome profile | 2026-10-24 |
| `mail13/` | attachments of the loop-13 mail | 27 MB | sent | 2026-10-24 |
| `scripts_junk/` | 765 one-off patch scripts (`_patch*`, `_rv_*`, `_fix*`), job-control scripts (`_kill*`, `_queue*`, `_wait*`), backups (`*.bak*`), `__pycache__2` (240 bytecode files) | 11 MB | applied long ago / obsolete; **`__pycache__2` keeps the compiled loop-17 modules - copies are in `experiment_execution/loop17_recovery/pyc/`, keep this folder until the loop-17 recovery is verified** | 2026-10-24 |
| `duplicates_literature/` | the original research notes (`lit16`, `lit17`, `lit_extra*`, `lit7`, `lit8`, ...) | 3 MB | copies are in `literature/sources/` | 2026-10-24 |
| `out_logs/`, `out_flags/` | setup and loop-1 logs of the former `out/` | 3 MB | superseded | 2026-10-24 |
| `tools_env/` | `_install12/13/14.ps1`, `delete_lidar_raw.sh` (destructive, must not be re-run by accident), `inventar.sh`, `lidar_inventory.sh`, `netspeed.sh`, `check_env.sh`, `gcar_build_rast2.sh` | 1 MB | one-off installers / checks of old loops | 2026-10-24 |
| `tools_restructure/` | transcript-replay and survey tools of the loop-17 recovery | 1 MB | **copies of the replay tools are in `experiment_execution/loop17_recovery/`** (needed until the recovery is finished); the migration scripts are in `C:\ML-Experiment\restructure\` | 2026-10-24 |
| `CLAUDE_root_draft_2026-10-24.md` | draft root CLAUDE.md of another chat | 1 kB | merged into `CLAUDE.md` | 2026-10-24 |

Not staged on purpose (look big, are needed): the WSL caches (`~/exp_gcar_b123`, 155 GB), `runs/` (512 entries the live code reads), `downloads/`, `data/`.
Also in the WSL home: `~/_review_v6_*`, `~/_verify_v4..v7` (140 MB of bundle test extractions) - not touched, delete them in WSL when you purge.
