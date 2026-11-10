# temp - scratch of the running loop

One-off scripts and temporary files of the loop that is running now. **Convention: `temp/loopNN/`** (subfolders such as `temp/loopNN/analysis/` or `temp/loopNN/launchers/` are fine). When the loop is filed (the experiment files have their summaries, the report is built, ledger and state written) run, from the problem folder,

```
wsl python3 ../autoresearch/tools/clean_temp.py loopNN            # dry run
wsl python3 ../autoresearch/tools/clean_temp.py loopNN --move     # commit + tag lcr/loopNN-scratch, then to _to_delete/temp_loopNN/ (or --delete)
```

The script refuses while `training/`, `evaluation/`, `tools/` or `reports/build/` import a module that exists only here (copy it to a role folder first) or while a file changed in the last 6 hours. It checks imports only: a script named by path in `data/rebuild.json`, a report builder or an experiment file that is still used must be moved to a role folder by hand first. `training/` and `evaluation/` never import from here; a script worth keeping moves to a role folder.

| folder | what |
|---|---|
| `loopNN/` | empty now: lane loops 19 - 22 were filed on 2026-11-01 (tags `lcr/loopNN-scratch` hold the scripts, the files are in `_to_delete/temp_loopNN/`); the next loop writes here |
| `STRUCTURE.md` | folder map with sizes, on demand (`tools/maintenance/make_structure.py`) |

History: on 2026-10-25 the analysis of loops 8 - 16, the archive of 472 historic scripts and the doc backup were staged in `_to_delete/`; modules that live code still imports were copied to `training/legacy/`, `training/data/prep/` and `training/search/`. The launchers of loops 15 and 16 are in `../experiment_execution/history/launchers/`, those of loop 17 in git (tag `lcr/loop17-scratch`). On 2026-10-31 `loop19_review/` was filed with `clean_temp.py` (tag `lcr/loop19_review-scratch`) and `literature_backup_2026-10-24/` moved to `_to_delete/cleanup_2026-10-31_review/temp/`.
