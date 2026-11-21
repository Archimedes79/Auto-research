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
| `e<id>/`, `day<NN>/`, `_session_scratch_<name>/` | what is written here now (since loop 22): the scratch of one experiment (`e<id>/`), of one day's chain of runs (`day<NN>/`) or of one chat (`_session_scratch_<name>/`: probes, critic scratch, message files); not in git (`.gitignore`); a script that a card, a report or other code names, or that is used a second time, moves into a code folder with its test and README row (`autoresearch/tools/check_git.py --used` lists them) |
| `STRUCTURE.md` | folder map with sizes, on demand (`tools/maintenance/make_structure.py`) |

History: on 2026-10-25 the analysis of loops 8 - 16, the archive of 472 historic scripts and the doc backup were staged in `_to_delete/`; modules that live code still imports were copied to `training/legacy/`, `training/data/prep/` and `training/search/` (`training/legacy/` was removed on 2026-11-03, commit 38d7c51f, `training/search/` in the clean-up of 2026-11-21; the prep scripts that no recipe of `data/rebuild.json` names were removed on 2026-11-21 too; all of it is in the git history). The launcher scripts of loops 15 - 18 (the former launchers folder of `experiment_execution/history/`) were removed from the tree on 2026-11-21 (the git history (commit `57b9eab1`)); the loop-17 launchers were in the scratch tag `lcr/loop17-scratch`, which no longer exists. On 2026-10-31 `loop19_review/` was filed with `clean_temp.py` (tag `lcr/loop19_review-scratch`) and `literature_backup_2026-10-24/` moved to `_to_delete/cleanup_2026-10-31_review/temp/`.
