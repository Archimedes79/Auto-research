# temp - scratch of the running loop

One-off scripts and temporary files of the loop that is running now. **Convention: `temp/loopNN/`** (subfolders such as `analysis/`, `launchers/` inside it are fine). When the loop is filed (`experiments/E<id>_*/README.md` has the result, the report is built, ledger and state written) run

```
wsl python3 tools/maintenance/clean_temp.py loopNN            # dry run
wsl python3 tools/maintenance/clean_temp.py loopNN --move     # to _to_delete/temp_loopNN/ (or --delete)
```

The script refuses while `training/`, `evaluation/`, `tools/` or `reports/build/` import a module that exists only here (copy it to a role folder first) or while a file changed in the last 6 hours. `training/` and `evaluation/` never import from here; a script worth keeping moves to a role folder.

| folder | what |
|---|---|
| `analysis/loop18`, `launchers/loop18` | scratch and launchers of loop 18 (older layout: by type, then loop; new loops use `temp/loopNN/`) |
| `launchers/loop17` | wrappers of loop 17, still named in HANDOVER.md until its report is final |
| `literature_backup_2026-10-24/` | safety copy made by the literature chat; remove when that work is filed |
| `STRUCTURE.md` | folder map with sizes, on demand (`tools/maintenance/make_structure.py`) |

History: on 2026-10-25 the analysis of loops 8 - 16, the archive of 472 historic scripts and the doc backup were staged in `_to_delete/`; modules that live code still imports were copied to `training/legacy/`, `training/data/prep/` and `training/search/`. The launchers of loops 15 and 16 are in `../experiment_execution/history/launchers/`.
