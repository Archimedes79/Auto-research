# temp - scratch of the running loop

One-off scripts and temporary files of the loop that is running now. **Convention: `temp/loopNN/`** (subfolders such as `analysis/`, `launchers/` inside it are fine). When the loop is filed (the experiment files have their summaries, the report is built, ledger and state written) run, from the problem folder,

```
wsl python3 ../autoresearch/tools/clean_temp.py loopNN            # dry run
wsl python3 ../autoresearch/tools/clean_temp.py loopNN --move     # commit + tag lcr/loopNN-scratch, then to _to_delete/temp_loopNN/ (or --delete)
```

The script refuses while `training/`, `evaluation/`, `tools/` or `reports/build/` import a module that exists only here (copy it to a role folder first) or while a file changed in the last 6 hours. It checks imports only: a script named by path in `data/rebuild.json`, a report builder or an experiment file that is still used must be moved to a role folder by hand first. `training/` and `evaluation/` never import from here; a script worth keeping moves to a role folder.

| folder | what |
|---|---|
| `loop19/` | scratch of loop 19 (filed); its b456 builders `make_b456_tokens.py` and `run_b456_extract.sh` moved to `training/data/prep/` on 2026-10-31 (D24, `data/rebuild.json` updated); `run_b456_gt.sh` (b456 ground truth, E235) is still here and not yet in `data/rebuild.json` |
| `loop20/` | scratch of loop 20 (filed); still used by loop 21: the E253 timing driver (`loop20/run_e253.sh`) and the E255 matched-vertex loader (`loop20/e255_matched.py`) |
| `loop21/` | scratch of the running loop 21 (the status report builder in `reports/build/reports/` uses its `fig_status_*.py`) |
| `STRUCTURE.md` | folder map with sizes, on demand (`tools/maintenance/make_structure.py`) |

History: on 2026-10-25 the analysis of loops 8 - 16, the archive of 472 historic scripts and the doc backup were staged in `_to_delete/`; modules that live code still imports were copied to `training/legacy/`, `training/data/prep/` and `training/search/`. The launchers of loops 15 and 16 are in `../experiment_execution/history/launchers/`, those of loop 17 in git (tag `lcr/loop17-scratch`). On 2026-10-31 `loop19_review/` was filed with `clean_temp.py` (tag `lcr/loop19_review-scratch`) and `literature_backup_2026-10-24/` moved to `_to_delete/cleanup_2026-10-31_review/temp/`.
