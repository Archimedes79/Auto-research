# tools - launchers of the BEV system

| file | what |
|---|---|
| `run_ar.sh` | trains one config under `autoresearch/tools/run.py`: unpacks `training/` and `../autoresearch/trainkit` of the run's snapshot commit (`$AR_COMMIT`) into `runs/<name>/code/`, writes the header line and the trainer's output into `runs/<name>/train.log`, ends with the trainer's exit code and touches `.done` only for 0; `project.json` `train` and `fingerprint_extra` name it; test `tests/test_run_ar.py` |
| `run_guard.py` | watches ONE long run (card t26, D102 stages) and stops it through `Training stop` (checkpoint, exit 75, HOLD) when C: has less than 6 GB free, the seconds per step (the median of the last 5 intervals between logged rows that do not touch a validation row: a hold gap or an evaluation is not a slow step) pass 14.4 on two looks in a row, radar_kept is below 0.2 at step 3000, or the log shows `--stage-end`; `--dry` = one look, nothing sent; log in `runs/_guards/<run id>.log`; standard library only; promoted from the scratch script `temp/t26/run_guard.py` (D140); test `tests/test_run_guard.py` |

It was a scratch script in `temp/train/` until D73 (2026-11-10): the pinned engine rested on one line that existed only in the working tree.
