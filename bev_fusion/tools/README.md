# tools - launchers of the BEV system

| file | what |
|---|---|
| `run_ar.sh` | trains one config under `autoresearch/tools/run.py`: unpacks `training/` and `../autoresearch/trainkit` of the run's snapshot commit (`$AR_COMMIT`) into `runs/<name>/code/`, writes the header line and the trainer's output into `runs/<name>/train.log`, ends with the trainer's exit code and touches `.done` only for 0; `project.json` `train` and `fingerprint_extra` name it; test `tests/test_run_ar.py` |

It was a scratch script in `temp/train/` until D73 (2026-11-10): the pinned engine rested on one line that existed only in the working tree.
