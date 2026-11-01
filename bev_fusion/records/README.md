# records - the run records of bev_fusion under autoresearch (D23, 2026-10-31)

Since D23 a bev_fusion training starts through the launcher of the research system, `../../autoresearch/tools/run.py`, like the runs of the lane problem. The launcher writes the card (hypothesis, prediction, rule) and the facts of the run into the ledger, tags the code it ran (`bf/<run id>`: `training/`, `evaluation/`, `autogt/`, `tests/`, `temp/train/run_cfg.sh` and `autoresearch/`, uncommitted changes included), waits for a GPU slot shared with the lane problem (2 slots on this machine), retries a CUDA "unknown error" and writes a done flag only for exit code 0. The paths are in `../project.json`.

| file / folder | what |
|---|---|
| `LEDGER.csv` / `LEDGER.md` | one row per run (`ledger.py`); made by the first run |
| `runs/<run id>/` | run record of the launcher: `run.json` (command, commit, tag, status, duration), `run.log`, `heartbeat`, `done` |
| `schedule.csv` | optional work list for `schedule.py` (not used yet) |
| `champion.json` | the base chain every card is compared against (`base` = a run name with its seeds, `open` = the arms on the board, `options` = adopted options, `promoted` = null until Mirko's OK); read by `../evaluation/board.py` |
| `CKPT_SLIM_2026-11-01.md` | what the cleanup of 2026-11-01 did: 34 finished runs slimmed to one checkpoint each (per-run sizes), the outdated caches deleted, what was kept on purpose and what is open |
| `morning_2026-11-01.md` | summary of the overnight round of 2026-11-01 (D28): t10_dual seed twin, the velocity fix t11 (arms B' / B2), the prototype command, incidents and open items |
| `sensor_coverage_2026-11-01.md` | proposal: one coverage rule per output type for training, evaluation and execution (what the code does today, findings, open points; Mirko's question of 2026-11-01) |
| `BOARD.md` | the board made by `python evaluation/board.py --write`: every open arm against the base, all heads, from the stored `metrics.json` files; regenerate it, do not edit it |

## Start a training
From `bev_fusion/` (the problem folder), in the WSL. `AR_PROJECT` names this problem (the tools default to the lane problem):

```
wsl bash -c 'AR_PROJECT=../bev_fusion nohup python3 ../autoresearch/tools/run.py --id t9.C --gpu --split train --hyp "..." --pred "..." --rule "..." -- bash temp/train/run_ar.sh training/configs/t9_C.json > runs/t9_C.launcher.log 2>&1 &'
```

- `temp/train/run_ar.sh` (not `run_cfg.sh`, which always exits 0) trains from a `git archive` of `training/` at `$AR_COMMIT`, the snapshot commit `run.py` made for this attempt (also on a resumed id, whose tag still names attempt 1), with the config copy inside that snapshot, and exits with `train.py`'s exit code, so a crash is a failed run and a CUDA "unknown error" is retried (`train.py` resumes from its last checkpoint).
- Not pinned: `autogt/` is imported from the live tree when the training starts. The tag records its state, and the run writes the GT signature it used to `runs/<name>/gt.json`; `evaluation/bev_suite.py compare` checks it.
- The training itself still writes `runs/<config name>/` (`train.log`, `log.jsonl`, checkpoints, `.done`) as before; the run record here holds the launcher's facts and the ledger row.
- Watch it with `python -m trainsuite --root ../bev_fusion/runs watch <config name>` (from `autoresearch/`).
- After the run: `ledger.py note <run id> --result "..." --decision "..."` (with the same `AR_PROJECT`).
- A planned row first (`ledger.py add <run id> --hyp ... --pred ... --rule ...`) lets the start line carry only `--id`; chains wait with `--after <run id>`.

Not yet under the launcher: the queue scripts in `temp/study/` (`lane*.sh`); they keep working, but their runs have no ledger row. The GT that a training reads is built from the live `autogt/` (the tag records its state; the GT signature in the run's config is item F3 of the repository review).
