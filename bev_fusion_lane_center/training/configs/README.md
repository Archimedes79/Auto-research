# configs - one run = one json file (the BEV system's way, adapted)

A config is the JSON of one run of a *design* (`training/startable.py` holds the table of the keys of each design: an unknown key, a wrong type or a value out of range is an error before a GPU is touched).
Start it with the shared tools, from `autoresearch/` (the project's `project.json` block `Training` says how):

```
AR_PROJECT=../bev_fusion_lane_center python -m Training start training/configs/e271_grid.json --id e271_a_s1 --preset seed1 --run-arg=--split --run-arg=val --hyp "..." --pred "..." --rule "..." [--go]
python -m Training --root ../bev_fusion_lane_center/runs status        # also health, curves, stop, watch, queue
```

**Split argument:** `Training start` always passes `--split train` to `run.py`, and this project's `splits` (protected, `project.json`) are val / test / confirm, so the launcher refuses it. Add `--run-arg=--split --run-arg=val` (the later `--split` wins; a training run that scores VAL at its end is a val run; it was tried end to end with the smoke `e271_cfg_smoke`, 2026-11-03). Adding `train` to `splits` would need a grant under a decided D-entry.

`start` composes the base with the presets (`presets/<name>.json`, deep-merged overrides) and `--set key=value`, runs the check (`training/startable.py`), prints the exact `run.py` command and with `--go` starts it; the same command again resumes the run.
Without the tools: `wsl bash tools/env/_py14.sh startable.py <config>` (the check, `--write-smoke DIR` writes the smoke twin) and `wsl bash tools/env/_py14.sh run_config.py <config>` (the training; the trainer's own command line stays as it is).
An arm is the base config with the fields changed (a new file or a preset, never an edit of a base that a finished run used).

| file | what | status |
|---|---|---|
| `e271_grid.json` | E271 arm a (grid head 1 m + sparsification, single frame, 4000 steps) | arms a_s0 / a_s1 ran by the command line of `grid_train.py` before configs existed (their `runs/<name>/config.json` holds the same values) |
| `e273_head.json` | the lane head of E273 = baseline B (design `qgraph_head`: phases 2 - 7 of `qgraph_train.py`, a gate after each; later arms add the flags of their card, key = flag) | `e273_p_s0` / `e273_p_s1` ran by the command line of `qgraph_train.py` (`temp/e273/start_p.sh`); the config reproduces that command |
| `jobs/e273_head_example.json` | the same as a job file of the Training class (`Training.run`), a smoke as a diagnostic | `Training.check` ok on 2026-11-10; not run (GPU window) |
| `presets/smoke.json` | 40 steps, 24 evaluation frames: the end-to-end smoke of a new `grid_head` version | |
| `presets/smoke_qgraph.json` | the smoke of a `qgraph_head` run (the trainer's own few steps per phase; `steps` of that design is a list) | |
| `presets/seed1.json` | seed 1 (the seed twin) | |

**The Training class (since 2026-11-10)**: the same run as a job file, `python -m Training Training.run job.json --project ../bev_fusion_lane_center` from `autoresearch/` (`Training.check` first: it composes, checks and starts nothing; `status`, `stop`, `resume`, `plot` work on the run folder; a `qgraph_head` run writes one folder per phase, `runs/<name>_p<k>`, the final model in `runs/<name>`). The job needs `"run_args": ["--split", "val"]` (the split argument above), and either `card` {hyp, pred, rule} / the arm's run id in the `run_ids` of its card, or `no_card: true` (a diagnostic never feeds a verdict). The shared job runner (`python -m jobrunner`) is not wired to this project: it lists the suites in a `jobrunner` block of `project.json`, which this project does not have, and the head has no `JOBS` table yet.
The GT the head reads is the lane GT of `GtGenerator` (`python -m GtGenerator key lane_center --set full`: release v12, lane store `0702a065c6` = l4, complete on 2026-11-10); `startable.py` notes when the store of this checkout is missing.

New designs (a hold, an assignment, a delivered graph) add a row to `DESIGNS` in `training/startable.py`, a branch in `training/train/run_config.py` and a file here.
