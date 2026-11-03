# configs - one run = one json file (the BEV system's way, adapted)

A config is the JSON of one run of a *design* (`training/startable.py` holds the table of the keys of each design: an unknown key, a wrong type or a value out of range is an error before a GPU is touched).
Start it with the shared tools, from `autoresearch/` (the project's `project.json` block `trainsuite` says how):

```
AR_PROJECT=../bev_fusion_lane_center python -m trainsuite start training/configs/e271_grid.json --id e271_a_s1 --preset seed1 --hyp "..." --pred "..." --rule "..." [--go]
python -m trainsuite --root ../bev_fusion_lane_center/runs status        # also health, curves, stop, watch, queue
```

`start` composes the base with the presets (`presets/<name>.json`, deep-merged overrides) and `--set key=value`, runs the check (`training/startable.py`), prints the exact `run.py` command and with `--go` starts it; the same command again resumes the run.
Without the tools: `wsl bash tools/env/_py14.sh startable.py <config>` (the check, `--write-smoke DIR` writes the smoke twin) and `wsl bash tools/env/_py14.sh run_config.py <config>` (the training; the trainer's own command line stays as it is).
An arm is the base config with the fields changed (a new file or a preset, never an edit of a base that a finished run used).

| file | what | status |
|---|---|---|
| `e271_grid.json` | E271 arm a (grid head 1 m + sparsification, single frame, 4000 steps) | arms a_s0 / a_s1 ran by the command line of `grid_train.py` before configs existed (their `runs/<name>/config.json` holds the same values) |
| `presets/smoke.json` | 40 steps, 24 evaluation frames: the end-to-end smoke of a new version | |
| `presets/seed1.json` | seed 1 (the seed twin) | |

New designs (a hold, an assignment, a delivered graph) add a row to `DESIGNS` in `training/startable.py`, a branch in `training/train/run_config.py` and a file here.
