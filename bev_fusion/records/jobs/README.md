# jobs - the job files of the training system, and where the Runner and the Evaluation are called from

One job file is one small json that says what to do and with what; the same file goes to `check` (looks, starts nothing), `run` (now) or `queue_add` (the queue starts it). The convention for all
suites is `../../../autoresearch/README.md`; each suite's own README is the reference for the fields.

| folder | suite (class) | how it is called | state |
|---|---|---|---|
| `training/` | `trainsuite` (`Training`) | job files of kind `train` | built (work of the Training chat, 2026-11-10, not yet committed) |
| `running/` | `bev_fusion/runner` (`Runner`) | no job files: `python -m runner Runner.predict` takes arguments (the network on frames, no ground truth, stored for the evaluation); the one example file belongs to the planner's runner `runner/bev_runner.py`, whose job layer `runsuite` is commented out (2026-11-10) | built 2026-11-10 (one class, a command line above it) |
| `evaluation/` | `evalsuite` (`Evaluation`) | no job files: `Evaluation.predict` runs the network over whole scenes (always warm, D70) and stores the predictions, `score` scores them against a ground truth, `run` does both | built (work of the Evaluation chat, 2026-11-10, not yet committed) |

```
training/*.json   --Training.run / queue_add-->   runs/<name>/ (checkpoint, config, code snapshot)
Runner.predict(network, frames)  -->  runs/<name>/eval/results/<id>   (the stored predictions: a ResultSet, evalsuite/results.py)
Evaluation.predict(network, frames) -->  runs/<name>/eval/results/<id>   (the stored predictions: a ResultSet, evalsuite/results.py)
Evaluation.score(results, frames, gt_key) --> the numbers, curves, worst cases, comparisons, pages
```

A job file in here is a recipe, small text, kept in git; what a job produces (runs, frames, tables) is not. A job that may feed a verdict still needs its card and a logged TEST row.
