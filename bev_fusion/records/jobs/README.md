# jobs - the job files of the training system, and where the other two suites are called from

One job file is one small json that says what to do and with what; the same file goes to `check` (looks, starts nothing), `run` (now) or `queue_add` (the queue starts it). The convention for all
suites is `../../../autoresearch/INTERFACE.md`; each suite's own README is the reference for the fields.

| folder | suite (class) | how it is called | state |
|---|---|---|---|
| `training/` | `trainsuite` (`Training`) | job files of kind `train` | built (work of the Training chat, 2026-11-10, not yet committed) |
| `running/` | `runsuite` (`RunJobs`) | job files of kind `run`: the planner-facing runner `runner/bev_runner.py` (boxes and dt-0 maps, no ground truth) in the same queue | built (same) |
| `evaluation/` | `evalsuite` (`Evaluation`) | no job files: `Evaluation.predict` runs the network over whole scenes (always warm, D70) and stores the predictions, `score` scores them against a ground truth, `run` does both | built (work of the Evaluation chat, 2026-11-10, not yet committed) |

```
training/*.json   --Training.run / queue_add-->   runs/<name>/ (checkpoint, config, code snapshot)
running/*.json    --RunJobs.run / queue_add---->   objects.json + maps.npz of chosen frames (the planner's data)
Evaluation.predict(network, frames) -->  runs/<name>/eval/results/<id>   (the stored predictions: a ResultSet, evalsuite/results.py)
Evaluation.score(results, frames, gt_key) --> the numbers, curves, worst cases, comparisons, pages
```

A job file in here is a recipe, small text, kept in git; what a job produces (runs, frames, tables) is not. A job that may feed a verdict still needs its card and a logged TEST row.
