# jobs - how the three classes are called on this project

Training, Runner and Evaluation are one class each with a command line above it (the generator is `autoresearch/suitecli.py`; every method of a class is a command, `methods` lists them). Only the
training takes a job file; the other two take arguments. The convention for all three is `../../../autoresearch/README.md`, section 'The interface of the suites'; each class's README is the reference.

| folder | class | how it is called |
|---|---|---|
| `training/` | `Training` (`autoresearch/Training`, `python -m Training`) | a job file: `Training.check` looks, `Training.run` starts; see `training/README.md` |
| `running/` | `Runner` (`bev_fusion/runner`, `python -m runner`) | arguments: `Runner.plan` (dry run), `Runner.predict` (the network on frames, stored); see `running/README.md` |
| `evaluation/` | `Evaluation` (`autoresearch/Evaluation`, `python -m Evaluation`; BEV adapter `evaluation/bev_eval.py`) | arguments: `Evaluation.run` (predict through the Runner, then score), `Evaluation.score` (only score what is stored); see `evaluation/README.md` |
| `chain/` | all of them, one after the other (`autoresearch/jobrunner`) | four job files: GT, training, predictions, numbers; the data named in the arguments; see `chain/README.md` |

```
training/*.json  --Training.run-->       runs/<name>/ (checkpoint, config, code snapshot)
Runner.predict(network, frames)  -->     runs/<name>/eval/results/<id>   (the stored predictions: a ResultSet, autoresearch/Evaluation/results.py)
Evaluation.score(results, frames, gt_key) --> the numbers, curves, worst cases, comparisons, pages
Evaluation.run(network, frames, gt_key)   =  Runner.predict + Evaluation.score   (the Evaluation always includes the project's Runner: evaluation/evaluation.json, "runner")
```

A job file in here is a recipe, small text, kept in git; what a job produces (runs, frames, tables) is not. A job that may feed a verdict still needs its card and a logged TEST row.
