# runsuite - run a trained network on frames and write the data for the evaluation

The interface is the one of the training (`autoresearch/INTERFACE.md`): the class `RunJobs(project)` with `check(job)`, `run(job)`, `status(i)`, `queue_add(job)`; the other queue methods
(`queue_list`, `queue_get`, `queue_remove`, `queue_start`, `queue_hold`, `queue_resume`) are the ones of `Training` (one queue for trainings and runner jobs). The head of `__init__.py` has the
example and the modes. Every method is a command: `python -m runsuite run job.json` (from `autoresearch/`), `python -m runsuite methods`.

| file | what |
|---|---|
| `__init__.py` | the interface and an example job |
| `api.py` | the class `RunJobs`: the job's fields become the runner's flags (`FLAGS`, `POSITIONAL`); `run` goes through `tools/run.py` (record, ledger row without card, GPU slot) |
| `__main__.py` | the command line (`trainsuite/apicli.py` turns every method into a command) |

A runner job never carries a card (`run.py --no-card`): it feeds no verdict by itself, and the holdout split (`split: test`) is refused by `run.py`, which wants a full card and the logged row there
(a run on the holdout split goes through `run.py --split test` by hand, hard rule 1). A runner job is no training: `queue_hold` leaves it alone.

The project names its runner in `project.json` (`"runsuite": {"command": "...{args}...", "runs": "runs"}`; `{args}` is where the mode and the flags go). Test: `python -m trainsuite.selftest_api`
(group 9, a toy runner in a throw-away project).
