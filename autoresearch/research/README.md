# research - experiments, runs, verdicts (the core of autoresearch)

`python -m research <verb> ...` from `autoresearch/` (or with `autoresearch/` on `PYTHONPATH`). Standard library only, except the evaluation step and the judge, which load the problem's metrics through `Evaluation` (numpy) and run where torch is.

```
 agent writes experiment.md (card)          research new | check | start | status | stop | resume | judge | adopt | register | test
        |
 start: gates -> ONE snapshot commit + tag <prefix>/E<n> -> per run: records/config.json, run.json -> one worker per run
        |
 worker: code = git archive of the tag (runs/<problem>/_code/<commit>) -> steps of the run in order (train, evaluate, ...)
        |        GPU slots and holds in ~/.research, kill rules on every row of the step's log.jsonl, stop = exit 75, resume skips finished steps
        v
 runs/<problem>/<run>/  (D:, not in git)          problems/<p>/experiments/E###_slug/runs/<run>/{config,run,result}.json  (git)
        |
 judge: card + problem.json read FROM THE TAG -> every run ok? -> control valid? -> paired scene bootstrap over the
        pooled seeds from the stored counts -> guards -> verdict.json; one commit of the records
        |
 adopt (the owner's decision): the arm's settings into the base config (its code is committed, default off, before the start),
        anchor = the evaluation repeated with the working tree gives the same numbers -> champions/<problem>/, one commit (anything
        that fails before the commit leaves base config and champions/ as they were); register: a measured network (kind measure,
        verdict DONE) as champion or comparison network; both need the seeds' run folders
```

Upkeep: `tests <problem>` (the problem's own tests, problem.json `tests`, with its interpreter and module path), `anchor <problem> <run>`
(a finished run's evaluation step run again with the working tree: it covers what that step runs; a problem that predicts in a step of its own
re-scores the stored predictions, so a change of its network needs a new run: lane_center), `compare <problem> <run> <run>` (a diagnostic,
never a verdict), `clean <problem>`
(the run folders nothing needs any more, written to a list file in the run root with a hash; the owner confirms the list with a decided row
that names the hash, then `clean <problem> --delete <list> --decision D<n>` deletes the runs of that list, each checked again: finished, no
live step, no arm KEEP, INCONCLUSIVE, TIE or MEASURED, named by no card (`from`, `needs`, a `{runs}/<run>` setting), no champion or
comparison seed), `index` (ledger.csv), `report`. Gates: `hold` / `release`
(no start of a problem), `grant` (a protected file for some hours under a decided decision). Git: `commit <paths>` (these paths only,
through a temporary index; paths as git reads them or from the top), `push-check`. `machine` shows or writes the machine file.

## Files a problem writes
| file | what |
|---|---|
| `problem.json` | name, tag prefix, python environment, base config, steps (a step that writes into a subfolder of the run folder names it, `dir`: its `log.jsonl` and `CONTROL` are there), score, guards, control floor, splits and holdout, protected files, the module path of older flat imports (`pythonpath`), the problem's own tests (`tests`) (schema: `spec.py`, `PROBLEM_KEYS`) |
| `training/train.py` | reads one composed config (json path as its argument), writes `log.jsonl` and checkpoints into `out_dir`; exit 0 done, 75 stopped (`trainkit` keeps this contract) |
| `evaluation/adapter.py` + `evaluation/evaluation.json` | the `Evaluation` adapter: `source`, `subject`, `task` (metrics with additive counts, views); the core gives defaults for everything else (`Evaluation/defaults.py`) |
| `experiments/E###_slug/experiment.md` | the card: a fenced `json card` block (schema: `spec.py`, `CARD_KEYS`), then the text |

## Records (one fact, one file, one writer)
| file | writer | holds |
|---|---|---|
| `experiment.md` | the agent | question, arms, rule; the conclusion after the verdict |
| `runs/<run>/config.json` | `start` | the composed config of the run |
| `runs/<run>/run.json` | `start` (queued), `resume` (queued again), the worker | state, steps with exit codes and times, commit, tag, machine |
| `runs/<run>/result.json` | the evaluation step | numbers, per-scene score, identities, where the counts and predictions are |
| `verdict.json` | the judge | the verdict, effect and interval, control check, guards, the rule as frozen, the judge's code (`judge_code`: the files that ran from the working tree and their hash) |
| `ledger.csv` | `index` | generated from the above, never edited |
| `champions/<problem>/` | `adopt`, `register` | champion and comparison networks |

## Rules the code enforces
- The rule is what the tag holds: the judge reads card and `problem.json` from the snapshot; later edits do not count.
- A run executes its snapshot, never the working tree. A second `start` of the same experiment (more seeds) reuses its tag.
- Only the worker of a registered run writes records. The evaluation step refuses to run outside a worker; `start` refuses uncommitted code
  of the problem and the core.
- Identity = code, config, weights, ground truth, data. Stored predictions are reused only under the same identity.
- The holdout split opens only for a run named in the problem's `TEST_LOG.md` under a decided `D<n>`; it is checked by `research test`, in the
  evaluation step, and wherever data are read through `Evaluation` (`Evaluation/defaults.py`: `holdout_open`, `guard` for a problem's own reader).
- Protected (the hook refuses an edit without a grant): the gate code, `.claude/settings*.json`, `champions/`, every problem's `problem.json`
  and `TEST_LOG.md`, a problem's `protected` list (`gates.py`; case does not matter), and anything in the machine's state folder.
- An invalid control (below the floor of `problem.json`, or seeds further apart than the card allows) makes the verdict INVALID.
- A broken guard makes a winning variant a TIE, with the gain and the guard in the verdict.

## Machine file (`~/.research/machine.json`, not in git)
```json
{"data": "/mnt/d/Auto-research/data", "runs": "/mnt/d/Auto-research/runs",
 "python": {"gcar": "/home/<user>/miniconda3/envs/gcar/bin/python"}, "gpu_slots": 2}
```
`research machine init ...` writes it. `RESEARCH_MACHINE` names another file (the tests use a throw-away one). Slots, holds and grants live next to it
(`state/`); only the research commands write there. A step, the computation of a verdict and an anchor test get this process's environment without
`RESEARCH_*`, `BFLC_ROOT`, `AR_PROJECT`, `PYTHONHOME`, `PYTHONSTARTUP` and git's own variables, then their own (`paths.child_env`).
