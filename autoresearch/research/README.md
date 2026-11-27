# research - experiments, runs, verdicts (the core of autoresearch)

`python -m research <verb> <problem> ...` from `autoresearch/` (or with `autoresearch/` on `PYTHONPATH`); `<verb> --help` lists the options.
Standard library only: what loads a problem's metrics (the evaluation step, the judge's numbers, `compare`, `anchor`, report pictures) runs
in a process with the problem's interpreter. Exit 0 done, 1 refused or failed (`check` too, when `start` would refuse), 2 wrong arguments;
`--json` before the verb prints json.

```
 agent writes experiment.md (card)          research new | check | start | status | stop | resume | judge | adopt | register | test
        |
 start: gates -> ONE snapshot commit + tag <prefix>/E<n> -> per run: records config.json, run.json -> one worker per run
        |
 worker: code = git archive of the tag (<runs root>/<problem>/_code/<commit>) -> the steps of the run in order (train, evaluate, ...)
        |        GPU slots of the machine, kill rules on every row of a step's log.jsonl, stop = exit 75, resume skips finished steps
        v
 <runs root>/<problem>/<run>/  (not in git)          problems/<p>/experiments/E###_slug/runs/<run>/{config,run,result}.json  (git)
        |
 judge: card + problem.json read FROM THE TAG -> every run ok? -> control valid? -> arms scored alike? -> paired scene bootstrap over
        the pooled seeds from the stored counts -> guards -> verdict.json; one commit of the records   (the standard rule: judge.py)
        |
 adopt (the owner's decision): the arm's settings become the defaults of the base config (its code is committed, default off, before the
        start) -> anchor: the evaluation repeated with the working tree gives the same numbers -> champions/<problem>/, one commit
        (a failure before the commit leaves base config and champions/ as they were)
 register (the owner's decision): a measured network (kind measure, verdict DONE) as champion or comparison network
        (adopt and register need the seeds' run folders)
```

## Upkeep
| verb | what |
|---|---|
| `tests <p> [--only text]` | the problem's own tests (`problem.json` `tests`) with its interpreter and module path and, when it has `training/architecture.json`, the drawing against the code; after any change of the problem's code |
| `anchor <p> <run>` | a finished run's evaluation step run again with the working tree: do its numbers come out (`anchor_tolerance`)? It covers what that step runs: a problem that predicts in a step of its own (`lane_center`) re-scores the stored predictions, so a change of its network needs a new run |
| `compare <p> <run> <run>` | two finished runs side by side: a diagnostic, never a verdict |
| `clean <p>` | writes the run folders nothing needs any more to a list file in the run root, with a hash. The owner confirms that list with a decided row that names the hash; `clean <p> --delete <list> --decision D<n>` then deletes its runs, each checked again: finished, no live step, a verdict with no arm KEEP, INCONCLUSIVE, TIE or MEASURED, named by no card (`from`, `needs`, a `{runs}/<run>` setting), no champion or comparison seed |
| `index <p> [--check]` | writes `ledger.csv`; `--check` exits 1 when it is not what the records give (a problem without runs needs none) |
| `report <p> <folder> [--strict]` | a report built again from the records and the stored predictions, into the run root (`../reporter/README.md`) |
| `hold <p> "<why>"`, `release <p>` | no start or resume of a problem until released |
| `grant <path> --decision D<n> --note "..." [--hours h]` | a protected file or folder (path ending in `/`) may be edited for some hours |
| `commit <paths> -m "..."`, `push-check` | commit these paths only, through a temporary index (paths as git reads them or from the top); what the unpushed commits add that never belongs into git |
| `machine [init]` | show or write the machine file |

## Files a problem writes
| file | what |
|---|---|
| `problem.json` | name, tag prefix, python environment, base config, steps (a step that writes into a subfolder of the run folder names it, `dir`: its `log.jsonl` and `CONTROL` are there), default steps, score, guards, control floor, evaluation split and metrics, holdout, protected files, the module path of flat imports (`pythonpath`), its tests (`tests`). Schema: `spec.py`, `PROBLEM_KEYS` |
| the base config | every setting an arm may change; has `name`, `seed`, `out_root` |
| `training/train.py` (or other steps) | reads one composed config (json path as its argument), writes `log.jsonl` and checkpoints into its run folder; exit 0 done, 75 stopped (`trainkit` keeps this contract) |
| `evaluation/adapter.py` + `evaluation/evaluation.json` | the `Evaluation` adapter: source, subject, task (metrics with additive counts, views); the core gives defaults for the rest (`Evaluation/defaults.py`) |
| `experiments/E###_slug/experiment.md` | the card: a fenced `json card` block (schema: `spec.py`, `CARD_KEYS`; template: `../templates/experiment.md`), then the text |

## Records (one fact, one file, one writer)
| file | writer | holds |
|---|---|---|
| `experiment.md` | the agent | question, arms, rule; the conclusion after the verdict |
| `runs/<run>/config.json` | `start` | the composed config of the run |
| `runs/<run>/run.json` | `start` (queued), `resume` (queued again), the worker | state, steps with exit codes and times, commit, tag, machine |
| `runs/<run>/result.json` (`test.json`) | the evaluation step (of `test`) | numbers, per-scene score, identities, where the counts and predictions are |
| `verdict.json` | the judge | the verdict, effect and interval, control check, guards, the rule as frozen, the judge's own code (`judge_code`) |
| `ledger.csv` | `index` | generated from the above, never edited |
| `champions/<problem>/` | `adopt`, `register` | champion and comparison networks (`../../champions/README.md`) |

## Rules the code enforces
- The rule is what the tag holds: the judge reads card and `problem.json` from the snapshot; later edits do not count.
- A run executes its snapshot, never the working tree. A second `start` of the same experiment (more seeds) reuses its tag.
- Only the worker of a registered run writes records: the evaluation step refuses to run outside a worker. `start` refuses uncommitted code
  of the problem and the core.
- Identity = code, config, weights, ground truth, data. Stored predictions are reused only under the same identity.
- Arms are compared only when scored alike (split, frames, ground truth by its content signature, evaluator, metrics), else INVALID; records
  that hold no signature are compared by the ground truth's name, with a warning in the verdict.
- A control below the floor of `problem.json`, or with seeds further apart than the card allows, makes the verdict INVALID. A broken guard
  makes a winning variant a TIE, with the gain and the guard in the verdict.
- TEST and protected files: `../METHOD.md`, hard rules 1 and 2. The holdout gate is read by `research test`, in the evaluation step and
  wherever data are read through `Evaluation` (`Evaluation/defaults.py`: `holdout_open`; `guard` for a problem's own reader).

## Machine file (`~/.research/machine.json`, not in git)
```json
{"data": "<data root>", "runs": "<runs root>", "python": {"gcar": "<interpreter>"}, "gpu_slots": 2}
```
`research machine init ...` writes it; `RESEARCH_MACHINE` names another file (the tests use a throw-away one). Slots, holds and grants live
next to it (`state/`); only the research commands write there. A step, the computation of a verdict and an anchor test get the caller's
environment without the variables that could point them at another tree (`paths.child_env`), then their own.
