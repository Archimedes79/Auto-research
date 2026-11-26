# research - experiments, runs, verdicts (the core of autoresearch)

`python -m research <verb> ...` from `autoresearch/` (or with `autoresearch/` on `PYTHONPATH`). Standard library only, except the evaluation step and the judge, which load the problem's metrics through `Evaluation` (numpy) and run where torch is.

```
 agent writes experiment.md (card)          research new | check | start | status | stop | resume | judge | adopt | compare | index | report
        |
 start: gates -> ONE snapshot commit + tag <prefix>/E<n> -> per run: records/config.json, run.json -> one worker per run
        |
 worker: code = git archive of the tag (runs/<problem>/_code/<commit>) -> steps of the run in order (train, evaluate, ...)
        |        GPU slots and holds in ~/.research, kill rules on log.jsonl, stop = exit 75, resume skips finished steps
        v
 runs/<problem>/<run>/  (D:, not in git)          problems/<p>/experiments/E###_slug/runs/<run>/{config,run,result}.json  (git)
        |
 judge: card + problem.json read FROM THE TAG -> every run ok? -> control valid? -> paired scene bootstrap over the
        pooled seeds from the stored counts -> guards -> verdict.json; one commit of the records
        |
 adopt (the owner's decision): the experiment's code onto the branch, the arm's settings into the base config,
        anchor = the evaluation repeated with the new code gives the same numbers -> champions/<problem>/
```

## Files a problem writes
| file | what |
|---|---|
| `problem.json` | name, tag prefix, python environment, base config, steps, score, guards, control floor, splits and holdout, protected files (schema: `spec.py`, `PROBLEM_KEYS`) |
| `training/train.py` | reads one composed config (json path as its argument), writes `log.jsonl` and checkpoints into `out_dir`; exit 0 done, 75 stopped (`trainkit` keeps this contract) |
| `evaluation/adapter.py` + `evaluation/evaluation.json` | the `Evaluation` adapter: `source`, `subject`, `task` (metrics with additive counts, views); the core gives defaults for everything else (`Evaluation/defaults.py`) |
| `experiments/E###_slug/experiment.md` | the card: a fenced `json card` block (schema: `spec.py`, `CARD_KEYS`), then the text |

## Records (one fact, one file, one writer)
| file | writer | holds |
|---|---|---|
| `experiment.md` | the agent | question, arms, rule; the conclusion after the verdict |
| `runs/<run>/config.json` | `start` | the composed config of the run |
| `runs/<run>/run.json` | the worker | state, steps with exit codes and times, commit, tag, machine |
| `runs/<run>/result.json` | the evaluation step | numbers, per-scene score, identities, where the counts and predictions are |
| `verdict.json` | the judge | the verdict, effect and interval, control check, guards, the rule as frozen |
| `ledger.csv` | `index` | generated from the above, never edited |
| `champions/<problem>/` | `adopt` | champion and comparison networks |

## Rules the code enforces
- The rule is what the tag holds: the judge reads card and `problem.json` from the snapshot; later edits do not count.
- A run executes its snapshot, never the working tree. A second `start` of the same experiment (more seeds) reuses its tag.
- Only the worker of a registered run writes records. A call by hand writes into `temp/` and is never a record.
- Identity = code, config, weights, ground truth, data. Stored predictions are reused only under the same identity.
- The holdout split opens only for a run named in the problem's `TEST_LOG.md` under a decided `D<n>`; it is checked at the start and again where the data are read.
- An invalid control (below the floor of `problem.json`, or seeds further apart than the card allows) makes the verdict INVALID.
- A broken guard makes a winning variant a TIE, with the gain and the guard in the verdict.

## Machine file (`~/.research/machine.json`, not in git)
```json
{"data": "/mnt/d/Auto-research/data", "runs": "/mnt/d/Auto-research/runs",
 "python": {"gcar": "/home/<user>/miniconda3/envs/gcar/bin/python"}, "gpu_slots": 2}
```
`research machine init ...` writes it. `RESEARCH_MACHINE` names another file (the tests use a throw-away one). Slots, holds and grants live next to it.
