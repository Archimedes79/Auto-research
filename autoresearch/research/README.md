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
 adopt (the owner's decision): a KEEP must come out again, judged once more from the records (minutes; not by hand) -> the arm's settings
        become the defaults of the base config (its code is committed, default off, before the start) -> anchor: the evaluation repeated with
        the working tree gives the same numbers -> champions/<problem>/, one commit (a failure before the commit leaves base config and
        champions/ as they were)
 register (the owner's decision): a measured network (kind measure, verdict DONE) as champion or comparison network
        (adopt and register need the seeds' run folders)
```

## Upkeep
| verb | what |
|---|---|
| `tests <p> [--only text]` | the problem's own tests (`problem.json` `tests`: modules of its package, started with `-P -m`) with its interpreter and module path and, when it has `training/architecture.json`, the drawing against the code; after any change of the problem's code |
| `py <p> [python arguments]` | python by hand as a test gets it: the problem's interpreter and the module path of the working tree, from any folder (`py toy -m toy.tests.test_toy --quick`), with `-P` when the problem is a package (one that is none yet starts without, so a script finds the modules next to it); a diagnostic, never a run |
| `anchor <p> <run>` | a finished run's evaluation step run again with the working tree: do its numbers come out (`anchor_tolerance`)? It covers what that step runs: a problem that predicts in a step of its own (`lane_center`) re-scores the stored predictions, so a change of its network needs a new run |
| `compare <p> <run> <run>` | two finished runs side by side: a diagnostic, never a verdict |
| `clean <p>` | writes the run folders nothing needs any more to a list file in the run root, with a hash. The owner confirms that list with a decided row that names the hash; `clean <p> --delete <list> --decision D<n>` then deletes its runs, each checked again: finished, no live step, a verdict with no arm KEEP, INCONCLUSIVE, TIE or MEASURED, named by no card (`from`, `needs`, a `{runs}/<run>` setting), no champion or comparison seed |
| `index <p> [--check]` | writes `ledger.csv`; `--check` exits 1 when it is not what the records give (a problem without runs needs none) |
| `report <p> <folder> [--strict]` | a report built again from the records and the stored predictions, into the run root (`../reporter/README.md`) |
| `hold <p> "<why>"`, `release <p>` | no start or resume of a problem until released |
| `grant <path> --decision D<n> --note "..." [--hours h]` | a protected file or folder (path ending in `/`) may be edited for some hours |
| `commit <paths> -m "..."` | commit these paths only, through a temporary index (paths as git reads them or from the top) |
| `push-check [--no-tags]` | what a push would publish that never belongs into git, in the unpushed commits and in the commits only the tags carry that the remote lacks (it asks the remote; `--no-tags`: the branch only): forbidden files, and matches of the regular expressions in `.private_patterns` at the top (one a line, not in git; searched in authors, messages, file names, added lines, new tags and the branch name; a match is never printed). Without that file it warns that it did not look; other binary files than UTF-16 text are listed as not scanned |
| `machine [init]` | show or write the machine file |

## Files a problem writes
| file | what |
|---|---|
| `problem.json` | name, tag prefix, python environment, base config, steps (a step that writes into a subfolder of the run folder names it, `dir`: its `log.jsonl` and `CONTROL` are there), default steps, score, guards, control floor, evaluation split and metrics, holdout, protected files, its tests (`tests`). Schema: `spec.py`, `PROBLEM_KEYS` |
| `__init__.py`, empty, in the problem folder and every code folder | the problem is one package named like its folder: `from <problem>.net.model.net import DiscNet`; steps and tests are modules (`{python} -P -m <problem>.<module>`), nothing edits `sys.path`. Checked by `pkgcheck.py`, which the problem's own test calls. A problem that is no package yet (no `__init__.py`) keeps script paths and the key `pythonpath`; in a package that key is refused |
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
- The rule is what the tag holds: the evaluation step and the judge read card and `problem.json` from the snapshot; later edits do not count.
- Callers of `new` in one checkout get different experiment numbers; two clones can still pick the same one.
- A run executes its snapshot, never the working tree. A second `start` of the same experiment reuses its tag; more seeds: append them to
  the card's `seeds`, then `start`.
- Code is linked per process, never installed: a step, a test and the judge's computation get the module path `paths.module_path` builds
  from ONE tree (a run: its snapshot): `problems/`, `autoresearch/` and the top of the repository. The core's own steps (the evaluation,
  the judge's computation, `anchor`, `compare`, report pictures) start python with `-P` (`paths.core_step`); a problem's steps start as
  its `problem.json` writes them, which for a package is `-P -m` (`pkgcheck.py` refuses another form), and its tests and `research py`
  get `-P` when it is a package. The evaluation step refuses when a module of the problem's package, `Evaluation`, `trainkit`,
  `research` or `GtGenerator` was loaded from outside the snapshot: before it predicts, and again before it writes its record (what it
  stored is then removed). `Evaluation` refuses an adapter that was imported from outside its project folder. An install of a problem
  or of the core into the environment would make the working tree importable by every process; it is not done.
- Only the worker of a registered run writes records: the evaluation step refuses to run outside a worker. `start` refuses uncommitted code
  of the problem and the core (records and `literature/` are not code).
- Identity = code, config, weights, ground truth, data. Stored predictions are reused only under the same identity.
- Arms are compared only when scored alike (split, frames, ground truth by its content signature, evaluator, metrics), else INVALID; records
  that hold no signature are compared by the ground truth's name, with a warning in the verdict.
- A control below the floor of `problem.json`, or with seeds further apart than the card allows, makes the verdict INVALID. A broken guard
  makes a winning variant a TIE, with the gain and the guard in the verdict.
- A KEEP of the standard rule needs two seeds that agree in sign: from one seed, or from seeds that disagree, it is INCONCLUSIVE (the
  interval is a bootstrap over the scenes; only a second seed shows what training alone moves). A card's own rule script decides after it.
- A GPU step waits for one of the machine's slots; a step with `gpu_mb` also waits, holding its slot, until that much memory is free on
  its GPU and no other such step started in the last 45 s; waiting steps have no order. It reads what is allocated now, so memory that
  jobs outside this system hold counts, but nothing is reserved: a job that grows later, or one that started a moment ago outside this
  system, is not seen. Its own GPU counts when the worker bound it (a machine with several GPUs) or the caller set
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`; otherwise every GPU of the machine must have the memory. More than the GPU has at all fails the step.
  When `nvidia-smi` cannot be asked, it does not wait; `run.json` records the free memory the step saw (`gpu_free_mb`).
- TEST and protected files: `../METHOD.md`, hard rules 1 and 2. The holdout gate is read by `research test`, in the evaluation step and
  wherever data are read through `Evaluation` (`Evaluation/defaults.py`: `holdout_open`; `guard` for a problem's own reader).

## Machine file (`~/.research/machine.json`, not in git)
```json
{"data": "<data root>", "runs": "<runs root>", "python": {"gcar": "<interpreter>"}, "gpu_slots": 2}
```
`research machine init ...` writes it; `RESEARCH_MACHINE` names another file (the tests use a throw-away one). Slots, holds and grants live
next to it (`state/`); only the research commands write there. With several GPUs, a step of one GPU sees the device of its slot (slot k:
device k modulo their number, from the caller's `CUDA_VISIBLE_DEVICES`, else `nvidia-smi`; callers that show different devices can still
meet on one). A step, the computation of a verdict and an anchor test get the caller's
environment without the variables that could point them at another tree (`paths.child_env`), then their own.
