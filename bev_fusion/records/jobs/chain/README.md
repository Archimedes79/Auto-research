# jobs/chain - the whole way on one data set: GT -> training -> predictions -> numbers

Four job files, one call each (the interface of every suite is the same: a class, a method, arguments; the data are arguments). They were run end to end on 2026-11-10 as `chain.*` and, on the final tree of the clean-base round, as `chain2.*` (GT store complete, 30 training steps, 378 frames of `small.val`, 10 scenes, scored; 11:55 - 12:14, RTX 3060, warm caches). The training here is a smoke test, not a result: the point is that the way works and that what runs on which data is written in the files.

| step | file | call | data it names | waits for |
|---|---|---|---|---|
| 1 | `gt.json` | `GtGenerator.generate_bev_gt` | `dataroot` (the nuScenes folder), `set_name` (a GtGenerator set: `full` or `small`; `$PY -m GtGenerator sets` from the workspace root). A complete store is not rebuilt | - |
| 2 | `train.json` | `Training.run` (a trainsuite job file) | `config`, `set.split` (a BEV split: `small`, `large`, `xl`; `splits/*.json`), `preset` (length) | `demo.gt` |
| 3 | `predict.json` | `Runner.predict` | `network` (the run folder of step 2: run id `demo.t20` = `runs/demo_t20`), `frames` | `demo.t20` |
| 4 | `score.json` | `Evaluation.score` | `results` (`{"$result": "demo.predict:results"}`: the id step 3 returned), `frames` (the same as step 3), `gt_key` (default: the GT version of the store) | `demo.predict` |

## Run it
From the workspace root inside WSL (`wsl`), the environment is `gcar` (torch). Every submit returns at once; the job waits in run.py's line (GPU slot, RAM gate, `after`):
```
cd autoresearch
export AR_PROJECT=../bev_fusion
PY=~/miniconda3/envs/gcar/bin/python
for j in gt train predict score; do
  python3 -m jobrunner JobRunner.submit ../bev_fusion/records/jobs/chain/$j.json --project ../bev_fusion --python $PY || break
done
python3 -m jobrunner JobRunner.list --project ../bev_fusion                  # states, one line per job
python3 -m jobrunner JobRunner.status demo.score --project ../bev_fusion     # state, why, last log lines, the result of the call
```
- **Submit in this order.** `check` of a later job says "no such run and no job with that id" (`after`, `reads`) or "$result ...: no such job" until the earlier ones are submitted (it resolves them against the jobs it knows); that is the reason for the loop, and a first look at a single file is `JobRunner.check <file>` (add `--deep` once the jobs before it exist: it runs the suite's own dry run, `Training.check`, `Runner.plan`, `Evaluation.check`).
- **A new id for a new run.** A job that is done or in progress is not started twice. Copy the folder for another run (`chain_large/`, then the submit loop names it) and change, in all four files, the id and every reference to an id: `after` in steps 2 - 4, `network` in step 3 (the run folder: the dots of the run id become underscores, `lg1.t20` = `lg1_t20`) and `$result` in step 4 (`lg1.predict:results`). The ids are `demo.*` here, a real run gets a name that says what it is (`t24.A_s0`).
- **Name the run folder, not the run id.** `network` is `demo_t20` (the folder `runs/demo_t20`): `Runner.predict` with the dotted id `demo.t20` ends `failed: no run folder with a config.json` (that is how the first attempt `chain.predict` ended; `chain.predict2` with the folder name worked).
- `JobRunner.check` of the gt job prints `est_hours: 6`: that is the cost of a store that is NOT complete; a complete store ends in seconds (chain2: skipped).
- The numbers: `$PY -m evalsuite Evaluation.show-numbers <evaluation id>` from `autoresearch/` (`$PY` as above; the system `python3` has no numpy). The result of step 4 (`JobRunner.status demo.score`) holds two ids: `id` (`<network>@e<hash>`, the evaluation: this one goes to `show-numbers`) and `results` (`<network>@r<hash>`, the stored predictions of step 3: `show-numbers` refuses it); the predictions of step 3 lie in `runs/demo_t20/eval/results/<id>` (not in git); the records of the jobs in `records/runs/.jobs/<id>/` (`job.json`, `args.json`, `result.json`, `launch.log`).

## Another data set: what changes
The data are named in the arguments and nowhere else. To run the chain on `large`:

| where | field | small (here) | large |
|---|---|---|---|
| `gt.json` | `args.set_name` | `full` (also `small`: the 10 mini sequences of `GtGenerator/sets/small.txt`) | `full` |
| `train.json` | `set` and `preset` | `set.split` `small`, preset `small`, and the smoke values `steps` 30, `warmup`, `eval_every`, `eval_batches`, `save_every` | delete `preset` and the whole `set` (`t20_base.json` already says `"split": "large"` and `steps` 6000, `warmup` 300: a real run of hours on the RTX 3060); a shorter one: `set.steps`. Changing only `set.split` keeps the 30-step smoke values and trains 30 steps on `large` without a word (`check` accepts it). `xl`: also `feature_cache` set values, `splits/README.md` |
| `predict.json` | `args.frames` | `small.val` | `large.val` |
| `score.json` | `args.frames` | `small.val` | `large.val` |

Not measured in this round: what `Runner.predict` on `large.val` costs. `splits/README.md` says that of the 3,109 val samples of `large` 1,673 had their caches on 2026-11-03 and "the rest is written when an evaluation first reads it"; whether the GT-free Runner builds the missing feature caches itself, and how long and how much disk that takes, was not looked at. Take a `spread:N:large.val` first (a small number of frames) and read the time.

The grammar of `frames` (Runner and Evaluation are the same): `<split>.<part>` = a file of `splits/` and its `train` or `val` list; `val`, `train`; `spread:N:<set>` (N frames spread over the set); `scenes:a,b:val`; `random:<n>:<seed>:val`; `frames:scene-0061/12:val`. The step 4 `frames` must be the frames of step 3 (what was not predicted cannot be scored). The TEST split is not a name here: a job that names it is refused by `check` unless it says `split` = the holdout, which goes through run.py's TEST-log gate.

**One name, two things.** The GtGenerator set `small` (10 mini sequences) is not the BEV split `small` (60 train / 10 val scenes of the full store): `set_name` is the first, `split` / `frames` the second. The BEV splits need the GT of the full store, which is why step 1 names `full` for a `small` training.

**Not in this chain.** A network trained before the D48 redesign (the kept nets of `keep3.json`: `t11_vel_hurdle_acc2`, `t10_dual_*`, `t8_ctl_*`) does not run GT-free through the Runner; evaluate it with `bev_suite.py eval` on GT (`evaluation/README.md`). A training that feeds a verdict gets its card in the job (`card = {hyp, pred, rule}`) instead of `no_card`, and its TEST row before it starts.

The reference for the job fields, the states and the catalogue is `../../../../autoresearch/jobrunner/README.md`; the design history is `../JOB_LAYER_PROPOSAL.md`.
