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
- **Submit in this order.** `check` of a later job says "no such job" until the earlier ones are submitted (it resolves `after` and `$result` against the jobs it knows); that is the reason for the loop, and a first look at a single file is `JobRunner.check <file>` (add `--deep` once the jobs before it exist: it runs the suite's own dry run, `Training.check`, `Runner.plan`, `Evaluation.check`).
- **A new id for a new run.** A job that is done or in progress is not started twice. Copy the folder (or change the ids and `network`) for another run; the ids are `demo.*` here, a real run gets a name that says what it is (`t24.A_s0`).
- **Name the run folder, not the run id.** `network` is `demo_t20` (the folder `runs/demo_t20`): `Runner.predict` with the dotted id `demo.t20` ends `failed: no run folder with a config.json` (that is how the first attempt `chain.predict` ended; `chain.predict2` with the folder name worked).
- The numbers: `$PY -m evalsuite Evaluation.show-numbers <evaluation id>` from `autoresearch/` (`$PY` as above; the system `python3` has no numpy; the id is in the result of step 4: `JobRunner.status demo.score`); the predictions of step 3 lie in `runs/demo_t20/eval/results/<id>` (not in git); the records of the jobs in `records/runs/.jobs/<id>/` (`job.json`, `args.json`, `result.json`, `launch.log`).

## Another data set: what changes
The data are named in the arguments and nowhere else. To run the chain on `large`:

| where | field | small (here) | large |
|---|---|---|---|
| `gt.json` | `args.set_name` | `full` (also `small`: the 10 mini sequences of `GtGenerator/sets/small.txt`) | `full` |
| `train.json` | `set.split` | `small` | `large` (`xl`: also `feature_cache` set values, `splits/README.md`); `steps` and the preset for the length |
| `predict.json` | `args.frames` | `small.val` | `large.val` |
| `score.json` | `args.frames` | `small.val` | `large.val` |

The grammar of `frames` (Runner and Evaluation are the same): `<split>.<part>` = a file of `splits/` and its `train` or `val` list; `val`, `train`; `spread:N:<set>` (N frames spread over the set); `scenes:a,b:val`; `random:<n>:<seed>:val`; `frames:scene-0061/12:val`. The step 4 `frames` must be the frames of step 3 (what was not predicted cannot be scored). The TEST split is not a name here: a job that names it is refused by `check` unless it says `split` = the holdout, which goes through run.py's TEST-log gate.

**One name, two things.** The GtGenerator set `small` (10 mini sequences) is not the BEV split `small` (60 train / 10 val scenes of the full store): `set_name` is the first, `split` / `frames` the second. The BEV splits need the GT of the full store, which is why step 1 names `full` for a `small` training.

**Not in this chain.** A network trained before the D48 redesign (the kept nets of `keep3.json`: `t11_vel_hurdle_acc2`, `t10_dual_*`, `t8_ctl_*`) does not run GT-free through the Runner; evaluate it with `bev_suite.py eval` on GT (`evaluation/README.md`). A training that feeds a verdict gets its card in the job (`card = {hyp, pred, rule}`) instead of `no_card`, and its TEST row before it starts.

The reference for the job fields, the states and the catalogue is `../../../autoresearch/jobrunner/README.md`; the design history is `../JOB_LAYER_PROPOSAL.md`.
