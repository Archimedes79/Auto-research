# jobrunner - run a method of a suite as a job, with the data named in the job

One call of a suite (`Evaluation.run`, `Runner.predict`, `Training.run`, any method you list) becomes a **job**: a small json that says WHAT to call and on WHICH data. `submit` checks it, writes it down and starts it under `run.py`; `list` / `status` show what became of it. Nothing here schedules: `run.py` stays the queue (its first-come line for GPU slots, RAM gate, machine lock, `--after`, ledger row, card and TEST gates). The state of a job is read from run.py's own record with run.py's own functions, there is no second store. Design history and the reasons: `../../bev_fusion/records/jobs/JOB_LAYER_PROPOSAL.md`.

```
python3 -m jobrunner JobRunner.check  job.json      # look only: problems, what it needs, what it reads and writes, the run.py command
python3 -m jobrunner JobRunner.submit job.json      # write it down and start it (it waits in run.py's line)
python3 -m jobrunner JobRunner.list                 # all jobs and states        python3 -m jobrunner JobRunner.status <id>
python3 -m jobrunner methods                        # the commands; `selftest` checks the suite on a throw-away project (~40 s)
```
Run from `autoresearch/` under WSL (it needs `/proc`) or with `autoresearch/` on `PYTHONPATH`; `--project DIR` names the problem folder (default `AR_PROJECT`). Standard library only; the suites it calls need their own environment: **the interpreter of the child processes is an option, nothing has to be set in `project.json`**: `--python PATH` (Python: `JobRunner(project, python=PATH)`), else the environment variable `JOBRUNNER_PYTHON`, else `project.json` `jobrunner.python` if someone set it, else the interpreter that runs the job runner. A catalogue entry's or a job's own `python` wins over all of them. Example for the BEV system (torch is in the `gcar` environment): `python3 -m jobrunner JobRunner.check job.json --project ../bev_fusion --python ~/miniconda3/envs/gcar/bin/python`.

## The interface: class `JobRunner(project)`
| method | does | returns |
|---|---|---|
| `catalogue()` | the calls that are jobs, how each runs, what it takes, what it reads and writes, its parameters | list of dicts |
| `check(job, deep=False)` | looks and starts nothing. `deep`: also the suite's own dry run (`Evaluation.check`, `Runner.plan`, `Training.check`: the entry's `dry_run`) in the interpreter of the job, with the arguments it knows: it finds what only the suite knows (a network name that is not a run folder, data that does not open); seconds to a minute of CPU and RAM, outside run.py's gates, so it is asked for, not the default | `{ok, id, call, how, state, problems, warnings, needs, reads, writes, after, diagnostic, command, in_process}`; exit 1 when not ok |
| `submit(job, start=True, deep=False)` | check (`deep`: with the dry run), write the job, start it; the same job again is not started twice; a failed, blocked, stopped or dead one is started anew, also with changed arguments (the earlier attempt stays as `job.1.json`, `args.1.json`, `result.1.json`); a job that is done or in progress needs a new id | `{ok, id, state, started, pid}` |
| `start(i)` | start a job submitted with `start=False`, or one that ended failed, blocked, stopped or dead | as `submit` |
| `run(job, wait=0)` | `submit` and wait until it ends (or `wait` seconds) | as `status`, plus `ok` = the job is done (the command exits 1 otherwise) |
| `list(state="")` | every job, oldest first | list of `{id, call, state, submitted, writes, why}` |
| `status(i)` | one job: state and why, exit code, data, last log lines, the result of the call | dict |

Every method is also a command (the convention of the suites, `../README.md`); an error is `error: ...` on stderr with exit 2 (`JobError`, `JOBRUNNER_TRACEBACK=1` shows the traceback).

## The job: what to do on which data
```json
{"id": "t23.C_s1.eval",
 "call": "Evaluation.run",
 "args": {"network": "t23.C_s1", "frames": "large.val"},
 "after": [], "needs": {}, "card": null, "no_card": false, "split": "", "note": ""}
```
| field | meaning |
|---|---|
| `id` | the job and, for a `process` job, the run id in run.py's records (letters, digits, `. _ -`); may be left out when the catalogue entry takes it from the arguments (`Training.run`: `job.id`) |
| `call` | `Class.method` of a catalogue entry (`catalogue` lists them) |
| `args` | **exactly the keyword arguments of that method**, checked against its signature (unknown name, missing required one, a wrong basic type are problems). The data are arguments: a network (`network`), a set of frames (`frames`: `val`, `large.val`, ...), a ground-truth key, a data root |
| `needs` | what it takes from the machine, over the entry's defaults: `gpu` (true / false), `gpu_max_used_mb` (start only when the GPU memory in use is below this: run.py `--gpu-max-mb`), `ram_gb` (GPU jobs: run.py `--ram-mb`), `cpu`, `est_hours` (recorded); numbers, not texts |
| `reads`, `writes` | data references (below); normally derived from `args` by the entry, may be given |
| `after` | job or run ids that must be done first; ids of runs the job reads, and of jobs that write what it reads, are added automatically |
| `card` / `no_card` | `card = {hyp, pred, rule}` registers a decision rule for a call that may feed a verdict; without a card the run is a **diagnostic** (`run.py --no-card --no-git`) and never feeds a verdict. A call that never feeds a verdict refuses a card |
| `split` | passed to `run.py --split`. The holdout split (TEST) is behind run.py's logged-row gate: `check` applies it too |
| `retries`, `timeout` | run.py `--retries` (not passed unless given: run.py's own default stays, the CUDA retry) and `--timeout` (minutes) |
| `python`, `run_args`, `priority`, `note` | the interpreter of the child process; more run.py flags; recorded only; the ledger hypothesis of a diagnostic (default: the call and its arguments) |

**The result of one job as the argument of the next:** an argument may be `{"$result": "<job id>:<path>"}`, e.g. `{"results": {"$result": "chain.predict:results"}}` (the path is dotted into the other job's `result.json`, list items by number: `nested.list.1`). The job waits for that job (`after` is added), the value is filled in when the call is made, and a path that is not in the result is an error that says where it stopped. Such an argument counts as given and is not type-checked (it is known only later); the dry run of `check --deep` is skipped for it. This is how a chain is written down in advance: GT, training, `Runner.predict`, `Evaluation.score`.

Keys starting with `_` are comments, an unknown key is an error. A `launch` call (`Training.run`) starts a run of its own and takes `after`, `timeout`, `split`, `retries`, `run_args`, `python` and a card in its arguments: giving them here is an error, not silently ignored. A training job file of `trainsuite` (it has `config`, no `call`) is accepted as it is: it becomes `Training.run` with that file as `args.job`.

**Data references** are typed `kind:name`, built from the arguments by the entry (`run:{network}`, `set:{frames}`, `eval:{network}@{frames}`):
| kind | checked | effect |
|---|---|---|
| `run:X` | a run may be named as `t23.C_s1`, `t23_C_s1` or `runs/t23_C_s1/`; a done run, a run or job in progress or a run folder must exist; a failed, blocked or stopped run is a problem | a run in progress, or a job X, is added to `after` (run.py waits for its done flag) |
| `path:P` | must exist (relative to the problem folder) | none |
| any other (`set:`, `eval:`, `results:`, `predictions:` ...) | not verified here (the suite checks its own data in its `check`) | a job in progress that writes the same reference is added to `after`; two jobs in progress must not write the same reference. The outputs of a finished job are the suite's business (it refuses to overwrite). The built-in `Runner.predict` writes `predictions:...` while `Evaluation.score` reads `results:{results}` (an id the Runner returns): that chain needs an explicit `after` |

**What `check` refuses before run.py would**, because the job form hides it from run.py: the holdout split named in the arguments that name a split or a set of frames (run.py reads only its own command line, the arguments are in a file; the entry's `splits` lists those arguments, default `split frames set sets subset`; free text, a card text or a path is never scanned) unless the job says `split` = the holdout, which sends it to run.py's TEST-log check; a ledger row that carries a card for a `process` job that is submitted as a diagnostic (a `launch` call such as `Training.run` has its card in its arguments and a planned row first is its normal case); a HOLD marker on the id; a twin run folder (`X.a` / `X_a`) that runs or is finished; an id that names a run this runner did not make; two ids that differ only in case; `run_args` that contradict the job (`--id`, `--after`, `--split`, `--hyp` ...); an `after` that names nothing. For a job that is already in progress or done, what has changed around it since is no news: `submit` answers "already submitted".

## What is a job: the catalogue (a suite needs no code change)
`catalogue.py` holds the built-in entries (`Training.run`, `Evaluation.run`, `Evaluation.score`, `Runner.predict`). An entry says `suite` (module), `cls`, `method`, `how`, `may_feed_verdict`, `needs`, `reads`, `writes`, `python`, `id_from`, `after_from` (where the run ids a `launch` call waits for stand in its arguments, `job.after` for `Training.run`: `check` then shows them in `after` and finds a typo before the start; `check` of a `launch` call has no run.py `command`, it says `in_process: Training(project).run(**args)`), `splits`. A suite is added at three levels, from least to most work:
1. nothing: the built-in table already names the four methods; the class is built as `cls(project_folder)` (the convention of the suites) and the method is called with `**args`;
2. the project's `project.json` (optional, the interpreter can be given as an option instead): `"jobrunner": {"python": "~/miniconda3/envs/gcar/bin/python", "catalogue": {"Toy.work": {"suite": "toysuite", "cls": "Toy", "method": "work", "how": "process", "reads": ["path:{src}"]}}}` adds or overrides entries and names the interpreter of the children (needed where the suite imports torch; the signature of the method is then read by that interpreter too);
3. the suite exports its own table as a module attribute `JOBS = {"Class.method": {...}}` (same format) and `project.json` lists the module in `"jobrunner": {"suites": ["mysuite"]}` (the four built-in suites are always read). A suite that cannot be imported here, or exits on import, just keeps its other entries.

`how = "process"`: `run.py ... -- <python> -m jobrunner call --suite S --cls C --method M --args-file A --out R --project P` (the child builds `C(project)`, calls `M(**args)`, writes the result json; exit 0 done, 1 the result says `ok: false`, 2 an error). For long or GPU methods: a crash does not take the runner down, and slots, RAM gate, lock and ledger row apply. `how = "launch"`: the method itself starts a run.py run and returns (`Training.run`); the runner calls it in its own process and then follows that run's record.

## States
From run.py's `state_of` (done flag first, heartbeat age, HOLD marker) on `<run_records>/<id>/`:
| state | when |
|---|---|
| `planned` | submitted with `start=False` |
| `queued` | run.py is waiting (`--after`, slot, RAM, lock; `why` says what it waits for, from the heartbeat), or the launcher lives and has not written a record yet, or the job was just started again and run.py has not written the new record |
| `running` | the command runs |
| `held` | a HOLD marker on the id (hold.py): it starts only after the person's words clear it |
| `stalled` | no heartbeat for `AR_STALE_S` (300) seconds; `why` says whether a process of it still lives (then it is not started again) or not (then it can be) |
| `blocked`, `stopped` | run.py's own: a dependency failed or the run was refused; a stop request. Both can be started again |
| `done` | the done flag (exit 0) |
| `failed` | run.py status failed, timeout or gpu_busy, or the launcher ended without a record (see the launch log in `status`), or the start itself raised |

## Where things are
`<run_records>/.jobs/<id>/`: `job.json` (the job as submitted, its run.py command, the launcher pid), `args.json`, `result.json` (what the call returned; an earlier attempt's result is renamed `result.1.json`, `result.2.json`, never deleted), `launch.log`, `submit.lock` (one submit or start of an id at a time: the second waits up to `JOBRUNNER_LOCK_WAIT_S` = 10 s and is then told "already submitted"; the file holds pid, start time and thread, so a dead owner's lock or a reused pid is taken over). The folder lies next to `.gpu_slots`; none of it is in git (`.gitignore`); a recipe that should be kept goes to the project's job folder, e.g. `records/jobs/`.

## What it does not do (version 1)
- no scheduler, priorities or daemon: run.py's line decides the order (`priority` is recorded only);
- no `stop` / `hold` / `resume` wrappers yet: use `tools/hold.py` for a waiting run and the suite's own stop (`Training.stop`) for a running one;
- no CPU gate (`needs.cpu` is recorded); `ram_gb` is applied by run.py to GPU jobs only;
- the ground-truth suite is not in the built-in table: it brings its own (level 3): `GtGenerator/jobs.py` exports `JOBS` with `GtGenerator.generate_bev_gt` and `GtGenerator.generate_lane_gt` (run through `run_nice.sh`, no card), and `bev_fusion/project.json` lists `GtGenerator` in `jobrunner.suites`. Two jobs that write the same store are found only by the reference names (`gt:lane.all@...` and `gt:lane.bev_fusion@...` do not collide);
- run.py's own refusals that depend on its state at the moment of the start (the machine lock, a full GPU) are not predicted; they show as `failed` or `blocked` with the reason from the record;
- the arguments are checked against the method's signature by importing the class here or, when that fails, in the interpreter of the job (a minute or two the first time for a heavy suite); if neither works a warning says so;
- a process that imports two projects' suites keeps the first one's modules in `sys.modules`: one JobRunner process per project;
- it uses run.py's own functions (`RUN_NAMES` in `api.py`: `state_of`, `record`, `command_alive`, `twin_refusal`, `is_holdout`, `ledger_row`, `test_log_refusal`, `HOLD`, ...; the selftest and the first use check that they exist): when run.py renames one, the error is a `JobError` that says which, not a traceback;
- `--result` of run.py is not forwarded: the ledger row says what was called (`--hyp` / `--basis`), the result of the call is in `.jobs/<id>/result.json`.

## Tests
`python3 -m jobrunner selftest` (`selftest.py`, 183 checks, ~1.5 min): the catalogue, the check of a job, the gates of run.py a job could slip past, the run.py command a job becomes, submit / start / restart / idempotence / failure / the lock (also two submits at once) with a fake launcher, the states from fabricated run.py records (including a zombie launcher, a `launch` method that is still running, a restarted `launch` job), and the whole way with the REAL `run.py` and the child process on a throw-away problem folder (a chain by `reads`, a failing call, a result with `ok: false`, a card job with its ledger rule and code tag, the launch form, the command line and its exit codes). Two review rounds by a fresh critic found 21 and then 9 defects; each fix has a test, and planted bugs (a flag dropped, no dependency inferred, `gpu_busy` read as done, the same job started twice, write conflicts ignored, unknown arguments allowed, the TEST split in the arguments not found, free text scanned, the lock not looked at again, a zombie launcher alive, ...) are each caught.
