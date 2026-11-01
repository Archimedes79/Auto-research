# trainsuite - see, stop and judge a training run from the files it writes

A training is a process that writes files. This suite reads them and tells you where the run is, whether it looks healthy and whether a kill criterion of its card is hit; it can ask the trainer to stop. It imports no project code, needs only the standard library (curves also matplotlib) and knows no network, loss or dataset. Run it from `autoresearch/`; `--root` is the folder of the run directories (default `./runs`, so from here e.g. `--root ../bev_fusion/runs`).

## Try it
```
python -m trainsuite selftest                                  # synthetic runs with known answers, a few seconds
python -m trainsuite --root ../bev_fusion/runs status          # runs active in the last 24 h, also one that died before its first row (--all: every run with a metrics file or a text log)
python -m trainsuite --root ../bev_fusion/runs health t8_bevfeat_s0:t8_ctl_s0 --kill "group.lines@1000 > 1.48"
python -m trainsuite --root ../bev_fusion/runs curves t8_bevfeat_s0 t8_ctl_s0      # png: total, every group.*, grad_norm
python -m trainsuite --root ../bev_fusion/runs watch t8_bevfeat_s0:t8_ctl_s0 t8_bevfeat_s1:t8_ctl_s1 --kill "group.lines@1000 > 1.48" --enforce
python -m trainsuite --root ../bev_fusion/runs stop t8_bevfeat_s0
python -m trainsuite --root ../bev_fusion/runs early --calibrate --better lower --band 0.2 --key val.total --out calib.json   # how well does an early gap predict the final one?
python -m trainsuite --root ../bev_fusion/runs early t3_radar_only --vs t2_ref_s0 --better lower --calib calib.json --at 0.5 --min-rows 1   # advice only
```
`run:reference` gives a run its own reference for the loss comparison (`--vs` is one reference for all). `status` with one run adds the phase and readiness, the findings, the kill rules, the loss groups, the module table (grad, change against the earlier rows, update ratio), the time split, the incidents and the last eval; `health` exits 1 on an alert or a kill hit. `watch` prints **one line per change** (state, new finding, kill rule, new eval) and ends when every run is finished: it is made for an agent's Monitor, which then sleeps until something happens. A run that is not running when the watch begins (not started, failed, stopped) is waited for up to `--give-up` minutes (default 180), because it may be restarted. A finding that comes and goes is reported once. Without `--enforce` it only reports; with it a kill hit asks the trainer to stop (`--honour-wait` seconds later, default 300, it writes the request once more if nothing happened, then gives up and says so).

## The contract: what a trainer writes into its run directory
| file | content | the suite uses it for |
|---|---|---|
| `config.json` | `{"steps": N, ...}` | % and ETA |
| `log.jsonl` | one JSON line per logged step: `{"step", "t" (unix time), "total", "lr", "grad_norm", "s_per_step", "gpu_gb", <any scalar, e.g. "group.lines">}`; an eval line `{"step", "val": {<scalars>}}`. After a resume the file may hold steps twice: the later one wins | loss and every other curve, speed, health |
| `train.log` | the text output | a traceback at its end = crashed; lines `non-finite ... skipped (k in a row)` |
| `ckpt_*.pt` | atomic checkpoints; **starting the same command again resumes** | nothing (the trainer's business) |
| `.done` | written only when the process exits with code 0 | state `done` |
| `incident_<step>.json` | optional: one file per incident (a skipped non-finite step, a spike): `{"step", "kind", "batch": [sample ids], "losses": {...}}` | finding `incident`; listed by `status RUN` |
| `CONTROL` / `STOPPED` | the suite writes `CONTROL`; the trainer answers with `STOPPED` | stop (below) |

Optional scalars in a `log.jsonl` row, for the module view: `grad.<module>` (gradient norm of one top-level module), `upd.<module>` (update-to-weight ratio of that module per step), `t_data` / `t_step` (seconds waiting for the data / the whole step). A module that is frozen on purpose is named in `config.json` (`"freeze": [prefix, ...]` or `"train_only": [prefix, ...]`; `fusion`, `fusion.` and `fusion.*` are one spelling, a single string is a list of one, anything else names nothing) and is not asked to learn; a `freeze` prefix inside a module (`head.gate`) does not exempt the whole module (the rest of it trains). A module that gets no gradient at all reports 0.0, which is the finding. A trainer that writes none of them loses nothing.

The loss key is `total` (`--loss KEY`). A project with other names puts a `trainsuite.json` into the run directory (`EXTENSIONS.md` 1).

**Stop.** At every step boundary the trainer looks at `CONTROL`. If it says `{"cmd": "stop"}` the trainer saves a checkpoint, writes `STOPPED` (`{"step", "reason"}`) and exits with code **75**, never 0, so no `.done` appears and nobody evaluates a half-trained model. The launcher `autoresearch/tools/run.py` knows the code: exit 75 is the run status `stopped` (not `failed`, not counted as a failure in `metrics.py`), is never retried even when the log tail holds an older CUDA error, blocks its `--after` dependents and is resumed by starting the same id again. At its **start** the trainer moves `CONTROL` and `STOPPED` of an earlier stop aside (`*.old`), so an old request never stops a resume; no clocks are compared. A request written before the trainer's `train()` begins (imports take 10-40 s) is moved aside too; `watch` notices that nothing happened and writes it again. About 20 lines inlined in the trainer (a run trains from a pinned code snapshot that does not contain this package): `control.stop_requested`, `control.clear_stale`, used in `bev_fusion/training/train.py`; the selftest runs that copy against this module. **A trainer older than this contract ignores CONTROL**: `stop` and `--enforce` look into `<run>/code/train.py` and refuse (`stop` exits 2) when the check is not there; a stop of such a run is a decision about the process. `resume <run>` moves the stop files aside by hand.

## Readiness: how far is the run, and when is it ready?
`status` shows `ready` (0 - 100 %), the ETA and the clock time at which the last step should be logged (`last step`; the final checkpoint and the evaluation follow it, the percentage stays at 99 until then); `status RUN` adds the phase; `status --json` has `phase`, `ready_pct`, `ready_at` (the time of the last step); `watch` prints one line per 10 %. Ready is the share of the steps that are logged, held at **99 % until the process has exited with code 0** (`.done`): a run whose last step is logged and whose final checkpoint is still being written is not ready. A stalled, failed or stopped run keeps the share it reached; a run without `steps` (or with a total below 1) has no percentage. One look gives one answer: the fields of a summary come from one reading of the state. Phases: `new` (nothing written) - `starting` (output, no row yet: imports, caches, first batch) - `training` - `finishing` (all steps logged, no exit 0 yet) - `done` / `failed` / `stopped` / `stalled` / `ended`.

## States and what is checked
`new` (no output yet) - `running` - `stalled` (no output for 15 min or 10 rows' time, `TS_STALE_S` seconds) - `failed` (a traceback at the end of `train.log` that no `step` line follows; seen at once) - `stopped` - `ended` (all steps logged, no `.done`) - `done`. A process killed by the OOM killer leaves no traceback: it shows `running` until it is `stalled`.

| code | level | meaning |
|---|---|---|
| `failed`, `stalled` | alert | crashed (last line of the traceback) / no output |
| `nonfinite`, `skipped` | alert / warn | NaN or inf in a logged row / steps the trainer skipped (3 in a row = alert) |
| `spike` | warn | a value 8 robust sigmas above the recent median (`watch` tests every row since its last look; after a restart from an earlier checkpoint the rows from the resume step on count as new, also when the run has passed the old step again) |
| `shift` | warn after 50 % of the steps, else info | the last 10 rows of `total` jumped by more than 30 % and 5 standard errors over the 30 before, and stay there |
| `rising` | same | window mean of `total` or a `group.*` rose by more than 3 standard errors |
| `grad_norm`, `gpu`, `slow` | warn | grad norm > 10 x its median / the trainer's peak GPU allocation > 11 GB / s per step 1.5 x the typical one of the current attempt (rows with a time `t` only, running runs only) |
| `overfit` | warn | `val.total` rose by 5 % while the training loss fell |
| `vs_ref` | warn if 10 % **and** 3 standard errors worse over the last 50 rows after 20 % of the steps, else info | the loss against a reference run at the same step; a reference with another step total is on another schedule (warm-up, learning rate): info "not compared" |
| `dead_module` | warn | a module's gradient norm is zero over the last 20 rows (detached, frozen by mistake, or unused on purpose: a switched-off branch, a loss weight of 0; modules named in `freeze` / `train_only` are exempt) |
| `grad_module` | warn | a module's gradient norm is 10 times its own earlier median (a single module blows up while the total looks fine) |
| `update_ratio` | warn | one step changes a module's weights by less than 1e-7 or more than 20 % of their size (measured against at least 0.01, so a module that starts at zero does not show millions); Adam moves each weight by about the learning rate, so 1e-2 is normal; the thresholds are not calibrated on a real probe run yet |
| `data_bound` | warn | more than 30 % of a step waits for the data loader (median of the last 50 rows with `t_data` and `t_step`) |
| `incident` | warn | `incident_<step>.json` files exist: the trainer dumped the step and the batch ids of a failure |
| `flat`, `no_metrics` | info | the loss did not move over the last rows / nothing logged yet |
| `kill_rule`, `kill_key` | alert / warn | a line of `kill.txt` or `--kill` is not a rule / the run passed the rule's step and never logged its key (a typo) |

**Healthy is not good.** These checks say the training is sound; whether the network is good is the evaluation (`evalsuite`). And the loss of a multi-task run is not a number that must fall: the bev_fusion `total` rises from 8.5 to 12.5 between steps 500 and 1300 in every run, reference runs included (the warm-up of the dt = +-0.5 loss terms explains it: they enter between 20 % and 40 % of the steps, steps 600-1200 of 3000, and each group sum grows by up to a factor 2; the timing matches, no ablation was run). The reliable early signal is the comparison with a **reference run of the same code and ground truth** (`--vs`, `curves` with two runs) and a kill rule. A loss that jumps and stays is `shift`; a loss that is simply dead (flat) is only `flat` (info).

## Kill rules
The kill criterion of an experiment card as a line a program can check, from `--kill "..."` (repeatable) and `<run>/kill.txt` (UTF-8, with or without BOM, or UTF-16 with a BOM as PowerShell 5.1 writes it; bytes that are no text are one `kill_rule` alert, never a crash); `#` starts a comment:
```
group.lines@1000 > 1.48              # mean of the last 10 logged values up to step 1000
total@500 > 30 window 5              # the last 5 values
val.total@2000 >= 13.5               # the eval row at step 2000 (or the first after it)
```
A rule is `pending` until the run has reached its step, then `ok` or `hit` (a non-finite value is a hit) or `unknown` (the key was never logged). A bad line is reported and the other rules still work.

## Early advice (`early`, advisory)
`early <run> --vs <control>` takes the gap to the control run (signed so that positive = the run is better; the direction is **required**: `--better higher|lower` or `--lower KEY`; a logged `val.*` is mostly a loss) at the fraction of the steps the run has reached (or `--at F`), needs `--min-rows` (3) rows of the key, and compares it with the seed-noise band (`--band`, **required, in the units of the key**; with `--calib` it is the calibration's band and a different value is refused). `early --calibrate [root] --out calib.json` asks, over the finished pairs of a runs root (same step total, same name prefix before the first `_`, or `--pair variant:control`), how often the sign of the gap at f = 0.1 / 0.25 / 0.5 / 0.75 equals the sign of the final gap and how the two rank (Spearman); with fewer than 6 usable pairs it says `not enough pairs to calibrate` and gives no number. The verdict is one of: `behind control by G > band B ... predicted the final sign in X of N pairs ...: `consider stopping` (only if the 95 % Wilson lower bound of the agreement is >= `--min-agree` 0.8 **and** the pairs beyond the band number >= 15 and come from >= 8 distinct runs), `... lower bound below 0.8: no reliable signal`, `... not enough independent evidence`, `inside the band: no signal`, `ahead of the control ...`, `not calibrated: gap only`. `--as-rule` also prints a `key@step > value` line for `kill.txt` to review, only for `consider stopping`; nothing is written, the exit code is 0 (2 = usage error), only `watch --enforce` stops a run. Details and limits: `EXTENSIONS.md` 5.

## Start a training: modes are values in the config
The modes of a training are not code paths but values of the config file. `compose` builds the config of one run from a base file, presets and overrides, and prints every change with its old value; `start` does that, runs the project's static check and prints the exact `run.py` command (`--go` starts it detached and waits 3 s to see whether `run.py` refused the id: a held, running, finished or card-less id; **the same command again resumes the run**):
```
python -m trainsuite compose training/configs/t10_dual_s0.json --preset small --set seed=3 --out /tmp/c.json
AR_PROJECT=../bev_fusion python -m trainsuite start training/configs/t10_dual_s0.json --id t15.a --preset small --set seed=3 --hyp "..." --pred "..." --rule "..." [--go]
```
| mode | how |
|---|---|
| preprocessing on the fly | `--preset onfly` (all caches off) |
| training on data that was computed elsewhere | `--set feature_cache=<path>` (and the other cache keys) |
| small, fast | `--preset small` (400 steps, evaluation every 100) or `--preset smoke` (20 steps: the end-to-end smoke of a new version) |
| full | the base config as it is |
| one module, the rest frozen | `--set 'train_only=["fusion."]'` or `--set 'freeze=["obj_head."]'` (the module prefixes of the model; needs the trainer's freeze support) |
A preset is a json file of overrides in the project's presets folder, merged in the order given; `--set a.b=c` reads `c` as json when it parses; an object (`--set 'model={"x": 1}'`) sets its keys one by one, a value for a group of keys (`--set model=5`) is refused. A key the base file did not have is marked **NEW** (a typo shows there, with the nearest key), and the project's check (for bev_fusion `training/startable.py`) refuses an unknown key before a GPU is touched. `start` needs a `trainsuite` block in the project's `project.json`: `presets`, `out` (where the composed configs go: scratch), `runs`, `check` and `train` (commands with `{config}`); the composed file is named after the run folder (id with `.` as `_`) and the key `name` of the config is set to it (a `--set name=` is refused: the trainer writes to `runs/<name>` and the launcher script names the run after the config file, they must agree). Without `--go` the composed config goes to `<id>.preview.json`, so a preview never changes the file a running or retrying run reads; for an id that already has a run, a different config is refused (a resume needs the same one; the trainer copies the config it ran with into the run folder). The card (`--hyp --pred --rule`, or `--no-card` for a diagnostic run) is required, as for `run.py`.

## Not in this suite (yet)
A queue of runs (`run.py` does the process side: record, GPU slots, retry), a run fingerprint that refuses comparisons across different code / ground truth / caches, and a ledger row for a killed run. More in `EXTENSIONS.md`.
