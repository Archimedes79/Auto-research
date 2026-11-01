# trainsuite - see, stop and judge a training run from the files it writes

A training is a process that writes files. This suite reads them and tells you where the run is, whether it looks healthy and whether a kill criterion of its card is hit; it can ask the trainer to stop. It imports no project code, needs only the standard library (curves also matplotlib) and knows no network, loss or dataset. Run it from `autoresearch/`; `--root` is the folder of the run directories (default `./runs`, so from here e.g. `--root ../bev_fusion/runs`).

## Try it
```
python -m trainsuite selftest                                  # synthetic runs with known answers, a few seconds
python -m trainsuite --root ../bev_fusion/runs status          # runs active in the last 24 h (--all: every run with a metrics file)
python -m trainsuite --root ../bev_fusion/runs health t8_bevfeat_s0:t8_ctl_s0 --kill "group.lines@1000 > 1.48"
python -m trainsuite --root ../bev_fusion/runs curves t8_bevfeat_s0 t8_ctl_s0      # png: total, every group.*, grad_norm
python -m trainsuite --root ../bev_fusion/runs watch t8_bevfeat_s0:t8_ctl_s0 t8_bevfeat_s1:t8_ctl_s1 --kill "group.lines@1000 > 1.48" --enforce
python -m trainsuite --root ../bev_fusion/runs stop t8_bevfeat_s0
```
`run:reference` gives a run its own reference for the loss comparison (`--vs` is one reference for all). `status` with one run adds the findings, the kill rules and the last eval; `health` exits 1 on an alert or a kill hit. `watch` prints **one line per change** (state, new finding, kill rule, new eval) and ends when every run is finished: it is made for an agent's Monitor, which then sleeps until something happens. A run that is not running when the watch begins (not started, failed, stopped) is waited for up to `--give-up` minutes (default 180), because it may be restarted. A finding that comes and goes is reported once. Without `--enforce` it only reports; with it a kill hit asks the trainer to stop (`--honour-wait` seconds later, default 300, it writes the request once more if nothing happened, then gives up and says so).

## The contract: what a trainer writes into its run directory
| file | content | the suite uses it for |
|---|---|---|
| `config.json` | `{"steps": N, ...}` | % and ETA |
| `log.jsonl` | one JSON line per logged step: `{"step", "t" (unix time), "total", "lr", "grad_norm", "s_per_step", "gpu_gb", <any scalar, e.g. "group.lines">}`; an eval line `{"step", "val": {<scalars>}}`. After a resume the file may hold steps twice: the later one wins | loss and every other curve, speed, health |
| `train.log` | the text output | a traceback at its end = crashed; lines `non-finite ... skipped (k in a row)` |
| `ckpt_*.pt` | atomic checkpoints; **starting the same command again resumes** | nothing (the trainer's business) |
| `.done` | written only when the process exits with code 0 | state `done` |
| `CONTROL` / `STOPPED` | the suite writes `CONTROL`; the trainer answers with `STOPPED` | stop (below) |

The loss key is `total` (`--loss KEY`). A project with other names puts a `trainsuite.json` into the run directory (`EXTENSIONS.md` 1).

**Stop.** At every step boundary the trainer looks at `CONTROL`. If it says `{"cmd": "stop"}` the trainer saves a checkpoint, writes `STOPPED` (`{"step", "reason"}`) and exits with code **75**, never 0, so no `.done` appears and nobody evaluates a half-trained model. The launcher `autoresearch/tools/run.py` knows the code: exit 75 is the run status `stopped` (not `failed`, not counted as a failure in `metrics.py`), is never retried even when the log tail holds an older CUDA error, blocks its `--after` dependents and is resumed by starting the same id again. At its **start** the trainer moves `CONTROL` and `STOPPED` of an earlier stop aside (`*.old`), so an old request never stops a resume; no clocks are compared. A request written before the trainer's `train()` begins (imports take 10-40 s) is moved aside too; `watch` notices that nothing happened and writes it again. About 20 lines inlined in the trainer (a run trains from a pinned code snapshot that does not contain this package): `control.stop_requested`, `control.clear_stale`, used in `bev_fusion/training/train.py`; the selftest runs that copy against this module. **A trainer older than this contract ignores CONTROL**: `stop` and `--enforce` look into `<run>/code/train.py` and refuse (`stop` exits 2) when the check is not there; a stop of such a run is a decision about the process. `resume <run>` moves the stop files aside by hand.

## States and what is checked
`new` (no output yet) - `running` - `stalled` (no output for 15 min or 10 rows' time, `TS_STALE_S` seconds) - `failed` (a traceback at the end of `train.log` that no `step` line follows; seen at once) - `stopped` - `ended` (all steps logged, no `.done`) - `done`. A process killed by the OOM killer leaves no traceback: it shows `running` until it is `stalled`.

| code | level | meaning |
|---|---|---|
| `failed`, `stalled` | alert | crashed (last line of the traceback) / no output |
| `nonfinite`, `skipped` | alert / warn | NaN or inf in a logged row / steps the trainer skipped (3 in a row = alert) |
| `spike` | warn | a value 8 robust sigmas above the recent median (`watch` tests every row since its last look) |
| `shift` | warn after 50 % of the steps, else info | the last 10 rows of `total` jumped by more than 30 % and 5 standard errors over the 30 before, and stay there |
| `rising` | same | window mean of `total` or a `group.*` rose by more than 3 standard errors |
| `grad_norm`, `gpu`, `slow` | warn | grad norm > 10 x its median / the trainer's peak GPU allocation > 11 GB / s per step 1.5 x the typical one of the current attempt (rows with a time `t` only, running runs only) |
| `overfit` | warn | `val.total` rose by 5 % while the training loss fell |
| `vs_ref` | warn if 10 % **and** 3 standard errors worse over the last 50 rows after 20 % of the steps, else info | the loss against a reference run at the same step |
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

## Not in this suite (yet)
Starting, queueing and resuming runs (`run.py` does the process side: record, GPU slots, retry; a queue is planned), a run fingerprint that refuses comparisons across different code / ground truth / caches, and a ledger row for a killed run. More in `EXTENSIONS.md`.
