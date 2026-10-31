# trainsuite - see, stop and judge a training run from the files it writes

A training is a process that writes files. This suite reads them and tells you where the run is, whether it looks healthy and whether a kill criterion of its card is hit; it can ask the trainer to stop. It imports no project code, needs only the standard library (curves also matplotlib) and knows no network, loss or dataset. Run it from `autoresearch/`; `--root` is the folder of the run directories (default `runs`).

## Try it
```
python -m trainsuite selftest                                  # synthetic runs with known answers, a few seconds
python -m trainsuite --root ../bev_fusion/runs status          # all runs active in the last 24 h (--all: every run)
python -m trainsuite --root ../bev_fusion/runs health t8_bevfeat_s0 --vs t8_ctl_s0 --kill "group.lines@1000 > 1.48"
python -m trainsuite --root ../bev_fusion/runs curves t8_bevfeat_s0 t8_ctl_s0      # png: total, every group.*, grad_norm
python -m trainsuite --root ../bev_fusion/runs watch t8_bevfeat_s0 t8_ctl_s0 --kill "group.lines@1000 > 1.48" --enforce
python -m trainsuite --root ../bev_fusion/runs stop t8_bevfeat_s0
```
`status` with one run adds the findings, the kill rules and the last eval. `health` exits 1 on an alert or a kill hit. `watch` prints **one line per change** (state, new finding, kill rule, new eval) and ends when every run is finished: it is made for an agent's Monitor, which then sleeps until something happens. Without `--enforce` it only reports; with it a kill hit asks the trainer to stop.

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

**Stop.** The trainer looks at `CONTROL` at a step boundary. If the file is **newer than the start of the process** and says `{"cmd": "stop"}`, it saves a checkpoint, writes `STOPPED` (`{"step", "reason"}`) and exits with code **75**, never 0, so no `.done` appears and nobody evaluates a half-trained model. An old `CONTROL` never stops a resume. A trainer needs about 12 lines for this, inlined (a run trains from a pinned code snapshot that does not contain this package): see `control.stop_requested` and the use in `bev_fusion/training/train.py`.

## States and what is checked
`new` (no output yet) - `running` - `stalled` (no output for 15 min, `TS_STALE_S` seconds) - `failed` (traceback at the end of `train.log`; seen at once, not after 15 min) - `stopped` - `ended` (all steps logged, no `.done`) - `done`.

| code | level | meaning |
|---|---|---|
| `failed`, `stalled` | alert | crashed (last line of the traceback) / no output |
| `nonfinite`, `skipped` | alert / warn | NaN or inf in a logged row / steps the trainer skipped (3 in a row = alert) |
| `spike` | warn | one value 8 robust sigmas above the recent median |
| `rising` | warn after 50 % of the steps, else info | window mean of `total` or a `group.*` rose by more than 3 standard errors |
| `grad_norm`, `gpu`, `slow` | warn | grad norm > 10 x its median / GPU > 11 GB / s per step 1.5 x the typical one (a shared box) |
| `overfit` | warn | `val.total` rose by 5 % while the training loss fell |
| `vs_ref` | warn if 10 % worse after 20 % of the steps, else info | the loss against a reference run at the same step (`--vs`) |

**Healthy is not good.** These checks say the training is sound; whether the network is good is the evaluation (`evalsuite`). And the loss of a multi-task run is not a number that must fall: the bev_fusion `total` rises from 8.5 to 12.5 between steps 500 and 1300 in every run (probably the warm-up of the time weights and the learned task weights; not analysed), reference runs included. The reliable early signal is the comparison with a reference run (`--vs`, `curves` with two runs) and a kill rule.

## Kill rules
The kill criterion of an experiment card as a line a program can check, from `--kill "..."` (repeatable) and `<run>/kill.txt`:
```
group.lines@1000 > 1.48              # mean of the last 10 logged values up to step 1000
total@500 > 30 window 5              # the last 5 values
val.total@2000 >= 13.5               # the eval row at step 2000 (or the first after it)
```
A rule is `pending` until the run has reached its step, then `ok` or `hit`.

## Not in this suite (yet)
Starting, queueing and resuming runs (`run.py` does the process side: record, GPU slots, retry; a queue is planned), a run fingerprint that refuses comparisons across different code / ground truth / caches, and a ledger row for a killed run. More in `EXTENSIONS.md`.
