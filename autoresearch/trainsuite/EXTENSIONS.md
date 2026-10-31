# trainsuite - extensions

`README.md` is complete for a simple project. Read only the chapter you need.

| chapter | you need it when |
|---|---|
| 1 Another layout of the run directory | the trainer names its files differently |
| 2 Thresholds | a check is too loud or too quiet for your runs |
| 3 Kill rules and `watch` in an agent loop | you want a run cut by its own card |
| 4 Reading curves: the bev_fusion case | the loss does not simply fall |
| 5 What comes next | you wonder where start / queue / fingerprint are |

## 1. Another layout of the run directory
A `trainsuite.json` in the run directory names what differs; every key is optional:
```
{"log": "metrics.jsonl", "text_log": "run.log", "loss": "loss", "steps": 3000, "progress": "it "}
```
`log` is the metrics stream, `text_log` the text output (crash detection, skipped steps), `loss` the optimised value, `steps` the total (else `config.json["steps"]`), `progress` how a progress line of the text log starts (default `"step "`, the bev_fusion trainer): a traceback counts as a crash unless such a line follows it, so a trainer that logs a handled exception and goes on is not reported failed when its own prefix is named here. Rows must be flat or nested one level (`{"val": {...}}` becomes `val.total`); strings, lists and booleans are ignored. A row without `step` is skipped. Without `t` in the rows the speed is the trainer's own `s_per_step` (an average since the process start, so a slowdown shows late).

## 2. Thresholds
Module constants in `health.py`: `SPIKE_K` 8, `RISE_K` 3, `SHIFT_K` 5 / `SHIFT_REL` 0.3, `RAMP_FRAC` 0.5, `SLOW_FACTOR` 1.5, `GPU_LIMIT_GB` 11, `REF_TOL` 0.10 / `REF_K` 3 / `REF_ROWS` 50, `MIN_ROWS` 40; `TS_STALE_S` (environment) 900 s for `stalled`, raised automatically to 10 row intervals when the rows are far apart. A validation pass or a checkpoint of a large model can be quiet for longer: raise `TS_STALE_S` for that project. The age of the last output is the file mtime against the clock of the machine running the suite: after a host sleep the WSL clock can be off by minutes and show a false `stalled` (or hide a real one); the stop protocol does not depend on clocks. Every test compares window means and uses the spread of the window, because the loss of one batch is noisy (bev_fusion: sd about 2 on a mean of 11).

## 3. Kill rules and `watch` in an agent loop
Write the kill criterion of the card as a rule in `<run>/kill.txt` before the run starts (the run directory may not exist yet: `--kill` on the command line works too). Start `watch <runs> --enforce` as a Monitor: it prints a line only on a change, so the agent is woken by a kill hit, a crash, a stall or a finished eval, not by the clock. An enforced kill writes `CONTROL` (only when the run's code snapshot has the stop check), the trainer stops at the next step with a checkpoint and exit code 75; the run is then `stopped`, never `done`. A lane script that starts the next job after a run without `.done` goes on with the next run: a stop frees the GPU for it, it does not pause the lane. The ledger note and the provisional decision (CUT) are written by the agent, not by the suite: the suite does not know the ledger. Events are also appended to `<run>/trainsuite_events.jsonl`.
Runs from a code snapshot older than this contract (no stop check in the trainer, i.e. everything pinned before commit 5d3bdae of 2026-10-31) ignore `CONTROL`: the suite reports them, refuses to write the request and says so. Rows of such a trainer have no time `t`: the speed is the trainer's own average (startup and evals included) and there is no slowdown check.

## 4. Reading curves: the bev_fusion case
`curves A B` draws `total`, every `group.*` and `grad_norm` of both runs (moving mean of 10 rows, raw faint). In the bev_fusion runs `total` is the Kendall-weighted sum with warm-ups: it falls to about 8.5 at step 500, rises to about 12.5 at step 1300 and stays there, in the reference runs too. That is not a failure (cause not analysed). The question to ask is whether a run follows its **reference** (`health --vs`), per group, and the reference must have the same ground truth and code: `t8_bevfeat_s0` is to be compared with `t8_ctl_s0`, not with `t2_ref_s0` (another ground-truth signature). Training curves are not results: the numbers come from the evaluation.

## 5. What comes next
Planned, not built (see the proposal in the answer of 2026-10-31 and the DECISIONS entry): `start` / `queue` / `resume` on top of `run.py` (card, GPU slots, retry on CUDA error, one queue file consumed by a persistent worker instead of lane shell scripts), a `run.json` fingerprint (code commit, config hash, ground-truth signature, cache id, seed) with a refusal to compare runs whose fingerprints differ, the progress heartbeat of `run.py` taken from the metrics stream instead of the launcher, a quick validation at checkpoints through `evalsuite`, and the generic loop (`fit`) shared by the projects.
