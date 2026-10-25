# Review of the experiment system (2026-10-25, after loop 18)

Question: where is the system (literature -> plan -> run -> evaluate -> decide -> report -> handover) flawed, what should the reusable bundle (data, training, losses, evaluation) look like, and how do we get faster? Evidence: ledger of loop 18 (34 rows), the code base, this weekend's clean-up, a loader benchmark.

## 1. What works (keep it)

- **Pre-registered decisions.** 20 of 34 loop-18 runs carry a numeric prediction and a decision rule written before the run, and `decide.py` turns them into KEEP / CUT / TIE / INCONCLUSIVE. This caught real things. For example, the equal-budget control showed that most of the Q gain of the recall fine-tune was extra training, not the recall weight.
- **TEST discipline as code.** `run.py --split test` refuses unless the run id is logged. TEST was used once per design.
- **Checks of important code.** 18 builder spec tests, metric-response tests, and "fast path == reference graphs" (873 / 873 bit-identical) before a speed-up is accepted.
- **Successive halving** (cheap rung on VAL250, full VAL for survivors), one shared metric library (`card` / `card_any` called in about 60 places), paired scene bootstrap.
- **Loop 18 throughput:** about 30 decided runs in about 6 GPU hours, and a real result (fork pruning closes the J gap, TEST confirms).

## 2. Where the system is flawed (ordered by cost)

| # | flaw | evidence | cost |
|---|---|---|---|
| F1 | **No version control.** Code is shared, mutable and unversioned; `code_hash` in a run record says *that* code changed, not *what*, and cannot restore it | a rollback deleted the loop-17 code; another chat's edit of `run.py` truncated `LEDGER.csv` mid-run; an edit of `p16_loss.py` broke a running TEST job; evaluation was reorganised twice within 10 minutes by two chats; a deck disappeared from `reports/weekly/` | hours per incident, and lost provenance |
| F2 | **Running jobs import live code.** A job reads `training/` and `evaluation/` while other chats edit them | pitfalls 1, 4, 5 of loop 18 | crashes, silent changes inside a run |
| F3 | **The general trainer is not used.** `training/trainer.py` (resume, NaN skip, roll-back, done flag) is used by 0 of 23 training scripts; only 3 of them can resume | E209 lost about 2 h at step 3500 / 4800; each script repeats its own loop, AMP, logging | lost GPU hours, divergent conventions |
| F4 | **Evaluation is re-written per loop.** 27 evaluation entry points (`p16_final_eval`, `p17_eval`, `p17_route`, `p18_fork_prune_eval`, `_pipeline18b.sh`, ...); the standard settings live in a memory note ("use `fork_prune=0.3`, `fast=2` from now on"), not in code | every loop writes a driver; a forgotten flag gives a different number | time and comparability |
| F5 | **Experiments are shell wrappers in `temp/`.** 12 of 34 ledger rows are `bash temp/launchers/...sh`; one row hides several steps; waiting loops (`until [ -f done ]; sleep 30`) are counted as run time (E232.eval: 89 min, of which the evaluation took about 2 min); when `temp/` is cleaned the record of what ran is gone | `LEDGER.csv`, `_eval_j3.sh` | provenance, wrong durations |
| F6 | **Too much state to read.** A new chat or wake-up reads CLAUDE.md, HANDOVER, PROGRAM, PLAN, LOOP18_STATE (30 KB), TRIAGE_STATE (78 KB), STATE (81 KB), the ledger and memory notes, some of which are in another project's memory folder: about 220 KB, about 55k tokens | file sizes | tokens per wake-up, contradictions between files |
| F7 | **Repeated negatives.** Joint end-to-end fine-tuning (route loss into the builder) was tested three times (E202, E214, E232), negative each time, at about 1 to 1.5 GPU-hours per attempt | ledger | GPU hours |
| F8 | **The goal metric does not respond.** "Planner-ready" has no planner-level metric that moves: path error sits at a floor of 0.27 m whatever the graph; S was gamed by sparsity until Q and J were added | E222.b, loop 16 / 18 | optimising the wrong thing |
| F9 | **Report numbers are re-typed.** The weekly reports were built from digests typed by agents; the fresh review found 81 errors; `check_numbers.py` is advisory | review_week1 / 2 | review rounds |
| F10 | **Several chats on one folder without ownership.** 4 to 6 chats at once; nothing says who owns which folder | this weekend | collisions (F1, F2) |

## 3. The bundle: what to standardise (small, no framework)

Keep it simple: no package rewrite, no renaming of the `p<loop>_` modules. Four thin entry points over the existing code, each replacing a family of per-loop scripts:

1. **`models/registry.json`**: name -> checkpoint, design / config, assembly settings, for example `champion: e123_D7_r3_f1, M 288, K0 144, fork_prune 0.3, fast 2`; `fast: j1`; `recall: rc1`; `control: e94_a + e120`. The standard settings live here, not in memory notes (fixes F4).
2. **`evaluation/evaluate.py <name> --split val|test --series clean,dark,... --map exact|none|wrong|err2`**: one entry point that writes `card.json` (S, Q, J, F1, lanes found, route success, ms) next to the run. It is built from the existing functions (`p16_infer.assemble`, `p17_series`, `sparse_metrics.card`, `bootstrap.paired`). Raw network outputs are cached per checkpoint and split (as `p18_asm_check` did ad hoc), so new assembly or pruning settings re-score in seconds. The per-loop drivers stay for history.
3. **`training/train.py`** = `p16_train.py` on `trainer.fit` (resume, roll-back, done flag), ending with a call of `evaluate.py`. Migrate the two workhorses first (`p16_train.py`: 10 of 34 runs; `p18_joint.py` / `p17_mapft.py`). Others only when touched.
4. **`losses/`**: one tested module per head (builder, route, dense). Add three cheap tests per loss: zero (or minimal) at the ground truth, gradient reaches every head, scale stays within a stated range. Today losses are defined in 8 files.

Decisions and records stay as they are (`run.py`, `ledger.py`, `decide.py`, `check_numbers.py`), with three small additions to `run.py` (after loop 18):
- `--after E232.a0,E232.c0`: wait for done flags inside the launcher, so the waiting is not counted and each step is its own ledger row (fixes F5).
- **Snapshot**: copy `training/`, `evaluation/`, `tools/` (about 1.5 MB) and the wrapper script into `experiment_execution/runs/<id>/code/` and run with that `PYTHONPATH`. A running job can then not be broken by an edit, and the record survives a `temp/` clean-up (fixes F2 and F5). `git` makes this a commit hash instead of a copy.
- An `--est-min` field so the ledger can compare planned against actual cost.

## 4. Getting faster

Measured: training of the builder consumes about 25 frames/s (8 u: 9,600 iterations x 8 frames in 3,001 s). The data loader alone delivers about 100 frames/s cold (57 with the previous frame). So **training is GPU-bound, not data-bound**. Evaluation steps take 2-10 min; a loop decided about 5 runs per GPU hour.

| lever | expected gain | effort | how to check |
|---|---|---|---|
| **Stop losing hours**: resume in every training (F3), snapshot runs (F2), no early shutdown (loop 18 lost about 4 h) | the largest; 2-5 h per loop | small | ledger idle gaps |
| **Faster training step**: first measure GPU utilisation during `p16_train` (`nvidia-smi dmon`); the builder has many small ops (tree, sampling, children), so it is likely launch-bound. Then `torch.compile` / CUDA graphs for the training step (the inference path already uses CUDA graphs) and `channels_last` | 1.5-2x if launch-bound | half a day | it/s, same loss curve on one seed |
| **Smaller input**: the builder reads 42 input channels at 200 x 200 (fp16); PCA bases of the features exist (`~/exp_gcar_b123/pca_gcar.npz`, `pca_gcar96.npz`); test about 16 channels on the builder at rung 0 | up to 2x memory and data, unknown quality | one rung | Q at VAL250 within the band |
| **Early kill by learning curve**: the control curve is flat from 2 u; cut a run at 25-50 % of its budget when its VAL250 Q is below the control's curve at the same step by more than the band | about 30 % fewer GPU hours in a search | small (rule in `decide.py`) | count of runs that would have been wrongly cut, replayed on loop 16 / 18 |
| **Prediction cache** in `evaluate.py` (section 3) | assembly / pruning / threshold sweeps in seconds instead of minutes | part of item 2 | |
| **One GPU job + CPU work in parallel** instead of three GPU jobs (three stalled each other); evaluation and analysis on the CPU while a training runs | fewer stalls | rule | |
| **Closed questions**: before a run, search the ledger for the same hypothesis; two negatives close a question unless the card names a new mechanism (F7) | 1-1.5 GPU h per avoided repeat | rule in the card template | |
| **Less state per wake-up**: HANDOVER (narrative, <= 3 KB), PLAN (tree + status), LEDGER (runs) are the only state; loop state files become the PLAN status column plus ledger notes; STATE and TRIAGE_STATE go to `history/`; memory notes only for facts outside the project (F6) | about 40k tokens per wake-up, fewer contradictions | small | |

## 5. Science side (what the system should push next)

- **A goal metric that responds (F8).** Before optimising for "planner-ready", build the planner-level metric and run the same metric-response test as for S: controlled graph errors (missing branch, false fork, 1 m shift) must move it. Path error at a floor of 0.27 m whatever the graph says the consumer, not the graph, limits it. Until such a metric exists, Q, J, lanes found and route success are the targets.
- **Far range** (58 % of the recall gap beyond 40 m) is the open quality gap. Next candidates are a range-aware input (higher-resolution features far out) and temporal accumulation of far evidence, not loss weights (rc2 / rc4 showed the weight is spent).
- **Reports from data (F9).** The tables of a report are generated from `card.json` and the ledger; hand-written text only around them; `check_numbers.py` becomes a gate (exit 1) for the tables.

## 6. Order of work (after loop 18)

1. `git init` with a `.gitignore` for `runs/`, `data/`, `_to_delete/`, caches (half an hour; decision waiting for the user since 2026-10-24). One branch or worktree per chat, merge at the end of its task. This alone removes most of F1, F2 and F10.
2. `run.py`: snapshot + `--after` + copy of the wrapper script (one hour; then `selftest.py`).
3. `models/registry.json` + `evaluation/evaluate.py` with the prediction cache, validated by reproducing the loop-18 VAL and TEST cards of `f1 + prune 0.3`, `j1`, `rc1` and the control (one day).
4. `p16_train.py` and `p18_joint.py` on `trainer.fit` (half a day; one seed must reproduce its loss curve).
5. Measure GPU utilisation; if launch-bound, compile the training step (half a day).
6. State diet (F6) and the closed-question rule (F7) in `templates/` and CLAUDE.md (one hour).
7. Planner-level metric with a response test (a loop of its own).

Folder ownership while several chats run: one line per chat in `HANDOVER.md` ("chat X owns `reports/` until ..."), or simply one git worktree per chat.
