# trainkit - the project-independent training engine

The part of a training that does not depend on the network, the loss or the data: the fit loop with accumulation, EMA, the learning-rate schedule (an own rate per module), gradient clipping and the optimiser step with the non-finite rules, AMP with the GradScaler, atomic checkpoints with resume and warm start, the log files, and the stop protocol with the trainsuite. Plain torch, no project code. The BEV system (`bev_fusion/training/train.py`) is the first user (D48); the lane project has a trainer of its own and does not use it. `bev_fusion/tests/test_trainkit_equivalence.py` runs the trainer from before the move and the live one on the same toy and requires the same rows, weights, files and messages; it also kills eight mutants of this package.

## The contract: model, loss, data, config - and two functions of yours
```python
from trainkit.config import Config            # 4. CONFIG: a dataclass; a project extends it with its own fields (and overrides RESUME_FREE, RESUME_SWITCHES, RESUME_ENV)
from trainkit.loop import fit
from trainkit.util import prepare_run_dir

model, loss = Net(), Loss()                   # 1. MODEL and 2. LOSS: nn.Modules. The loss module holds learnable loss weights (an empty nn.Module() if there are none);
                                              #    its parameters train, are saved and are warm-started with the model
batches = [...]                               # 3. DATA: any re-iterable of batches (a DataLoader, a list); the loop starts over when it ends. With `iter_from(epoch, pos)`
                                              #    (`trainkit.data.EpochLoader`: a shuffling DataLoader whose pass order is a function of (seed, epoch)) a resume continues the pass it was in

def step_fn(batch, info):                     # one micro-batch: forward AND backward -> (terms, finite?, extra)
    l = loss(model.fc(batch["x"]), batch["y"])
    info.backward(l)                          # divides by accum, applies the loss scale under fp16 (a step that needs more may call .backward itself)
    return {"total": float(l.detach())}, bool(torch.isfinite(l)), None

cfg = Config(name="toy", out_root="runs", steps=12, accum=2, lr=0.05, warmup=2, device="cpu", save_every=6)
out_dir = prepare_run_dir(cfg)                # runs/toy, created; the CONTROL / STOPPED files of an earlier stop moved aside
model, loss, ema, last = fit(cfg, model, loss, batches, step_fn, out_dir, "cpu")     # `selftest.py` runs this example
```
`terms` = `{name: float}` with at least `total` (averaged over the micro-batches into the row; every other name is a column of `log.jsonl`); `finite? = False` skips the step; `extra` = None or a dict that goes into the row and forces it to be written (the BEV gradient-conflict diagnostic). The autocast context of the forward pass (`trainkit.amp.amp_context`) and the data pipeline are yours: they live in `step_fn` and in the batches.

Further arguments of `fit`: `evaluate(ema) -> {name: float}` (a validation under the averaged weights: `with ema.applied(model): ...`; it is logged as `{"step", "val": {...}}`), `hooks`, `frozen` (state keys whose average stays the value itself: a freeze plan), `external` (state-key prefixes that are not in the checkpoints because they come from elsewhere: a frozen pre-trained part), `stop_check(out_dir)` (default `control.stop_requested`).

## Hooks (`loop.Hooks`, every method optional)
| method | when | use |
|---|---|---|
| `begin_step(n)` | before the micro-batches of step n (1-based) | return a probe with `grads()` (between the backward and the clip, unscaled gradients) and `finish()` (after the optimiser step -> a dict added to the row), or None |
| `data_wait()` | around every batch fetch | a context manager that times the wait for the data |
| `batch(item)` | after every fetch | remember the samples (for an incident dump) |
| `skipped(step, what, gn, terms, step_frac)` | a step skipped as non-finite (`what` = loss / gradient) | dump the incident |
| `end_step()` | at the end of every step | return a dict added to the row (timings), tick a profiler |
| `close()` | after the last step, before the final checkpoint | write the profile |
The BEV probes (`training/probe.py`) are one implementation: `ProbeHooks` in `train.py`.

## What the loop guarantees, in this order, for every optimiser step
1. The stop request comes first: a `CONTROL` file with `{"cmd": "stop"}` ends the run with a checkpoint, a `STOPPED` file and exit code 75 (never 0, so no `.done` and nobody evaluates half a model); the next start moves both files aside (`control.py` is the one implementation, the trainsuite re-exports it).
2. The learning rate of every group follows the schedule (warm-up from lr / 3, cosine to `lr_min_frac`; an `lr_groups` prefix has its own rate and warm-up).
3. `accum` micro-batches go through `step_fn`; a non-finite loss skips the step. Then the gradients are clipped (one global clip, or per `grad_clip_split` group) and the optimiser steps; a non-finite gradient skips the step (fp16: an overflow halves the scale and counts only below scale 1); `max_bad_steps` skipped steps in a row abort the run. The lines `step N: non-finite ..., skipped (k in a row)` are what the trainsuite's health check counts.
4. The EMA is updated after a step that was made; a row goes to `log.jsonl` at step 1, every `log_every` steps and when `extra` is given; a checkpoint every `save_every` steps (atomic: write, fsync, rename; the newest `keep` stay), then the evaluation every `eval_every` steps, so a crash in the evaluation costs nothing.
5. At the start: resume from the newest readable checkpoint of the run folder (a damaged one is kept as `.damaged`; the configuration must be the checkpoint's except `RESUME_FREE`, and the switches in `RESUME_SWITCHES` must not be turned on in the middle of a run; the environment variable `RESUME_ENV` = 1 overrides), or a warm start from `init_from`; log rows above the resumed step move to `log_superseded.jsonl`; an evaluation that a crash cut is made up. A finished run is written slim (`slim_at_end`: weights only, evaluates and warm-starts but cannot resume).
A resume is the uninterrupted run (bit for bit on the CPU) when the data has `iter_from`: the checkpoint holds the random streams (torch, numpy and, when CUDA is up, every CUDA device; older checkpoints lack the CUDA key and load as before) and, in `extra["data"]`, the pass and the number of micro-batches taken from it, with the number of samples and the batch size it is valid for; the loop opens that pass at that place (a position that does not fit these data, and a checkpoint without one, are printed and the pass starts again; so is a position that plain data cannot use). On the GPU the float atomics of some layers are not deterministic: bit for bit is what the CPU test shows. Data without `iter_from` (a list, a plain DataLoader) starts a pass afresh on a resume, as before. Randomness inside a dataset's `__getitem__` in a worker process is not restored.
Not in the loop: a roll-back to an earlier checkpoint on divergence (a run that skips `max_bad_steps` steps stops; the trainer resumes by starting the same command again).

## Files
| file | content |
|---|---|
| `loop.py` | `fit`, `Hooks`, `StepInfo` |
| `config.py` | `Config`: the fields the engine reads (steps, lr, warm-up, accum, clip, EMA, AMP, cadence of logging and saving, seed, warm start, device) and the rules of a resume |
| `ema.py` | `EMA` (weights and buffers, decay ramp, frozen keys stay the value itself), `ema.applied(model)` |
| `schedule.py` | `lr_factor`, `scheduled_lr` |
| `optim.py` | `param_groups` (decay / no decay, one pair per `lr_groups` prefix), `clip_groups` / `clip_gradients`, `optimizer_step`, `bad_steps_after`, `poisoned_buffers` |
| `amp.py` | `amp_context`, `make_scaler` (GradScaler for fp16), `scaler_extra` |
| `checkpoint.py` | `save_checkpoint` / `load_checkpoint` (state without the `external` prefixes, optimiser, EMA, RNG: `capture_rng` / `restore_rng`, torch + numpy + CUDA), `resume`, `check_resume`, `refuse_slim_resume`, `warm_start`, `prune_checkpoints` |
| `data.py` | `EpochLoader` (a DataLoader with `iter_from(epoch, pos)`), `EpochSampler` (pass order = numpy permutation seeded by (seed, epoch)) |
| `runlog.py` | `supersede_log`, `val_logged` (the `log.jsonl` rules) |
| `control.py` | the stop protocol, standard library only (`trainsuite/control.py` re-exports it) |
| `util.py` | `seed_all`, `resolve_device`, `prepare_run_dir`, `detach_tree` |
| `selftest.py` | known answers for the schedule, the EMA, the clip and step rules, the stop protocol, the CUDA random state (stubbed), `EpochLoader`, a stop + resume that equals the uninterrupted run, and the example above; `python -m trainkit.selftest` from `autoresearch/` (needs torch) |

## A run trains from a code snapshot
`run.py` snapshots `training/` (and, through `code_dirs`, `autoresearch/trainkit`) into the run's tag, and the BEV launcher `bev_fusion/tools/run_ar.sh` (D73) starts `train.py` from a `git archive` of `training/` with a `git archive` of `autoresearch/trainkit` unpacked next to it. `train.py` finds this package by walking up from its own file to the folder that holds `autoresearch/trainkit` (the live copy), unless a `trainkit/` folder lies next to it (the pinned copy, which the launcher puts there; by hand: `git archive <commit> autoresearch/trainkit | tar -x --strip-components=1 -C <run>/code`). Started as a script, `train.py` prints one line first that says which trainkit runs (the pinned copy, or the live one when no copy lies next to it: an old snapshot or another launcher) and exits when a copy lies next to it but another one was imported (`train.check_engine`). Old runs have an old `train.py` of their own and never import this package.
