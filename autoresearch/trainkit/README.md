# trainkit - the project-independent training engine

What a training needs and that does not depend on the network, the loss or the data: the fit loop (accumulation, EMA, schedule, clipping, AMP), atomic checkpoints with resume and warm start, the log, the probes and the stop protocol of the research worker. Plain torch. The BEV problem and the toy train on `fit`; the lane problem has its own loop (`problems/lane_center/training/trainer.py`) and takes `control` and `probe` from here. A run imports the trainkit of its code snapshot.

## Minimal example (`selftest.py` runs it)
```python
import torch
from torch import nn
from trainkit.config import Config            # a dataclass; a project adds its fields (and overrides RESUME_FREE, RESUME_SWITCHES, RESUME_ENV)
from trainkit.loop import fit
from trainkit.util import prepare_run_dir

model, loss = nn.Linear(3, 1), nn.Module()    # nn.Modules; the loss holds the learnable loss weights (an empty one if none): they train and are saved
batches = [{"x": torch.randn(4, 3), "y": torch.zeros(4, 1)} for _ in range(8)]     # any re-iterable; `data.EpochLoader`: a resume continues its pass

def step_fn(batch, info):                     # one micro-batch: forward AND backward
    l = ((model(batch["x"]) - batch["y"]) ** 2).mean()
    info.backward(l)                          # divides by accum, applies the loss scale under fp16
    return {"total": float(l.detach())}, bool(torch.isfinite(l)), None      # terms (at least "total"), finite?, extra fields of the log row or None

cfg = Config(name="toy", out_root="runs", steps=12, accum=2, lr=0.05, warmup=2, device="cpu", save_every=6)
model, loss, ema, last = fit(cfg, model, loss, batches, step_fn, prepare_run_dir(cfg), "cpu")
```
`fit` also takes `evaluate(ema) -> {name: float}` (under the average: `with ema.applied(model): ...`), `hooks` (`loop.Hooks`: probes around a step), `frozen` (state keys the EMA leaves alone), `external` (state-key prefixes that are not saved) and `stop_check`. The autocast of the forward pass (`amp.amp_context`) and the data are yours.

## What the loop guarantees
1. **Stop**: a `CONTROL` file `{"cmd": "stop"}` (the worker writes it on `research stop`) ends the run at a step boundary: checkpoint, `STOPPED`, exit 75 (never 0). The next start moves both files aside.
2. **Bad steps**: a non-finite loss or gradient skips the step, `max_bad_steps` in a row abort; non-finite batch-norm statistics abort at once; an fp16 overflow halves the scale and counts only below scale 1.
3. **Saved**: the EMA after every step that was made; a row of `log.jsonl` every `log_every`; a checkpoint every `save_every` (atomic; the newest `keep` and the `keep_steps` stay), then the evaluation.
4. **Start**: resumes from the newest readable checkpoint, only with the same config (`RESUME_FREE` aside; `RESUME_ENV`=1 overrides); log rows above it move to `log_superseded.jsonl`, a cut evaluation is made up. A run without one warm-starts from `init_from`.
5. With `EpochLoader` a resumed run is the uninterrupted one, bit for bit on the CPU. Not restored: GPU float atomics, randomness inside worker processes. No roll-back on divergence.

## Files
| file | content |
|---|---|
| `loop.py` | `fit`, `Hooks`, `StepInfo` |
| `config.py` | `Config`, `read_config` |
| `optim.py`, `schedule.py`, `ema.py`, `amp.py` | parameter groups, clipping and the step; the learning rate; the average; mixed precision |
| `checkpoint.py` | save, load, the rules of a resume, warm start, pruning |
| `data.py`, `runlog.py`, `util.py` | `EpochLoader`; the log rules; seed, device, run folder |
| `control.py` | the stop protocol, standard library only |
| `probe.py` | gradient and update size per module, data wait, profile of a few steps, incident dumps |
| `selftest.py` | known answers of all of it: `python -m trainkit.selftest` from `autoresearch/` (needs torch, seconds) |
