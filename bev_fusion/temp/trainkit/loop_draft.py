"""The fit loop: the part of a training that does not depend on the network, the loss or the data.

    fit(cfg, model, loss, batches, step_fn, out_dir, device, evaluate=..., hooks=...)

The four parts of the contract (README.md): the MODEL and the LOSS are `torch.nn.Module`s (the loss module holds learnable loss weights; an empty `nn.Module()` if there are none),
the DATA is any re-iterable of batches (a DataLoader, a list), the CONFIG is a `trainkit.config.Config`. The project supplies how one micro-batch is computed (`step_fn`), how a
validation is made (`evaluate`) and, if it wants, probes (`Hooks`). `fit` guarantees, in this order, for every optimiser step:
  - the stop request (CONTROL file) is looked at first: checkpoint, STOPPED file, exit code 75
  - the learning rate of every group follows the schedule (warm-up, cosine, the own rate of an lr_groups prefix)
  - `accum` micro-batches are computed by `step_fn` (forward and backward inside it); a micro-batch with a non-finite loss skips the step
  - the gradients are clipped (globally or per `grad_clip_split` group) and the optimiser steps; a non-finite gradient skips the step (fp16: an overflow halves the scale);
    `max_bad_steps` skipped steps in a row abort the run
  - the EMA is updated after a step that was made; a row goes to log.jsonl every `log_every` steps; a checkpoint every `save_every` (atomic, the newest `keep` stay);
    an evaluation every `eval_every` steps (after the checkpoint, so a crash in it costs nothing)
and at the start: a resume from the newest readable checkpoint of out_dir (the same configuration is required), or a warm start from `init_from`; log rows above the resumed step
move to log_superseded.jsonl.
"""
import contextlib
import dataclasses
import json
import os
import time

import numpy as np
import torch

from . import checkpoint, control, optim, runlog, schedule
from .amp import make_scaler, scaler_extra
from .config import RESUME_FREE
from .ema import EMA


@dataclasses.dataclass
class StepInfo:
    """what `step_fn` is told about the micro-batch it computes"""
    step: int                    # the optimiser step (0-based)
    frac: float                  # step / steps: the position in the schedule
    micro: int                   # the micro-batch inside the step (0 .. accum - 1)
    accum: int                   # micro-batches per optimiser step
    scaler: object = None        # the GradScaler of amp "fp16", else None

    def backward(self, loss, retain_graph=False):
        """the backward pass of one micro-batch: the loss divided by accum (the step is the mean of its micro-batches) and, under fp16, scaled"""
        loss = loss / self.accum
        (loss if self.scaler is None else self.scaler.scale(loss)).backward(retain_graph=retain_graph)


class Hooks:
    """what a project can watch or add to the loop; every method is optional (these are the defaults). The BEV probes (grad / update norms per module, step timer, profiler,
    incident dumps) are one implementation (training/train.py ProbeHooks)."""

    def begin_step(self, n):
        """before the micro-batches of step number n (1-based, the number its row gets) -> None, or a probe with `grads()` (called between the backward and the clip, the
        gradients unscaled) and `finish()` (after the optimiser step -> a dict that is added to the row)"""
        return None

    def data_wait(self):
        """a context manager around the fetch of every batch (to time the wait for the data)"""
        return contextlib.nullcontext()

    def batch(self, item):
        """every micro-batch right after it was fetched"""

    def skipped(self, step, what, gn, terms, step_frac):
        """a step that was skipped as non-finite; what = "loss" or "gradient", gn = the gradient norm (None for a loss), terms = the loss terms so far"""

    def end_step(self):
        """at the end of every step -> a dict that is added to its row (e.g. timings)"""
        return {}

    def close(self):
        """after the last step, before the final checkpoint"""


def _skip_report(step, status, gn, bad, was_bad, n_overflow, scaler):
    """the lines the trainsuite's health check counts (`non-finite ... skipped (k in a row)`); -> the number of fp16 overflows so far"""
    if status == "overflow" and bad == was_bad:             # fp16: expected now and then, the scaler halved its scale
        n_overflow += 1
        print(f"step {step}: fp16 gradient overflow, skipped, loss scale -> {scaler.get_scale():g} ({n_overflow} since this start)", flush=True)
    elif status == "overflow":                              # the scale is below 1: a diverged run, in the words the health check counts
        n_overflow += 1
        print(f"step {step}: non-finite gradient (fp16 overflow, loss scale {scaler.get_scale():g}), skipped ({bad} in a row)", flush=True)
    elif status != "ok":
        print(f"step {step}: non-finite {'loss' if gn is None else 'gradient'}, skipped ({bad} in a row)", flush=True)
    return n_overflow


def fit(cfg, model, loss, batches, step_fn, out_dir, device, evaluate=None, hooks=None, frozen=(), external=(), stop_check=None,
        resume_free=RESUME_FREE, switches=(), resume_env="TRAINKIT_RESUME_ANY"):
    """train `model` (and the parameters of `loss`) for cfg.steps optimiser steps -> (model, loss, ema, the last logged row).
    step_fn(batch, info: StepInfo) -> (terms, ok, extra): forward and backward of one micro-batch (info.backward); terms = {name: float} with at least "total" (averaged over the
    micro-batches into the row), ok = False when the loss is not finite (the step is skipped), extra = None or a dict that is added to the row and forces it to be written.
    evaluate(ema) -> {name: float}: a validation under the averaged weights (`with ema.applied(model):`), logged as {"step", "val"}; None = no evaluation.
    frozen: state keys whose average stays the value itself. external: state-key prefixes that are not in the checkpoints. stop_check(out_dir) -> reason or None (default:
    control.stop_requested). resume_free / switches / resume_env: checkpoint.check_resume."""
    hooks = hooks or Hooks()
    stop_check = stop_check or control.stop_requested
    clip_split = optim.clip_groups(model, cfg.grad_clip_split)
    opt = torch.optim.AdamW(optim.param_groups(model, loss, cfg.lr, cfg.weight_decay, cfg.lr_groups), lr=cfg.lr, betas=(0.9, 0.999))
    scaler = make_scaler(cfg, device)                           # amp "fp16" only
    ema = EMA(model, cfg.ema, skip=frozen)
    resumed = checkpoint.resume(out_dir, model, loss, opt, ema, scaler, cfg, external, resume_free, switches, resume_env)
    step, was_slim = resumed or (0, False)
    if resumed is None and cfg.init_from:                       # a new run warm-started from another run's checkpoint
        fresh = checkpoint.warm_start(cfg.init_from, model, loss, reset=cfg.init_reset)
        lost = sorted(set(fresh) & set(frozen))
        if lost:                                                # a frozen weight that the checkpoint does not have stays what it was initialised to
            print(f"WARNING: {len(lost)} frozen state key(s) are not in the warm-start checkpoint (or under init_reset) and stay at their random initialisation, e.g. {lost[:3]}", flush=True)
        ema = EMA(model, cfg.ema, skip=frozen)
        torch.manual_seed(cfg.seed)                             # the same data order and flips whatever the new branches drew
        np.random.seed(cfg.seed)
        print(f"warm start from {cfg.init_from}; fresh keys: {fresh}", flush=True)
    runlog.supersede_log(out_dir, step)                         # also after a crash before the first save (rows of the lost steps)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=1)
    log = open(os.path.join(out_dir, "log.jsonl"), "a")

    def save(n, slim=False):
        checkpoint.save_checkpoint(os.path.join(out_dir, f"ckpt_{n:08d}.pt"), model, loss, opt, ema, n, cfg, extra=scaler_extra(scaler), slim=slim, external=external)

    def validate(n, note=""):
        ev = evaluate(ema)
        log.write(json.dumps({"step": n, "val": ev}) + "\n")
        log.flush()
        print(f"step {n}: val total {ev.get('total', float('nan')):.4f}{note}", flush=True)

    if evaluate is not None and step and cfg.eval_every and step % cfg.eval_every == 0 and not runlog.val_logged(out_dir, step):   # the run died inside the evaluation
        validate(step, " (the evaluation a crash had cut)")                                                                        # of this step (the checkpoint is saved before it)
    model.train()
    it = iter(batches)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(batches)
            return next(it)
    params = [p for g in opt.param_groups for p in g["params"]]
    bad, n_overflow, t0, step0, last = 0, 0, time.time(), step, {}
    while step < cfg.steps:
        reason = stop_check(out_dir)                    # `python -m trainsuite stop <run>` (autoresearch/trainsuite)
        if reason is not None:
            save(step)
            checkpoint.prune_checkpoints(out_dir, cfg.keep)
            control.mark_stopped(out_dir, step, reason)
            print(f"step {step}: stop requested ({reason}), checkpoint saved, exit {control.EXIT_STOPPED}", flush=True)
            raise SystemExit(control.EXIT_STOPPED)
        f = schedule.lr_factor(step, cfg.warmup, cfg.steps, cfg.lr_min_frac)
        for g in opt.param_groups:
            g["lr"] = schedule.scheduled_lr(g, step, cfg, f)
        opt.zero_grad(set_to_none=True)
        acc, finite, forced, prow = {}, True, False, {}
        pstep = hooks.begin_step(step + 1)               # a probe only on a step that is logged
        for m in range(cfg.accum):                       # effective batch = batch size * accum
            with hooks.data_wait():                      # the wait for the data loader
                item = next_batch()
            hooks.batch(item)
            out, ok, extra = step_fn(item, StepInfo(step, step / max(1, cfg.steps), m, cfg.accum, scaler))
            if not ok:
                finite = False
                break
            for k, v in out.items():
                acc[k] = acc.get(k, 0.0) + v / cfg.accum
            if extra:
                acc.update(extra)
                forced = True
        if finite:                                       # the probe's gradient norm per module comes BEFORE clipping (a blow-up must not hide behind the clip), with the weights before the step
            gn, status = optim.optimizer_step(opt, params, cfg.grad_clip, scaler, before_clip=pstep.grads if pstep is not None else None, split=clip_split)
        else:
            gn, status = None, "nonfinite"
            opt.zero_grad(set_to_none=True)
        skipped = status != "ok"
        if status == "nonfinite" and gn is None and scaler is not None:      # a non-finite LOSS under fp16: the forward may have overflowed and written inf / nan into the batch-norm statistics
            poisoned = optim.poisoned_buffers(model)
            if poisoned:
                raise RuntimeError(f"step {step}: the forward pass left non-finite batch-norm statistics ({len(poisoned)} buffers, e.g. {poisoned[:3]}): an fp16 overflow of the activations "
                                   "(above 65504), which no loss scale cures; resume from the last checkpoint with amp bf16 or off")
        was_bad, bad = bad, optim.bad_steps_after(status, bad, scaler)
        n_overflow = _skip_report(step, status, gn, bad, was_bad, n_overflow, scaler)
        if skipped and status != "overflow":
            hooks.skipped(step, "loss" if gn is None else "gradient", gn, acc, step / max(1, cfg.steps))
        if skipped:
            if bad >= cfg.max_bad_steps:
                raise RuntimeError("too many non-finite steps in a row")
        else:
            ema.update(model, step)
            if pstep is not None:
                prow = pstep.finish()
        step += 1
        trow = hooks.end_step()
        if not skipped and (step % cfg.log_every == 0 or step == 1 or forced):
            rec = {"step": step, "t": time.time(), "lr": cfg.lr * f, "grad_norm": float(gn),
                   "s_per_step": (time.time() - t0) / max(1, step - step0), **acc, **prow, **trow}
            rec.update({f"lr.{g['lr_group']}": g["lr"] for g in opt.param_groups if "lr_group" in g})          # the own learning rate of every lr_groups prefix
            if device.startswith("cuda"):
                rec["gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
            if scaler is not None:
                rec["loss_scale"], rec["fp16_overflows"] = scaler.get_scale(), n_overflow
            last = rec
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"step {step}: total {rec['total']:.4f} lr {rec['lr']:.2e} grad {rec['grad_norm']:.2f}", flush=True)
        if cfg.save_every and step % cfg.save_every == 0:       # before the evaluation (and also after a skipped step): an evaluation
            save(step)                                          # that dies no longer costs the steps since the last save
            checkpoint.prune_checkpoints(out_dir, cfg.keep)
        if evaluate is not None and cfg.eval_every and step % cfg.eval_every == 0:
            validate(step)
    hooks.close()                                               # a run shorter than the profiler's warm-up + window writes what it has (before the final save: its wall time must not hold the checkpoint write)
    save(step, slim=cfg.slim_at_end or was_slim)
    checkpoint.prune_checkpoints(out_dir, 1 if (cfg.slim_at_end or was_slim) else cfg.keep)         # a finished run: the final weights only
    return model, loss, ema, last
