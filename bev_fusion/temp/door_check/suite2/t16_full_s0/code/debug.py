"""Debug tools for a training that does not behave: they need one batch and a loss function, no data pipeline and no GPU.

    losses = overfit_one_batch(model, lambda m, b: forward_loss(m, loss, b, 0.5, "cpu", "off")["total"], batch)
    assert overfit_ok(losses)                   # the loss must fall towards 0 on ONE batch; if it does not, the bug is in the loss, the targets or the wiring
    changed, moved = frozen_stayed(model, loss_fn, batch, train_only=["fusion."])        # the frozen part is bit-equal after a few steps, the trained part moved
    with anomaly():                             # torch names the operation that produced the first NaN
        loss_fn(model, batch).backward()
    samples = replay_ids(read_incident("runs/x/incident_800.json"), lookup)    # the samples of a failure, rebuilt from their tokens by the dataset's own lookup (token -> sample);
                                                                                    # run them on the checkpoint BEFORE that step: the dump holds no weights and no random state

The hot-fix loop of a run is: `trainsuite stop RUN`, fix the code, start the same command again (it resumes from the last checkpoint on a new snapshot).
"""
import copy
from contextlib import contextmanager

import torch

try:
    from . import freeze
except ImportError:                                   # run as a script, like train.py
    import freeze


def overfit_one_batch(model, loss_fn, batch, steps=60, lr=1e-3, optimizer=None):
    """train the model on one batch for `steps` steps; the list of losses. The model is changed (use a copy if it matters)"""
    model.train()
    opt = optimizer or torch.optim.AdamW(freeze.trainable(model), lr=lr, weight_decay=0.0)
    out = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch)
        loss.backward()
        opt.step()
        out.append(float(loss.detach()))
    return out


def overfit_ok(losses, factor=0.3):
    """True when the loss ended below `factor` of where it started and never went non-finite"""
    return bool(losses) and all(x == x and abs(x) != float("inf") for x in losses) and losses[-1] < factor * losses[0]


def frozen_stayed(model, loss_fn, batch, train_only=(), freeze_=(), steps=3, lr=1e-2):
    """freeze as asked, take `steps` optimiser steps on one batch, and return (frozen weights / buffers that CHANGED, trainable parameters that MOVED).
    The first list must be empty (also for BatchNorm statistics); the second must not be: nothing moved means nothing trains. Works on a copy of the model"""
    m = copy.deepcopy(model)
    freeze.apply(m, train_only, freeze_)
    snap = freeze.snapshot(m)
    start = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    overfit_one_batch(m, loss_fn, batch, steps=steps, lr=lr)
    cur = dict(m.named_parameters())
    moved = sorted(n for n, t in start.items() if not torch.equal(cur[n].detach(), t))
    return freeze.unchanged(m, snap), moved


@contextmanager
def anomaly():
    """torch anomaly mode: a backward that meets a NaN raises with the traceback of the forward operation that made it (slow: for a replay, not for a run)"""
    with torch.autograd.set_detect_anomaly(True):
        yield


def replay_ids(incident, get_sample):
    """the samples of an incident dump (its `batch` ids) through the dataset's own lookup; ids the dump did not keep (None) are an error"""
    ids = incident.get("batch")
    if not ids:
        raise ValueError("the incident dump has no batch ids (the trainer did not pass them): rerun the step number from the checkpoint before it")
    return [get_sample(i) for i in ids]
