"""See inside a training step: per-module gradient and update sizes, data wait against compute time, a dump of the batch when a step goes wrong.

The numbers go into the same log row as the loss (keys the trainsuite reads: `grad.<module>`, `upd.<module>`, `t_data`, `t_step`) and the dump is a file next
to the log (`incident_<step>.json`); `python -m trainsuite status RUN` shows them and `health` warns (dead module, a module that blows up, weights that do not
move or move too fast, a data loader that starves the GPU). Nothing here imports the trainer: it takes the model, the step number and plain values.

    probe = ModuleProbe(model, every=cfg.log_every)                 # once, after the model is frozen / built
    timer = StepTimer()
    ...
    with timer.data():
        batch = next(it)                                            # the wait for the data loader
    p = probe.begin(step)                                           # None unless this step is logged
    loss.backward(); (clip, unscale)
    if p: p.grads()                                                 # gradient norm per module BEFORE clipping, and a copy of the weights (only on logged steps)
    opt.step()
    if p: row.update(p.finish())                                    # update-to-weight ratio per module, the grad. / upd. keys
    row.update(timer.row())                                         # t_data, t_step of the step just finished
    # on a skipped non-finite step or a loss spike:
    dump_incident(run_dir, step, "nonfinite", batch_ids=batch_ids, losses=loss_dict)

A module is the first `depth` dotted parts of a parameter name (depth 1: `obj_head`, `fusion`, `lane`). Only parameters that train (requires_grad) are measured:
a frozen module has no keys, which is what the suite expects.
"""
import json
import os
import time
from contextlib import contextmanager

import torch

UPD_FLOOR = 1e-2            # the update is measured against at least this weight norm: a module that starts at zero must not show a ratio of millions
MAX_INCIDENTS = 50        # a run that fails on every step must not fill the disk with dumps


def module_of(name, depth=1):
    return ".".join(name.split(".")[:depth])


class StepProbe:
    def __init__(self, probe, step):
        self.probe, self.step = probe, step
        self.before = {}                                                              # the weights before the optimiser step, copied in grads() (after the backward: the
        self.g = {}                                                                   # activations are gone then, the peak memory is lower) and only on logged steps

    def grads(self):
        """L2 norm of the gradient per module, and a copy of the weights for the update ratio (call between backward and the optimiser step)"""
        self.before = {n: p.detach().clone() for n, p in self.probe.params()}
        sq = {}
        for n, p in self.probe.params():
            m = module_of(n, self.probe.depth)
            sq[m] = sq.get(m, 0.0)                                                   # a module whose parameters got no gradient at all (detached, unused) reports 0.0: that is the finding
            if p.grad is not None:
                sq[m] += float(p.grad.detach().float().pow(2).sum())
        self.g = {m: v ** 0.5 for m, v in sq.items()}

    def finish(self):
        """{grad.<m>: norm, upd.<m>: |w_after - w_before| / |w_before|} for the step; call after the optimiser step"""
        d, w = {}, {}
        for n, p in self.probe.params():
            if n not in self.before:                                                  # grads() was not called: no update ratio, the grad. keys (if any) stay
                continue
            m = module_of(n, self.probe.depth)
            b = self.before[n].float()
            dn, wn = w.get(m, (0.0, 0.0))
            w[m] = (dn + float((p.detach().float() - b).pow(2).sum()), wn + float(b.pow(2).sum()))
        out = {}
        for m, (dn, wn) in w.items():
            out["upd." + m] = (dn ** 0.5) / max(wn ** 0.5, UPD_FLOOR)
        for m, v in self.g.items():
            out["grad." + m] = v
        self.before = {}                                                              # give the copy back
        return out


class ModuleProbe:
    def __init__(self, model, every=20, depth=1):
        self.model, self.every, self.depth = model, max(1, int(every)), depth

    def params(self):
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    def due(self, step):
        return step % self.every == 0

    def begin(self, step):
        """a StepProbe when this step is logged, else None"""
        return StepProbe(self, step) if self.due(step) else None


class StepTimer:
    """seconds a step waited for its data (`with timer.data():` around the fetch) and the whole step (from the end of the previous `row()`; the first step has no
    start and gives no t_step). Call `row()` once per step at its end; `sync` makes the CUDA work count (a synchronisation per call: use it on logged steps only)"""
    def __init__(self, sync=False):
        self.sync, self._t0, self._data = sync, None, 0.0

    @contextmanager
    def data(self):
        t = time.perf_counter()
        try:
            yield
        finally:
            self._data += time.perf_counter() - t

    def row(self):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        out = {"t_data": self._data, "t_step": now - self._t0} if self._t0 is not None else {}
        self._t0, self._data = now, 0.0
        return out


def batch_tokens(item):
    """the sample tokens of one micro-batch (a collated batch, or a list of clips of them) as a flat list of strings; [] for anything odd. Never raises"""
    try:
        items = item if isinstance(item, list) else [item]
        return [str(t) for it in items for t in it.get("tokens", [])]
    except Exception:                                                                # noqa: BLE001
        return []


def dump_incident(run_dir, step, kind, batch_ids=None, losses=None, extra=None):
    """write run_dir/incident_<step>.json ({step, kind, batch, losses, ...}) and return its path; None when there are already MAX_INCIDENTS (the first ones are the
    informative ones) or the dump itself fails (a dump must never stop the training). `batch_ids` = the sample tokens of the step (all micro-batches so far); `extra` =
    whatever helps to rebuild it (micro-batch index, the tokens per micro-batch, the fraction of the schedule, the gradient norm). It names the step and the samples; it
    does not hold the weights or the random state: to look at the step again, load the checkpoint before it and run these samples with debug.anomaly()"""
    try:
        existing = [f for f in os.listdir(run_dir) if f.startswith("incident_") and f.endswith(".json")]
        if len(existing) >= MAX_INCIDENTS:
            return None
        path = os.path.join(run_dir, f"incident_{int(step)}.json")
        rec = {"step": int(step), "kind": kind, "time": time.time(), "batch": list(batch_ids) if batch_ids is not None else None,
               "losses": {k: (float(v) if isinstance(v, (int, float)) or hasattr(v, "item") else str(v)) for k, v in (losses or {}).items()}}
        rec.update(extra or {})
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, default=str)
        os.replace(tmp, path)
        return path
    except Exception:                                                                    # noqa: BLE001 - see the docstring
        return None


def read_incident(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
