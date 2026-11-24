"""Train one part of the network and keep the rest fixed: `train_only` / `freeze` by module prefix.

    plan = freeze.apply(model, train_only=["fusion."], freeze=[])       # before the optimiser is built
    opt = AdamW(freeze.trainable(model), lr=...)                        # only what learns
    print(freeze.report(model))                                         # one line per top-level module: trainable / frozen parameters

A prefix is the dotted name of a module (`obj_head.head.vel.`, `lane.`): it selects every parameter below it (`fusion` and `fusion.` are the same).
`train_only` makes everything else frozen; `freeze` freezes the listed modules; with both, `train_only` is applied first and `freeze` takes from it.
A prefix that matches no parameter is an ERROR (a typo would train the wrong thing silently), with the nearest names.

What "frozen" means here, and the traps it avoids:
- `requires_grad = False` on the parameters, so no gradient is computed or stored for them and the optimiser never sees them
- a frozen module is kept in eval mode whatever the trainer does (`model.train()` before every epoch or evaluation would switch BatchNorm back to
  updating its running statistics, which changes the "frozen" weights: `unchanged` finds that); the mode is forced by a forward pre-hook, so it holds
  across every later `model.train()` call
- buffers (BatchNorm statistics) are covered by `unchanged` too, not only parameters
- the resume of a run is exact only with the same train_only / freeze (the optimiser state has one slot per trainable parameter): `signature` is a short
  text of the plan to put into the run's config copy and to compare on resume
"""
import difflib
import hashlib

import torch
from torch import nn


def _under(name, prefix):
    p = prefix.rstrip(".")
    return name == p or name.startswith(p + ".")


def select(model, prefixes):
    """names of the parameters under any of the prefixes"""
    return [n for n, _ in model.named_parameters() if any(_under(n, p) for p in prefixes)]


def check_prefixes(model, prefixes):
    """ValueError for a prefix that matches no parameter, naming the nearest module names"""
    names = [n for n, _ in model.named_parameters()]
    modules = sorted({n.rsplit(".", 1)[0] for n in names if "." in n} | {m for m, _ in model.named_modules() if m})
    for p in prefixes:
        if not isinstance(p, str) or not p.strip(".") or not any(_under(n, p) for n in names):
            near = difflib.get_close_matches(str(p).rstrip("."), modules, 3)
            raise ValueError(f"freeze prefix {p!r} matches no parameter of the model" + (f" (did you mean {', '.join(near)}?)" if near else f" (top-level modules: {', '.join(sorted({n.split('.')[0] for n in names}))})"))


def plan(model, train_only=(), freeze=()):
    """{parameter name: trainable?} without touching the model; the rules are in the module docstring"""
    train_only, freeze = list(train_only or ()), list(freeze or ())
    check_prefixes(model, train_only + freeze)
    only = set(select(model, train_only)) if train_only else None
    frozen = set(select(model, freeze))
    return {n: (only is None or n in only) and n not in frozen for n, _ in model.named_parameters()}


def _force_eval(module, _inputs):
    if module.training:
        module.eval()


def apply(model, train_only=(), freeze=()):
    """freeze the model as planned (see the module docstring); returns the plan {name: trainable?}"""
    pl = plan(model, train_only, freeze)
    for n, p in model.named_parameters():
        p.requires_grad_(pl[n])
    for m in model.modules():
        ps = list(m.parameters(recurse=True))
        if ps and not any(p.requires_grad for p in ps) and not getattr(m, "_frozen_eval_hook", False):
            m.register_forward_pre_hook(_force_eval)
            m._frozen_eval_hook = True
            m.eval()
    if not any(pl.values()):
        raise ValueError("nothing is left to train: train_only / freeze froze every parameter")
    return pl


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def report(model, depth=1):
    """text, one line per module at the given depth: trainable / frozen parameter counts"""
    rows = {}
    for n, p in model.named_parameters():
        k = ".".join(n.split(".")[:depth])
        t, f = rows.get(k, (0, 0))
        rows[k] = (t + p.numel(), f) if p.requires_grad else (t, f + p.numel())
    tt, ff = sum(t for t, _ in rows.values()), sum(f for _, f in rows.values())
    lines = [f"  {k:<28}{t:>12,} trainable{f:>12,} frozen" for k, (t, f) in sorted(rows.items())]
    return "\n".join(lines + [f"  {'total':<28}{tt:>12,} trainable{ff:>12,} frozen"])


def signature(train_only=(), freeze=()):
    """a short text of the plan for the run's config copy (compare it on resume): '-' when nothing is frozen"""
    if not train_only and not freeze:
        return "-"
    return "train_only=" + ",".join(sorted(train_only or ())) + ";freeze=" + ",".join(sorted(freeze or ()))


def snapshot(model, names=None):
    """{name: tensor copy} of the parameters and buffers (default: the frozen parameters and all buffers), to be compared by `unchanged` after some steps"""
    out = {}
    for n, p in model.named_parameters():
        if (names is None and not p.requires_grad) or (names is not None and n in names):
            out[n] = p.detach().clone()
    for n, b in model.named_buffers():
        if names is None or n in names:
            out[n] = b.detach().clone()
    return out


def unchanged(model, snap):
    """names (parameters and buffers) whose values differ from the snapshot, bit for bit; [] when the frozen part really stayed"""
    cur = dict(model.named_parameters())
    cur.update(dict(model.named_buffers()))
    return sorted(n for n, t in snap.items() if n not in cur or not torch.equal(cur[n].detach(), t))


def fingerprint(model):
    """short hash of all parameters and buffers (a quick 'is this the same weights' for logs)"""
    h = hashlib.sha1()
    for n, t in list(model.state_dict().items()):
        h.update(n.encode())
        h.update(t.detach().cpu().double().contiguous().numpy().tobytes())          # double: bf16 has no numpy form
    return h.hexdigest()[:12]


def frozen_keys(model):
    """state_dict keys that a plan froze: the parameters without gradient and the buffers (BatchNorm statistics) of the frozen modules. Their values must stay what
    they were: the trainer's EMA skips them (an average of a constant is the constant only up to rounding, and 'rest frozen' should mean bit-equal in the EMA
    weights that are evaluated, too)"""
    keys = {n for n, p in model.named_parameters() if not p.requires_grad}
    for mn, m in model.named_modules():
        if getattr(m, "_frozen_eval_hook", False):
            keys |= {(mn + "." if mn else "") + bn for bn, _ in m.named_buffers()}
    return keys
