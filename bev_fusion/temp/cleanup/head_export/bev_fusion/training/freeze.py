"""Train one part of the network and keep the rest fixed: `train_only` / `freeze` by module prefix.

    plan = freeze.apply(model, train_only=["fusion."], freeze=[])       # before the optimiser is built
    opt = AdamW(freeze.trainable(model), lr=...)                        # only what learns
    print(freeze.report(model))                                         # one line per top-level module: trainable / frozen parameters

A prefix is the dotted name of a module (`obj_head.head.vel.`, `lane.`): it selects every parameter below it (`fusion` and `fusion.` are the same).
`train_only` makes everything else frozen; `freeze` freezes the listed modules; with both, `train_only` is applied first and `freeze` takes from it.
A prefix that matches no parameter is an ERROR (a typo would train the wrong thing silently), with the nearest names.

What "frozen" means here, and the traps it avoids:
- `requires_grad = False` on the parameters, so no gradient is computed or stored for them and the optimiser never sees them. The plan only ever FREEZES more: a parameter
  that the model keeps frozen itself (the camera heads with freeze_camera, the res4 lane features) stays frozen, and a `train_only` prefix that selects only such parameters
  is an error (it would claim to train what it cannot train)
- a parameter that has several names (a module registered under two parents, e.g. the TimeMover of the heads) is ONE parameter: it trains when any of its names is selected by
  `train_only` (or there is no `train_only`) and is frozen when any of its names is listed in `freeze`. Prefixes may use any of its names
- a frozen module is kept in eval mode whatever the trainer does (`model.train()` before every epoch or evaluation would switch BatchNorm back to
  updating its running statistics, which changes the "frozen" weights: `unchanged` finds that); the mode is forced by a forward pre-hook, so it holds
  across every later `model.train()` call. Hooks fire when the module is CALLED (a module whose forward is not used through `__call__` is not covered)
- buffers are covered by `frozen_keys`, `snapshot` and `unchanged` too, not only parameters, and by every name they have: the BatchNorm statistics of the frozen modules and every
  constant buffer; only the running statistics of normalisation layers in the TRAINED part are left out (they change)
- the resume of a run is exact only with the same train_only / freeze (the optimiser state has one slot per trainable parameter): `signature` is a short
  text of the plan to put into the run's config copy and to compare on resume
- a frozen module that is also a modality (a frozen fusion) switches modality dropout off in eval mode: that is the meaning of frozen
"""
import difflib
import hashlib

import torch
from torch import nn


def as_list(x):
    """a plan value as a list of prefixes: a list / tuple, one string is ONE prefix (not its characters), None and () are none"""
    return [x] if isinstance(x, str) else list(x or ())


def _under(name, prefix):
    p = prefix.rstrip(".").rstrip("*").rstrip(".")
    return name == p or name.startswith(p + ".")


def _aliases(model):
    """[(parameter, [every name it has])] in the order of first appearance: a parameter shared by two parents is listed once with both names"""
    by_id = {}
    for n, p in model.named_parameters(remove_duplicate=False):
        by_id.setdefault(id(p), (p, []))[1].append(n)
    return list(by_id.values())


def select(model, prefixes):
    """the (first) names of the parameters that any of the prefixes selects through any of their names"""
    return [names[0] for p, names in _aliases(model) if any(_under(n, x) for n in names for x in as_list(prefixes))]


def check_prefixes(model, prefixes, trainable_only=False):
    """ValueError for a prefix that matches no parameter (or, for train_only, none that the model lets train), naming the nearest module names"""
    al = _aliases(model)
    names = [n for _, ns in al for n in ns]
    modules = sorted({n.rsplit(".", 1)[0] for n in names if "." in n} | {m for m, _ in model.named_modules() if m})
    for p in as_list(prefixes):
        if not isinstance(p, str) or not p.strip(".*") or not any(_under(n, p) for n in names):
            near = difflib.get_close_matches(str(p).rstrip(".*"), modules, 3)
            raise ValueError(f"freeze prefix {p!r} matches no parameter of the model" + (f" (did you mean {', '.join(near)}?)" if near else f" (top-level modules: {', '.join(sorted({n.split('.')[0] for n in names}))})"))
        if trainable_only and not any(q.requires_grad for q, ns in al if any(_under(n, p) for n in ns)):
            raise ValueError(f"train_only prefix {p!r} selects only parameters that the model keeps frozen itself (freeze_camera, a pretrained lane branch): they cannot be trained from here")


def plan(model, train_only=(), freeze=()):
    """{parameter name: trainable?} (first names) without touching the model; the rules are in the module docstring. Never True for what is frozen already"""
    train_only, freeze = as_list(train_only), as_list(freeze)
    check_prefixes(model, train_only, trainable_only=True)
    check_prefixes(model, freeze)
    out = {}
    for p, names in _aliases(model):
        chosen = not train_only or any(_under(n, x) for n in names for x in train_only)
        taken = any(_under(n, x) for n in names for x in freeze)
        out[names[0]] = bool(p.requires_grad and chosen and not taken)
    return out


def _force_eval(module, _inputs):
    if module.training:
        module.eval()


def _fully_frozen(m):
    ps = list(m.parameters(recurse=True))
    return bool(ps) and not any(p.requires_grad for p in ps)


def apply(model, train_only=(), freeze=()):
    """freeze the model as planned (see the module docstring); returns the plan {name: trainable?}"""
    pl = plan(model, train_only, freeze)
    for p, names in _aliases(model):
        if not pl[names[0]]:
            p.requires_grad_(False)                              # only ever False: what the model froze itself stays frozen
    for m in model.modules():
        if _fully_frozen(m) and not getattr(m, "_frozen_eval_hook", False):
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
    train_only, freeze = as_list(train_only), as_list(freeze)
    if not train_only and not freeze:
        return "-"
    return "train_only=" + ",".join(sorted(train_only)) + ";freeze=" + ",".join(sorted(freeze))


def frozen_keys(model):
    """state_dict keys (under every name they have) that a plan keeps fixed: the parameters without gradient, the buffers of the modules whose parameters are all frozen, and
    every constant buffer (grids, depth bins: anything that is not the running statistics of a normalisation layer outside a frozen module). Their values must stay what
    they were: the trainer's EMA skips them (an average of a constant is the constant only up to rounding, and 'rest frozen' should mean bit-equal in the EMA weights that
    are evaluated, too), and `snapshot` / `unchanged` watch them. Only the running statistics of normalisation layers in the TRAINED part are left out: they change"""
    frozen = {id(p) for p in model.parameters() if not p.requires_grad}
    in_frozen = set()
    for m in model.modules():
        if _fully_frozen(m):
            in_frozen |= {id(b) for b in m.buffers(recurse=True)}
    live = set()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._NormBase) and m.track_running_stats:
            live |= {id(b) for b in m.buffers(recurse=False)} - in_frozen
    frozen |= {id(b) for b in model.buffers()} - live
    return {k for k, v in model.state_dict(keep_vars=True).items() if id(v) in frozen}


def snapshot(model, names=None):
    """{key: tensor copy} of the frozen part (`frozen_keys`) or of the given state_dict keys, to be compared by `unchanged` after some steps"""
    sd = model.state_dict(keep_vars=True)
    keys = frozen_keys(model) if names is None else set(names)
    return {k: v.detach().clone() for k, v in sd.items() if k in keys}


def unchanged(model, snap):
    """keys whose values differ from the snapshot, bit for bit; [] when the frozen part really stayed"""
    cur = model.state_dict(keep_vars=True)
    return sorted(k for k, t in snap.items() if k not in cur or not torch.equal(cur[k].detach(), t))


def fingerprint(model):
    """short hash of all parameters and buffers (a quick 'is this the same weights' for logs)"""
    h = hashlib.sha1()
    for n, t in list(model.state_dict().items()):
        h.update(n.encode())
        h.update(t.detach().cpu().double().contiguous().numpy().tobytes())          # double: bf16 has no numpy form
    return h.hexdigest()[:12]
