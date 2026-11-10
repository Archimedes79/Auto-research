"""Checkpoints of a run: atomic save, load, resume from the newest readable one, warm start from another run, pruning, and the rules that refuse a resume.

A checkpoint is {"state": model.* and loss.* tensors, "opt", "ema", "step", "cfg" (the config as a dict), "rng" (torch, numpy and, when CUDA was up, "cuda": one state per device),
"extra" ("scaler"; "data" = the position in the data, written when the data can be positioned: loop.py, data.py)}. `external` = state-key prefixes that are NOT saved
because they come from elsewhere (a frozen pre-trained part of the model): they may be missing on a load, nothing else may."""
import dataclasses
import json
import os
import pickle
import zipfile

import numpy as np
import torch

from .optim import retag_lr_groups


def model_state(model, loss, external=()):
    """everything that trains and is not external: "model.<key>" and "loss.<key>" tensors"""
    external = tuple(external)
    sd = {"model." + k: v for k, v in model.state_dict().items() if not k.startswith(external)}
    sd.update({"loss." + k: v for k, v in loss.state_dict().items()})
    return sd


def capture_rng():
    """the random streams a resume must go on with: torch (CPU), numpy and, when CUDA is up in this process, the generator of every CUDA device (a random op on the GPU, e.g. the BEV
    modality dropout, draws from it). Not initialising CUDA here keeps a CPU run off the GPU"""
    rng = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if torch.cuda.is_initialized():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    return rng


def restore_rng(rng):
    """the inverse of capture_rng; a checkpoint from before the CUDA key, or a process without CUDA, leaves the CUDA generators alone"""
    torch.set_rng_state(rng["torch"])
    np.random.set_state(rng["numpy"])
    if "cuda" in rng and torch.cuda.is_initialized():
        for i, state in enumerate(rng["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(state, i)


def save_checkpoint(path, model, loss, opt, ema, step, cfg, extra=None, slim=False, external=()):
    ck = {"state": model_state(model, loss, external), "opt": opt.state_dict(),
          "ema": {k: v.cpu() for k, v in ema.shadow.items()}, "step": step, "cfg": dataclasses.asdict(cfg),
          "rng": capture_rng(), "extra": extra or {}}
    if slim:                                                    # weights only (finished run): no optimiser state, no RNG
        del ck["opt"], ck["rng"]
        ck["slim"] = True
    tmp = path + ".tmp"
    torch.save(ck, tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())                                   # on the disk before os.replace and before older checkpoints are pruned
    os.replace(tmp, path)


def load_checkpoint(path, model, loss, opt, ema, ck=None, external=()):
    """model, loss, optimiser (if the checkpoint has one: a slim one has not), EMA and RNG state from the checkpoint -> (step, extra). A model key that the checkpoint lacks and
    that is not external, or a key the model does not have, is a RuntimeError"""
    ck = ck if ck is not None else torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state"]
    m = model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
    missing = [k for k in m.missing_keys if not k.startswith(tuple(external))]
    if missing or m.unexpected_keys:
        parts = sorted({k.split(".")[0] for k in missing})
        what = f"{parts[0]} weights" if len(parts) == 1 else "weights"
        raise RuntimeError(f"{path} has no {what} ({len(missing)} missing, e.g. {missing[:2]}; {len(m.unexpected_keys)} keys the model does not have) but this model needs them: "
                           "a checkpoint of another code version cannot be resumed; train again, or declare that part external (frozen, loaded from elsewhere)")
    loss.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("loss.")})
    if "opt" in ck:                                             # a slim checkpoint (finished run) has no optimiser state and no RNG
        opt.load_state_dict(ck["opt"])
    ema.shadow = {k: v.to(next(model.parameters()).device) for k, v in ck["ema"].items()}
    if "rng" in ck:
        restore_rng(ck["rng"])
    return ck["step"], ck.get("extra", {})


def warm_start(path, model, loss, ck=None, reset=()):
    """Config.init_from (t11): the model takes the checkpoint's EMA tensors (the evaluated weights; the integer buffers, which the
    EMA does not average, from its state) for every key it shares with them; a shape mismatch or a checkpoint key the model does not
    have raises; the model's keys the checkpoint lacks (a new branch) and the keys under a `reset` prefix (Config.init_reset; a
    prefix that names no key raises) keep their initialisation. The loss takes the checkpoint's `loss.*` (Kendall s). Optimiser, step
    and schedule are not touched. -> sorted list of the fresh keys"""
    ck = ck if ck is not None else torch.load(path, map_location="cpu", weights_only=False)
    src = {k[6:]: v for k, v in ck["state"].items() if k.startswith("model.")}
    src.update(ck["ema"])
    own = model.state_dict()
    extra = sorted(k for k in src if k not in own)
    bad = sorted(k for k in src if k in own and src[k].shape != own[k].shape)
    if extra or bad:
        raise ValueError(f"{path} does not fit the model: keys it lacks {extra[:5]} ({len(extra)}), other shapes {bad[:5]} ({len(bad)})")
    unused = [p for p in reset if not any(k.startswith(p) for k in own)]
    if unused:
        raise ValueError(f"init_reset prefixes {unused} name no key of the model")
    src = {k: v for k, v in src.items() if not k.startswith(tuple(reset))}
    model.load_state_dict(src, strict=False)
    loss.load_state_dict({k[5:]: v for k, v in ck["state"].items() if k.startswith("loss.")})
    return sorted(k for k in own if k not in src)


def prune_checkpoints(out_dir, keep):
    cks = sorted(f for f in os.listdir(out_dir) if f.startswith("ckpt_") and f.endswith(".pt"))
    for f in cks[:-keep]:
        os.remove(os.path.join(out_dir, f))
    for f in os.listdir(out_dir):                               # a write that failed (disk full) leaves ckpt_*.pt.tmp behind
        if f.startswith("ckpt_") and f.endswith(".pt.tmp"):
            os.remove(os.path.join(out_dir, f))


def check_resume(old, cfg, name):
    """refuses to resume from a checkpoint made with another configuration: a resumed run must be the run it continues. A changed
    `steps` restarts the cosine schedule high (lr factor 0.001 at step 2999 of 3000 -> 0.54 at step 3000 of 6000), a changed model,
    loss or data trains another experiment under the old name. Paths, workers and the logging / saving cadence (`cfg.RESUME_FREE`) may
    change; `cfg.RESUME_SWITCHES` = config keys that a checkpoint from before the switch lacks and that change the numbers when turned on in the middle of a run; the
    environment variable `cfg.RESUME_ENV` = 1 overrides the refusal (review 2026-10-31 pass 2)."""
    norm = lambda d: json.loads(json.dumps(d, default=str))                 # noqa: E731 - tuples and lists compare equal
    a, b = norm(old), norm(dataclasses.asdict(cfg))
    diff = sorted(k for k in a if k not in cfg.RESUME_FREE and a[k] != b.get(k))
    diff += [k for k in cfg.RESUME_SWITCHES if k not in a and b.get(k)]     # a checkpoint from before the switch ran without it: a new plan cannot resume it
    if diff and os.environ.get(cfg.RESUME_ENV) != "1":
        raise SystemExit(f"{name}: the configuration differs from the checkpoint's in {diff}; start a new run name "
                         f"(or set {cfg.RESUME_ENV}=1)")


def refuse_slim_resume(ck, step, cfg, name):
    """a slim checkpoint (a finished run: no optimiser state) evaluates and warm-starts but cannot continue training: a resume below the
    configured end is refused, at the end (a re-invocation of a finished run) it is accepted"""
    if ck.get("slim") and step < cfg.steps:
        raise SystemExit(f"{name} is a slim checkpoint (no optimiser state): it evaluates and warm-starts (init_from), it cannot "
                         f"continue training; start a new run name")


def resume(out_dir, model, loss, opt, ema, scaler, cfg, external=()):
    """continue from the newest READABLE checkpoint of out_dir -> (step, was_slim, the saved position in the data: {"epoch", "pos"} or {}), or None when the folder has none. A damaged
    one is kept as <name>.damaged and the one before is tried; when none is readable the run stops (a restart at step 0 would mix two runs in one folder). The configuration must
    be the checkpoint's (check_resume)."""
    damaged = []
    ckpts = sorted(f for f in os.listdir(out_dir) if f.startswith("ckpt_") and f.endswith(".pt"))
    for name in reversed(ckpts):
        try:
            ck = torch.load(os.path.join(out_dir, name), map_location="cpu", weights_only=False)
        except (zipfile.BadZipFile, pickle.UnpicklingError, EOFError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "zip archive" not in str(e) and "PytorchStreamReader" not in str(e):
                raise                                           # not a damaged file (e.g. out of memory): no fallback
            os.replace(os.path.join(out_dir, name), os.path.join(out_dir, name + ".damaged"))   # kept, no longer listed or pruned
            damaged.append(name)
            print(f"WARNING: {name} is damaged ({type(e).__name__}: {e}), kept as {name}.damaged; trying the one before", flush=True)
            continue
        check_resume(ck.get("cfg", {}), cfg, name)
        step, ck_extra = load_checkpoint(os.path.join(out_dir, name), model, loss, opt, ema, ck=ck, external=external)
        retag_lr_groups(opt, cfg)
        if scaler is not None and ck_extra.get("scaler"):        # the scale it had reached, not the start scale again
            scaler.load_state_dict(ck_extra["scaler"])
        refuse_slim_resume(ck, step, cfg, name)
        print(f"resumed from {name} at step {step}", flush=True)
        return step, bool(ck.get("slim")), ck_extra.get("data", {})
    if damaged:
        raise SystemExit(f"no readable checkpoint in {out_dir} (damaged: {damaged}): a restart at step 0 would mix two runs in "
                         f"one folder")
    return None
