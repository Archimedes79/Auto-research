"""Optimiser groups, gradient clipping and the optimiser step of one training step (with the non-finite rules)."""
import torch


def _prefix_params(model, pre):
    """the trainable parameters of one module prefix: the module itself and its children, never `motion_blockx` for `motion_block`"""
    return [p for n, p in model.named_parameters() if p.requires_grad and (n == pre or n.startswith(pre.rstrip(".") + "."))]


def lr_group_params(model, lr_groups):
    """{prefix: [trainable parameters]} of `Config.lr_groups`, in sorted prefix order: module prefix -> {"lr": float, "warmup": int} (either key optional). A prefix matches the module itself and
    its children, never `motion_blockx` (the rule of `grad_clip_split`); a prefix that names no trainable parameter, two prefixes that overlap, an unknown key or a bad value are errors"""
    if not isinstance(lr_groups, dict):
        raise ValueError(f"lr_groups must be a dict prefix -> {{'lr': ..., 'warmup': ...}}, got {lr_groups!r}")
    out, seen = {}, {}
    for pre in sorted(lr_groups):                                # sorted: the group order, and with it the pairing of Adam moments and groups on a resume, does not depend on the order of the keys
        spec = lr_groups[pre]
        if not isinstance(spec, dict) or not spec or set(spec) - {"lr", "warmup"}:
            raise ValueError(f"lr_groups[{pre!r}] must be a dict with the keys lr and / or warmup, got {spec!r}")
        if "lr" in spec and not (isinstance(spec["lr"], (int, float)) and not isinstance(spec["lr"], bool) and spec["lr"] > 0):
            raise ValueError(f"lr_groups[{pre!r}]['lr'] must be a positive number, got {spec['lr']!r}")
        if "warmup" in spec and not (isinstance(spec["warmup"], int) and not isinstance(spec["warmup"], bool) and spec["warmup"] >= 0):
            raise ValueError(f"lr_groups[{pre!r}]['warmup'] must be a non-negative integer (steps), got {spec['warmup']!r}")
        ps = _prefix_params(model, pre)
        if not ps:
            raise ValueError(f"lr_groups: the prefix {pre!r} names no trainable parameter")
        for p in ps:
            if id(p) in seen:
                raise ValueError(f"lr_groups: the prefixes {seen[id(p)]!r} and {pre!r} overlap (a parameter would have two learning rates)")
            seen[id(p)] = pre
        out[pre] = ps
    return out


def param_groups(model, loss, lr, wd, lr_groups=None):
    """the optimiser's groups: decay (weights of dim >= 2) and no decay. Without lr_groups exactly these two, in this order (the optimiser state layout of every stored run). With lr_groups
    (module prefix -> {"lr", "warmup"}): the same two for everything the prefixes do not claim, then a decay / no-decay pair per prefix, tagged base_lr (the group's own peak lr),
    warmup (if given) and lr_group (the prefix); the schedule of the loop reads the tags"""
    taken = lr_group_params(model, lr_groups) if lr_groups else {}
    claimed = {id(p) for ps in taken.values() for p in ps}
    decay, no_decay = [], []
    for name, p in list(model.named_parameters()) + [("loss." + n, p) for n, p in loss.named_parameters()]:
        if p.requires_grad and id(p) not in claimed:
            (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    if not taken:
        return groups
    for pre, ps in taken.items():
        tag = {"base_lr": float(lr_groups[pre].get("lr", lr)), "lr_group": pre}
        if "warmup" in lr_groups[pre]:
            tag["warmup"] = int(lr_groups[pre]["warmup"])
        groups += [{"params": [p for p in ps if p.ndim >= 2], "weight_decay": wd, **tag}, {"params": [p for p in ps if p.ndim < 2], "weight_decay": 0.0, **tag}]
    return [g for g in groups if g["params"]]


def retag_lr_groups(opt, cfg):
    """after opt.load_state_dict: the groups of lr_groups carry the tags of the CONFIG, not the ones the checkpoint saved (they can differ only when the resume override
    let a changed lr_groups pass check_resume; without this the new value would be recorded in config.json and the old one used)"""
    for g in opt.param_groups:
        pre = g.get("lr_group")
        if pre is None:
            continue
        spec = cfg.lr_groups.get(pre)
        if spec is None:
            raise ValueError(f"the checkpoint's optimiser has a learning-rate group for {pre!r} that lr_groups no longer names")
        g["base_lr"] = float(spec.get("lr", cfg.lr))
        if "warmup" in spec:
            g["warmup"] = int(spec["warmup"])
        else:
            g.pop("warmup", None)


def clip_groups(model, prefixes):
    """the trainable parameters of each module prefix of `grad_clip_split` (a prefix matches the module itself and its children, never `motion_blockx`); a prefix that names none is an error"""
    if isinstance(prefixes, str):
        raise ValueError(f"grad_clip_split must be a list of module prefixes, got the string {prefixes!r}")
    groups, seen = [], {}
    for pre in prefixes:
        ps = _prefix_params(model, pre)
        if not ps:
            raise ValueError(f"grad_clip_split: the prefix {pre!r} names no trainable parameter")
        for p in ps:
            if id(p) in seen:
                raise ValueError(f"grad_clip_split: the prefixes {seen[id(p)]!r} and {pre!r} overlap (a parameter would be clipped twice)")
            seen[id(p)] = pre
        groups.append(ps)
    return groups


def clip_gradients(params, grad_clip, split=()):
    """clip the gradients to `grad_clip` and return the norm of ALL of them before the clip. split = lists of parameters (disjoint, from clip_groups): each list and the rest are clipped on
    their own, so a group with a much larger gradient does not scale the others down; without split it is the one global torch call, the same bits as before"""
    if not split:
        return torch.nn.utils.clip_grad_norm_(params, grad_clip)
    taken = {id(p) for g in split for p in g}
    rest = [p for p in params if id(p) not in taken]
    norms = [torch.nn.utils.clip_grad_norm_(g, grad_clip) for g in split] + ([torch.nn.utils.clip_grad_norm_(rest, grad_clip)] if rest else [])
    dev = params[0].device                                  # a group without any gradient returns a CPU zero: bring every norm to one device before the stack
    return torch.linalg.vector_norm(torch.stack([n.reshape(()).to(dev) for n in norms]))


def optimizer_step(opt, params, grad_clip, scaler=None, before_clip=None, split=()):
    """gradient clip and optimiser step of one training step. -> (grad norm before the clip, status). Without a scaler this is what the loop did inline: clip, and step unless the
    norm is not finite ("nonfinite": no step, gradients dropped). With a scaler the gradients are unscaled first (before_clip, the probe, sees them unscaled), and a non-finite
    gradient is an "overflow": no step, the scale is halved (an expected event of fp16, not a diverged run). "ok": the step was made (and the scale updated).
    split: parameter lists of `grad_clip_split` (clip_groups), each clipped to grad_clip on its own with the rest as one group; () = one global clip"""
    if scaler is not None:
        scaler.unscale_(opt)
    if before_clip is not None:
        before_clip()
    gn = clip_gradients(params, grad_clip, split)
    if not torch.isfinite(gn):
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.update(scaler.get_scale() * scaler.get_backoff_factor())
            scaler.load_state_dict({**scaler.state_dict(), "_growth_tracker": 0})            # update(new_scale) leaves the count of clean steps; an overflow restarts it, as in the stock path
            return gn, "overflow"
        return gn, "nonfinite"
    if scaler is not None:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()
    return gn, "ok"


def bad_steps_after(status, bad, scaler=None):
    """the number of bad steps in a row after a step with this status: "ok" resets it; a non-finite loss or gradient counts; an fp16 "overflow" counts only once the scale is
    below 1 (then it is a diverged run, not the scaler finding its scale: it starts at 1024 and halves)"""
    if status == "ok":
        return 0
    if status == "overflow" and scaler is not None and scaler.get_scale() >= 1.0:
        return bad
    return bad + 1


def poisoned_buffers(model):
    """names of the float buffers (batch-norm statistics ...) that are not finite: a forward pass that overflowed in fp16 writes inf / nan into them, and the EMA would carry it on"""
    return [k for k, v in model.named_buffers() if v.is_floating_point() and not torch.isfinite(v).all()]
