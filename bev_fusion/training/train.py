"""Trainer for the whole camera + radar BEV chain on nuScenes (plan.md sections 8.4, 11).

    FrontRadarDataset (or the cached variant: frozen camera heads precomputed, training/feature_cache.py)
      -> BEVFusionModel.forward(batch, dts)   with dts = the target times of the GT (0 and +-0.5 s: past AND future)
      -> BEVLoss (training/losses.py, plan 8.4: masks, task groups with Kendall weights, dt weights with a warm-up)
      -> AdamW (decay only on weights of dim >= 2) + linear warm-up + cosine, grad clipping, EMA, optional bf16

What this file adds to the blocks (it edits none of them): the model's head layout taken from the LossConfig (the model
returns predictions in the loss's layout), the targets from the keyframe GT (`losses.sample_targets`: previous, current,
next keyframe = past AND future), parameter groups, schedule (warm-up from lr / 3, cosine), gradient accumulation
(effective batch = batch_size * accum), EMA, task balancing arms (Kendall / fixed / UW-SO, `loss_extras`), the optional
train-only cross-time consistency of the predictions at -0.5 / 0 / +0.5 s (`loss_extras.cross_time_consistency`, own
warm-up), an optional gradient-conflict diagnostic of the task groups,
atomic checkpoints with resume (only what trains is saved: the frozen camera heads come from the pre-trained file),
a non-finite guard, JSONL logging, validation under the EMA weights. Derived data lives in this project folder (not in
git): GT cache `data/cache/gt_*`, feature cache `data/cache/cam_feats*`, runs `runs/<name>`.
Sequence training (`seq_len` > 1): clips of consecutive keyframes of one scene (`ClipDataset`), the world memory of the
temporal fusion carried from keyframe to keyframe with the GT ego step between them (`clip_step`), detached at every
keyframe or, with `seq_bptt`, kept in the graph through the clip; the first keyframe of a clip starts cold, the later ones
warm, so both cases are trained. With `seq_len` = 1 (default) every keyframe is independent, the world memory starts
empty per sample and `cam_temporal = window_memory` is only the 3-frame window, at train AND test time. Validation of
clips reports the last keyframe warm and cold (`total_last_warm` / `total_last_cold`, the same frames).
Not here yet: BEV augmentation, the metric evaluation of evaluation/metrics.py (`eval_fn` hook). BatchNorm statistics come from the micro-batch (`batch_size`, not the
effective batch): use the largest micro-batch that fits (GPU memory not measured yet).
Validation: a fixed subset of `eval_batches * batch_size` samples spread over all val scenes (not the first scenes).
Usage:  python training/train.py <config.json>   (fields of TrainConfig; unknown keys are an error)
Self-test (nuScenes mini, CPU, needs data/cache/cam_feats_mini): python training/train.py
"""
import copy
import dataclasses
import functools
import inspect
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

try:
    from .feature_cache import CachedBEVFusionModel, CachedFrontRadarDataset, FeatureStore, collate_cached
    from .loss_extras import UWSO, cross_time_consistency, group_gradient_cosines
    from .losses import BEVLoss, LossConfig, build_layer_targets, sample_targets
    from .model import BEVFusionModel, ModelConfig
    from .nuscenes_dataset import FrontRadarDataset, available_scenes, collate
except ImportError:                                   # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from feature_cache import CachedBEVFusionModel, CachedFrontRadarDataset, FeatureStore, collate_cached
    from loss_extras import UWSO, cross_time_consistency, group_gradient_cosines
    from losses import BEVLoss, LossConfig, build_layer_targets, sample_targets
    from model import BEVFusionModel, ModelConfig
    from nuscenes_dataset import FrontRadarDataset, available_scenes, collate


@dataclass
class TrainConfig:
    name: str = "run"
    out_root: str = "runs"                       # runs/<name>/ (derived data, not in git)
    data_root: str = "~/data/nuscenes_full"
    version: str = "v1.0-trainval"
    train_split: str = "train"
    val_split: str = "val"
    train_scenes: tuple = ()                     # scene names (restrict; empty = the whole split)
    val_scenes: tuple = ()
    feature_cache: str = ""                      # path of the frozen-head feature cache ("" = run the camera heads)
    gt_cache: str = "data/cache/gt"              # GT cache folder of the dataset ("" = off)
    radar_cache: str = ""                        # per-sample radar cache folder (feature_cache.py; "" = off; exact)
    n_cam: int = 3
    cam_stride: int = 2
    batch_size: int = 2
    accum: int = 8                               # micro-batches per optimiser step: effective batch = batch_size * accum
    workers: int = 2
    steps: int = 20000                           # OPTIMISER steps
    lr: float = 3e-4                             # published 2e-4 .. 4e-4 (BEVDet family, not scaled with the batch); fresh
                                                 # trunk / heads with BatchNorm: <= 6e-4 (lane project's derail note)
    lr_min_frac: float = 1e-3
    warmup: int = 500                            # linear from lr / 3 (scan 2026-10-26)
    balancing: str = "kendall"                   # task groups: "kendall" | "fixed" (loss.task_weighting) | "uwso" (UW-SO)
    uwso_T: float = 5.0                          # UW-SO temperature, search {1, 2, 5, 10, 20} on quarter-length runs
    consistency: float = 0.0                     # weight of the train-only cross-time consistency (loss_extras), 0 = off
    consistency_warmup: tuple = (0.2, 0.2)       # (hold, ramp) fractions: off for the first 20 %, then linear in
    grad_diag_every: int = 0                     # log the cosines of the task-group gradients at the BEV feature every N steps
    seq_len: int = 1                             # keyframes per clip: > 1 trains on clips of consecutive keyframes of one scene with
                                                 # the world memory carried from keyframe to keyframe (batch_size = clips per
                                                 # micro-batch, so a micro-batch has batch_size * seq_len samples); 1 = independent
    seq_bptt: bool = False                       # backpropagate through the carried memory over the clip (graphs of all keyframes
                                                 # stay in memory); False = the state is detached at every keyframe
    gap_ego: bool = True                         # decode dt = +-0.5 s with each sample's true keyframe gap and gap-mean ego
                                                 # motion (needs `ego=` in the model's forward; 0.2-0.35 m of label error otherwise)
    weight_decay: float = 1e-2
    grad_clip: float = 5.0
    ema: float = 0.999
    amp: str = "off"                             # "off" | "bf16" (CUDA only)
    log_every: int = 20
    eval_every: int = 1000
    eval_batches: int = 50
    save_every: int = 500
    keep: int = 3
    max_bad_steps: int = 20                      # consecutive non-finite steps before aborting
    seed: int = 0
    device: str = "cuda"
    model: dict = field(default_factory=dict)    # ModelConfig fields
    loss: dict = field(default_factory=dict)     # LossConfig fields


# ------------------------------------------------------------------------------------------------ glue
def build(cfg):
    """(model, loss, cached?). The head layout of the model follows the loss's layer spec: the flags that define it
    (centrelines, dense_objects, radar_stationary) are taken from the LossConfig, so the two cannot disagree; the model
    returns predictions in the loss's layout (res["out"] = {dt: {"sem", "obj", "mass", ...}})."""
    lc = LossConfig(**{**cfg.loss, **({"task_weighting": "fixed"} if cfg.balancing == "fixed" else {})})
    loss = BEVLoss(lc)
    if cfg.balancing == "uwso":
        loss.task = UWSO(cfg.uwso_T)                          # UW-SO (2408.07985) instead of the Kendall weights
    elif cfg.balancing not in ("kendall", "fixed"):
        raise ValueError(f"balancing {cfg.balancing!r}")
    layout = {"centrelines": lc.centrelines, "dense_objects": lc.dense_objects, "radar_stationary": lc.radar_stationary}
    mcfg = ModelConfig(**{**layout, **cfg.model})
    for k, v in layout.items():
        if getattr(mcfg, k) != v:
            raise ValueError(f"model.{k} = {getattr(mcfg, k)} but loss.{k} = {v}: the head layout must match the loss")
    cached = bool(cfg.feature_cache)
    model = CachedBEVFusionModel(mcfg) if cached else BEVFusionModel(mcfg)
    return model, loss, cached


def make_datasets(cfg):
    from nuscenes.nuscenes import NuScenes
    root = os.path.expanduser(cfg.data_root)
    nusc = NuScenes(cfg.version, dataroot=root, verbose=False)

    def scenes(split, names):
        if cfg.version.endswith("mini") and split not in ("mini_train", "mini_val"):
            split = "mini_" + split
        toks = available_scenes(nusc, split)
        if names:
            toks = [t for t in toks if nusc.get("scene", t)["name"] in set(names)]
        return toks
    kw = dict(n_cam=cfg.n_cam, cam_stride=cfg.cam_stride, with_gt=True, nusc=nusc,
              cache_dir=cfg.gt_cache or None)
    out = []
    for split, names in ((cfg.train_split, cfg.train_scenes), (cfg.val_split, cfg.val_scenes)):
        if cfg.feature_cache:
            out.append(CachedFrontRadarDataset(root, cfg.version, scenes(split, names),
                                               store=FeatureStore(cfg.feature_cache), radar_cache=cfg.radar_cache or None, **kw))
        else:
            out.append(FrontRadarDataset(root, cfg.version, scenes(split, names), **kw))
    return out


class ClipDataset(torch.utils.data.Dataset):
    """Clips of `length` consecutive keyframes of ONE scene (sequence training): item = [base[i], base[i + 1], ...], a clip
    starts at every sample whose next `length - 1` keyframes are also in `base.samples` (the dataset already leaves out the
    first and last keyframe of a scene, so the chain stays inside the scene)."""

    def __init__(self, base, length):
        self.base, self.length = base, length
        pos = {t: i for i, t in enumerate(base.samples)}
        self.starts = []
        for t in base.samples:
            chain, cur = [pos[t]], t
            while len(chain) < length:
                cur = base.nusc.get("sample", cur)["next"]
                if cur not in pos:
                    break
                chain.append(pos[cur])
            if len(chain) == length:
                self.starts.append(chain)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        return [self.base[j] for j in self.starts[i]]


def collate_clips(clips, fn):
    """list of clips (lists of items) -> list over time of batches: [fn([clip[t] for clip in clips]) for t]."""
    return [fn([c[t] for c in clips]) for t in range(len(clips[0]))]


def make_loader(ds, cfg, train):
    fn = collate_cached if cfg.feature_cache else collate
    if cfg.seq_len > 1:
        fn = functools.partial(collate_clips, fn=fn)
    return torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=train, num_workers=cfg.workers,
                                       collate_fn=fn, drop_last=train, persistent_workers=cfg.workers > 0)


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif k == "radar":
            out[k] = {a: b.to(device) for a, b in v.items()}
        elif k == "steps":
            out[k] = [[tuple(t.to(device) for t in s) for s in fr] for fr in v]
        else:
            out[k] = v
    return out


# ------------------------------------------------------------------------------------------------ optimisation
def param_groups(model, loss, lr, wd):
    decay, no_decay = [], []
    for name, p in list(model.named_parameters()) + [("loss." + n, p) for n, p in loss.named_parameters()]:
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def lr_factor(step, warmup, total, min_frac):
    if step < warmup:
        return 1 / 3 + (2 / 3) * (step + 1) / warmup                      # from lr / 3 to lr
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t))


class EMA:
    """Exponential moving average of the trainable weights and the buffers (BatchNorm statistics)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in self._tensors(model).items()}
        self.backup = None

    @staticmethod
    def _tensors(model):
        return {k: v for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))
        for k, v in self._tensors(model).items():
            self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)

    def apply(self, model):
        self.backup = {k: v.detach().clone() for k, v in self._tensors(model).items()}
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=True)

    def restore(self, model):
        model.load_state_dict({**model.state_dict(), **self.backup}, strict=True)
        self.backup = None


def trainable_state(model, loss):
    """Everything that is not the frozen camera heads (those come from the pre-trained file)."""
    sd = {"model." + k: v for k, v in model.state_dict().items() if not k.startswith("camera.")}
    sd.update({"loss." + k: v for k, v in loss.state_dict().items()})
    return sd


def save_checkpoint(path, model, loss, opt, ema, step, cfg, extra=None):
    ck = {"state": trainable_state(model, loss), "opt": opt.state_dict(),
          "ema": {k: v.cpu() for k, v in ema.shadow.items()}, "step": step, "cfg": dataclasses.asdict(cfg),
          "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}, "extra": extra or {}}
    tmp = path + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, loss, opt, ema):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state"]
    m = model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
    assert all(k.startswith("camera.") for k in m.missing_keys) and not m.unexpected_keys, m
    loss.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("loss.")})
    opt.load_state_dict(ck["opt"])
    ema.shadow = {k: v.to(next(model.parameters()).device) for k, v in ck["ema"].items()}
    torch.set_rng_state(ck["rng"]["torch"])
    np.random.set_state(ck["rng"]["numpy"])
    return ck["step"], ck.get("extra", {})


def prune_checkpoints(out_dir, keep):
    cks = sorted(f for f in os.listdir(out_dir) if f.startswith("ckpt_") and f.endswith(".pt"))
    for f in cks[:-keep]:
        os.remove(os.path.join(out_dir, f))


# ------------------------------------------------------------------------------------------------ loop
def _f(v):
    return float(v.detach()) if torch.is_tensor(v) else float(v)


def consistency_weight(cfg, step_frac):
    """cfg.consistency x a linear ramp: 0 during the hold (first `hold` of the steps), 1 after hold + ramp."""
    hold, ramp = cfg.consistency_warmup
    return cfg.consistency * min(1.0, max(0.0, (step_frac - hold) / max(ramp, 1e-6)))


def consistency_term(model, loss, res, gts, gt_samples, device):
    """Train-only cross-time consistency of the predictions at dt = 0 and dt = +-0.5 s (loss_extras): the layers whose
    kind is area / line / object, the visibility of the cells = camera and not occluded, the ego steps and the motion
    field of the batch. -> the result dict of cross_time_consistency."""
    sel = [i for i, (_, kind, _) in enumerate(loss.spec) if kind in ("area", "line", "object")]
    out = res["out"]
    probs = {dt: torch.sigmoid(o["sem"][:, sel].float()) for dt, o in out.items()}
    weights = {dt: build_layer_targets(gts[dt], loss.cfg, out[dt].get("mass"), dt, device)["w_cam"] for dt in out}
    steps = {dt: [s["prev" if dt < 0 else "next"]["meta"]["steps"] for s in gt_samples] for dt in out if dt != 0.0}
    return cross_time_consistency(probs, steps, out[0.0]["v_hat"].float(), model.grid, weights)


def gap_ego_motion(steps):
    """The ego steps [(dt_i, v_i, omega_i)] of the GT meta (time 0 -> the target keyframe, negative dt backwards) as ONE
    splat step (dt, v, omega) that lands a static point where the chain does: the chain is composed with the splat's own
    motion model p' = R(-omega dt)(p - d), d = (v dt, v omega dt^2 / 2); omega = heading change / dt, v = forward chord /
    dt (what splat.ego_speed_yawrate gives from the two poses). dt is the TRUE gap (0.40-0.60 s on nuScenes), not the
    nominal 0.5 s."""
    x = y = th = T = 0.0                                   # the start ego origin in the current ego frame, heading, time
    for dt, v, w in steps:
        px, py = x - v * dt, y - 0.5 * v * w * dt * dt
        c, s = math.cos(-w * dt), math.sin(-w * dt)
        x, y = c * px - s * py, s * px + c * py
        th += w * dt
        T += dt
    if abs(T) < 1e-9:
        return 0.0, 0.0, 0.0
    chord_x = -(math.cos(th) * x - math.sin(th) * y)       # the end ego in the start frame (x component)
    return T, chord_x / T, th / T


def batch_ego(gt_samples, dts, device):
    """{nominal dt != 0: (dt, v, omega)} as (B,) tensors: the true gap and the gap-mean ego motion of every sample."""
    ego = {}
    for dt in dts:
        if abs(dt) < 1e-6:
            continue
        key = "prev" if dt < 0 else "next"
        vals = [gap_ego_motion(s[key]["meta"]["steps"]) for s in gt_samples]
        ego[dt] = tuple(torch.tensor([v[i] for v in vals], dtype=torch.float32, device=device) for i in range(3))
    return ego


def supports_ego(model):
    return "ego" in inspect.signature(model.forward).parameters


def val_subset(ds, n):
    """n samples spread evenly over the whole validation set (all scenes), not its first batches."""
    if n >= len(ds):
        return ds
    return torch.utils.data.Subset(ds, sorted({int(i) for i in np.linspace(0, len(ds) - 1, n).round()}))


def detach_tree(x):
    """The model state (nested dicts / lists of tensors, None allowed) without its graph."""
    if torch.is_tensor(x):
        return x.detach()
    if isinstance(x, dict):
        return {k: detach_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(detach_tree(v) for v in x)
    return x


def clip_step(prev_batch, device):
    """(dt, v, omega) as (B,) tensors from the previous keyframe of each clip to the current one: the GT ego steps of the
    previous batch toward its next keyframe, composed into one splat step; the model's memory pose advances by it."""
    vals = [gap_ego_motion(s["next"]["meta"]["steps"]) for s in prev_batch["gt"]]
    return tuple(torch.tensor([v[i] for v in vals], dtype=torch.float32, device=device) for i in range(3))


def forward_loss(model, loss, batch, step_frac, device, amp, cfg=None, diag=False, train_terms=True, state=None, step=None):
    """-> (loss dict, model result, GT targets). With cfg.gap_ego the model decodes dt = +-0.5 s with the TRUE gap of each
    sample and its gap-mean ego motion (the GT is the keyframe at the true gap), when the model takes `ego`. With
    cfg.consistency > 0 (and train_terms) the cross-time consistency is added (with its own warm-up); with diag the
    cosines of the task-group gradients at the BEV feature are put into out['diag']. state / step: the model state of the
    previous keyframe of the clip and the ego step since then (sequence training)."""
    gts = sample_targets(batch["gt"])
    dts = tuple(sorted(gts))
    ego = batch_ego(batch["gt"], dts, device) if cfg is not None and cfg.gap_ego and supports_ego(model) else None
    batch = to_device(batch, device)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp == "bf16" and device.startswith("cuda") else torch.autocast("cpu", enabled=False)
    kw = {k: v for k, v in (("ego", ego), ("state", state), ("step", step)) if v is not None}
    with ctx:
        res = model(batch, dts=dts, **kw)
    with torch.autocast("cpu", enabled=False), torch.autocast("cuda", enabled=False):
        preds = {dt: {k: ({a: b.float() for a, b in v.items()} if isinstance(v, dict) else v.float()) for k, v in p.items()}
                 for dt, p in res["out"].items()}                                    # already in the loss's layout
        out = loss(preds, gts, step_frac)
        out["total_main"] = out["total"].detach()                                 # without the train-only terms
        wc = consistency_weight(cfg, step_frac) if cfg is not None and train_terms and len(dts) > 1 else 0.0
        if wc > 0:                                                                 # nothing is computed during the hold
            ct = consistency_term(model, loss, res, gts, batch["gt"], device)
            out["consistency"] = ct["total"]
            out["total"] = out["total"] + wc * ct["total"]
        if diag:
            groups = {k[len("group."):]: v for k, v in out.items() if k.startswith("group.") and torch.is_tensor(v) and v.requires_grad}
            if wc > 0 and out["consistency"].requires_grad:      # its gradient norm at the BEV feature calibrates its weight
                groups["consistency"] = out["consistency"]
            if len(groups) > 1:
                names, C, nrm = group_gradient_cosines(groups, res["features"])
                out["diag"] = {"groups": names, "cos": C.tolist(), "norm": nrm.tolist()}
    return out, res, gts


def run_clip(model, loss, clip, step_frac, device, amp, cfg, diag=False, train_terms=True, backward=False):
    """One clip = a list of batches (consecutive keyframes of the same scenes; a single batch = a clip of length 1). The
    model state (world memory) is carried from keyframe to keyframe with the ego step between them: detached at every
    keyframe, or with cfg.seq_bptt kept in the graph through the clip. backward: backpropagate the mean loss of the clip
    divided by cfg.accum, right after each keyframe (with seq_bptt the earlier graphs are retained for the state path of
    the later keyframes, so the gradient is that of the clip's mean loss).
    -> (dict of the scalar terms averaged over the keyframes, finite flag, diag of the last keyframe or None, list of the
    per-keyframe totals)."""
    L, state, prev, acc, totals, dg = len(clip), None, None, {}, [], None
    bptt, accum = bool(cfg is not None and cfg.seq_bptt), (cfg.accum if cfg is not None else 1)
    for t, batch in enumerate(clip):
        step = clip_step(prev, device) if prev is not None else None
        out, res, _ = forward_loss(model, loss, batch, step_frac, device, amp, cfg, diag=diag and t == L - 1,
                                   train_terms=train_terms, state=state, step=step)
        if not torch.isfinite(out["total"]):
            return acc, False, dg, totals
        totals.append(_f(out["total"]))
        if backward:           # backward at once: the loss of a later keyframe changes the Kendall weights in place (clamp_)
            (out["total"] / (L * accum)).backward(retain_graph=bptt and t < L - 1)    # bptt: the graph is still needed for the state path
        for k, v in out.items():
            if k == "total" or k == "consistency" or k.startswith(("group.", "s.")):
                acc[k] = acc.get(k, 0.0) + _f(v) / L
        if "diag" in out:
            dg = out["diag"]
        state = res["state"] if bptt else detach_tree(res["state"])
        prev = batch
    return acc, True, dg, totals


def evaluate(model, loss, ema, loader, n_batches, device, amp, eval_fn=None, cfg=None):
    """Validation loss (and groups) under the EMA weights, decoded like in training (cfg.gap_ego), without the train-only
    terms; eval_fn(model, loader) -> dict is an optional metrics hook. With clips (seq_len > 1) the state is carried through
    the clip, and the LAST keyframe is also run cold (no memory): total_last_warm / total_last_cold are the same frames with
    and without the carried memory (paired)."""
    ema.apply(model)
    model.eval()
    tot, n = {}, 0
    try:
        with torch.no_grad():
            for i, item in enumerate(loader):
                if i >= n_batches:
                    break
                clip = item if isinstance(item, list) else [item]
                out, ok, _, totals = run_clip(model, loss, clip, 1.0, device, amp, cfg, train_terms=False)
                if not ok:
                    continue
                for k, v in out.items():
                    if k == "total" or k.startswith("group."):
                        tot[k] = tot.get(k, 0.0) + v
                if len(clip) > 1:
                    cold, _, _ = forward_loss(model, loss, clip[-1], 1.0, device, amp, cfg, train_terms=False)
                    tot["total_last_warm"] = tot.get("total_last_warm", 0.0) + totals[-1]
                    tot["total_last_cold"] = tot.get("total_last_cold", 0.0) + _f(cold["total"])
                n += 1
        res = {k: v / max(1, n) for k, v in tot.items()}
        if eval_fn is not None:
            res.update(eval_fn(model, loader))
    finally:
        ema.restore(model)
        model.train()
    return res


def train(cfg, eval_fn=None, datasets=None):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    out_dir = os.path.join(cfg.out_root, cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    model, loss, _ = build(cfg)
    model, loss = model.to(device), loss.to(device)
    train_ds, val_ds = datasets or make_datasets(cfg)
    if cfg.seq_len > 1:                                         # clips of consecutive keyframes, the memory carried through
        train_ds, val_ds = ClipDataset(train_ds, cfg.seq_len), ClipDataset(val_ds, cfg.seq_len)
    train_loader = make_loader(train_ds, cfg, True)
    val_loader = make_loader(val_subset(val_ds, cfg.eval_batches * cfg.batch_size), cfg, False)     # fixed, spread over all scenes
    if cfg.gap_ego and not supports_ego(model):
        print("WARNING: gap_ego is on but model.forward has no `ego` argument: dt = +-0.5 s is decoded with the nominal "
              "gap and the last step's ego motion (0.2-0.35 m of label error)", flush=True)
    opt = torch.optim.AdamW(param_groups(model, loss, cfg.lr, cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.999))
    ema = EMA(model, cfg.ema)
    step = 0
    ckpts = sorted(f for f in os.listdir(out_dir) if f.startswith("ckpt_") and f.endswith(".pt"))
    if ckpts:
        step, _ = load_checkpoint(os.path.join(out_dir, ckpts[-1]), model, loss, opt, ema)
        print(f"resumed from {ckpts[-1]} at step {step}", flush=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=1)
    log = open(os.path.join(out_dir, "log.jsonl"), "a")
    model.train()
    it = iter(train_loader)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(train_loader)
            return next(it)
    params = [p for g in opt.param_groups for p in g["params"]]
    bad, t0, step0, last = 0, time.time(), step, {}
    while step < cfg.steps:
        f = lr_factor(step, cfg.warmup, cfg.steps, cfg.lr_min_frac)
        for g in opt.param_groups:
            g["lr"] = cfg.lr * f
        diag = bool(cfg.grad_diag_every) and (step + 1) % cfg.grad_diag_every == 0
        opt.zero_grad(set_to_none=True)
        acc, finite = {}, True
        for m in range(cfg.accum):                              # effective batch = batch_size * seq_len * accum samples
            item = next_batch()
            out, ok, dg, _ = run_clip(model, loss, item if isinstance(item, list) else [item], step / max(1, cfg.steps),
                                      device, cfg.amp, cfg, diag=diag and m == 0, backward=True)
            if not ok:
                finite = False
                break
            for k, v in out.items():
                acc[k] = acc.get(k, 0.0) + v / cfg.accum
            if dg:
                acc["diag"] = dg
        gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip) if finite else None
        if gn is None or not torch.isfinite(gn):
            bad += 1
            opt.zero_grad(set_to_none=True)
            print(f"step {step}: non-finite {'loss' if gn is None else 'gradient'}, skipped ({bad} in a row)", flush=True)
            if bad >= cfg.max_bad_steps:
                raise RuntimeError("too many non-finite steps in a row")
            step += 1
            continue
        bad = 0
        opt.step()
        ema.update(model, step)
        step += 1
        if step % cfg.log_every == 0 or step == 1 or "diag" in acc:
            rec = {"step": step, "lr": cfg.lr * f, "grad_norm": float(gn),
                   "s_per_step": (time.time() - t0) / max(1, step - step0), **acc}
            if device.startswith("cuda"):
                rec["gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
            last = rec
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"step {step}: total {rec['total']:.4f} lr {rec['lr']:.2e} grad {rec['grad_norm']:.2f}", flush=True)
        if cfg.eval_every and step % cfg.eval_every == 0:
            ev = evaluate(model, loss, ema, val_loader, 10 ** 9, device, cfg.amp, eval_fn, cfg)
            log.write(json.dumps({"step": step, "val": ev}) + "\n")
            log.flush()
            print(f"step {step}: val total {ev.get('total', float('nan')):.4f}", flush=True)
        if cfg.save_every and step % cfg.save_every == 0:
            save_checkpoint(os.path.join(out_dir, f"ckpt_{step:08d}.pt"), model, loss, opt, ema, step, cfg)
            prune_checkpoints(out_dir, cfg.keep)
    save_checkpoint(os.path.join(out_dir, f"ckpt_{step:08d}.pt"), model, loss, opt, ema, step, cfg)
    prune_checkpoints(out_dir, cfg.keep)
    return model, loss, ema, last


# ------------------------------------------------------------------------------------------------ self-test
def _selftest():
    import shutil
    import tempfile
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))                 # the bev_fusion folder
    torch.set_num_threads(4)                                   # the machine is shared: do not take all cores
    cache = "data/cache/cam_feats_mini"
    assert os.path.isdir(cache) and len(FeatureStore(cache)) > 0, \
        "build the mini cache first: python training/feature_cache.py build ~/data/nuscenes v1.0-mini data/cache/cam_feats_mini --limit 6"
    store = FeatureStore(cache)
    tmp = tempfile.mkdtemp()
    try:
        cfg = TrainConfig(name="selftest", out_root=tmp, data_root="~/data/nuscenes", version="v1.0-mini",
                          train_split="train", val_split="val", feature_cache=cache, gt_cache=os.path.join(tmp, "gt"),
                          batch_size=2, accum=2, workers=0, steps=4, warmup=2, log_every=1, eval_every=0, save_every=2, keep=2,
                          device="cpu", model={"trunk_width": 0.25})
        # the dataset restricted to the samples whose frames are cached
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(cfg.version, dataroot=os.path.expanduser(cfg.data_root), verbose=False)
        base = CachedFrontRadarDataset(os.path.expanduser(cfg.data_root), cfg.version, store=store, n_cam=3, cam_stride=2,
                                       with_gt=True, nusc=nusc, cache_dir=cfg.gt_cache)
        from feature_cache import used_frames
        ok = [t for t in base.samples if all(store.has(x) for v in used_frames(base, [t]).values() for x in v)]
        assert len(ok) >= 4, f"only {len(ok)} cached samples"
        base.samples = ok[:4]
        # 1 glue: GT targets at three times, model outputs -> loss inputs, the loss is finite and has the 8.4 structure
        model, loss, cached = build(cfg)
        assert cached and model.aux_cam.net[-1].out_channels == len(loss.cam_spec) and \
            model.aux_rad.net[-1].out_channels == len(loss.radar_spec)
        batch = collate_cached([base[0], base[1]])
        gts = sample_targets(batch["gt"])
        assert sorted(gts) == [-0.5, 0.0, 0.5], sorted(gts)
        # 0 the true gap and the gap-mean ego motion of the ego steps (nominal 0.5 s, true 0.40-0.60 s)
        ego = batch_ego(batch["gt"], sorted(gts), "cpu")
        assert sorted(ego) == [-0.5, 0.5] and all(t.shape == (2,) for e in ego.values() for t in e), ego
        assert all(0.35 < abs(float(d)) < 0.65 for e in ego.values() for d in e[0]) and bool((ego[-0.5][0] < 0).all()) \
            and bool((ego[0.5][0] > 0).all())
        T, v, w = gap_ego_motion([(1 / 12, 10.0, 0.0)] * 6)
        assert abs(T - 0.5) < 1e-9 and abs(v - 10) < 1e-9 and abs(w) < 1e-9, (T, v, w)
        T, v, w = gap_ego_motion([(-1 / 12, 10.0, 0.0)] * 6)
        assert abs(T + 0.5) < 1e-9 and abs(v - 10) < 1e-9, (T, v, w)
        for sgn in (1, -1):                       # an arc, forward and backward: chord = v / omega sin(omega T), lateral offset
            T, v, w = gap_ego_motion([(sgn / 12, 10.0, 0.2)] * 6)
            assert abs(v * T - sgn * 50 * math.sin(0.1)) < 2e-3 and abs(w - 0.2) < 1e-9, (sgn, T, v, w)
            assert abs(0.5 * v * w * T * T - 50 * (1 - math.cos(0.1))) < 2e-3, "the splat's lateral term matches the arc"
        assert gap_ego_motion([]) == (0.0, 0.0, 0.0)
        sub = val_subset(list(range(100)), 10)
        assert len(sub) == 10 and sub[0] == 0 and sub[-1] == 99 and val_subset(list(range(5)), 10) == list(range(5))
        model.train()
        out, res, _ = forward_loss(model, loss, batch, 0.5, "cpu", "off")
        assert torch.isfinite(out["total"]) and any(k.startswith("group.") for k in out)
        if supports_ego(model):                                  # true gap + gap-mean ego motion through the model
            out_e, _, _ = forward_loss(model, loss, batch, 0.5, "cpu", "off", cfg)
            assert torch.isfinite(out_e["total"]) and _f(out_e["total"]) != _f(out["total"]), "ego changes the decode"
        # 1b balancing arms: kendall (default), fixed, UW-SO; the consistency term reaches the motion field; the
        # gradient-conflict diagnostic returns a cosine matrix with a unit diagonal
        assert loss.task.mode == "kendall"
        assert build(dataclasses.replace(cfg, balancing="fixed"))[1].task.mode == "fixed"
        m_uw, l_uw, _ = build(dataclasses.replace(cfg, balancing="uwso", uwso_T=2.0))
        assert isinstance(l_uw.task, UWSO)
        try:
            build(dataclasses.replace(cfg, balancing="nope"))
            raise AssertionError("an unknown balancing must raise")
        except ValueError:
            pass
        out_u, _, _ = forward_loss(m_uw.train(), l_uw, batch, 0.5, "cpu", "off")
        assert torch.isfinite(out_u["total"]) and abs(sum(_f(v) for k, v in out_u.items() if k.startswith("s.")) - 1.0) < 1e-4, \
            "UW-SO weights are a softmax"
        model.zero_grad()
        ccfg = dataclasses.replace(cfg, consistency=1.0, consistency_warmup=(0.0, 0.1))
        out_c, _, _ = forward_loss(model, loss, batch, 0.9, "cpu", "off", ccfg, diag=True)
        wc = consistency_weight(ccfg, 0.9)
        assert wc == 1.0 and torch.isfinite(out_c["consistency"]) and \
            abs(_f(out_c["total"]) - _f(out_c["total_main"]) - wc * _f(out_c["consistency"])) < 1e-4 * max(1.0, _f(out_c["total"])), \
            "total = main total + weight x consistency"
        names, C = out_c["diag"]["groups"], torch.tensor(out_c["diag"]["cos"])
        d = torch.diagonal(C)      # 1 for a group with a gradient at the feature; 0 for the motion group: its head starts at zero
        assert len(names) > 1 and bool((((d - 1).abs() < 1e-3) | (d.abs() < 1e-6)).all()) and int((d > 0.5).sum()) >= 3, (names, C)
        out_c["consistency"].backward()
        mg = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters() if n.startswith("mover.") and p.grad is not None)
        assert mg > 0, "the consistency term must give the motion head a gradient"
        model.zero_grad()
        held = dataclasses.replace(cfg, consistency=1.0, consistency_warmup=(0.5, 0.1))
        assert all(abs(consistency_weight(held, f) - w) < 1e-9 for f, w in ((0.3, 0.0), (0.5, 0.0), (0.55, 0.5), (0.6, 1.0), (0.9, 1.0)))
        out_h, _, _ = forward_loss(model, loss, batch, 0.3, "cpu", "off", held)           # step_frac 0.3 is inside the hold
        assert "consistency" not in out_h and _f(out_h["total"]) == _f(out_h["total_main"]), "hold = weight 0, nothing computed"
        # 1c sequences: clips of consecutive keyframes, the world memory carried with the ego step between them
        cds = ClipDataset(base, 3)
        assert len(cds) == 2 and cds.starts == [[0, 1, 2], [1, 2, 3]], cds.starts     # 4 consecutive samples -> 2 clips of 3
        toks = [base.samples[i] for i in cds.starts[0]]
        assert [nusc.get("sample", t)["next"] for t in toks[:-1]] == toks[1:], "consecutive keyframes of one scene"
        cl = collate_clips([cds[0], cds[1]], collate_cached)
        assert len(cl) == 3 and all(b["prob"].shape[0] == 2 for b in cl)
        step1 = clip_step(cl[0], "cpu")                                                  # keyframe 0 -> 1
        gap_back = [-sum(x[0] for x in s["prev"]["meta"]["steps"]) for s in cl[1]["gt"]]   # the same gap from keyframe 1 back to 0
        assert all(abs(float(a) - b) < 1e-6 for a, b in zip(step1[0], gap_back)), (step1[0], gap_back)
        with torch.no_grad():                               # the carried memory changes the prediction (no graphs: CPU memory)
            _, r0, _ = forward_loss(model, loss, cl[0], 0.5, "cpu", "off", cfg)
            assert float(r0["state"]["cam"]["W"].max()) > 0, "the first keyframe writes the memory"
            _, rw, _ = forward_loss(model, loss, cl[1], 0.5, "cpu", "off", cfg, state=r0["state"], step=step1)
            _, rc, _ = forward_loss(model, loss, cl[1], 0.5, "cpu", "off", cfg)
            assert float((rw["out"][0.0]["sem"] - rc["out"][0.0]["sem"]).abs().max()) > 1e-6, "the carried memory changes the prediction"
        del r0, rw, rc
        c1 = collate_clips([cds[0]], collate_cached)        # ONE clip of 3 keyframes (batch 1) for the gradient tests
        s1 = clip_step(c1[0], "cpu")
        p0 = next(p for n, p in model.named_parameters() if n.startswith("cam_reduce") and p.ndim >= 2)

        def grad_second(attached):                         # gradient of the second keyframe's loss at a camera-branch weight
            _, ra, _ = forward_loss(model, loss, c1[0], 0.5, "cpu", "off", cfg)
            o2, _, _ = forward_loss(model, loss, c1[1], 0.5, "cpu", "off", cfg, state=ra["state"] if attached else detach_tree(ra["state"]),
                                    step=s1)
            return torch.autograd.grad(o2["total"], p0, allow_unused=True)[0]
        g_det, g_att = grad_second(False), grad_second(True)
        assert g_det is not None and g_att is not None and not torch.allclose(g_det, g_att, rtol=1e-3, atol=1e-9),             "with the state attached the second loss also reaches the first keyframe's camera planes"
        model.zero_grad()
        for bptt, clip in ((False, c1), (True, c1[:2])):    # run_clip with backward: finite, gradients on the weights
            o, ok, _, tot_t = run_clip(model, loss, clip, 0.5, "cpu", "off", dataclasses.replace(cfg, seq_bptt=bptt), backward=True)
            assert ok and len(tot_t) == len(clip) and p0.grad is not None and float(p0.grad.abs().sum()) > 0, bptt
            model.zero_grad()
        # 2 four steps: loss finite, parameters change, frozen heads are not part of the optimiser or the checkpoint
        before = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("cam_reduce")}
        model2, loss2, ema, last = train(cfg, datasets=(base, base))
        assert last["total"] == last["total"], "logged"
        ckpts = sorted(f for f in os.listdir(os.path.join(tmp, "selftest")) if f.endswith(".pt"))
        assert len(ckpts) == 2, ckpts                                              # keep = 2
        ck = torch.load(os.path.join(tmp, "selftest", ckpts[-1]), weights_only=False)
        assert ck["step"] == 4 and not any(k.startswith("model.camera.") for k in ck["state"])
        changed = any(not torch.equal(before[k], v) for k, v in model2.state_dict().items() if k in before)
        assert changed, "parameters changed"
        # 3 resume: a run with steps = 6 continues from step 4 and ends at 6; the EMA differs from the weights
        cfg2 = dataclasses.replace(cfg, steps=6)
        _, _, ema2, last2 = train(cfg2, datasets=(base, base))
        ck2 = sorted(f for f in os.listdir(os.path.join(tmp, "selftest")) if f.endswith(".pt"))[-1]
        assert torch.load(os.path.join(tmp, "selftest", ck2), weights_only=False)["step"] == 6
        # 4 validation under the EMA weights restores the training weights afterwards
        w_before = {k: v.clone() for k, v in model2.state_dict().items() if k.startswith("cam_reduce")}
        loader = make_loader(base, dataclasses.replace(cfg, batch_size=2), False)
        ev = evaluate(model2, loss2, ema2, loader, 1, "cpu", "off")
        assert "total" in ev and all(torch.equal(w_before[k], v) for k, v in model2.state_dict().items() if k in w_before)
        # 4b sequence training: train() on clips of 2 keyframes; validation with the paired warm / cold last keyframe
        scfg = dataclasses.replace(cfg, name="seqrun", seq_len=2, batch_size=1, accum=1, steps=2, save_every=0, eval_every=0, warmup=1)
        model_s, loss_s, ema_s, last_s = train(scfg, datasets=(base, base))
        assert last_s["total"] == last_s["total"], "logged"
        sl = make_loader(ClipDataset(base, 2), dataclasses.replace(scfg, batch_size=2), False)
        ev_s = evaluate(model_s, loss_s, ema_s, sl, 1, "cpu", "off", None, scfg)
        assert "total_last_warm" in ev_s and "total_last_cold" in ev_s and ev_s["total_last_warm"] != ev_s["total_last_cold"], ev_s
        # 5 schedule
        assert abs(lr_factor(0, 10, 100, 0.01) - (1 / 3 + 2 / 30)) < 1e-9 and abs(lr_factor(9, 10, 100, 0.01) - 1.0) < 1e-9
        assert abs(lr_factor(100, 10, 100, 0.01) - 0.01) < 1e-9
        print(f"train selftest ok: GT at three times (-0.5 / 0 / +0.5 s) -> loss, balancing arms (Kendall / fixed / UW-SO), "
              f"cross-time consistency (hold = weight 0, gradient reaches the motion head), gradient-conflict cosines, "
              f"sequence training (clips of consecutive keyframes, memory carried and detached or kept in the graph, paired "
              f"warm / cold validation), 2 micro-batches per step, 4 steps with checkpoints (frozen heads "
              f"not saved), resume 4 -> 6, validation under EMA restores the weights, schedule; "
              f"{last['s_per_step']:.1f} s per step on the CPU at batch 2 (cached features, width 1/4)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as fh:
            raw = json.load(fh)
        train(TrainConfig(**raw))
    else:
        _selftest()
