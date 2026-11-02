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
import contextlib
import copy
import dataclasses
import functools
import inspect
import json
import math
import os
import pickle
import sys
import time
import zipfile
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

try:
    from .augment import flip_batch
    from .feature_cache import CachedBEVFusionModel, CachedFrontRadarDataset, FeatureStore, collate_cached
    from . import freeze as freeze_mod
    from .probe import ModuleProbe, StepProfiler, StepTimer, batch_tokens, dump_incident
    from .interface import check_inputs, check_outputs
    from .loss_extras import UWSO, cross_time_consistency, group_gradient_cosines
    from .losses import BEVLoss, LossConfig, build_layer_targets, sample_targets
    from .model import CAMERAS, BEVFusionModel, ModelConfig
    from .nuscenes_dataset import FrontRadarDataset, available_scenes, collate
    from .radar_encoder import aux_weight
except ImportError:                                   # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from augment import flip_batch
    from feature_cache import CachedBEVFusionModel, CachedFrontRadarDataset, FeatureStore, collate_cached
    import freeze as freeze_mod
    from probe import ModuleProbe, StepProfiler, StepTimer, batch_tokens, dump_incident
    from interface import check_inputs, check_outputs
    from loss_extras import UWSO, cross_time_consistency, group_gradient_cosines
    from losses import BEVLoss, LossConfig, build_layer_targets, sample_targets
    from model import CAMERAS, BEVFusionModel, ModelConfig
    from nuscenes_dataset import FrontRadarDataset, available_scenes, collate
    from radar_encoder import aux_weight


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
    feature_cache2: str = ""                     # second feature cache (BEVFusion, camera_features "dual", t10; "" = off)
    radar_cache: str = ""                        # per-sample radar cache folder (feature_cache.py; "" = off; exact)
    lane_images: str = ""                        # key-frame image cache of the lane branch (lane_branch.py, t7; "" = off)
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
    radar_aux: float = 0.0                       # radar_kind "delta_slots": weight of the radar encoder's reconstruction SIDE loss at the start of
                                                 # the training (radar_encoder.aux_weight: floor + (start - floor) * 0.5 ** (progress / half)); it
                                                 # helps the first phase, the task losses are the goal; 0 = off (kept-fraction not computed either)
    radar_aux_floor: float = 0.05                # weight it decays to
    radar_aux_half: float = 0.2                  # half-life in fractions of the training
    grad_diag_every: int = 0                     # log the cosines of the task-group gradients at the BEV feature every N steps
    seq_len: int = 1                             # keyframes per clip: > 1 trains on clips of consecutive keyframes of one scene with
                                                 # the world memory carried from keyframe to keyframe (batch_size = clips per
                                                 # micro-batch, so a micro-batch has batch_size * seq_len samples); 1 = independent
    seq_bptt: bool = False                       # backpropagate through the carried memory over the clip (graphs of all keyframes
                                                 # stay in memory); False = the state is detached at every keyframe
    gap_ego: bool = True                         # decode dt = +-0.5 s with each sample's true keyframe gap and gap-mean ego
                                                 # motion (needs `ego=` in the model's forward; 0.2-0.35 m of label error otherwise)
    flip_aug: float = 0.0                        # probability that a sample (a whole clip) is mirrored left-right in training
                                                 # (training/augment.py: BEV, radar, ego motion and every target); 0 = off
    weight_decay: float = 1e-2
    grad_clip: float = 5.0
    grad_clip_split: tuple = ()                  # module prefixes (as freeze / train_only) whose gradients are clipped to grad_clip ON THEIR OWN, the rest as one group: a block whose
                                                 # gradient is far above the others' (the motion block in t16: 36 - 131 against 0.1 - 17) no longer sets the clip factor of the whole
                                                 # network; () = one global clip as before (bit-identical)
    ema: float = 0.999
    amp: str = "off"                             # "off" | "bf16" | "fp16" (CUDA only). fp16 trains with a GradScaler (dynamic loss scaling). bf16 as it is: ledger row speed.c2 (an 8-step diagnostic, with cudnn.benchmark): no speed-up, gradient norm 25x fp32
    fp16_scale: float = 1024.0                   # amp "fp16": the start scale of the GradScaler (halved on an overflow; a power of two scales exactly)
    fp16_growth_interval: int = 2000             # amp "fp16": the scale doubles after that many clean steps in a row (an overflow restarts the count); the torch default
    log_every: int = 20
    eval_every: int = 1000
    eval_batches: int = 50
    save_every: int = 500
    keep: int = 3
    slim_at_end: bool = False                    # True: a run that finishes keeps ONE checkpoint without optimiser state and RNG (about half
                                                 # the size, no older ones): it evaluates and warm-starts but cannot resume training.
                                                 # False (default; Mirko 2026-11-01: keep the full data to resume later): `keep` complete ones
    max_bad_steps: int = 20                     # consecutive non-finite steps before aborting
    train_only: tuple = ()                       # module prefixes that train, everything else is frozen (freeze.py; () = everything trains)
    freeze: tuple = ()                           # module prefixes that are frozen (freeze.py); with train_only it takes from it. Not in RESUME_FREE: a resume needs the same plan
    pin_memory: bool = False                     # DataLoader with pinned host memory and non-blocking copies to the GPU (cuda only; same numbers; NOT measured: the pinning thread shares the GIL with the host-bound main thread)
    prefetch_factor: int = 0                     # batches each loader worker prepares ahead (0 = the torch default, 2); only with workers > 0
    cudnn_benchmark: bool = False                # torch.backends.cudnn.benchmark: the fastest convolution algorithm per input shape (shapes are static here); the last bits may differ
    tf32: bool = False                           # torch.set_float32_matmul_precision("high"): TF32 for fp32 matrix products (torch already runs convolutions in TF32 by default); changes the numbers slightly
    profile_steps: int = 0                       # > 0: torch.profiler over that many optimiser steps (at most 8: 30 steps held 22 GB of host RAM) after 10 warm-up steps; writes profile.txt into the run folder (probe.StepProfiler)
    probe: bool = False                          # log grad.<module> / upd.<module> (gradient norm, update-to-weight ratio) and t_data / t_step in the rows of log_every,
                                                 # dump incident_<step>.json on a skipped step (probe.py; costs one copy of the trainable weights per logged step)
    seed: int = 0
    init_from: str = ""                          # checkpoint of another run to start from (warm_start; only while this run has
                                                 # no checkpoint of its own; optimiser, step and schedule start fresh), "" = off
    init_reset: tuple = ()                       # key prefixes that keep their fresh initialisation in that warm start (t11: a dead
                                                 # branch, e.g. "obj_head.head.vel.")
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
    if mcfg.camera_features != "dualbev" and not cached:
        raise ValueError(f"camera_features={mcfg.camera_features!r} needs a feature_cache")
    if (("feature_cache2" in {s.cache for s in CAMERAS[mcfg.camera_features]}) != bool(cfg.feature_cache2)):
        raise ValueError(f"camera_features {mcfg.camera_features!r} and feature_cache2 do not match: the kinds with a second feature source "
                         "('dual') need feature_cache2 (the BEVFusion cache), and feature_cache2 needs such a kind")
    if mcfg.lane_branch and cached and not cfg.lane_images:
        raise ValueError("model.lane_branch with a feature_cache needs lane_images (lane_branch.py build-cache)")
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
                                               store=FeatureStore(cfg.feature_cache), radar_cache=cfg.radar_cache or None,
                                               camera_features=cfg.model.get("camera_features", "dualbev"),
                                               lane_images=cfg.lane_images or None,
                                               store2=FeatureStore(cfg.feature_cache2) if cfg.feature_cache2 else None, **kw))
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


def make_loader(ds, cfg, train, workers=None, generator=None):
    """workers: default cfg.workers. The validation loader of train() gets at most 2 workers that are not kept alive between evaluations,
    and its own generator: a worker that builds a missing GT loads the map files, and t7_rgb_s1 died of ENOMEM in validation worker 4 at
    its first evaluation (2026-10-31); with the generator an evaluation no longer draws from the training's random stream."""
    fn = collate_cached if cfg.feature_cache else collate
    if cfg.seq_len > 1:
        fn = functools.partial(collate_clips, fn=fn)
    n = cfg.workers if workers is None else workers
    extra = {}
    if cfg.pin_memory and torch.cuda.is_available():
        extra["pin_memory"] = True
    if cfg.prefetch_factor > 0 and n > 0:
        extra["prefetch_factor"] = cfg.prefetch_factor
    return torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=train, num_workers=n, collate_fn=fn, drop_last=train,
                                       persistent_workers=train and n > 0, generator=generator, **extra)


def to_device(batch, device, non_blocking=False):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        elif k == "radar":
            out[k] = {a: b.to(device, non_blocking=non_blocking) for a, b in v.items()}
        elif k == "steps":
            out[k] = [[tuple(t.to(device, non_blocking=non_blocking) for t in s) for s in fr] for fr in v]
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

    def __init__(self, model, decay, skip=()):
        self.decay = decay
        self.skip = frozenset(skip)                              # state keys that a freeze plan fixed: their average stays the value itself (freeze.frozen_keys)
        self.shadow = {k: v.detach().clone() for k, v in self._tensors(model).items()}
        self.backup = None

    @staticmethod
    def _tensors(model):
        return {k: v for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))
        live = [(k, v) for k, v in self._tensors(model).items() if k not in self.skip]
        full = [(k, v) for k, v in live if v.dtype in (torch.float32, torch.float64)]
        if full:                                                 # two multi-tensor calls instead of two kernels per state tensor (218 float tensors of the t10 model: 436 launches of the ~20000 per step)
            shadow = [self.shadow[k] for k, _ in full]
            torch._foreach_mul_(shadow, d)
            torch._foreach_add_(shadow, [v.detach() for _, v in full], alpha=1 - d)
        for k, v in live:                                        # a multi-tensor call rounds the scalar to the tensor's dtype (bf16 / fp16 state would be averaged with 0.98828125 for 0.99): those take the loop
            if v.dtype not in (torch.float32, torch.float64):
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)

    def apply(self, model):
        self.backup = {k: v.detach().clone() for k, v in self._tensors(model).items()}
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=True)

    def restore(self, model):
        model.load_state_dict({**model.state_dict(), **self.backup}, strict=True)
        self.backup = None


def camera_frozen(model):
    """True if the camera heads are frozen (they come from the pre-trained file); a model without that switch counts as frozen"""
    return bool(getattr(getattr(model, "cfg", None), "freeze_camera", True))


def trainable_state(model, loss):
    """Everything that is not the frozen camera heads (those come from the pre-trained file). With freeze_camera false the camera
    weights are trained, so they belong to the state: a resume must restore them, not start them from the pre-trained file again."""
    frozen = camera_frozen(model)
    sd = {"model." + k: v for k, v in model.state_dict().items() if not (frozen and k.startswith("camera."))}
    sd.update({"loss." + k: v for k, v in loss.state_dict().items()})
    return sd


def save_checkpoint(path, model, loss, opt, ema, step, cfg, extra=None, slim=False):
    ck = {"state": trainable_state(model, loss), "opt": opt.state_dict(),
          "ema": {k: v.cpu() for k, v in ema.shadow.items()}, "step": step, "cfg": dataclasses.asdict(cfg),
          "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}, "extra": extra or {}}
    if slim:                                                    # weights only (finished run): no optimiser state, no RNG
        del ck["opt"], ck["rng"]
        ck["slim"] = True
    tmp = path + ".tmp"
    torch.save(ck, tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())                                   # on the disk before os.replace and before older checkpoints are pruned
    os.replace(tmp, path)


def load_checkpoint(path, model, loss, opt, ema, ck=None):
    ck = ck if ck is not None else torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state"]
    m = model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
    assert all(k.startswith("camera.") for k in m.missing_keys) and not m.unexpected_keys, m
    if m.missing_keys and not camera_frozen(model):
        raise RuntimeError(f"{path} has no camera weights ({len(m.missing_keys)} missing) but this model trains them (freeze_camera false): "
                           "a checkpoint of the code from before 2026-10-31 cannot be resumed, train again or freeze the camera")
    loss.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("loss.")})
    if "opt" in ck:                                             # a slim checkpoint (finished run) has no optimiser state and no RNG
        opt.load_state_dict(ck["opt"])
    ema.shadow = {k: v.to(next(model.parameters()).device) for k, v in ck["ema"].items()}
    if "rng" in ck:
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
    return ck["step"], ck.get("extra", {})


def warm_start(path, model, loss, ck=None, reset=()):
    """TrainConfig.init_from (t11): the model takes the checkpoint's EMA tensors (the evaluated weights; the integer buffers, which the
    EMA does not average, from its state) for every key it shares with them; a shape mismatch or a checkpoint key the model does not
    have raises; the model's keys the checkpoint lacks (a new branch) and the keys under a `reset` prefix (TrainConfig.init_reset; a
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


RESUME_FREE = ("name", "out_root", "data_root", "feature_cache", "gt_cache", "feature_cache2", "radar_cache", "lane_images", "workers",
               "device", "log_every", "eval_every", "eval_batches", "save_every", "keep", "slim_at_end", "max_bad_steps", "grad_diag_every", "probe", "pin_memory", "prefetch_factor", "profile_steps")


def check_resume(old, cfg, name):
    """refuses to resume from a checkpoint made with another configuration: a resumed run must be the run it continues. A changed
    `steps` restarts the cosine schedule high (lr factor 0.001 at step 2999 of 3000 -> 0.54 at step 3000 of 6000), a changed model,
    loss or data trains another experiment under the old name. Paths, workers and the logging / saving cadence (RESUME_FREE) may
    change; BF_RESUME_ANY=1 overrides the refusal (review 2026-10-31 pass 2)."""
    norm = lambda d: json.loads(json.dumps(d, default=str))                 # noqa: E731 - tuples and lists compare equal
    a, b = norm(old), norm(dataclasses.asdict(cfg))
    diff = sorted(k for k in a if k not in RESUME_FREE and a[k] != b.get(k))
    diff += [k for k in ("train_only", "freeze") if k not in a and b.get(k)]               # a checkpoint from before freeze.py has no plan: a new one cannot resume it (its optimiser state has a slot per parameter)
    diff += [k for k in ("cudnn_benchmark", "tf32") if k not in a and b.get(k)]            # a checkpoint from before these switches ran without them: turning one on changes the numbers in the middle of a run
    if diff and os.environ.get("BF_RESUME_ANY") != "1":
        raise SystemExit(f"{name}: the configuration differs from the checkpoint's in {diff}; start a new run name "
                         f"(or set BF_RESUME_ANY=1)")


def refuse_slim_resume(ck, step, cfg, name):
    """a slim checkpoint (a finished run: no optimiser state) evaluates and warm-starts but cannot continue training: a resume below the
    configured end is refused, at the end (a re-invocation of a finished run) it is accepted"""
    if ck.get("slim") and step < cfg.steps:
        raise SystemExit(f"{name} is a slim checkpoint (no optimiser state): it evaluates and warm-starts (init_from), it cannot "
                         f"continue training; start a new run name")


def supersede_log(out_dir, step):
    """on a resume from `step`: the rows of log.jsonl above it (the steps redone after a crash) move to log_superseded.jsonl, so the log
    has one row per step again (before 2026-10-31 a resumed run had steps 520 - 1000 twice)"""
    path = os.path.join(out_dir, "log.jsonl")
    if not os.path.exists(path):
        return
    keep, gone = [], []
    for ln in open(path, encoding="utf-8"):
        try:
            s = json.loads(ln).get("step", 0)
        except ValueError:                                     # a row cut by the crash
            s = step + 1
        (gone if s > step else keep).append(ln if ln.endswith("\n") else ln + "\n")
    if gone:
        with open(os.path.join(out_dir, "log_superseded.jsonl"), "a", encoding="utf-8") as f:
            f.writelines(gone)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            f.writelines(keep)
        try:
            os.replace(path + ".tmp", path)
        except OSError as e:                                   # a reader on the Windows side holds log.jsonl open: keep appending
            os.remove(path + ".tmp")
            print(f"WARNING: log.jsonl not rewritten ({type(e).__name__}: {e}); its rows above step {step} stay in it", flush=True)


def val_logged(out_dir, step):
    """True if log.jsonl has the validation row of `step`"""
    path = os.path.join(out_dir, "log.jsonl")
    if not os.path.exists(path):
        return False
    for ln in open(path, encoding="utf-8"):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("step") == step and "val" in r:
            return True
    return False


# ------------------------------------------------------------------------------------------------ loop
def _f(v):
    return float(v.detach()) if torch.is_tensor(v) else float(v)


def _pack(out, selected):
    """snapshot of the logged scalar terms of ONE keyframe: (order, packed). The tensors are stacked NOW into one device tensor (a term that is a view of a parameter, which a
    later keyframe changes in place, must be read as it is at this point; the stack is a copy), numbers are kept. Nothing is read back here: the old code called float() per
    term, and every call waited for the GPU, which was still busy with the backward of this keyframe, so the host could not queue the next keyframe meanwhile. The terms are
    widened to float64 (exact for float16 / bfloat16 / float32 / float64, as float(v) was)"""
    order, tens = [], []
    for k, v in out.items():
        if selected(k):
            if torch.is_tensor(v):
                order.append((k, True, len(tens)))
                tens.append(v.detach().double().reshape(()))
            else:
                order.append((k, False, float(v)))
    return order, (torch.stack(tens) if tens else None)


def _unpack(kf, L):
    """-> (acc, totals) of the packed keyframes with ONE device-to-host transfer. Bit-identical to the per-term floats: the same float32 values widened to Python floats, summed in
    the same order (keyframe by keyframe, term by term)"""
    packs = [p for _, p in kf if p is not None]
    flat = torch.cat(packs).cpu().tolist() if packs else []
    acc, totals, pos = {}, [], 0
    for order, p in kf:
        base = pos
        pos += 0 if p is None else p.numel()
        for k, is_t, x in order:
            val = flat[base + x] if is_t else x
            acc[k] = acc.get(k, 0.0) + val / L
            if k == "total":
                totals.append(val)
    return acc, totals


def consistency_weight(cfg, step_frac):
    """cfg.consistency x a linear ramp: 0 during the hold (first `hold` of the steps), 1 after hold + ramp."""
    hold, ramp = cfg.consistency_warmup
    return cfg.consistency * min(1.0, max(0.0, (step_frac - hold) / max(ramp, 1e-6)))


def consistency_term(model, loss, res, gts, gap_steps, device):
    """Train-only cross-time consistency of the predictions at dt = 0 and dt = +-0.5 s (loss_extras): the layers whose
    kind is area / line / object, the visibility of the cells = camera and not occluded, the ego steps (batch["gap_steps"]) and the
    motion field of the batch. -> the result dict of cross_time_consistency."""
    sel = [i for i, (_, kind, _) in enumerate(loss.spec) if kind in ("area", "line", "object")]
    out = res["out"]
    probs = {dt: torch.sigmoid(o["sem"][:, sel].float()) for dt, o in out.items()}
    weights = {dt: build_layer_targets(gts[dt], loss.cfg, out[dt].get("mass"), dt, device)["w_cam"] for dt in out}
    steps = {dt: [g["prev" if dt < 0 else "next"] for g in gap_steps] for dt in out if dt != 0.0}
    return cross_time_consistency(probs, steps, out[0.0]["v_hat"].float(), model.grid, weights)


def gap_ego_motion(steps):
    """The ego steps [(dt_i, v_i, omega_i)] (batch["gap_steps"]: time 0 -> the target keyframe, negative dt backwards) as ONE
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


def batch_ego(gap_steps, dts, device):
    """{nominal dt != 0: (dt, v, omega)} as (B,) tensors: the true gap and the gap-mean ego motion of every sample, from batch["gap_steps"]
    (the dataset makes them from the poses: the network input needs no ground truth)."""
    ego = {}
    for dt in dts:
        if abs(dt) < 1e-6:
            continue
        key = "prev" if dt < 0 else "next"
        vals = [gap_ego_motion(g[key]) for g in gap_steps]
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
    """(dt, v, omega) as (B,) tensors from the previous keyframe of each clip to the current one: the ego steps of the
    previous batch toward its next keyframe (batch["gap_steps"], from the poses), composed into one splat step; the model's memory pose
    advances by it."""
    vals = [gap_ego_motion(g["next"]) for g in prev_batch["gap_steps"]]
    return tuple(torch.tensor([v[i] for v in vals], dtype=torch.float32, device=device) for i in range(3))


AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def amp_context(amp, device):
    """the autocast context of the forward pass: bf16 / fp16 on CUDA, nothing otherwise. An unknown name is an error (it used to mean `off`)"""
    if amp != "off" and amp not in AMP_DTYPES:
        raise ValueError(f"amp must be one of off, {', '.join(AMP_DTYPES)}; got {amp!r}")
    if amp in AMP_DTYPES and device.startswith("cuda"):
        return torch.autocast("cuda", dtype=AMP_DTYPES[amp])
    return torch.autocast("cpu", enabled=False)


def make_scaler(cfg, device):
    """amp 'fp16': a GradScaler (the fp16 range is small: gradients below ~6e-8 vanish, values above 65504 overflow, so the loss is scaled and the gradients are
    unscaled before the clip); None otherwise (bf16 has the range of fp32). An unknown amp name is an error here, before the model and the data are built.
    NOT covered: an overflow in the FORWARD pass (activations above 65504) is no scaler event, it gives a non-finite loss and leaves non-finite batch-norm statistics (see poisoned_buffers)"""
    amp_context(cfg.amp, device)
    if cfg.amp != "fp16":
        return None
    return torch.amp.GradScaler("cuda" if device.startswith("cuda") else "cpu", init_scale=cfg.fp16_scale, growth_interval=cfg.fp16_growth_interval)


def poisoned_buffers(model):
    """names of the float buffers (batch-norm statistics ...) that are not finite: a forward pass that overflowed in fp16 writes inf / nan into them, and the EMA would carry it on"""
    return [k for k, v in model.named_buffers() if v.is_floating_point() and not torch.isfinite(v).all()]


def scaler_extra(scaler):
    """the part of a checkpoint's `extra` that holds the scaler (its scale and growth counter), {} without one"""
    return {"scaler": scaler.state_dict()} if scaler is not None else {}


def clip_groups(model, prefixes):
    """the trainable parameters of each module prefix of `grad_clip_split` (a prefix matches the module itself and its children, never `motion_blockx`); a prefix that names none is an error"""
    groups = []
    for pre in prefixes:
        ps = [p for n, p in model.named_parameters() if p.requires_grad and (n == pre or n.startswith(pre.rstrip(".") + "."))]
        if not ps:
            raise ValueError(f"grad_clip_split: the prefix {pre!r} names no trainable parameter")
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
    return torch.linalg.vector_norm(torch.stack([n.reshape(()) for n in norms]))


def optimizer_step(opt, params, grad_clip, scaler=None, before_clip=None, split=()):
    """gradient clip and optimiser step of one training step. -> (grad norm before the clip, status). Without a scaler this is what the loop did inline: clip, and step unless the
    norm is not finite ("nonfinite": no step, gradients dropped). With a scaler the gradients are unscaled first (before_clip, the probe, sees them unscaled), and a non-finite
    gradient is an "overflow": no step, the scale is halved (an expected event of fp16, not a diverged run). "ok": the step was made (and the scale updated)"""
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


_CONTRACT_CHECKED = []      # interface.check_inputs / check_outputs run on the first forward_loss call of the process


def forward_loss(model, loss, batch, step_frac, device, amp, cfg=None, diag=False, train_terms=True, state=None, step=None):
    """-> (loss dict, model result, GT targets). With cfg.gap_ego the model decodes dt = +-0.5 s with the TRUE gap of each
    sample and its gap-mean ego motion (the GT is the keyframe at the true gap), when the model takes `ego`. With
    cfg.consistency > 0 (and train_terms) the cross-time consistency is added (with its own warm-up); with diag the
    cosines of the task-group gradients at the BEV feature are put into out['diag']. state / step: the model state of the
    previous keyframe of the clip and the ego step since then (sequence training)."""
    gts = sample_targets(batch["gt"])
    dts = tuple(sorted(gts))
    ego = batch_ego(batch["gap_steps"], dts, device) if cfg is not None and cfg.gap_ego and supports_ego(model) else None
    batch = to_device(batch, device, non_blocking=bool(cfg is not None and cfg.pin_memory))
    ctx = amp_context(amp, device)
    kw = {k: v for k, v in (("ego", ego), ("state", state), ("step", step)) if v is not None}
    first = not _CONTRACT_CHECKED
    if first:                                                  # the interface contract on the first real batch of the process
        check_inputs(batch, dts=dts, ego=ego, raise_on=True)   # (training, validation and the evaluation adapter all pass here)
    with ctx:
        res = model(batch, dts=dts, **kw)
    if first:
        check_outputs(res, dts, raise_on=True)
        _CONTRACT_CHECKED.append(True)
    with torch.autocast("cpu", enabled=False), torch.autocast("cuda", enabled=False):
        preds = {dt: {k: ({a: b.float() for a, b in v.items()} if isinstance(v, dict) else v.float()) for k, v in p.items()}
                 for dt, p in res["out"].items()}                                    # already in the loss's layout
        out = loss(preds, gts, step_frac)
        out["total_main"] = out["total"].detach()                                 # without the train-only terms
        wc = consistency_weight(cfg, step_frac) if cfg is not None and train_terms and len(dts) > 1 else 0.0
        if wc > 0:                                                                 # nothing is computed during the hold
            ct = consistency_term(model, loss, res, gts, batch["gap_steps"], device)
            out["consistency"] = ct["total"]
            out["total"] = out["total"] + wc * ct["total"]
        side = res.get("radar_side")                                               # radar_kind "delta_slots" only
        if side is not None and cfg is not None and cfg.radar_aux > 0:
            enc = model.radar
            f_side, s_side = side["features"].float(), side["state"].float()
            if train_terms:                                                        # the SIDE loss: helps early, decays to a floor
                la = enc.aux_loss(f_side, s_side)
                out["radar_aux"] = la
                out["total"] = out["total"] + aux_weight(step_frac, cfg.radar_aux, cfg.radar_aux_floor, cfg.radar_aux_half) * la
            else:                                                                  # validation: how much of the detections the features hold
                out["radar_kept"] = enc.kept_fraction(batch["radar"], f_side.shape[0], features=f_side)[0]
        if diag:
            groups = {k[len("group."):]: v for k, v in out.items() if k.startswith("group.") and torch.is_tensor(v) and v.requires_grad}
            if wc > 0 and out["consistency"].requires_grad:      # its gradient norm at the BEV feature calibrates its weight
                groups["consistency"] = out["consistency"]
            if len(groups) > 1 and res["features"].requires_grad:      # a frozen trunk gives the groups no common feature to differentiate by
                names, C, nrm = group_gradient_cosines(groups, res["features"])
                out["diag"] = {"groups": names, "cos": C.tolist(), "norm": nrm.tolist()}
    return out, res, gts


LOGGED_TERMS = ("objhead.vel@+0.0", "objhead.mov@+0.0", "motion@+0.0", "radar_aux", "radar_kept")     # single terms logged (train and val) next to the groups:
                                                                             # the velocity terms (t11), hidden in group.objects / motion


def run_clip(model, loss, clip, step_frac, device, amp, cfg, diag=False, train_terms=True, backward=False, augment=False, scaler=None):
    """One clip = a list of batches (consecutive keyframes of the same scenes; a single batch = a clip of length 1). The
    model state (world memory) is carried from keyframe to keyframe with the ego step between them: detached at every
    keyframe, or with cfg.seq_bptt kept in the graph through the clip. backward: backpropagate the mean loss of the clip
    divided by cfg.accum, right after each keyframe (with seq_bptt the earlier graphs are retained for the state path of
    the later keyframes, so the gradient is that of the clip's mean loss). scaler: the GradScaler of amp "fp16": the loss is scaled before the backward (the gradients stay
    scaled until optimizer_step unscales them); the logged terms are the unscaled ones.
    -> (dict of the scalar terms averaged over the keyframes, finite flag, diag of the last keyframe or None, list of the
    per-keyframe totals)."""
    L, state, prev, kf, dg = len(clip), None, None, [], None
    selected = lambda k: k == "total" or k == "consistency" or k.startswith(("group.", "s.")) or k in LOGGED_TERMS          # noqa: E731
    bptt, accum = bool(cfg is not None and cfg.seq_bptt), (cfg.accum if cfg is not None else 1)
    if augment and cfg is not None and cfg.flip_aug > 0:         # mirror a sample in ALL its keyframes (same memory frame)
        flip = torch.rand(len(clip[0]["tokens"])) < cfg.flip_aug
        for batch in clip:
            flip_batch(batch, flip)
    for t, batch in enumerate(clip):
        step = clip_step(prev, device) if prev is not None else None
        out, res, _ = forward_loss(model, loss, batch, step_frac, device, amp, cfg, diag=diag and t == L - 1,
                                   train_terms=train_terms, state=state, step=step)
        if not torch.isfinite(out["total"]):
            acc, totals = _unpack(kf, L)
            return acc, False, dg, totals
        kf.append(_pack(out, selected))                           # before the backward, which changes the Kendall weights in place (clamp_)
        if backward:           # backward at once: the loss of a later keyframe changes the Kendall weights in place (clamp_)
            mean_loss = out["total"] / (L * accum)
            (mean_loss if scaler is None else scaler.scale(mean_loss)).backward(retain_graph=bptt and t < L - 1)    # bptt: the graph is still needed for the state path
        if "diag" in out:
            dg = out["diag"]
        state = res["state"] if bptt else detach_tree(res["state"])
        prev = batch
    acc, totals = _unpack(kf, L)                                  # the only synchronisation of the clip besides the finite check of each keyframe
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
                    if k == "total" or k.startswith("group.") or k in LOGGED_TERMS:
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


EXIT_STOPPED = 75                      # a stopped run exits with this code and never with 0: the flag `.done` means "finished correctly"


def _stop_requested(run_dir):
    """reason (a string) when the file CONTROL of the run directory says {"cmd": "stop"}, else None. Same code as
    autoresearch/trainsuite/control.py (inlined: a run trains from a code snapshot that does not contain that package)."""
    try:
        with open(os.path.join(run_dir, "CONTROL"), encoding="utf-8") as fh:
            d = json.load(fh)
        return str(d.get("reason", "")) if d.get("cmd") == "stop" else None
    except (OSError, ValueError, AttributeError):
        return None


def _clear_control(run_dir):
    """at the start: move CONTROL / STOPPED of an earlier stop aside (*.old), so a new process never obeys an old request (no clocks)"""
    for n in ("CONTROL", "STOPPED"):
        p = os.path.join(run_dir, n)
        if os.path.exists(p):
            os.replace(p, p + ".old")


def record_gt(out_dir, datasets):
    """runs/<name>/gt.json = {"gt_signature": ...}: the identity of the GT the run trains and validates on (FrontRadarDataset.gt_signature:
    grid res, supersampling, autogt source + flags). A pinned code snapshot does not pin the GT (autogt is read live), so this file is
    what `evaluation/bev_suite.py compare` checks. A resume on another GT keeps the earlier signatures and warns. -> the signature(s)"""
    sigs = sorted({d.gt_signature() for d in datasets if hasattr(d, "gt_signature")})
    if not sigs:
        return None
    sig = sigs[0] if len(sigs) == 1 else sigs
    path = os.path.join(out_dir, "gt.json")
    try:                                                      # D29: the human key of the GT version (v5, ...) next to the signature; none for an unreleased source
        from autogt.versions import version_of_key
        ver = [version_of_key(s) for s in sigs]
        ver = None if None in ver else (ver[0] if len(ver) == 1 else ver)
    except ImportError:
        ver = None
    rec = {"gt_signature": sig, **({"gt_version": ver} if ver else {})}
    if not os.path.exists(path) and any(f.startswith("ckpt_") and f.endswith(".pt") for f in os.listdir(out_dir)):
        rec["earlier"] = ["unknown: the run has checkpoints from before gt.json existed"]   # a resume of a run started before 2026-10-31
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        rec["earlier"] = list(old.get("earlier", []))
        if old.get("gt_signature") != sig:
            rec["earlier"].append(old.get("gt_signature"))
            print(f"WARNING: the GT changed since this run started ({old.get('gt_signature')} -> {sig}): the run mixes two GT versions",
                  flush=True)
        if not rec["earlier"]:
            del rec["earlier"]
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(path + ".tmp", path)
    return sig


def check_grids(model, *datasets):
    """The model and the GT of its datasets must share one cell grid (extents and resolution). `cfg.model["grid_kwargs"]` changes the
    model's grid only; the datasets are built with the default front grid, so another model grid would train on labels cut for
    another area, with the same shapes and no error."""
    g = model.grid
    for ds in datasets:
        d = getattr(ds, "grid", None) or getattr(getattr(ds, "base", None), "grid", None)
        if d is not None and (d.x_min, d.x_max, d.y_min, d.y_max, d.res) != (g.x_min, g.x_max, g.y_min, g.y_max, g.res):
            raise ValueError(f"the model grid x {g.x_min}..{g.x_max} y {g.y_min}..{g.y_max} res {g.res} is not the grid of the GT "
                             f"x {d.x_min}..{d.x_max} y {d.y_min}..{d.y_max} res {d.res} (make_datasets builds the default grid)")


def apply_speed_flags(cfg):
    """the opt-in speed switches of the config; all off by default and then nothing is touched. They are process-wide and set once at the start of train() (an eval_fn in the
    same process runs with them). -> {name: value} of what was set"""
    done = {}
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        done["cudnn.benchmark"] = True
    if cfg.tf32:
        torch.set_float32_matmul_precision("high")
        done["tf32"] = "high"
    return done


def train(cfg, eval_fn=None, datasets=None):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    speed = apply_speed_flags(cfg)
    if speed:
        print("speed flags:", speed, flush=True)
    out_dir = os.path.join(cfg.out_root, cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    _clear_control(out_dir)
    model, loss, _ = build(cfg)
    model, loss = model.to(device), loss.to(device)
    if cfg.train_only or cfg.freeze:                            # one module trains, the rest is frozen (freeze.py): before the optimiser is built
        freeze_mod.apply(model, cfg.train_only, cfg.freeze)
        print("freeze:", freeze_mod.signature(cfg.train_only, cfg.freeze), flush=True)
        print(freeze_mod.report(model), flush=True)
    train_ds, val_ds = datasets or make_datasets(cfg)
    check_grids(model, train_ds, val_ds)
    record_gt(out_dir, (train_ds, val_ds))                      # the GT is not pinned by the code snapshot: record which one is used
    if cfg.seq_len > 1:                                         # clips of consecutive keyframes, the memory carried through
        train_ds, val_ds = ClipDataset(train_ds, cfg.seq_len), ClipDataset(val_ds, cfg.seq_len)
    train_loader = make_loader(train_ds, cfg, True)
    val_loader = make_loader(val_subset(val_ds, cfg.eval_batches * cfg.batch_size), cfg, False,      # fixed, spread over all scenes
                             workers=min(cfg.workers, 2), generator=torch.Generator().manual_seed(cfg.seed))
    if cfg.gap_ego and not supports_ego(model):
        print("WARNING: gap_ego is on but model.forward has no `ego` argument: dt = +-0.5 s is decoded with the nominal "
              "gap and the last step's ego motion (0.2-0.35 m of label error)", flush=True)
    opt = torch.optim.AdamW(param_groups(model, loss, cfg.lr, cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.999))
    scaler = make_scaler(cfg, device)                           # amp "fp16" only
    if scaler is not None and cfg.grad_diag_every:
        print("WARNING: amp fp16: grad_diag_every is ignored (the group gradients would be taken unscaled and underflow)", flush=True)
    frozen = freeze_mod.frozen_keys(model) if (cfg.train_only or cfg.freeze) else ()
    ema = EMA(model, cfg.ema, skip=frozen)
    step, damaged, was_slim = 0, [], False
    ckpts = sorted(f for f in os.listdir(out_dir) if f.startswith("ckpt_") and f.endswith(".pt"))
    for name in reversed(ckpts):                                # the newest READABLE checkpoint: a damaged one falls back to the one before
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
        step, ck_extra = load_checkpoint(os.path.join(out_dir, name), model, loss, opt, ema, ck=ck)
        if scaler is not None and ck_extra.get("scaler"):        # the scale it had reached, not the start scale again
            scaler.load_state_dict(ck_extra["scaler"])
        refuse_slim_resume(ck, step, cfg, name)
        was_slim = bool(ck.get("slim"))
        print(f"resumed from {name} at step {step}", flush=True)
        break
    else:
        if damaged:
            raise SystemExit(f"no readable checkpoint in {out_dir} (damaged: {damaged}): a restart at step 0 would mix two runs in "
                             f"one folder")
        if cfg.init_from:                                       # a new run warm-started from another run's checkpoint
            fresh = warm_start(cfg.init_from, model, loss, reset=cfg.init_reset)
            lost = sorted(set(fresh) & set(frozen))
            if lost:                                            # a frozen weight that the checkpoint does not have stays what it was initialised to
                print(f"WARNING: {len(lost)} frozen state key(s) are not in the warm-start checkpoint (or under init_reset) and stay at their random initialisation, e.g. {lost[:3]}", flush=True)
            ema = EMA(model, cfg.ema, skip=frozen)
            torch.manual_seed(cfg.seed)                         # the same data order and flips whatever the new branches drew
            np.random.seed(cfg.seed)
            print(f"warm start from {cfg.init_from}; fresh keys: {fresh}", flush=True)
    supersede_log(out_dir, step)                                # also after a crash before the first save (rows of the lost steps)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=1)
    log = open(os.path.join(out_dir, "log.jsonl"), "a")
    if step and cfg.eval_every and step % cfg.eval_every == 0 and not val_logged(out_dir, step):   # the run died inside the evaluation
        ev = evaluate(model, loss, ema, val_loader, 10 ** 9, device, cfg.amp, eval_fn, cfg)            # of this step (the checkpoint is
        log.write(json.dumps({"step": step, "val": ev}) + "\n")                                         # saved before it): evaluate now
        log.flush()
        print(f"step {step}: val total {ev.get('total', float('nan')):.4f} (the evaluation a crash had cut)", flush=True)
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
    bad, n_overflow, t0, step0, last = 0, 0, time.time(), step, {}
    probe, timer = (ModuleProbe(model, every=cfg.log_every), StepTimer()) if cfg.probe else (None, None)
    prof = StepProfiler(out_dir, cfg.profile_steps, device=device) if cfg.profile_steps > 0 else None
    while step < cfg.steps:
        reason = _stop_requested(out_dir)               # `python -m trainsuite stop <run>` (autoresearch/trainsuite)
        if reason is not None:
            save_checkpoint(os.path.join(out_dir, f"ckpt_{step:08d}.pt"), model, loss, opt, ema, step, cfg, extra=scaler_extra(scaler))
            prune_checkpoints(out_dir, cfg.keep)
            with open(os.path.join(out_dir, "STOPPED"), "w", encoding="utf-8") as fh:
                json.dump({"step": step, "reason": reason, "t": time.time()}, fh)
            print(f"step {step}: stop requested ({reason}), checkpoint saved, exit {EXIT_STOPPED}", flush=True)
            raise SystemExit(EXIT_STOPPED)
        f = lr_factor(step, cfg.warmup, cfg.steps, cfg.lr_min_frac)
        for g in opt.param_groups:
            g["lr"] = cfg.lr * f
        diag = bool(cfg.grad_diag_every) and scaler is None and (step + 1) % cfg.grad_diag_every == 0         # not under fp16: the group gradients would be taken unscaled and underflow
        opt.zero_grad(set_to_none=True)
        acc, finite, prow, step_ids = {}, True, {}, []
        pstep = probe.begin(step + 1) if probe is not None else None    # a StepProbe only on a step that is logged
        for m in range(cfg.accum):                              # effective batch = batch_size * seq_len * accum samples
            with (timer.data() if timer is not None else contextlib.nullcontext()):        # the wait for the data loader
                item = next_batch()
            if probe is not None:
                step_ids.append(batch_tokens(item))                 # the samples of every micro-batch of the step, for an incident dump
            out, ok, dg, _ = run_clip(model, loss, item if isinstance(item, list) else [item], step / max(1, cfg.steps),
                                      device, cfg.amp, cfg, diag=diag and m == 0, backward=True, augment=True, scaler=scaler)
            if not ok:
                finite = False
                break
            for k, v in out.items():
                acc[k] = acc.get(k, 0.0) + v / cfg.accum
            if dg:
                acc["diag"] = dg
        if finite:                                              # the probe's gradient norm per module comes BEFORE clipping (a blow-up must not hide behind the clip), with the weights before the step
            gn, status = optimizer_step(opt, params, cfg.grad_clip, scaler, before_clip=pstep.grads if pstep is not None else None)
        else:
            gn, status = None, "nonfinite"
            opt.zero_grad(set_to_none=True)
        skipped = status != "ok"
        if status == "nonfinite" and gn is None and scaler is not None:      # a non-finite LOSS under fp16: the forward may have overflowed and written inf / nan into the batch-norm statistics
            poisoned = poisoned_buffers(model)
            if poisoned:
                raise RuntimeError(f"step {step}: the forward pass left non-finite batch-norm statistics ({len(poisoned)} buffers, e.g. {poisoned[:3]}): an fp16 overflow of the activations "
                                   "(above 65504), which no loss scale cures; resume from the last checkpoint with amp bf16 or off")
        was_bad, bad = bad, bad_steps_after(status, bad, scaler)
        if status == "overflow" and bad == was_bad:             # fp16: expected now and then, the scaler halved its scale
            n_overflow += 1
            print(f"step {step}: fp16 gradient overflow, skipped, loss scale -> {scaler.get_scale():g} ({n_overflow} since this start)", flush=True)
        elif status == "overflow":                              # the scale is below 1: a diverged run, in the words the health check counts
            n_overflow += 1
            print(f"step {step}: non-finite gradient (fp16 overflow, loss scale {scaler.get_scale():g}), skipped ({bad} in a row)", flush=True)
        elif skipped:
            print(f"step {step}: non-finite {'loss' if gn is None else 'gradient'}, skipped ({bad} in a row)", flush=True)
            if probe is not None:                               # what a replay needs: the step and the samples of the micro-batches so far (the last one is the one that failed)
                dump_incident(out_dir, step, "nonfinite", batch_ids=[t for ids in step_ids for t in ids], losses=acc,
                              extra={"what": "loss" if gn is None else "gradient", "micro": len(step_ids), "micro_batches": step_ids,
                                     "step_frac": step / max(1, cfg.steps), "grad_norm": None if gn is None else float(gn)})
        if skipped:
            if bad >= cfg.max_bad_steps:
                raise RuntimeError("too many non-finite steps in a row")
        else:
            ema.update(model, step)
            if pstep is not None:
                prow = pstep.finish()
        step += 1
        trow = timer.row() if timer is not None else {}
        if prof is not None:
            prof.tick()
        if not skipped and (step % cfg.log_every == 0 or step == 1 or "diag" in acc):
            rec = {"step": step, "t": time.time(), "lr": cfg.lr * f, "grad_norm": float(gn),
                   "s_per_step": (time.time() - t0) / max(1, step - step0), **acc, **prow, **trow}
            if device.startswith("cuda"):
                rec["gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
            if scaler is not None:
                rec["loss_scale"], rec["fp16_overflows"] = scaler.get_scale(), n_overflow
            last = rec
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"step {step}: total {rec['total']:.4f} lr {rec['lr']:.2e} grad {rec['grad_norm']:.2f}", flush=True)
        if cfg.save_every and step % cfg.save_every == 0:       # before the evaluation (and also after a skipped step): an evaluation
            save_checkpoint(os.path.join(out_dir, f"ckpt_{step:08d}.pt"), model, loss, opt, ema, step, cfg, extra=scaler_extra(scaler))   # that dies no longer
            prune_checkpoints(out_dir, cfg.keep)                                                                 # costs the steps since the last save
        if cfg.eval_every and step % cfg.eval_every == 0:
            ev = evaluate(model, loss, ema, val_loader, 10 ** 9, device, cfg.amp, eval_fn, cfg)
            log.write(json.dumps({"step": step, "val": ev}) + "\n")
            log.flush()
            print(f"step {step}: val total {ev.get('total', float('nan')):.4f}", flush=True)
    if prof is not None:
        prof.close()                                            # a run shorter than warm-up + window writes what it has (before the final save: its wall time must not hold the checkpoint write)
    save_checkpoint(os.path.join(out_dir, f"ckpt_{step:08d}.pt"), model, loss, opt, ema, step, cfg, extra=scaler_extra(scaler), slim=cfg.slim_at_end or was_slim)
    prune_checkpoints(out_dir, 1 if (cfg.slim_at_end or was_slim) else cfg.keep)         # a finished run: the final weights only
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
        assert all(g[k] == s[k]["meta"]["steps"] for g, s in zip(batch["gap_steps"], batch["gt"]) for k in ("prev", "next")), \
            "the dataset's gap steps (from the poses) equal the ones the GT carries"
        ego = batch_ego(batch["gap_steps"], sorted(gts), "cpu")
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
        gt_rec = json.load(open(os.path.join(tmp, "selftest", "gt.json")))          # the GT identity of the run (F3 of the review)
        from autogt.versions import version_of_key                                  # D29: the version key next to the signature, for a released one
        want_rec = {"gt_signature": base.gt_signature(), **({"gt_version": version_of_key(base.gt_signature())} if version_of_key(base.gt_signature()) else {})}
        assert gt_rec == want_rec and base.gt_signature().startswith("0.5_4_"), gt_rec
        ckpts = sorted(f for f in os.listdir(os.path.join(tmp, "selftest")) if f.endswith(".pt"))
        assert len(ckpts) == 2, ckpts                                              # keep = 2
        ck = torch.load(os.path.join(tmp, "selftest", ckpts[-1]), weights_only=False)
        assert ck["step"] == 4 and not any(k.startswith("model.camera.") for k in ck["state"])
        assert "opt" in ck and "rng" in ck and not ck.get("slim"), "the default keeps complete checkpoints (Adam moments and RNG)"
        changed = any(not torch.equal(before[k], v) for k, v in model2.state_dict().items() if k in before)
        assert changed, "parameters changed"
        # 2b slim_at_end (the default): a finished run keeps ONE checkpoint without optimiser state and RNG, and evaluation still loads it
        scfg = dataclasses.replace(cfg, name="slimrun", slim_at_end=True)
        _, _, sema, _ = train(scfg, datasets=(base, base))
        sdir = os.path.join(tmp, "slimrun")
        assert sorted(f for f in os.listdir(sdir) if f.startswith("ckpt_")) == ["ckpt_00000004.pt"], os.listdir(sdir)
        sck = torch.load(os.path.join(sdir, "ckpt_00000004.pt"), weights_only=False)
        assert sck["slim"] and "opt" not in sck and "rng" not in sck and sck["step"] == 4
        assert os.path.getsize(os.path.join(sdir, "ckpt_00000004.pt")) < 0.75 * os.path.getsize(os.path.join(tmp, "selftest", ckpts[-1]))
        m3, l3, _ = build(scfg)
        o3 = torch.optim.AdamW(param_groups(m3, l3, scfg.lr, scfg.weight_decay), lr=scfg.lr)
        e3 = EMA(m3, scfg.ema)
        assert load_checkpoint(os.path.join(sdir, "ckpt_00000004.pt"), m3, l3, o3, e3)[0] == 4
        assert all(torch.equal(e3.shadow[k].cpu(), sema.shadow[k].cpu()) for k in sema.shadow), "the EMA weights survive the slimming"
        # a slim checkpoint is refused as a resume below the configured end and accepted at the end
        for end, refused in ((6, True), (4, False)):
            try:
                refuse_slim_resume(sck, 4, dataclasses.replace(scfg, steps=end), "slimrun")
                got = False
            except SystemExit:
                got = True
            assert got == refused, (end, got)
        # re-invoking the finished slim run, even with slimming off, trains nothing and does not turn it into a "full" checkpoint with an
        # empty optimiser
        train(dataclasses.replace(scfg, slim_at_end=False), datasets=(base, base))
        sck2 = torch.load(os.path.join(sdir, "ckpt_00000004.pt"), weights_only=False)
        assert sck2["slim"] and "opt" not in sck2 and sorted(f for f in os.listdir(sdir) if f.startswith("ckpt_")) == ["ckpt_00000004.pt"]
        # the leftover of a failed checkpoint write is removed by the next prune, the checkpoint itself stays
        open(os.path.join(sdir, "ckpt_00000009.pt.tmp"), "wb").write(b"half written")
        prune_checkpoints(sdir, 1)
        assert not os.path.exists(os.path.join(sdir, "ckpt_00000009.pt.tmp")) and os.path.exists(os.path.join(sdir, "ckpt_00000004.pt"))
        # 3 resume: a resume with another configuration (here steps 4 -> 6, which would restart the cosine LR) is refused; with
        # BF_RESUME_ANY=1 the run continues from step 4 and ends at 6; the EMA differs from the weights. A damaged newest checkpoint
        # falls back to the one before, and the log rows above the resumed step move to log_superseded.jsonl
        cfg2 = dataclasses.replace(cfg, steps=6)
        msg = ""
        try:
            train(cfg2, datasets=(base, base))
        except SystemExit as e:
            msg = str(e.code)
        assert "steps" in msg, f"a resume with another `steps` must be refused: {msg!r}"
        os.environ["BF_RESUME_ANY"] = "1"
        try:
            _, _, ema2, last2 = train(cfg2, datasets=(base, base))
        finally:
            os.environ.pop("BF_RESUME_ANY")
        ck2 = sorted(f for f in os.listdir(os.path.join(tmp, "selftest")) if f.endswith(".pt"))[-1]
        assert torch.load(os.path.join(tmp, "selftest", ck2), weights_only=False)["step"] == 6
        rdir = os.path.join(tmp, "selftest")
        with open(os.path.join(rdir, "log.jsonl"), "a") as fh:     # rows of steps a crash would make the run redo
            fh.write(json.dumps({"step": 7, "total": 1.0}) + "\n" + '{"step": 8, "tot')
        with open(os.path.join(rdir, ck2), "wb") as fh:           # the newest checkpoint damaged (a cut write)
            fh.write(b"PK\x03\x04 cut")
        cfg3 = dataclasses.replace(cfg2, eval_every=0, save_every=0)
        os.environ["BF_RESUME_ANY"] = "1"                         # (eval_every / save_every are free anyway; steps differ from ckpt 4)
        try:
            train(cfg3, datasets=(base, base))                    # falls back to ckpt 4 and trains to 6 again
        finally:
            os.environ.pop("BF_RESUME_ANY")
        steps_logged = [json.loads(ln)["step"] for ln in open(os.path.join(rdir, "log.jsonl")) if '"val"' not in ln]
        assert max(steps_logged) == 6 and len(steps_logged) == len(set(steps_logged)), steps_logged
        assert any('"step": 7' in ln for ln in open(os.path.join(rdir, "log_superseded.jsonl"))), "superseded rows kept aside"
        assert os.path.exists(os.path.join(rdir, ck2 + ".damaged")), "the damaged checkpoint is kept aside, not pruned"
        # 3c a run that dies inside the evaluation of step 2 (t7_rgb_s1: ENOMEM in a validation worker) keeps ckpt 2, saved before
        # the evaluation, and makes the evaluation of step 2 up when it resumes
        ev_cfg = dataclasses.replace(cfg, name="evalcrash", steps=2, eval_every=2, save_every=2)
        edir, real_eval, crash = os.path.join(tmp, "evalcrash"), globals()["evaluate"], [True]

        def eval_crash_once(*a, **k):
            if crash[0]:
                crash[0] = False
                raise MemoryError("simulated ENOMEM in a validation worker")
            return real_eval(*a, **k)
        globals()["evaluate"] = eval_crash_once
        try:
            try:
                train(ev_cfg, datasets=(base, base))
                raise AssertionError("the simulated crash must end the run")
            except MemoryError:
                pass
            assert os.path.exists(os.path.join(edir, "ckpt_00000002.pt")) and not val_logged(edir, 2), "saved before the evaluation"
            train(ev_cfg, datasets=(base, base))
        finally:
            globals()["evaluate"] = real_eval
        assert val_logged(edir, 2), "the evaluation the crash cut is made up on the resume"
        # 3b stop request (autoresearch/trainsuite): CONTROL / STOPPED of an earlier stop are moved aside at the start (no clocks compared); a
        # request that arrives while training stops the run at a step boundary with a checkpoint, a STOPPED marker and exit code 75 (not 0:
        # no `.done`); the run resumes from that checkpoint; the log rows carry the time `t`
        stop_cfg = dataclasses.replace(cfg, name="stoprun", steps=1, save_every=0, eval_every=0)
        sdir = os.path.join(tmp, "stoprun")
        os.makedirs(sdir)
        for n in ("CONTROL", "STOPPED"):
            with open(os.path.join(sdir, n), "w") as fh:
                json.dump({"cmd": "stop", "reason": "old"}, fh)
        train(stop_cfg, datasets=(base, base))
        assert not os.path.exists(os.path.join(sdir, "STOPPED")) and not os.path.exists(os.path.join(sdir, "CONTROL")), "old files not moved"
        assert os.path.exists(os.path.join(sdir, "CONTROL.old")) and os.path.exists(os.path.join(sdir, "STOPPED.old"))
        assert all("t" in json.loads(ln) for ln in open(os.path.join(sdir, "log.jsonl")) if '"val"' not in ln), "log rows carry t"
        # 3d one module trains, the rest is frozen (freeze.py) and the probe logs the module keys (probe.py); with the defaults neither exists
        assert not any(k.startswith(("grad.", "upd.", "t_data")) for ln in open(os.path.join(sdir, "log.jsonl")) if '"val"' not in ln for k in json.loads(ln)), "the probe is off by default"
        # the average of a frozen weight is the weight: after 200 updates a plain EMA of a constant has drifted by rounding, the one with `skip` has not
        e_net = nn.Linear(64, 64)
        e_plain, e_skip = EMA(e_net, 0.99), EMA(e_net, 0.99, skip={"weight"})
        for i in range(200):
            e_plain.update(e_net, i)
            e_skip.update(e_net, i)
        assert not torch.equal(e_plain.shadow["weight"], e_net.weight.detach()) and torch.equal(e_skip.shadow["weight"], e_net.weight.detach()), "EMA skip"
        fz_cfg = dataclasses.replace(cfg, name="freezerun", steps=4, save_every=4, eval_every=0, log_every=2, grad_diag_every=1, grad_clip=1e-3, train_only=("aux_rad.",), probe=True)
        torch.manual_seed(fz_cfg.seed)
        init_model, _, _ = build(fz_cfg)
        init = {k: v.clone() for k, v in init_model.state_dict().items()}
        train(fz_cfg, datasets=(base, base))                    # grad_diag_every=1 with a frozen trunk: the diagnostics are skipped, not a crash
        fdir = os.path.join(tmp, "freezerun")
        ck_f = torch.load(os.path.join(fdir, "ckpt_00000004.pt"), map_location="cpu", weights_only=False)
        moved = [k for k in init if "model." + k in ck_f["state"] and not torch.equal(ck_f["state"]["model." + k], init[k])]
        assert moved and all(k.startswith("aux_rad.") for k in moved), moved[:5]
        assert all(torch.equal(ck_f["ema"][k], init[k]) for k in init if not k.startswith("aux_rad.") and k in ck_f["ema"]), "the average of a frozen weight is the weight itself"
        f_rows = [json.loads(ln) for ln in open(os.path.join(fdir, "log.jsonl")) if '"val"' not in ln]
        even = [r for r in f_rows if r["step"] % 2 == 0]
        assert even and all("grad.aux_rad" in r and "upd.aux_rad" in r for r in even), [sorted(r) for r in f_rows][:3]       # log_every 2: the numbering of begin(step + 1) is right
        assert not any(k.startswith(("grad.", "upd.")) and "aux_rad" not in k for r in f_rows for k in r) and all("diag" not in r for r in f_rows), "frozen modules have no keys, no diag"
        assert all(r["grad.aux_rad"] > fz_cfg.grad_clip for r in even), "the gradient norm is measured BEFORE clipping (clipped to 1e-3 it could not exceed 1e-3)"
        assert any(r.get("t_data", 0) > 0 and r.get("t_step", 0) >= r["t_data"] for r in f_rows), "the wait for the data is measured"
        try:
            train(dataclasses.replace(fz_cfg, train_only=("aux_cam.",)), datasets=(base, base))
            raise AssertionError("a resume with another freeze plan must be refused")
        except SystemExit as e:
            assert "train_only" in str(e), e
        train(dataclasses.replace(fz_cfg, probe=False), datasets=(base, base))        # probe is a logging switch: not part of the plan a resume checks
        # probe on or off: the weights are bit-equal (the probe only looks)
        ck_pair = []
        for nm, pr in (("probe_on", True), ("probe_off", False)):
            train(dataclasses.replace(cfg, name=nm, steps=2, save_every=2, eval_every=0, log_every=1, probe=pr), datasets=(base, base))
            ck_pair.append(torch.load(os.path.join(tmp, nm, "ckpt_00000002.pt"), map_location="cpu", weights_only=False)["state"])
        assert all(torch.equal(ck_pair[0][k], ck_pair[1][k]) for k in ck_pair[1]), "the probe changes no weight"
        # a non-finite step with the probe on: the dump names the step and the samples
        real_clip, calls_ = globals()["run_clip"], [0]

        def clip_fails_once(*a, **k):
            calls_[0] += 1
            if calls_[0] == 1:
                return {}, False, None, None
            return real_clip(*a, **k)
        globals()["run_clip"] = clip_fails_once
        try:
            train(dataclasses.replace(cfg, name="incrun", steps=1, save_every=0, eval_every=0, log_every=1, probe=True), datasets=(base, base))
        finally:
            globals()["run_clip"] = real_clip
        inc = json.load(open(os.path.join(tmp, "incrun", "incident_0.json")))
        assert inc["kind"] == "nonfinite" and inc["what"] == "loss" and inc["micro"] == 1 and inc["batch"] and all(isinstance(t, str) for t in inc["batch"]) and inc["micro_batches"] == [inc["batch"]], inc
        real_stop, calls = globals()["_stop_requested"], [0]

        def stop_third(run_dir):                              # the stop request arrives while the run trains: seen at the third step
            calls[0] += 1
            return "selftest" if calls[0] > 2 else None
        stop_cfg2 = dataclasses.replace(stop_cfg, name="stoprun2", steps=4)
        globals()["_stop_requested"] = stop_third
        try:
            train(stop_cfg2, datasets=(base, base))
            raise AssertionError("the stop request must end the run")
        except SystemExit as e:
            assert e.code == EXIT_STOPPED, e.code
        finally:
            globals()["_stop_requested"] = real_stop
        d2 = os.path.join(tmp, "stoprun2")
        st = json.load(open(os.path.join(d2, "STOPPED")))
        assert st["step"] == 2 and st["reason"] == "selftest" and os.path.exists(os.path.join(d2, "ckpt_00000002.pt")), st
        assert not os.path.exists(os.path.join(d2, ".done")), "a stopped run is never done"
        train(stop_cfg2, datasets=(base, base))               # resume: from step 2 to 4, the old STOPPED is moved aside, no second stop
        assert os.path.exists(os.path.join(d2, "ckpt_00000004.pt")) and os.path.exists(os.path.join(d2, "STOPPED.old"))
        assert not os.path.exists(os.path.join(d2, "STOPPED")), "STOPPED of the earlier stop must not stay"
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


def read_config(path):
    """TrainConfig of a config file; keys starting with "_" are comments for the reader (e.g. "_about"), every other unknown key is an error"""
    with open(path) as fh:
        return TrainConfig(**{k: v for k, v in json.load(fh).items() if not k.startswith("_")})


if __name__ == "__main__":
    if len(sys.argv) > 1:
        train(read_config(sys.argv[1]))
    else:
        _selftest()
