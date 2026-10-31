"""Probing a training run from a one-off script (the library behind temp/ diagnostics; replaces `import evaluate_run as E` and
`import observe_run as ob`).

    from evaluation import bev_probe as P
    cfg, model, loss, step, ds, device = P.load_all("runs/t1_r4_soa", "cuda")      # config, EMA model, loss, val dataset
    batch, res, gts, tap, vis, n = P.run_frame(model, loss, cfg, ds, i, device, cold=False)

What is here: the pinned code of the run on sys.path (the first command-line argument that is a run folder with a `code` snapshot
selects it, as before), the chain's functions by their usual names (forward_loss, make_datasets, make_loader, val_subset,
TrainConfig, NUSC_GROUPS, decode, ... resolved from the snapshot on first use), and the BEV helpers (load_run, load_all, run_frame,
predictions, layer_index, masks, gt_boxes, add_sample, prior_maps, perturb). Evaluation itself is `bev_suite.py`, not here.
"""
from __future__ import annotations

import os
import sys

from evaluation import bev_adapter as _A


def _scan_argv():
    return next((a for a in sys.argv[1:] if not a.startswith("-") and os.path.isdir(os.path.join(a, "code"))), None)


CODE = _A.setup_paths(_scan_argv())
ROOT = _A.ROOT
_NS = None


def chain():
    """the chain's code of the selected snapshot: .T (the train module), .NUSC_GROUPS, .decode, .sample_targets, .clip_step, ..."""
    global _NS
    if _NS is None:
        _NS = _A._chain()
    return _NS


_CHAIN_T = ("TrainConfig", "make_datasets", "make_loader", "val_subset", "forward_loss", "build", "param_groups", "EMA",
            "load_checkpoint", "to_device", "collate", "collate_cached", "clip_step", "detach_tree", "ClipDataset", "collate_clips")
_CHAIN_NS = ("NUSC_GROUPS", "sample_targets", "decode")


def __getattr__(name):                                     # lazy: the snapshot is imported when the name is first used
    if name in _CHAIN_T:
        return getattr(chain().T, name)
    if name in _CHAIN_NS:
        return getattr(chain(), name)
    raise AttributeError(name)


# ------------------------------------------------------------------------------------------------ BEV helpers
from evaluation.bev_adapter import layer_index, predictions, prior_maps, freeze_scenes  # noqa: E402,F401
from evaluation.bev_task import add_sample  # noqa: E402,F401
from evaluation.bev_views import composite, masks  # noqa: E402,F401
from evaluation.bev_views import gt_boxes as _gt_boxes  # noqa: E402


def gt_boxes(g, groups=None):
    """the GT boxes of one sample (training/gt_bev.py meta) of the NUSC_GROUPS classes"""
    return _gt_boxes(g, [x for x, _ in chain().NUSC_GROUPS] if groups is None else groups)


def load_run(run, device, ckpt_dir=None):
    """(cfg, model, loss, step, checkpoint): the last checkpoint under its EMA weights, in eval mode"""
    return _A.load_run(run, device, ckpt_dir, chain())


def load_all(run, device="cuda"):
    """everything a probe needs, loaded once: (cfg, model, loss, step, val dataset, device)"""
    import torch
    os.chdir(ROOT)
    device = device if torch.cuda.is_available() else "cpu"
    cfg, model, loss, step, _ = load_run(run, device)
    cfg.workers = 0
    freeze_scenes(cfg)
    _, val_ds = chain().T.make_datasets(cfg)
    return cfg, model, loss, step, val_ds, device


def run_frame(model, loss, cfg, ds, i, device, cold):
    """(batch of the target frame, model result, GTs, tap arrays {name: array of sample 0}, camera visibility (X, Y) or None, number
    of frames run) of val sample i, through its clip (the previous keyframes first) unless cold"""
    import torch
    from evalsuite.torch_taps import TapRecorder
    ns = chain()
    T = ns.T
    toks, pos = [ds.samples[i]], {t: k for k, t in enumerate(ds.samples)}
    while not cold and len(toks) < max(1, cfg.seq_len):
        prev = ds.nusc.get("sample", toks[0])["prev"]
        if not prev or prev not in pos:
            break
        toks.insert(0, prev)
    fn = T.collate_cached if cfg.feature_cache else T.collate
    hooks = _A.bev_hooks()
    rec = TapRecorder(model, hooks)
    state, prev_batch, res = None, None, None
    try:
        for tok in toks:
            batch = fn([ds[pos[tok]]])
            kw = {"state": state, "step": ns.clip_step(prev_batch, device)} if prev_batch is not None else {}
            rec.arm(list(hooks))
            with torch.no_grad():
                _, res, gts = T.forward_loss(model, loss, batch, 1.0, device, cfg.amp, cfg, train_terms=False, **kw)
            state, prev_batch = res["state"], batch
        tap = rec.read(0)
    finally:
        rec.close()
    vis = None
    if hasattr(model, "camera_visibility"):
        with torch.no_grad():
            vis = model.camera_visibility(T.to_device(batch, device), res["features"])[0, 0].cpu().numpy()
    return batch, res, gts, tap, vis, len(toks)


def perturb(batch, control):
    """in place: the sanity controls on a collated batch of two or more samples (the model must lose what the control takes away)"""
    import torch
    B = len(batch["tokens"])
    if control == "shuffle_radar":
        batch["radar"]["b"] = (batch["radar"]["b"] + 1) % B
    elif control == "zero_radar":
        batch["radar"] = {k: v[:0] for k, v in batch["radar"].items()}
    elif control in ("shuffle_cam", "zero_cam"):          # every camera input of the batch: both feature caches (t10), the lane image (t7)
        for k in ("prob", "ctx", "prob2", "ctx2", "img", "lane_img"):
            if k in batch:
                batch[k] = torch.roll(batch[k], 1, 0) if control == "shuffle_cam" else torch.zeros_like(batch[k])
    else:
        raise ValueError(f"unknown control {control}")
