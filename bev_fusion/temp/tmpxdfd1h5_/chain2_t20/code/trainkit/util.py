"""Small helpers of a training start."""
import os

import numpy as np
import torch

from . import control


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def resolve_device(device):
    """the configured device; "cuda" without CUDA raises (it used to fall back to "cpu" without a word: a failed CUDA initialisation reads as 'not available', and the trainer then
    trained on the CPU for hours holding a GPU slot). The message carries the text run.py retries on (CUDA error: unknown error)"""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device cuda was asked for and CUDA is not available (a failed CUDA initialisation reads the same: CUDA error: unknown error; run.py retries it)")
    return device


def prepare_run_dir(cfg):
    """<out_root>/<name>, created; the CONTROL / STOPPED files of an earlier stop are moved aside (control.clear_stale) -> the path"""
    out_dir = os.path.join(cfg.out_root, cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    control.clear_stale(out_dir)
    return out_dir


def detach_tree(x):
    """The model state (nested dicts / lists of tensors, None allowed) without its graph."""
    if torch.is_tensor(x):
        return x.detach()
    if isinstance(x, dict):
        return {k: detach_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(detach_tree(v) for v in x)
    return x
