"""bytes of every part of a checkpoint file: usage  sizes.py CKPT [CKPT ...]  (CPU, run from bev_fusion/)"""
import sys

import torch


def nbytes(x):
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, dict):
        return sum(nbytes(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(nbytes(v) for v in x)
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.nbytes
    except ImportError:
        pass
    return 0


for path in sys.argv[1:]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    print(f"== {path}")
    total = 0
    for k, v in ck.items():
        n = nbytes(v)
        total += n
        extra = ""
        if k == "opt" and isinstance(v, dict):
            extra = f"  ({len(v.get('state', {}))} parameters with moments, {len(v.get('param_groups', []))} param groups)"
        if k in ("state", "ema"):
            extra = f"  ({len(v)} tensors)"
        print(f"  {k:6s} {n / 1e6:8.1f} MB{extra}")
    print(f"  tensors total {total / 1e6:.1f} MB; keys: {sorted(ck)}")
