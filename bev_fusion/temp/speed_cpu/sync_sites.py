"""where do the sync-like ops of one keyframe step (forward, loss, backward) come from: python file:line of every aten op that would stop a CUDA stream when its input is on the GPU.
usage: sync_sites.py [B]    (small grid; counts per keyframe-batch; 'index(bool)' = boolean-mask indexing = a nonzero)"""
import collections
import sys
import traceback

import common as C
import torch
from torch.utils._python_dispatch import TorchDispatchMode

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1
cfg, model, loss, batch, _ = C.make(B=B, n_boxes=15, grid_kwargs={"x_max": 24.0, "y_min": -12.0, "y_max": 12.0})
from trainkit.util import detach_tree  # noqa: E402

SYNC = {"aten.nonzero.default", "aten._local_scalar_dense.default", "aten.is_nonzero.default", "aten.unique_consecutive.default", "aten._unique2.default",
        "aten.masked_select.default", "aten.equal.default", "aten.allclose.default", "aten.bincount.default", "aten.repeat_interleave.Tensor", "aten.argwhere.default"}


class Rec(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.sites = collections.Counter()
        self.phase = "fwd"
        self.devhint = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        kind = None
        if name in SYNC:
            kind = name.split(".")[1]
        elif name == "aten.index.Tensor":
            idx = args[1]
            if any(i is not None and i.dtype in (torch.bool, torch.uint8) for i in idx):
                kind = "index(bool)"
        elif name in ("aten.index_put_.default", "aten._index_put_impl_.default", "aten.index_put.default"):
            idx = args[1]
            if any(i is not None and i.dtype in (torch.bool, torch.uint8) for i in idx):
                kind = "index_put(bool)"
        elif name == "aten.index_select.default" or name == "aten.masked_fill_.Tensor":
            pass
        if kind:
            fr = [f for f in traceback.extract_stack() if "/bev_fusion/training/" in f.filename or "/autoresearch/trainkit/" in f.filename]
            fr = fr[-2:]
            where = " < ".join(f"{f.filename.split('/')[-1]}:{f.lineno}({f.name})" for f in reversed(fr)) if fr else "(no repo frame)"
            self.sites[(self.phase, kind, where)] += 1
        return func(*args, **kwargs)


rec = Rec()
# cold keyframe, then the warm one is analysed
out, res, _ = C.T.forward_loss(model, loss, batch, 0.5, "cpu", "off", cfg)
out["total"].backward()
state = detach_tree(res["state"])
model.zero_grad(set_to_none=True)
with rec:
    out, res, _ = C.T.forward_loss(model, loss, batch, 0.5, "cpu", "off", cfg, state=state, step=(1 / 12, 8.0, 0.05))
    rec.phase = "bwd"
    out["total"].backward()
print(f"warm keyframe, B={B}, 15 boxes per sample: sync-like op sites (count per keyframe-batch)")
for (ph, kind, where), n in sorted(rec.sites.items(), key=lambda kv: (kv[0][0], -kv[1])):
    print(f"  {ph} {kind:22s} x{n:3d}  {where}")
