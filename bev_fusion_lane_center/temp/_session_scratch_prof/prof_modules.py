"""scratch (report chat, diagnostic): the forward + backward time of the token head by part, on one real batch (pair tensors: where does the GPU time go?).
Forward parts by cuda events in hooks; the backward share of the pair MLPs by a run with the pair MLPs replaced by a zero-cost stub (the difference is their cost)."""
import json
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(2)
R = os.environ["BFLC_ROOT"]
for sub_ in ("training", "training/grid", "training/data", "training/train", "evaluation"):
    sys.path.insert(0, os.path.join(R, sub_))
import jtoken as JK  # noqa: E402
import jtok as JT  # noqa: E402
import jtoken_train as JT2  # noqa: E402
import lane_ptok_data as LP  # noqa: E402

dev = "cuda"
ds, dl = LP.loader("train", 0, 8, 0)
b = JT2.to_device(LP.collate([ds[j] for j in range(8)]), dev)
model = JT2.build_head(dict(JT2.ARCH_DEFAULTS)).to(dev).train()
model._step_py = 10 ** 6
res = {}


def tm(fn, n=15):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return round((time.time() - t) / n * 1e3, 2)


def fwd():
    with torch.autocast("cuda", torch.bfloat16):
        return model(b)


def fwdbwd():
    with torch.autocast("cuda", torch.bfloat16):
        out = model(b)
        K = out["cand"].shape[-1]
        l = out["logp"][..., :K].clamp(min=-30).mean() + out["logp"][..., K].mean() + out["types"].float().mean() + out["scale"].float().mean()
    l.backward()


res["total_forward_ms"] = tm(fwd)
res["total_forward_backward_ms"] = tm(fwdbwd)

# parts of the forward
pos, dirv, valid, e = b["pos"].float(), b["dirv"].float(), b["valid"].bool(), b["e"].float().sigmoid()
res["pair_geometry_ms"] = tm(lambda: JT.pair_geometry(pos, dirv, valid, e=e))
g, allowed = JT.pair_geometry(pos, dirv, valid, e=e)
g = g.detach()
with torch.autocast("cuda", torch.bfloat16):
    for n_, mod in (("layer_bias_mlp(one of 3)", model.layers[0].bias), ("wbias", model.wbias), ("sc_g", model.sc_g)):
        res[f"fwd_{n_}_ms"] = tm(lambda mod=mod: mod(g))
    x = torch.randn(8, 512, 64, device=dev)
    res["fwd_scale_branch_full_ms"] = tm(lambda: model.sc_o(torch.nn.functional.gelu(model.sc_i(x)[:, :, None] + model.sc_j(x)[:, None] + model.sc_g(g))))
    res["fwd_layer0_total_ms"] = tm(lambda: model.layers[0](x, g, torch.zeros(8, 512, device=dev), valid))
    q = torch.randn(8, 2, 512, 32, device=dev)
    res["fwd_wiring_logp_ms"] = tm(lambda: JK.wiring_logp(torch.randn(8, 2, 512, 512, device=dev), torch.randn(8, 2, 512, device=dev), torch.stack([allowed, allowed], 1)))

# the cost of all five pair MLPs + the scale branch, forward + backward: replace their outputs by zeros of the same shape and compare the total
import types as _types  # noqa: E402

full = res["total_forward_backward_ms"]
orig = {}


class Zero(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, g):
        return torch.zeros(*g.shape[:-1], self.n, device=g.device, dtype=g.dtype) + 0.0 * g[..., :1]


for l in model.layers:
    orig[id(l)] = l.bias
    l.bias = Zero(4)
wb, scg, sco = model.wbias, model.sc_g, model.sc_o
model.wbias, model.sc_g = Zero(2), Zero(32)
res["fwdbwd_without_pair_mlps_but_scale_ms"] = tm(fwdbwd)
model.sc_o = torch.nn.Sequential(torch.nn.Identity())
model.sc_o.forward = lambda z: z[..., :2]
model.sc_i, model.sc_j = torch.nn.Identity(), torch.nn.Identity()
res["note"] = "second variant not run (kept simple)"
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prof_modules.json"), "w"), indent=1)
