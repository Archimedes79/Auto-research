"""A toy to run the REAL train() on the CPU in about a second (the data path stubbed): a tiny network with a batch norm, a loss with a Kendall-like parameter, batches with a `mode` per
batch (ok | inf_grad | nan_loss | bn_poison). Used by test_lr_groups.py and test_train_guards.py; test_grad_scaler.py takes ToyNet, ToyLoss and same_tree from here and keeps its own forward and batches.
    import toy_train as TT; res = TT.run_train(T, root, "name", {}, lr_groups={...})   # -> dict(out=stdout, rows=log rows, ck=newest checkpoint, exc=text of the exception or None)"""
import contextlib
import io
import json
import os

import torch
from torch import nn


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Linear(6, 8), nn.BatchNorm1d(8), nn.Tanh(), nn.Linear(8, 3))
        self.grid = None

    def forward(self, batch, dts=None):
        return self.net(batch["x"])


class ToyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(2))


def toy_forward(model, loss, batch, step_frac, device, amp, cfg=None, diag=False, train_terms=True, state=None, step=None):
    h = model.net[2](model.net[1](model.net[0](batch["x"])))
    tot = ((model.net[3](h) - batch["y"]) ** 2).mean() + 0.01 * loss.s.pow(2).sum()
    return {"total": tot, "group.a": tot.detach()}, {"state": {"m": h.detach()}}, None


def toy_batches(n=40):
    g = torch.Generator().manual_seed(7)
    return [{"x": torch.randn(16, 6, generator=g), "y": torch.randn(16, 3, generator=g), "tokens": [f"t{i}"], "mode": "ok"} for i in range(n)]


def run_train(T, root, name, loader=None, **kw):
    """T = the train module. Patches build / forward_loss / make_loader / val_subset for the call and restores them."""
    saved = {k: getattr(T, k) for k in ("build", "forward_loss", "make_loader", "val_subset")}
    T.build = lambda cfg: (ToyNet(), ToyLoss(), True)
    T.forward_loss = toy_forward
    T.make_loader = loader or (lambda ds, cfg, train, workers=None, generator=None: ds)
    T.val_subset = lambda ds, n: ds
    base = dict(name=name, out_root=root, steps=12, accum=2, warmup=2, log_every=1, eval_every=0, save_every=4, keep=3, device="cpu", workers=0, max_bad_steps=3,
                flip_aug=0.0, seq_len=1, seed=0, lr=1e-2, grad_clip=0.5)
    base.update(kw)
    cfg = T.TrainConfig(**base)
    data = toy_batches()
    buf, exc = io.StringIO(), None
    try:
        with contextlib.redirect_stdout(buf):
            T.train(cfg, datasets=(data, data))
    except BaseException as e:  # noqa: BLE001
        exc = f"{type(e).__name__}: {e}"
    finally:
        for k, v in saved.items():
            setattr(T, k, v)
    d = os.path.join(root, name)
    lp = os.path.join(d, "log.jsonl")
    rows = [json.loads(ln) for ln in open(lp)] if os.path.exists(lp) else []
    cks = sorted(f for f in os.listdir(d) if f.startswith("ckpt_") and f.endswith(".pt")) if os.path.isdir(d) else []
    ck = torch.load(os.path.join(d, cks[-1]), map_location="cpu", weights_only=False) if cks else None
    return dict(out=buf.getvalue(), rows=rows, ck=ck, exc=exc)


def same_tree(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.shape == b.shape and torch.equal(a, b)
    return a == b
