"""Can the dense velocity head learn on the trunk features at all?  (review: the head predicts ~0 for movers, val and train.)

Frozen trunk output S of 48 train and 24 val frames (cold, no memory), then a FRESH MotionHead (same class, same loss as
training: losses.motion_term) is trained on the train features only. If it overfits the train frames and generalises somewhat
to val, the information and the target are consistent and the problem is the training of the shared model; if it cannot
even overfit, the target or the features are inconsistent (bug).
usage: velocity_overfit.py <run>"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from evaluation import bev_probe as ob  # noqa: E402
from losses import build_layer_targets, motion_term  # noqa: E402
from timed_head import MotionHead  # noqa: E402
from train import forward_loss, make_datasets, to_device, collate_cached, collate  # noqa: E402

torch.manual_seed(0)
run = sys.argv[1] if len(sys.argv) > 1 else "runs/t1_r4_soa"
cfg, model, loss, step, val_ds, device = ob.load_all(run, "cpu")
train_ds = make_datasets(cfg)[0]


def gather(ds, n):
    by = {}
    for k, tok in enumerate(ds.samples):
        by.setdefault(ds.nusc.get("sample", tok)["scene_token"], []).append(k)
    mids = [v[len(v) // 3] for v in by.values()]
    idx = [mids[int(j)] for j in np.linspace(0, len(mids) - 1, n).round()]
    out = []
    fn = collate_cached if cfg.feature_cache else collate
    for i in idx:
        try:
            batch = fn([ds[i]])
        except KeyError:
            continue
        with torch.no_grad():
            _, res, gts = forward_loss(model, loss, batch, 1.0, device, cfg.amp, cfg, train_terms=False)
            T = build_layer_targets(gts[0.0], loss.cfg, None, 0.0, torch.device("cpu"))
        rad = batch["radar"]
        g = model.grid
        ix = ((rad["x"] - g.x_min) / g.res).floor().long().clamp(0, g.nx - 1)
        iy = ((rad["y"] - g.y_min) / g.res).floor().long().clamp(0, g.ny - 1)
        r = torch.hypot(rad["x"], rad["y"]).clamp(min=1.0)
        cnt = torch.zeros(g.nx, g.ny)
        vx, vy = torch.zeros(g.nx, g.ny), torch.zeros(g.nx, g.ny)
        cnt.index_put_((ix, iy), torch.ones_like(r), accumulate=True)
        vx.index_put_((ix, iy), rad["vr"] * rad["x"] / r, accumulate=True)       # Doppler along the line of sight, as a vector
        vy.index_put_((ix, iy), rad["vr"] * rad["y"] / r, accumulate=True)
        dop = torch.stack([vx / cnt.clamp(min=1), vy / cnt.clamp(min=1), (cnt > 0).float()])[None]
        out.append((res["features"].detach().float(), {k: T[k] for k in ("velocity", "obj_cells", "w_obj")}, dop))
    return out


def batch_loss(head, items, dop=False):
    tot = 0.0
    for S, T, D in items:
        tot = tot + motion_term(head(torch.cat([S, D], 1) if dop else S), T, loss.cfg)
    return tot / len(items)


tr, va = gather(train_ds, 48), gather(val_ds, 24)
print(f"features: {len(tr)} train, {len(va)} val frames", flush=True)
c = tr[0][0].shape[1]
zero = lambda items: float(sum(motion_term(torch.zeros(1, 2, *S.shape[2:]), T, loss.cfg) for S, T, D in items) / len(items))
trained = lambda items: float(batch_loss(model.mover.motion, items).detach())
print(f"loss of the zero predictor: train {zero(tr):.3f} val {zero(va):.3f}")
print(f"loss of the trained model head: train {trained(tr):.3f} val {trained(va):.3f}")
for dop in (False, True):
    head = MotionHead(c + (3 if dop else 0))
    opt = torch.optim.Adam(head.parameters(), 1e-3)
    for it in range(601):
        perm = np.random.permutation(len(tr))[:8]
        l = batch_loss(head, [tr[i] for i in perm], dop)
        opt.zero_grad()
        l.backward()
        opt.step()
        if it % 200 == 0:
            with torch.no_grad():
                print(f"fresh head {'S + Doppler vector' if dop else 'S only         '} step {it:3d}: train {float(batch_loss(head, tr, dop)):.3f} "
                      f"val {float(batch_loss(head, va, dop)):.3f}", flush=True)
