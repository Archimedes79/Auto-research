"""Does the loss weighting of the dense velocity head matter, on frozen trunk features?  (review: moving objects are ~20 % of the
supervised objects and the head is trained with L1, whose optimum on cells it cannot tell apart is the median = 0.)
Fresh MotionHead on the frozen trunk output S of 48 train frames, evaluated on 24 val frames, variants of the loss on the object cells:
  L1            the training loss (losses.motion_term)
  L1 x4 moving  moving cells (GT speed >= 1 m/s) weighted by the inverse moving share
  MSE x4        squared error, same weights
  Huber x4      Huber (delta 2 m/s), same weights
Reported on the val frames' moving / static object cells: mean |v - GT| (per component), mean predicted speed on static cells, share of
moving cells predicted below 0.5 m/s; zero predictor for reference.   usage: velocity_weighting.py <run>"""
import sys

import numpy as np
import torch

sys.path.insert(0, "evaluation")
import observe_run as ob  # noqa: E402
from losses import build_layer_targets  # noqa: E402
from timed_head import MotionHead  # noqa: E402
from train import collate, collate_cached, forward_loss, make_datasets  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
run = sys.argv[1] if len(sys.argv) > 1 else "runs/t1_r4_soa"
cfg, model, loss, step, val_ds, device = ob.load_all(run, "cpu")
train_ds = make_datasets(cfg)[0]
model.eval()


def gather(ds, n):
    by = {}
    for k, tok in enumerate(ds.samples):
        by.setdefault(ds.nusc.get("sample", tok)["scene_token"], []).append(k)
    mids = [v[len(v) // 3] for v in by.values()]
    idx = [mids[int(j)] for j in np.linspace(0, len(mids) - 1, n).round()]
    fn = collate_cached if cfg.feature_cache else collate
    out = []
    for i in idx:
        try:
            batch = fn([ds[i]])
        except KeyError:
            continue
        with torch.no_grad():
            _, res, gts = forward_loss(model, loss, batch, 1.0, device, cfg.amp, cfg, train_terms=False)
            T = build_layer_targets(gts[0.0], loss.cfg, None, 0.0, torch.device("cpu"))
        valid = (T["w_obj"][:, 0] > 0)
        obj = (T["obj_cells"][:, 0] > 0.5) & valid
        gv = T["velocity"][0]
        gv = torch.where(gv.norm(dim=0, keepdim=True) < 1.0, torch.zeros_like(gv), gv)        # the training target (min speed 1 m/s)
        out.append((res["features"].detach().float(), gv, obj[0]))
    return out


tr, va = gather(train_ds, 40), gather(val_ds, 20)
n_mov = sum(int(((g.norm(dim=0) > 0) & o).sum()) for _, g, o in tr)
n_obj = sum(int(o.sum()) for _, _, o in tr)
w_mov = n_obj / max(n_mov, 1)
print(f"train object cells {n_obj}, moving {n_mov} ({100 * n_mov / n_obj:.0f} %), weight of a moving cell {w_mov:.1f}", flush=True)


def make_loss(kind):
    def f(v, gv, obj):
        err = v[0] - gv                                              # (2, X, Y)
        mov = (gv.norm(dim=0) > 0)
        w = obj.float() * torch.where(mov, torch.full_like(obj.float(), w_mov if kind != "L1" else 1.0), torch.ones_like(obj.float()))
        if kind in ("L1", "L1x4"):
            e = err.abs().sum(0)
        elif kind == "MSEx4":
            e = (err ** 2).sum(0)
        else:
            a = err.abs()
            e = torch.where(a < 2, 0.5 * a ** 2, 2 * (a - 1)).sum(0)
        bg = (v[0].abs().sum(0) * (~obj).float()).sum() / (~obj).float().sum().clamp(min=1)
        return (e * w).sum() / w.sum().clamp(min=1) + 0.1 * bg
    return f


def evaluate(head, items):
    em = es = 0.0
    nm = ns = low = 0
    for S, gv, obj in items:
        with torch.no_grad():
            v = head(S)[0]
        mov = (gv.norm(dim=0) > 0) & obj
        stat = (gv.norm(dim=0) == 0) & obj
        if mov.any():
            em += float((v[:, mov] - gv[:, mov]).abs().mean()) * int(mov.sum())
            low += int((v[:, mov].norm(dim=0) < 0.5).sum())
            nm += int(mov.sum())
        if stat.any():
            es += float(v[:, stat].norm(dim=0).sum())
            ns += int(stat.sum())
    return em / max(nm, 1), es / max(ns, 1), 100 * low / max(nm, 1)


z = lambda items: sum(float(gv[:, (gv.norm(dim=0) > 0) & o].abs().mean()) * int(((gv.norm(dim=0) > 0) & o).sum()) for _, gv, o in items if ((gv.norm(dim=0) > 0) & o).any()) / max(sum(int(((gv.norm(dim=0) > 0) & o).sum()) for _, gv, o in items), 1)
print(f"zero predictor: moving-cell error train {z(tr):.2f} val {z(va):.2f} m/s per component")
c = tr[0][0].shape[1]
for kind in ("L1", "L1x4", "MSEx4", "Huberx4"):
    head = MotionHead(c)
    opt = torch.optim.Adam(head.parameters(), 1e-3)
    lf = make_loss(kind)
    for it in range(601):
        perm = np.random.permutation(len(tr))[:8]
        l = sum(lf(head(tr[i][0]), tr[i][1], tr[i][2]) for i in perm) / len(perm)
        opt.zero_grad()
        l.backward()
        opt.step()
    tm, ts, tl = evaluate(head, tr)
    vm, vs, vl = evaluate(head, va)
    print(f"{kind:8s} train: moving error {tm:.2f}, static speed {ts:.2f}, moving<0.5 {tl:.0f} % | val: moving error {vm:.2f}, static speed {vs:.2f}, moving<0.5 {vl:.0f} %", flush=True)
