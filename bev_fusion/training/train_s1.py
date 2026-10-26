"""Training and evaluation driver of the S1 camera-only module (training/s1_camera.py) - ONE driver for the synthetic test and
the real-data run (experiments/s1a_lift_pd_vs_p): it only sees batch dicts (prob, ctx, K, T_cam_car, hw, ida, gt, mask, scene).

- loss (plan 8.4, camera-visible cells only): per class soft BCE + soft-IoU on the sigmoid (losses.soft_bce / soft_iou_loss),
  cell weight = the camera mask (1 on cells the camera sees, 0 elsewhere); AdamW (weight decay on weights only), linear warm-up
  + cosine, gradient clipping, lr <= 6e-4 (fresh trunk with BatchNorm, the lane project's derail note)
- evaluation: one evaluation/metrics.SegMeter PER SCENE (counts per range band and tolerance 0 / 0.5 m), so every number
  can be summed over scenes and resampled by the paired scene bootstrap (`paired_bootstrap`); mask = the camera-visible cells
- null baseline: the per-cell training-mean map of each class thresholded at the best strict-IoU threshold on TRAIN (a prior
  map: what a model without the image could do), fed through the same SegMeter
Run the self-test: python training/train_s1.py
"""
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from evaluation.metrics import RANGE_BINS, SegMeter, bin_label  # noqa: E402
from losses import soft_bce, soft_iou_loss  # noqa: E402

TOL = (0.0, 0.5)                                         # strict and one cell (0.5 m at 0.5 m resolution)
BINS = ["all"] + [bin_label(i) for i in range(len(RANGE_BINS))]


def s1_loss(logits, gt, mask):
    """logits (B, K, X, Y), gt (B, K, X, Y) coverage, mask (B, 1, X, Y) -> scalar and per-class (K,) terms."""
    bce = soft_bce(logits, gt, mask)
    iou = soft_iou_loss(logits.sigmoid(), gt, mask)
    return (bce + iou).mean(), {"bce": bce.detach(), "iou": iou.detach()}


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train(model, make_batch, n_samples, steps, bs=8, lr=5e-4, seed=0, device="cuda", warmup=100, log_every=250, log=print):
    """make_batch(indices) -> batch dict (CPU); n_samples: size of the training set. Returns the final loss average."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model.to(device).train()
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    plain = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-4}, {"params": plain, "weight_decay": 0.0}], lr=lr)
    t0, run = time.time(), []
    for it in range(steps):
        f = min(1.0, (it + 1) / warmup) if it < warmup else 0.5 * (1 + math.cos(math.pi * (it - warmup) / max(1, steps - warmup)))
        for g in opt.param_groups:
            g["lr"] = lr * max(f, 0.01)
        b = to_device(make_batch([rng.randrange(n_samples) for _ in range(bs)]), device)
        loss, _ = s1_loss(model(b), b["gt"], b["mask"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        run.append(loss.item())
        if not math.isfinite(run[-1]):
            raise RuntimeError(f"non-finite loss at step {it}")
        if log_every and ((it + 1) % log_every == 0 or it == 0):
            log(f"    step {it + 1:5d} loss {sum(run[-log_every:]) / len(run[-log_every:]):.4f} ({time.time() - t0:.0f} s)")
    return sum(run[-50:]) / len(run[-50:])


@torch.no_grad()
def predict(model, make_batch, indices, device="cuda", bs=10):
    """yields (index list, sigmoid probabilities (B, K, X, Y) on the CPU, batch) per chunk, model in eval mode."""
    model.to(device).eval()
    for i in range(0, len(indices), bs):
        idx = indices[i:i + bs]
        b = to_device(make_batch(idx), device)
        yield idx, model(b).sigmoid().cpu(), b


def meter_batch(meters, probs, gt, mask, scenes, names, grid):
    """add every sample of a chunk to its own SegMeter (meters: dict scene -> SegMeter)."""
    for j, s in enumerate(scenes):
        m = meters.setdefault(s, SegMeter(grid, tolerances=TOL))
        m.add({n: probs[j, k] for k, n in enumerate(names)}, {n: gt[j, k] for k, n in enumerate(names)},
              mask=mask[j, 0] > 0.5)


def evaluate(model, make_batch, indices, names, grid, device="cuda"):
    meters = {}
    for idx, probs, b in predict(model, make_batch, indices, device):
        meter_batch(meters, probs, b["gt"].cpu(), b["mask"].cpu(), b["scene"], names, grid)
    return meters


def null_meters(train_gt, train_mask, make_batch, indices, names, grid):
    """prior-map baseline: per-cell mean GT over the training set (camera-visible samples), threshold per class = the best
    strict IoU on the training set; the same SegMeter on the held-out samples. train_gt (N, K, X, Y), train_mask (N, 1, X, Y)."""
    num = (train_gt * train_mask).sum(0)
    prior = num / train_mask.sum(0).clamp(min=1)                                         # (K, X, Y)
    thr = []
    for k in range(len(names)):
        best = (-1.0, 0.5)
        for t in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
            P = (prior[k] > t)[None] & (train_mask[:, 0] > 0.5)
            G = (train_gt[:, k] > 0.5) & (train_mask[:, 0] > 0.5)
            iou = (P & G).sum() / max(1, int((P | G).sum()))
            best = max(best, (float(iou), t)) if best[0] >= 0 else (float(iou), t)
        thr.append(best[1])
    meters = {}
    for i in range(0, len(indices), 20):
        b = make_batch(indices[i:i + 20])
        probs = torch.stack([(prior[k] > thr[k]).float().expand(len(b["scene"]), -1, -1) for k in range(len(names))], 1)
        meter_batch(meters, probs, b["gt"], b["mask"], b["scene"], names, grid)
    return meters, thr


# ------------------------------------------------------------------------------------------------ numbers
def scene_counts(meters, layer, rng_bin="all", dt=0.0):
    """{scene: (len(TOL), 4) counts of the layer in the range bin} (tp_pred, fp, tp_gt, fn per tolerance)."""
    key = (round(dt, 3), layer, "all", rng_bin)
    return {s: m.c[key]["cnt"] for s, m in meters.items() if key in m.c}


def iou_from_counts(c, tol_index):
    tp_p, fp, tp_g, fn = c[tol_index]
    tp = (tp_p + tp_g) / 2
    return float(tp / (tp + fp + fn)) if tp + fp + fn > 0 else float("nan")


def summary(meters, names):
    """{layer: {bin: (iou@0, iou@0.5)}} from the counts summed over scenes."""
    out = {}
    for n in names:
        out[n] = {}
        for b in BINS:
            cs = scene_counts(meters, n, b)
            tot = np.sum(list(cs.values()), 0) if cs else np.zeros((len(TOL), 4))
            out[n][b] = tuple(iou_from_counts(tot, i) for i in range(len(TOL)))
    return out


def class_mean(summ, names, b="all", tol_index=1):
    return float(np.mean([summ[n][b][tol_index] for n in names]))


def paired_bootstrap(meters_a, meters_b, layer, tol_index=1, rng_bin="all", n_boot=1000, seed=0):
    """b - a on the IoU of a layer, resampling SCENES (the same draw for both); returns (point, lo, hi) with a 95 % interval."""
    ca, cb = scene_counts(meters_a, layer, rng_bin), scene_counts(meters_b, layer, rng_bin)
    sc = sorted(set(ca) & set(cb))
    A, B = np.stack([ca[s] for s in sc]), np.stack([cb[s] for s in sc])
    point = iou_from_counts(B.sum(0), tol_index) - iou_from_counts(A.sum(0), tol_index)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(sc), len(sc))
        d.append(iou_from_counts(B[ix].sum(0), tol_index) - iou_from_counts(A[ix].sum(0), tol_index))
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return point, float(lo), float(hi)


# ----------------------------------------------------------------------------------------------- self-test
def _selftest():
    import copy
    from lift_pd import Grid
    grid = Grid()
    names = ("a", "b")
    X, Y = grid.nx, grid.ny
    g = torch.Generator().manual_seed(0)
    # loss: a perfect prediction has a lower loss than a wrong one; masked cells carry no weight
    gt = (torch.rand(2, 2, X, Y, generator=g) > 0.9).float()
    mask = torch.ones(2, 1, X, Y)
    lg_good, lg_bad = (gt * 20 - 10), -(gt * 20 - 10)
    assert s1_loss(lg_good, gt, mask)[0] < 0.01 < s1_loss(lg_bad, gt, mask)[0]
    m0 = mask.clone()
    m0[:, :, :60] = 0
    lg_half = lg_good.clone()
    lg_half[:, :, :60] = lg_bad[:, :, :60]
    assert abs(float(s1_loss(lg_half, gt, m0)[0] - s1_loss(lg_good, gt, m0)[0])) < 1e-5, "masked cells have no weight"
    # meters: a perfect prediction has IoU 1; a prediction shifted by one cell has strict IoU < 1 and IoU@0.5 m = 1
    scenes = list(range(6))
    gtl = torch.zeros(6, 2, X, Y)
    for s in scenes:
        gtl[s, :, 20 + 3 * s, 10:60] = 1.0
    one = torch.ones(6, 1, X, Y)
    M_ok, M_sh = {}, {}
    meter_batch(M_ok, gtl.clone(), gtl, one, scenes, names, grid)
    meter_batch(M_sh, gtl.roll(1, 2), gtl, one, scenes, names, grid)
    s_ok, s_sh = summary(M_ok, names), summary(M_sh, names)
    assert s_ok["a"]["all"] == (1.0, 1.0), s_ok
    assert s_sh["a"]["all"][0] == 0.0 and s_sh["a"]["all"][1] > 0.99, s_sh["a"]["all"]
    # bootstrap: identical meters -> 0 inside the interval; a model that is better in every scene -> interval above 0
    p, lo, hi = paired_bootstrap(M_ok, copy.deepcopy(M_ok), "a")
    assert p == 0 and lo <= 0 <= hi
    p2, lo2, hi2 = paired_bootstrap(M_sh, M_ok, "a", tol_index=0)
    assert p2 == 1.0 and lo2 > 0.9, (p2, lo2, hi2)
    # range bands: a line at x = 5 m is counted in the 0-20 m band only
    gtb = torch.zeros(1, 2, X, Y)
    gtb[0, :, 10, :] = 1.0
    Mb = {}
    meter_batch(Mb, gtb.clone(), gtb, one[:1], [0], names, grid)
    sb = summary(Mb, names)
    assert sb["a"]["0-20"][0] == 1.0 and math.isnan(sb["a"]["40-60"][0]), sb["a"]
    # null baseline: a fixed-position target is solved by the prior, a random one is not
    fixed = torch.zeros(20, 2, X, Y)
    fixed[:, :, 30:34, 20:80] = 1.0
    randg = (torch.rand(20, 2, X, Y, generator=g) > 0.99).float()
    ones = torch.ones(20, 1, X, Y)

    def mk(gt_all):
        return lambda idx: {"gt": gt_all[idx], "mask": ones[idx], "scene": list(idx)}
    Mf, _ = null_meters(fixed, ones, mk(fixed), list(range(20)), names, grid)
    Mr, _ = null_meters(randg, ones, mk(randg), list(range(20)), names, grid)
    assert summary(Mf, names)["a"]["all"][0] > 0.95 and summary(Mr, names)["a"]["all"][0] < 0.2
    # a few training steps on a tiny model reduce the loss (wiring of train / predict / evaluate)
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(1, 2, 3, padding=1)

        def forward(self, b):
            return self.c(b["gt"][:, :1] * 0 + b["x"])
    xs = torch.randn(16, 1, X, Y, generator=g)
    tg = torch.stack([(xs[:, 0] > 1).float(), (xs[:, 0] < -1).float()], 1)
    mkb = lambda idx: {"x": xs[idx], "gt": tg[idx], "mask": torch.ones(len(idx), 1, X, Y), "scene": list(idx)}  # noqa: E731
    net = Tiny()
    with torch.no_grad():
        before = float(s1_loss(net(mkb(list(range(16)))), tg, torch.ones(16, 1, X, Y))[0])
    train(net, mkb, 16, 300, bs=8, lr=1e-2, warmup=20, device="cpu", log_every=0)
    with torch.no_grad():
        after = float(s1_loss(net(mkb(list(range(16)))), tg, torch.ones(16, 1, X, Y))[0])
    assert after < 0.7 * before, (before, after)
    mm = evaluate(net, mkb, list(range(16)), names, grid, device="cpu")
    assert summary(mm, names)["a"]["all"][0] > 0.6, summary(mm, names)
    print(f"train_s1 selftest ok: loss (perfect < wrong, masked cells weigh 0), meters per scene (perfect 1.0, shift of one "
          f"cell: strict 0, 0.5 m tolerance 1), range bands, paired bootstrap (identical 0 in the interval; better in every "
          f"scene above 0), null prior map, tiny training {before:.2f} -> {after:.2f}")


if __name__ == "__main__":
    _selftest()
