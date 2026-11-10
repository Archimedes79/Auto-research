"""Information preservation of the radar encoder (Mirko, 2026-10-26): how much of the detection cloud survives encoding?

Three levels, cheap first:
  A  unit tests with known answers (no learning): the rasters are exact where they claim to be - one detection per cell
     (simple raster and slot raster), up to K per cell (slot raster) + the remainder mean, Doppler near the wrap, frames
  B  the ceiling of the REPRESENTATION on real nuScenes mini (multi-sweep cache, held-out scenes): which share of the
     detections can be recovered at all from the raster - simple raster (one per cell) and slot raster with K = 1..6,
     at 0.5 m and 0.25 m; computed from the cell occupancy (a detection is recoverable if it has its own slot, or is the
     only one in the remainder)
  C  (--learned) the LEARNED encoder (two 3x3 convs, C channels) probed by a small 1x1 decoder, trained on 8 scenes,
     measured on the 2 held-out scenes AND on the training scenes (the gap = overfitting), against the ceiling of B:
       plain      the L1 loss on the normalised raster channels (studies/early/radar_recon_test.py)
       tolweight  every channel weighted by 1 / its tolerance in normalised units (range 0.5 m, direction 3 deg,
                  Doppler 0.5 m/s, RCS 3 dB, age 0.1 s, position 0.25 m): the loss spends the capacity where the sensor is
                  precise - the test criterion and the loss then measure the same thing
       skip       the encoder's features concatenated with the lossless raster (RadarBEVSlots with_state=True): the
                  decoder can read the raster, so this reaches the ceiling by construction (the reference)
A detection counts as recovered if ALL its fields lie within the tolerances at once (studies/early/radar_recon_test.py).
Pass criteria: A exact; C: tolweight on held-out >= 0.9 x the ceiling and train - held-out gap <= 3 points.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python -u studies/early/radar_encoding.py [--learned [steps]]
"""
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "training")
import radar_bev as rb  # noqa: E402
import radar_recon_test as rt  # noqa: E402
from lift_pd import Grid  # noqa: E402

DEV = rt.DEV
fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def det(rows):
    """rows of (b, x, y, r, psi, vr, rcs, dt) -> detection dict on the device."""
    t = torch.tensor(rows, dtype=torch.float32, device=DEV)
    d = {k: t[:, i] for i, k in enumerate(["b", "x", "y", "r", "psi", "vr", "rcs", "dt"])}
    d["b"] = d["b"].long()
    return d


# ------------------------------------------------------------------------------------------------ A: unit tests
def level_a():
    print("A  unit tests with known answers")
    g = Grid()
    one = det([[0, 20.3, 1.1, 18.0, 0.05, 7.5, 12.0, 0.0], [0, 35.6, -4.4, 34.9, -0.13, -12.0, 3.0, 0.25],
               [1, 20.3, 1.1, 18.2, 0.06, 0.4, -5.0, 0.4]])
    m = rb.rasterize(one, 2, g)
    ix, iy = int((20.3 - g.x_min) / g.res), int((1.1 - g.y_min) / g.res)
    check(m[0, 0, ix, iy] == 1 and m[1, 0, ix, iy] == 1, "simple raster: one detection per cell, counted per frame")
    s = rb.rasterize_slots(one, 2, g, K=3)
    dec, rem = rb.decode_slots(s, g, K=3)
    check(len(dec["x"]) == 3 and len(rem["b"]) == 0, "slot raster: three single detections -> three slots, no remainder")
    for i in range(3):
        j = int(torch.argmin((dec["x"] - one["x"][i]).abs() + (dec["y"] - one["y"][i]).abs() + 10 * (dec["b"] - one["b"][i]).abs()))
        err = max(abs(dec[k][j].item() - one[k][i].item()) for k in ("x", "y", "r", "vr", "rcs", "dt"))
        dpsi = abs(math.remainder(dec["psi"][j].item() - one["psi"][i].item(), 2 * math.pi))
        check(err < 1e-4 and dpsi < 1e-5, f"slot raster: detection {i} decoded exactly (max error {err:.1e})")
    for n in (1, 3, 5):                                   # n detections in one cell, K = 3
        rows = [[0, 30.1 + 0.07 * i, 2.2 + 0.05 * i, 29.0 + i, 0.1, 2.0 * i, 5.0, 0.05 * i] for i in range(n)]
        s = rb.rasterize_slots(det(rows), 1, g, K=3)
        dec, rem = rb.decode_slots(s, g, K=3)
        exact = sum(1 for r in rows if ((dec["x"] - r[1]).abs() < 1e-4).any() and ((dec["vr"] - r[5]).abs() < 1e-3).any())
        check(exact == min(n, 3) and (len(rem["b"]) == (1 if n > 3 else 0)),
              f"slot raster: {n} in one cell -> {min(n, 3)} exact slots, remainder {'with ' + str(n - 3) if n > 3 else 'empty'}")
    wrap = det([[0, 25.2, 0.3, 25.0, 0.0, 29.9, 0.0, 0.0], [0, 26.2, 0.3, 26.0, 0.0, -29.9, 0.0, 0.0]])
    dec, _ = rb.decode_slots(rb.rasterize_slots(wrap, 1, g, K=3), g, K=3)
    check(sorted(round(v, 3) for v in dec["vr"].tolist()) == [-29.9, 29.9], "Doppler near the wrap (+-29.9 m/s) exact")


# ------------------------------------------------------------------------------------------------ B: ceilings
def level_b():
    print("B  ceiling of the representation on real nuScenes mini (held-out scenes 8, 9; 0.5 s of sweeps, all radars)")
    frames = rt.load({8, 9})
    rows = []
    for res in (0.5, 0.25):
        g = Grid(res=res)
        rt.GRID = g
        counts, ranks = [], []
        for i in range(0, len(frames), 8):
            d = rt.batch(frames[i:i + 8])
            idx, ix, iy, rank, _, _ = rt.det_slots(d)
            cell = (d["b"][idx] * g.nx + ix) * g.ny + iy
            _, inv, cnt = torch.unique(cell, return_inverse=True, return_counts=True)
            counts.append(cnt[inv])
            ranks.append(rank)
        cnt, rank = torch.cat(counts), torch.cat(ranks)
        n = len(cnt)
        row = {"grid": f"{res} m", "detections": n, "simple": (cnt == 1).float().mean().item()}
        for K in (1, 2, 3, 4, 6):
            row[f"K{K}"] = ((rank < K) | (cnt == K + 1)).float().mean().item()
        rows.append(row)
        hist = torch.bincount(cnt.clamp(max=10).cpu(), minlength=11)[1:].tolist()
        print(f"  {res} m grid: detections by cell occupancy 1..9, >=10: {hist}")
    rt.GRID = Grid()
    # B2: a detection is "represented" if ANY entry stored for its cell (its own slot, another slot, or the remainder
    # mean) matches it within all tolerances - repeated sweeps of one static reflector are redundant measurements, so
    # this is the information criterion; the per-detection count above is the strict one
    g = Grid()
    rep = {K: [] for K in (1, 2, 3)}
    for i in range(0, len(frames), 8):
        d = rt.batch(frames[i:i + 8])
        idx, ix, iy, rank, ox, oy = rt.det_slots(d)
        own = torch.cat([rb.properties({k: d[k][idx] for k in rt.KEYS}, rb.V_MAX, rb.HARMONICS), ox[:, None], oy[:, None]], 1)
        cell = (d["b"][idx] * g.nx + ix) * g.ny + iy
        uc, inv = torch.unique(cell, return_inverse=True)
        for K in rep:
            ok = rank < K
            stored = [torch.full((len(uc), own.shape[1]), float("nan"), device=DEV) for _ in range(K)]
            for k in range(K):
                m = rank == k
                stored[k][inv[m]] = own[m]
            mrem = rank >= K
            s = torch.zeros(len(uc), own.shape[1], device=DEV).index_add_(0, inv[mrem], own[mrem])
            n = torch.zeros(len(uc), device=DEV).index_add_(0, inv[mrem], torch.ones(int(mrem.sum()), device=DEV))
            stored.append(torch.where(n[:, None] > 0, s / n.clamp(min=1)[:, None], torch.full_like(s, float("nan"))))
            for st in stored:
                cand = st[inv]
                e = rt.errors(torch.nan_to_num(cand, nan=1e3), own)
                within = torch.ones_like(ok)
                for k2, t in rt.TOL.items():
                    within &= e[k2] <= t
                ok |= within & ~torch.isnan(cand[:, 0])
            rep[K].append(ok)
    print("  B2 represented share (some stored entry of the cell within all tolerances), 0.5 m: " +
          ", ".join(f"K={K} {100 * torch.cat(v).float().mean().item():.1f}%" for K, v in rep.items()))
    print("  recoverable share | simple raster | slots K=1 | K=2 | K=3 | K=4 | K=6")
    for r in rows:
        print(f"  {r['grid']:>6s}, {r['detections']} det. | {100 * r['simple']:5.1f}% | " +
              " | ".join(f"{100 * r[f'K{K}']:5.1f}%" for K in (1, 2, 3, 4, 6)))
    return rows[0]["K3"]


# ------------------------------------------------------------------------------------------------ C: learned probes
# tolerance of every channel of a slot block (13 values: range, cos/sin psi, Doppler sin/cos for harmonics 1, 2, 4,
# rcs, age, sub-cell offsets) in its normalised unit
_TOL = [0.5 / rb.RANGE_SCALE, math.radians(3), math.radians(3)]
for _k in rb.HARMONICS:
    _TOL += [_k * math.pi * 0.5 / rb.V_MAX] * 2
_TOL += [3.0 / rb.RCS_SCALE, 0.1 / rb.AGE_SCALE, 0.25 / 0.5, 0.25 / 0.5]
W_TOL = torch.tensor([1.0 / t for t in _TOL], device=DEV)
W_TOL = 5.0 * W_TOL / W_TOL.mean()


class Skip(nn.Module):
    """encoder features + the raster itself into the 1x1 decoder (the lossless reference)."""

    def __init__(self, cin, C):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(cin, C, 3, padding=1), nn.ReLU(), nn.Conv2d(C, C, 3, padding=1), nn.ReLU())
        self.dec = nn.Sequential(nn.Conv2d(C + cin, C, 1), nn.ReLU(), nn.Conv2d(C, cin, 1))

    def forward(self, x):
        return self.dec(torch.cat([self.enc(x), x], 1))


def loss_tol(out, tgt):
    loss = F.l1_loss(out[:, 0], torch.log1p(tgt[:, 0]))
    for vi, v0, v1 in rt.blocks("S"):
        valid = (tgt[:, vi] > 0.5).float()
        loss = loss + F.binary_cross_entropy_with_logits(out[:, vi], valid)
        m = valid[:, None]
        loss = loss + ((out[:, v0:v1] - tgt[:, v0:v1]).abs() * W_TOL.view(1, -1, 1, 1) * m).sum() / (m.sum() * (v1 - v0)).clamp(min=1)
    return loss


def train(variant, C, steps, frames, seed=0):
    torch.manual_seed(seed)
    cin = 1 + (rt.K + 1) * rt.W
    net = (Skip(cin, C) if variant == "skip" else rt.AE(cin, C, cin)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = np.random.RandomState(seed)
    for _ in range(steps):
        fs = [frames[i] for i in g.choice(len(frames), 8, replace=False)]
        tgt = rt.raster("S", rt.batch(fs), 8)
        out = net(tgt)
        loss = loss_tol(out, tgt) if variant in ("tolweight", "skip") else rt.loss_fn(out, tgt, "S")
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


def level_c(steps, ceiling):
    print(f"C  learned encoder (2 x 3x3 conv) + 1x1 decoder, slot raster K = 3, {steps} steps; recovered = all fields "
          "within tolerance; ceiling (B, K = 3, 0.5 m) " + f"{100 * ceiling:.1f}%")
    tr, te = rt.load(set(range(8))), rt.load({8, 9})
    _, ok_or = rt.evaluate("S", None, te)                  # the oracle decodes the raster itself: own slot OR the remainder mean
    ceiling = ok_or.float().mean().item()                  # (the remainder matches the redundant sweeps: 95.0%, above the strict 87.7%)
    print(f"  oracle decode of the raster: {100 * ceiling:.1f}% - this is the ceiling used below")
    print("  variant           | held-out | train | gap  | held-out median: " + ", ".join(rt.TOL))
    res = {}
    for name, variant, C in (("plain C=64", "plain", 64), ("tolweight C=64", "tolweight", 64),
                             ("tolweight C=128", "tolweight", 128), ("skip C=32", "skip", 32)):
        net = train(variant, C, steps, tr)
        e, ok = rt.evaluate("S", net, te)
        _, ok_tr = rt.evaluate("S", net, tr)
        med = ", ".join(f"{np.median(e[k][torch.isfinite(e[k])].cpu().numpy()):.3f}" for k in rt.TOL)
        h, t = ok.float().mean().item(), ok_tr.float().mean().item()
        res[name] = (h, t)
        print(f"  {name:17s} | {100 * h:7.1f}% | {100 * t:4.1f}% | {100 * (t - h):+4.1f} | {med}", flush=True)
    h, t = res["tolweight C=64"]
    check(h >= 0.9 * ceiling, f"tolweight C=64 keeps >= 90 % of the ceiling on held-out scenes ({100 * h:.1f}% vs {100 * ceiling:.1f}%)")
    check(t - h <= 0.03, f"train - held-out gap <= 3 points ({100 * (t - h):+.1f})")


if __name__ == "__main__":
    level_a()
    ceiling = level_b()
    if "--learned" in sys.argv:
        i = sys.argv.index("--learned")
        steps = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 and sys.argv[i + 1].isdigit() else 2000
        level_c(steps, ceiling)
    print("RADAR ENCODING GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
    sys.exit(1 if fails else 0)
