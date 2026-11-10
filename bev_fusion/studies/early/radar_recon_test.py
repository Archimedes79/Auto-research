"""Reconstruction test (2026-10-25): do the radar BEV features keep the detections? "decoded to points, information is not lost".

Real nuScenes mini, cached multi-sweep detections (window 0.5 s, ego-aligned; temp/radar/radar_multisweep_cache.py), split by scene
(8 scenes train, 2 test). Two rasters of the same detections:
  M  the simple raster (count, mean properties, mean sub-cell offset; 14 channels)
  S  the slot raster (K = 3 exact slots + a remainder mean per cell; 57 channels)
For each: the block's two 3x3 convolutions (C channels) + a 1x1 decoder (2 layers) trained to reconstruct the raster from the features
(valid / occupancy logit by BCE, properties and offsets by L1 where valid). Then, PER ORIGINAL DETECTION of the test frames, the
error of the decoded vector at its cell (slot) against its own values - physical units - and the share recovered within a tolerance.
"oracle" rows decode the raster itself (no learning): the information ceiling of the representation.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python studies/early/radar_recon_test.py [steps]
"""
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "training")
import radar_bev as rb  # noqa: E402
from lift_pd import Grid  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID, K = Grid(), 3
P, W = rb.n_properties(), 1 + rb.n_properties() + 2          # 11, 14
KEYS = ["x", "y", "r", "psi", "vr", "rcs", "dt"]
TOL = {"pos (m)": 0.25, "dist (m)": 0.5, "dir (deg)": 3.0, "vr (m/s)": 0.5, "rcs (dB)": 3.0, "age (s)": 0.1}


def load(scenes):
    fr = [f for f in torch.load("temp/radar/cache/mini_multisweep.pt") if f["scene"] in scenes]
    out = []
    for f in fr:
        m = f["dt"] <= 0.5
        out.append({k: f[k][m].to(DEV) for k in KEYS})
    return out


def batch(frames):
    det = {k: torch.cat([f[k] for f in frames]) for k in KEYS}
    det["b"] = torch.cat([torch.full((len(f["x"]),), i, dtype=torch.long, device=DEV) for i, f in enumerate(frames)])
    return det


def det_slots(det):
    """ix, iy, rank (position in the cell by age, then distance) and inside mask per original detection (as rasterize_slots)."""
    fx, fy = (det["x"] - GRID.x_min) / GRID.res, (det["y"] - GRID.y_min) / GRID.res
    ix, iy = fx.floor().long(), fy.floor().long()
    inside = (ix >= 0) & (ix < GRID.nx) & (iy >= 0) & (iy < GRID.ny)
    idx = torch.nonzero(inside).squeeze(1)
    cell = (det["b"][idx] * GRID.nx + ix[idx]) * GRID.ny + iy[idx]
    order = torch.argsort(det["r"][idx], stable=True)
    order = order[torch.argsort(det["dt"][idx][order], stable=True)]
    order = order[torch.argsort(cell[order], stable=True)]
    _, inv, cnt = torch.unique_consecutive(cell[order], return_inverse=True, return_counts=True)
    rank_sorted = torch.arange(len(order), device=DEV) - (torch.cumsum(cnt, 0) - cnt)[inv]
    rank = torch.empty_like(idx)
    rank[order] = rank_sorted
    return idx, ix[idx], iy[idx], rank, fx[idx] - ix[idx] - 0.5, fy[idx] - iy[idx] - 0.5


def physical(v):
    """(n, 13) = 11 properties + 2 offsets -> dict of physical quantities."""
    H = len(rb.HARMONICS)
    return {"dist (m)": v[:, 0] * rb.RANGE_SCALE, "psi": torch.atan2(v[:, 2], v[:, 1]),
            "vr": torch.atan2(v[:, 3], v[:, 4]) * rb.V_MAX / math.pi, "rcs (dB)": v[:, 3 + 2 * H] * rb.RCS_SCALE,
            "age (s)": v[:, 4 + 2 * H] * rb.AGE_SCALE, "ox": v[:, P], "oy": v[:, P + 1]}


def errors(pred, own):
    a, b = physical(pred), physical(own)
    dpsi = torch.rad2deg(((a["psi"] - b["psi"]) + math.pi) % (2 * math.pi) - math.pi).abs()
    dvr = (((a["vr"] - b["vr"]) + rb.V_MAX) % (2 * rb.V_MAX) - rb.V_MAX).abs()
    return {"pos (m)": torch.hypot(a["ox"] - b["ox"], a["oy"] - b["oy"]) * GRID.res, "dist (m)": (a["dist (m)"] - b["dist (m)"]).abs(),
            "dir (deg)": dpsi, "vr (m/s)": dvr, "rcs (dB)": (a["rcs (dB)"] - b["rcs (dB)"]).abs(), "age (s)": (a["age (s)"] - b["age (s)"]).abs()}


class AE(nn.Module):
    def __init__(self, cin, C, cout):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(cin, C, 3, padding=1), nn.ReLU(), nn.Conv2d(C, C, 3, padding=1), nn.ReLU())
        self.dec = nn.Sequential(nn.Conv2d(C, C, 1), nn.ReLU(), nn.Conv2d(C, cout, 1))

    def forward(self, x):
        return self.dec(self.enc(x))


def blocks(kind):
    return [(0, 1, W)] if kind == "M" else [(1 + k * W, 2 + k * W, 1 + (k + 1) * W) for k in range(K + 1)]


def raster(kind, det, B):
    return rb.rasterize(det, B, GRID) if kind == "M" else rb.rasterize_slots(det, B, GRID, K)


def loss_fn(out, tgt, kind):
    loss = 0.0
    if kind == "S":
        loss = loss + F.l1_loss(out[:, 0], torch.log1p(tgt[:, 0]))
    for vi, v0, v1 in blocks(kind):
        valid = (tgt[:, vi] > 0.5).float()
        loss = loss + F.binary_cross_entropy_with_logits(out[:, vi], valid)
        m = valid[:, None]
        loss = loss + 5 * ((out[:, v0:v1] - tgt[:, v0:v1]).abs() * m).sum() / (m.sum() * (v1 - v0)).clamp(min=1)
    return loss


def train(kind, C, steps, train_frames, seed=0):
    torch.manual_seed(seed)
    cin = 1 + P + 2 if kind == "M" else 1 + (K + 1) * W
    cin = W if kind == "M" else cin
    net = AE(cin, C, cin).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = np.random.RandomState(seed)
    for it in range(steps):
        fs = [train_frames[i] for i in g.choice(len(train_frames), 8, replace=False)]
        det = batch(fs)
        tgt = raster(kind, det, 8)
        loss = loss_fn(net(tgt), tgt, kind)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net


@torch.no_grad()
def evaluate(kind, net, test_frames):
    acc = {k: [] for k in TOL}
    ok_valid = []
    for i in range(0, len(test_frames), 8):
        fs = test_frames[i:i + 8]
        det = batch(fs)
        tgt = raster(kind, det, len(fs))
        out = tgt if net is None else net(tgt)
        idx, ix, iy, rank, ox, oy = det_slots(det)
        b = det["b"][idx]
        blk = torch.zeros_like(rank) if kind == "M" else rank.clamp(max=K)
        vi = torch.zeros_like(rank) if kind == "M" else 1 + blk * W
        v0 = vi + 1
        cols = v0[:, None] + torch.arange(W - 1, device=DEV)[None]
        vec = out[b[:, None], cols, ix[:, None], iy[:, None]]                 # (n, 13) decoded vector at the detection's cell / slot
        valid = out[b, vi, ix, iy] > (0.5 if net is None else 0.0)             # logit > 0 (probability 0.5); raster channel > 0.5
        props = rb.properties({k: det[k][idx] for k in KEYS}, rb.V_MAX, rb.HARMONICS)
        own = torch.cat([props, ox[:, None], oy[:, None]], 1)
        e = errors(vec, own)
        for k in acc:
            acc[k].append(torch.where(valid, e[k], torch.full_like(e[k], float("inf"))))
        ok_valid.append(valid)
    e = {k: torch.cat(v) for k, v in acc.items()}
    ok = torch.ones_like(e["pos (m)"], dtype=torch.bool)
    for k, t in TOL.items():
        ok &= e[k] <= t
    return e, ok


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    tr, te = load(set(range(8))), load({8, 9})
    print(f"device {DEV}; train {len(tr)} frames, test {len(te)} frames; detections of the test frames inside the grid: "
          f"{sum(int(((f['x'] >= 0) & (f['x'] < 60) & (f['y'] >= -26) & (f['y'] < 26)).sum()) for f in te)}")
    print("share of detections recovered within " + ", ".join(f"{k} {v}" for k, v in TOL.items()) + " (all at once);")
    print(f"{'':26s}{'recovered':>10s} | " + " | ".join(f"{k.split(' ')[0] + ' med/p90':>16s}" for k in TOL))
    rows = [("M oracle (raster itself)", "M", None, 0), ("M learned, C = 32", "M", 32, 1), ("S oracle (raster itself)", "S", None, 0),
            ("S learned, C = 32", "S", 32, 1), ("S learned, C = 64", "S", 64, 1), ("S learned, C = 128", "S", 128, 1)]
    if len(sys.argv) > 2:                                                       # only the rows whose name contains one of the given texts
        rows = [r for r in rows if any(t in r[0] for t in sys.argv[2].split(","))]
    for name, kind, C, learn in rows:
        net = train(kind, C, steps, tr) if learn else None
        if net is not None:
            net.eval()
        e, ok = evaluate(kind, net, te)
        cells = []
        for k in TOL:
            v = e[k][torch.isfinite(e[k])].cpu().numpy()
            cells.append(f"{np.median(v):.3f} / {np.quantile(v, 0.9):.3f}" if len(v) else "n/a")
        print(f"{name:26s}{100 * ok.float().mean().item():9.1f}% | " + " | ".join(f"{c:>16s}" for c in cells), flush=True)
