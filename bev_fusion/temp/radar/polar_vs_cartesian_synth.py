"""Synthetic test (2026-10-25): where should the two 3x3 convolutions of the radar block run, does a soft footprint help,
does it still help when the footprint width is LEARNED (not the generator's exact noise), and does an area-averaged
(anti-aliased) polar-to-Cartesian pull fix the near-range weakness of the polar variants?

World: one radar at the origin looking along +x (field of view +-60 deg), point reflectors (pedestrian / pole like) at random
positions, 3 sweeps each (detection probability 0.8), plus clutter; the detections carry range / azimuth noise that grows with
range (sigma_r = 0.15 + 0.003 r m, sigma_phi = 0.6 + 0.02 r deg, scaled by a quality code 0.7 / 1.0 / 1.5 that is a feature).
Task: predict the true reflector positions (3 x 3 cells around them) on the 0.5 m Cartesian grid (x 0..60 m, y -26..26 m).
Doppler and RCS cues are reduced to one weak RCS feature: this isolates the decoding of the POSITION uncertainty.

Variants (same 10 input channels = [mass, mass-weighted 7 features, mass-weighted sub-cell offsets], 32-channel 3x3 convs):
  C2/C4  Cartesian grid, hard (nearest cell) raster, 2 / 4 convs      CS2  Cartesian, soft footprint (11 x 11 window), oracle sigma
  P2     polar grid (128 x 132: 0.5 m x 1 deg), hard raster, 2 convs, then a fixed bilinear gather to the Cartesian grid + 1x1 head
  PS2    polar, soft footprint (3 x 5 window), oracle sigma           PS2A  PS2 with an area-averaged pull (4 x 4 sub-samples per cell)
  PS2L / PS2AL / CS2L   the same with a LEARNED sigma: a linear layer on [log(r / 20), quality code one-hot] initialised at
                        (0.3 m, 1 deg) - no knowledge of the generator's noise model
Metrics on 512 fresh scenes, by range of the target cell: average precision (AP) and best-threshold IoU of the heatmap.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/radar/polar_vs_cartesian_synth.py [steps] [seeds] [names]

STATUS 2026-10-25 (parked, polar variants are a later option): the first five variants ran (results in radar_encoding.md 3b);
the added variants (learned sigma, area-averaged pull) have NO numbers: the self-test fails for the area-averaged pull
(assertion "polar -> Cartesian gather geometry (aa=4)", cause not yet diagnosed - the pull or the test tolerance).
"""
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NX, NY, RES, Y0 = 120, 104, 0.5, -26.0
NR, NPHI, DR, DPHI, PHI0 = 128, 132, 0.5, 1.0, -66.0
QS = torch.tensor([0.7, 1.0, 1.5], device=DEV)


def sig(r, q):
    s = QS[q]
    return (0.15 + 0.003 * r) * s, (0.6 + 0.02 * r) * s


def oracle_sigma(det):
    return sig(det["rd"].clamp(min=1.0), det["q"])


class LearnedSigma(nn.Module):
    """sigma_r [m], sigma_phi [deg] per detection from log range and the quality code; starts at (0.3 m, 1 deg)."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)
        nn.init.zeros_(self.lin.weight)
        self.lin.bias.data = torch.tensor([math.log(0.3), math.log(1.0)])

    def forward(self, det):
        x = torch.cat([torch.log(det["rd"].clamp(min=1.0) / 20)[:, None], F.one_hot(det["q"], 3).float()], -1)
        return torch.exp(self.lin(x)).unbind(-1)


def gen(B, g):
    K, S, M = 8, 3, 12
    r = lambda *s: torch.rand(*s, device=DEV, generator=g)
    n = lambda *s: torch.randn(*s, device=DEV, generator=g)
    tr, tp = 5 + 55 * r(B, K), -55 + 110 * r(B, K)
    tx, ty = tr * torch.cos(torch.deg2rad(tp)), tr * torch.sin(torch.deg2rad(tp))
    present = (r(B, K) < 0.7) & (tx < 59) & (ty.abs() < 25)
    q = (r(B, K, S) * 3).long().clamp(max=2)
    sr, sp = sig(tr[..., None].expand(B, K, S), q)
    rd, pd, rcs = tr[..., None] + sr * n(B, K, S), tp[..., None] + sp * n(B, K, S), 1 + n(B, K, S)
    ok = present[..., None] & (r(B, K, S) < 0.8)
    rc, pc, qc, rcsc = 3 + 61 * r(B, M), -60 + 120 * r(B, M), (r(B, M) * 3).long().clamp(max=2), -1 + n(B, M)
    bt = torch.arange(B, device=DEV)[:, None, None].expand(B, K, S)
    bc = torch.arange(B, device=DEV)[:, None].expand(B, M)
    b = torch.cat([bt[ok], bc.reshape(-1)])
    rdv, pdv = torch.cat([rd[ok], rc.reshape(-1)]), torch.cat([pd[ok], pc.reshape(-1)])
    qv, rcsv = torch.cat([q[ok], qc.reshape(-1)]), torch.cat([rcs[ok], rcsc.reshape(-1)])
    f = torch.stack([rdv / 60, torch.cos(torch.deg2rad(pdv)), torch.sin(torch.deg2rad(pdv)), rcsv,
                     (qv == 0).float(), (qv == 1).float(), (qv == 2).float()], -1)
    tgt = torch.zeros(B, NX, NY, device=DEV)
    ix, iy = (tx / RES).floor().long(), ((ty - Y0) / RES).floor().long()
    bb = torch.arange(B, device=DEV)[:, None].expand(B, K)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            jx, jy = ix + dx, iy + dy
            m = present & (jx >= 0) & (jx < NX) & (jy >= 0) & (jy < NY)
            tgt[bb[m], jx[m], jy[m]] = 1
    return dict(b=b, rd=rdv, pd=pdv, q=qv, f=f), tgt


def build(B, H, W, det, ih, iw, w, off):
    inside = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
    w = w * inside
    w = w / w.sum(1, keepdim=True).clamp(min=1e-9)
    flat = ((det["b"][:, None] * H + ih.clamp(0, H - 1)) * W + iw.clamp(0, W - 1)).reshape(-1)
    vals = torch.cat([w[..., None], w[..., None] * det["f"][:, None, :], w[..., None] * off], -1).reshape(-1, 10)
    out = torch.zeros(B * H * W, 10, device=DEV)
    out.index_add_(0, flat, vals)
    return out.view(B, H, W, 10).permute(0, 3, 1, 2).contiguous()


def gauss_w(rc, pc, det, sg):
    sr, sp = sg
    return torch.exp(-0.5 * (((rc - det["rd"][:, None]) / sr[:, None]) ** 2 + ((pc - det["pd"][:, None]) / sp[:, None]) ** 2))


def polar_hard(det, B, sg=None):
    ir, ip = (det["rd"] / DR).floor().long()[:, None], ((det["pd"] - PHI0) / DPHI).floor().long()[:, None]
    off = torch.stack([det["rd"][:, None] / DR - ir - 0.5, (det["pd"][:, None] - PHI0) / DPHI - ip - 0.5], -1)
    return build(B, NR, NPHI, det, ir, ip, torch.ones_like(ir, dtype=torch.float32), off)


def polar_soft(det, B, sg):
    ir, ip = (det["rd"] / DR).floor().long(), ((det["pd"] - PHI0) / DPHI).floor().long()
    dr, dp = torch.meshgrid(torch.arange(-1, 2, device=DEV), torch.arange(-2, 3, device=DEV), indexing="ij")
    jr, jp = ir[:, None] + dr.reshape(1, -1), ip[:, None] + dp.reshape(1, -1)
    w = gauss_w((jr + 0.5) * DR, PHI0 + (jp + 0.5) * DPHI, det, sg)
    off = torch.stack([det["rd"][:, None] / DR - jr - 0.5, (det["pd"][:, None] - PHI0) / DPHI - jp - 0.5], -1)
    return build(B, NR, NPHI, det, jr, jp, w, off)


def cart_xy(det):
    return det["rd"] * torch.cos(torch.deg2rad(det["pd"])), det["rd"] * torch.sin(torch.deg2rad(det["pd"]))


def cart_hard(det, B, sg=None):
    x, y = cart_xy(det)
    ix, iy = (x / RES).floor().long()[:, None], ((y - Y0) / RES).floor().long()[:, None]
    off = torch.stack([x[:, None] / RES - ix - 0.5, (y[:, None] - Y0) / RES - iy - 0.5], -1)
    return build(B, NX, NY, det, ix, iy, torch.ones_like(ix, dtype=torch.float32), off)


def cart_soft(det, B, sg, R=5):
    x, y = cart_xy(det)
    ix, iy = (x / RES).floor().long(), ((y - Y0) / RES).floor().long()
    dx, dy = torch.meshgrid(torch.arange(-R, R + 1, device=DEV), torch.arange(-R, R + 1, device=DEV), indexing="ij")
    jx, jy = ix[:, None] + dx.reshape(1, -1), iy[:, None] + dy.reshape(1, -1)
    xc, yc = (jx + 0.5) * RES, Y0 + (jy + 0.5) * RES
    w = gauss_w(torch.hypot(xc, yc), torch.rad2deg(torch.atan2(yc, xc)), det, sg)
    off = torch.stack([x[:, None] / RES - jx - 0.5, (y[:, None] - Y0) / RES - jy - 0.5], -1)
    return build(B, NX, NY, det, jx, jy, w, off)


class Net(nn.Module):
    def __init__(self, polar, n_conv, aa=1, learned=False):
        super().__init__()
        layers, c = [], 10
        for _ in range(n_conv):
            layers += [nn.Conv2d(c, 32, 3, padding=1), nn.ReLU()]
            c = 32
        self.body, self.head, self.polar, self.aa = nn.Sequential(*layers), nn.Conv2d(32, 1, 1), polar, aa
        self.sigma = LearnedSigma() if learned else None
        # sample positions of the pull: aa x aa sub-samples per Cartesian cell, averaged after the gather (aa = 1: cell centre)
        xs = ((torch.arange(NX * aa) + 0.5) / aa) * RES
        ys = Y0 + ((torch.arange(NY * aa) + 0.5) / aa) * RES
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        self.register_buffer("grid", torch.stack([2 * ((torch.rad2deg(torch.atan2(Y, X)) - PHI0) / DPHI) / NPHI - 1,
                                                 2 * (torch.hypot(X, Y) / DR) / NR - 1], -1)[None])

    def sigma_for(self, det):
        return self.sigma(det) if self.sigma is not None else oracle_sigma(det)

    def forward(self, x):
        h = self.body(x)
        if self.polar:
            h = F.grid_sample(h, self.grid.expand(len(h), -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=False)
            if self.aa > 1:
                h = F.avg_pool2d(h, self.aa)
        return self.head(h)[:, 0]


# name: (polar, convs, rasteriser, writes per detection, learned sigma, pull sub-samples)
VARIANTS = {"C2": (False, 2, cart_hard, 1, False, 1), "C4": (False, 4, cart_hard, 1, False, 1),
            "CS2": (False, 2, cart_soft, 121, False, 1), "CS2L": (False, 2, cart_soft, 121, True, 1),
            "P2": (True, 2, polar_hard, 1, False, 1), "PS2": (True, 2, polar_soft, 15, False, 1),
            "PS2L": (True, 2, polar_soft, 15, True, 1), "PS2A": (True, 2, polar_soft, 15, False, 4),
            "PS2AL": (True, 2, polar_soft, 15, True, 4)}
XS, YS = (torch.arange(NX, device=DEV) + 0.5) * RES, Y0 + (torch.arange(NY, device=DEV) + 0.5) * RES
RANGE = torch.hypot(XS[:, None], YS[None, :])
BINS = [(0, 20), (20, 40), (40, 61)]


def ap_iou(s, y):
    o = torch.argsort(s, descending=True)
    ys = y[o]
    tp = torch.cumsum(ys, 0)
    ap = ((tp / torch.arange(1, len(ys) + 1, device=s.device)) * ys).sum() / ys.sum().clamp(min=1)
    best = 0.0
    for t in (0.05, 0.1, 0.2, 0.3, 0.5):
        p = s >= t
        tpc = (p & (y > 0)).sum().item()
        best = max(best, tpc / max(1, p.sum().item() + (y > 0).sum().item() - tpc))
    return ap.item(), best


@torch.no_grad()
def evaluate(net, rast, tests):
    net.eval()
    sc = [[] for _ in BINS]
    lb = [[] for _ in BINS]
    for det, tgt, B in tests:
        p = torch.sigmoid(net(rast(det, B, net.sigma_for(det))))
        for i, (lo, hi) in enumerate(BINS):
            m = (RANGE >= lo) & (RANGE < hi)
            sc[i].append(p[:, m].flatten())
            lb[i].append(tgt[:, m].flatten())
    return [ap_iou(torch.cat(sc[i]), torch.cat(lb[i])) for i in range(len(BINS))]


def selftest():
    g = torch.Generator(device=DEV)
    g.manual_seed(5)
    det, tgt = gen(8, g)
    # gather geometry: a polar map whose value is the bin-centre range must come back as the Cartesian cell's range
    # (plain pull, and the area-averaged pull on this linear map)
    ph = torch.rad2deg(torch.atan2(YS[None, :], XS[:, None]))
    inner = (RANGE > 2) & (RANGE < 63) & (ph.abs() < 64)
    pm = ((torch.arange(NR, device=DEV).float() + 0.5) * DR)[None, None, :, None].expand(1, 1, NR, NPHI).contiguous()
    for aa in (1, 4):
        net = Net(True, 0, aa=aa).to(DEV)
        got = F.grid_sample(pm, net.grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if aa > 1:
            got = F.avg_pool2d(got, aa)
        assert (got[0, 0][inner] - RANGE[inner]).abs().max() < 0.06, f"polar -> Cartesian gather geometry (aa={aa})"
    # footprint weights sum to 1 (mass = number of detections), the hard raster counts every detection inside the grid
    for fn in (polar_soft, cart_soft):
        out = fn(det, 8, oracle_sigma(det))
        assert abs(out[:, 0].sum().item() - float(det["b"].numel())) < 0.02 * det["b"].numel() + 40, fn.__name__
    hard = cart_hard(det, 8)
    x, y = cart_xy(det)
    inside = (x >= 0) & (x < 60) & (y >= Y0) & (y < Y0 + 52)
    assert abs(hard[:, 0].sum().item() - inside.sum().item()) < 1e-3, "hard raster count"
    # the learned sigma gets a gradient from the loss
    ls = LearnedSigma().to(DEV)
    out = polar_soft(det, 8, ls(det))
    out[:, 1:].square().sum().backward()
    assert ls.lin.bias.grad is not None and ls.lin.bias.grad.abs().sum() > 0, "learned sigma gradient"
    print("selftest ok: gather geometry (plain and area-averaged), footprint mass, hard raster count, learned-sigma gradient")


def run(name, seed, steps, tests, B=16):
    polar, nconv, rast, writes, learned, aa = VARIANTS[name]
    torch.manual_seed(seed)
    g = torch.Generator(device=DEV)
    g.manual_seed(1000 + seed)
    net = Net(polar, nconv, aa=aa, learned=learned).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    posw = torch.tensor(5.0, device=DEV)
    for it in range(steps):
        net.train()
        det, tgt = gen(B, g)
        loss = F.binary_cross_entropy_with_logits(net(rast(det, B, net.sigma_for(det))), tgt, pos_weight=posw)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    res = evaluate(net, rast, tests)
    info = ""
    if learned:
        with torch.no_grad():
            d = {"rd": torch.tensor([10.0, 30.0, 50.0], device=DEV), "q": torch.tensor([1, 1, 1], device=DEV)}
            sr, sp = net.sigma(d)
            info = " learned sigma_r " + "/".join(f"{v:.2f}" for v in sr.tolist()) + " m, sigma_phi " + "/".join(f"{v:.2f}" for v in sp.tolist()) + " deg at 10/30/50 m (q=1)"
    g1 = torch.Generator(device=DEV)
    g1.manual_seed(3)
    d1, _ = gen(1, g1)
    net.eval()
    with torch.no_grad():
        for _ in range(10):
            net(rast(d1, 1, net.sigma_for(d1)))
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            net(rast(d1, 1, net.sigma_for(d1)))
        if DEV == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 20
    return res, sum(p.numel() for p in net.parameters()), ms, info


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else list(VARIANTS)
    selftest()
    gt = torch.Generator(device=DEV)
    gt.manual_seed(777)
    tests = [(*gen(128, gt), 128) for _ in range(4)]
    print(f"device {DEV}; {steps} steps x batch 16, {seeds} seeds; test: 512 scenes; AP / best IoU per range of the target cell")
    print(f"{'':6s}{'writes':>7s}{'params':>8s}{'ms/scene':>9s} | {'0-20 m':>13s} | {'20-40 m':>13s} | {'40-60 m':>13s}", flush=True)
    for name in names:
        rs, ms, params, info = [], [], 0, ""
        for s in range(seeds):
            r, params, m, info = run(name, s, steps, tests)
            rs.append(r)
            ms.append(m)
        cells = []
        for i in range(len(BINS)):
            ap = sum(r[i][0] for r in rs) / seeds
            iou = sum(r[i][1] for r in rs) / seeds
            cells.append(f"{ap:.3f} / {iou:.3f}")
        print(f"{name:6s}{VARIANTS[name][3]:7d}{params:8d}{sum(ms) / len(ms):9.2f} | " + " | ".join(f"{c:>13s}" for c in cells)
              + f"   (AP spread between seeds {max(r[2][0] for r in rs) - min(r[2][0] for r in rs):.3f} at 40-60 m){info}", flush=True)
