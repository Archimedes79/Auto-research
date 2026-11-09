"""Does the stride-1 detail path help the BEV trunk localise thin lines? (synthetic, 2026-10-25)

Task: the input is a blurred, noisy view of 3-6 random straight lines (some in parallel pairs 3 cells apart) on the 120 x 104
grid of 0.5 m cells plus 15 noise channels (the trunk has to find the signal), the target is the sharp ONE-CELL-wide line map
(a lane divider is 1-2 cells wide). Arms: T2 (width 1/2, light up-sampling: the plan's default) and T2 + detail path (32 channels),
same seeds, steps and data. Metrics on 64 fresh scenes: IoU of the one-cell lines, the tolerant F1 (within +-1 cell of the target
line), and the thickness (predicted cells / true cells at the best threshold).
Settings: blur sigma (cells) and signal noise; steps. Lines are drawn on the CPU (numpy).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/trunk_thin_lines.py <steps> <seeds> <sigma> <noise>
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "training")
from bev_trunk import BEVTrunk  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, W, CIN = 120, 104, 16


def lines_map(B, rng):
    m = np.zeros((B, H, W), np.float32)
    t = np.linspace(0, 1, 400)[:, None]
    for b in range(B):
        for _ in range(rng.randint(3, 7)):
            p0 = rng.rand(2) * [H, W]
            ang, length = 2 * np.pi * rng.rand(), 30 + 50 * rng.rand()
            d = np.array([np.cos(ang), np.sin(ang)])
            pts = p0 + t * length * d
            for o in [0.0] + ([3.0] if rng.rand() < 0.5 else []):                      # an optional parallel partner 3 cells away
                p = np.rint(pts + o * np.array([-d[1], d[0]])).astype(int)
                ok = (p[:, 0] >= 0) & (p[:, 0] < H) & (p[:, 1] >= 0) & (p[:, 1] < W)
                m[b, p[ok, 0], p[ok, 1]] = 1
    return torch.from_numpy(m)


def gaussian_kernel(sigma, r):
    ax = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (ax[:, None] ** 2 + ax[None] ** 2) / sigma ** 2)
    return (k / k.sum())[None, None].to(DEV)


def batch(B, rng, g, sigma, noise):
    tgt = lines_map(B, rng)[:, None].to(DEV)
    r = int(np.ceil(3 * sigma))
    sig = F.conv2d(tgt, gaussian_kernel(sigma, r), padding=r) * (3.0 * sigma)      # keep a similar peak height
    sig = sig + noise * torch.randn(B, 1, H, W, device=DEV, generator=g)
    x = torch.cat([sig, 0.3 * torch.randn(B, CIN - 1, H, W, device=DEV, generator=g)], 1)
    return x, tgt


def train_eval(detail, seed, steps, sigma, noise):
    torch.manual_seed(seed)
    rng, rt = np.random.RandomState(100 + seed), np.random.RandomState(999)
    g, gt = torch.Generator(device=DEV), torch.Generator(device=DEV)
    g.manual_seed(100 + seed)
    gt.manual_seed(999)
    net = BEVTrunk(cin=CIN, n_classes=1, width=0.5, light_up=True, detail=detail).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    posw = torch.tensor(10.0, device=DEV)
    for _ in range(steps):
        net.train()
        x, tgt = batch(8, rng, g, sigma, noise)
        loss = F.binary_cross_entropy_with_logits(net(x), tgt, pos_weight=posw)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    best = (0, 0, 0, 0)
    with torch.no_grad():
        xs, ts = [], []
        for _ in range(8):
            x, t = batch(8, rt, gt, sigma, noise)
            xs.append(torch.sigmoid(net(x)))
            ts.append(t)
        p, t = torch.cat(xs), torch.cat(ts)
        td = F.max_pool2d(t, 3, 1, 1)                                             # target dilated by 1 cell
        for th in (0.3, 0.5, 0.7):
            pb = (p > th).float()
            tp = (pb * t).sum()
            iou = tp / (pb.sum() + t.sum() - tp).clamp(min=1)
            pd = F.max_pool2d(pb, 3, 1, 1)
            prec = (pb * td).sum() / pb.sum().clamp(min=1)                         # predicted cells within 1 cell of a line
            rec = (t * pd).sum() / t.sum().clamp(min=1)                            # line cells with a prediction within 1 cell
            f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
            if iou.item() > best[0]:
                best = (iou.item(), f1.item(), (pb.sum() / t.sum()).item(), th)
    return best


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    noise = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
    print(f"device {DEV}; {steps} steps x batch 8, {seeds} seeds; blur sigma {sigma} cells, signal noise {noise}; 64 test scenes")
    print(f"{'arm':26s}{'IoU':>8s}{'F1 (+-1 cell)':>15s}{'thickness':>11s}")
    for name, detail in (("T2 (default)", 0), ("T2 + detail path (32 ch)", 32)):
        rs = [train_eval(detail, s, steps, sigma, noise) for s in range(seeds)]
        print(f"{name:26s}{sum(r[0] for r in rs) / seeds:8.3f}{sum(r[1] for r in rs) / seeds:15.3f}{sum(r[2] for r in rs) / seeds:11.2f}"
              f"   (IoU per seed " + ", ".join(f"{r[0]:.3f}" for r in rs) + ")", flush=True)
