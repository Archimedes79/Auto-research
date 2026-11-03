"""Gate, second attempt (plan.md 10.1 / 10.1.1): velocity field from two BEV frames with the fixes from existing work.

Same data as studies/early/flow_from_frames.py (synthetic world, frames 1/12 s apart, features lanes / vehicles /
pedestrians without a velocity channel; the newer frame ego-pulled into the older one, so the exact ego flow is
subtracted and only object motion is left). Changes (literature/fixes_for_gate_findings.md):
- network: a shared small encoder for both frames, a LOCAL CORRELATION volume over +-4 cells (81 displacements,
  enough for 15 m/s x 1/12 s = 2.5 cells), soft-argmax displacement -> velocity, plus a refinement CNN on
  [correlation, encoder features, soft-argmax] that adds a residual (zero start) and spreads edge evidence into the
  uniform interior (SelfOccFlow / ContrastMotion style correlation, RAFT-like)
- self-supervised loss by PULL (backward warping, the optical-flow standard; gives position gradients everywhere):
  the ego-aligned newer frame sampled at x + v dt must equal the older frame at x, on blurred maps (sigma 2, 1, 0),
  cells where the ego alignment left no mass masked out, a static-mask weight (cells unchanged between the aligned
  frames weigh 0.1, changed cells 1: SPML-style, re-implemented), plus a smoothness term inside occupied cells
- then supervised fine-tuning from the self-supervised weights; controls: supervised from scratch with the same
  number of steps as the fine-tuning, and with the full step budget
Evaluation as before: mean end-point error on moving-box cells, per box mean / median, against the null v = 0.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python studies/early/flow_correlation.py
(the pairs are cached in temp/cache/flow_pairs.pt; delete it after a change of the synthetic world)
"""
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "training")
sys.path.insert(0, "tests/support")
from lift_pd import Grid  # noqa: E402
from splat import splat_bev  # noqa: E402
from synthetic_world import gt_bev, random_scene  # noqa: E402
G = Grid()
DT = 1 / 12
R = 4
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # tiny network; the data is made on the CPU
CACHE = "temp/cache/flow_pairs.pt"                                    # generated pairs (not in git)
torch.set_num_threads(8)


def blur(x, sigma):
    if sigma <= 0:
        return x
    r = int(math.ceil(2.5 * sigma))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    C = x.shape[1]
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(C, 1, -1, 1), padding=(r, 0), groups=C)
    return F.conv2d(x, k.view(1, 1, 1, -1).expand(C, 1, 1, -1), padding=(0, r), groups=C)


def make_pair(rng):
    s0 = random_scene(rng)
    s1 = s0.step(DT)
    g0, g1 = gt_bev(s0, G), gt_bev(s1, G)
    f0 = torch.stack([g0["lanes"], g0["vehicles"], g0["pedestrians"]])
    f1 = torch.stack([g1["lanes"], g1["vehicles"], g1["pedestrians"]])
    with torch.no_grad():
        f1_in0, mass, _ = splat_bev(f1[None], G, -DT, s0.v, s0.omega)
    moving = (g0["box_id"] >= 0) & (g0["velocity"].norm(dim=0) > 0.5)
    return {"f0": f0, "f1_in0": f1_in0[0], "mass": mass[0, 0], "vel": g0["velocity"], "obj": g0["box_id"] >= 0,
            "box_id": g0["box_id"], "moving": moving}


def batch(pairs):
    return {k: torch.stack([p[k] for p in pairs]).to(DEV) for k in pairs[0]}


DISP = torch.tensor([(dx, dy) for dx in range(-R, R + 1) for dy in range(-R, R + 1)], dtype=torch.float32)


def correlation(a, b):
    """a, b (B, C, X, Y) -> (B, (2R+1)^2, X, Y): a(x) . b(x + d) / sqrt(C) for all displacements d (in cells)."""
    B, C, X, Y = a.shape
    bp = F.pad(b, (R, R, R, R))
    out = [(a * bp[:, :, R + dx:R + dx + X, R + dy:R + dy + Y]).sum(1) for dx, dy in DISP.long().tolist()]
    return torch.stack(out, 1) / C ** 0.5


class CorrFlow(nn.Module):
    def __init__(self, cin=3, ch=16, hid=48):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(cin, ch, 3, padding=1), nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1))
        self.temp = nn.Parameter(torch.tensor(5.0))
        n = (2 * R + 1) ** 2
        self.refine = nn.Sequential(
            nn.Conv2d(n + ch + 2, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(hid, 2, 3, padding=1))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, f0, f1_in0):
        e0, e1 = self.enc(f0), self.enc(f1_in0)
        corr = correlation(e0, e1)
        p = torch.softmax(self.temp * corr, 1)
        d = torch.einsum("bnxy,nk->bkxy", p, DISP.to(p))            # expected displacement (cells)
        v0 = d * G.res / DT                                                  # m/s
        v = v0 + self.refine(torch.cat([corr, e0, v0 / 10.0], 1))
        return 25.0 * torch.tanh(v / 25.0)


def pull(f, v):
    """sample f (B, C, X, Y) at x + v dt (v in m/s, (B, 2, X, Y): x forward along X, y left along Y)."""
    B, C, X, Y = f.shape
    ix = torch.arange(X, dtype=f.dtype, device=f.device).view(1, X, 1) + v[:, 0] * DT / G.res
    iy = torch.arange(Y, dtype=f.dtype, device=f.device).view(1, 1, Y) + v[:, 1] * DT / G.res
    grid = torch.stack([(iy + 0.5) / Y * 2 - 1, (ix + 0.5) / X * 2 - 1], -1)           # (B, X, Y, 2): (W, H)
    return F.grid_sample(f, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def loss_selfsup(net, b, sigmas=(2.0, 1.0, 0.0), w_smooth=0.02):
    v = net(b["f0"], b["f1_in0"])
    valid = (b["mass"] > 0.5).float().unsqueeze(1)
    changed = ((b["f0"] - b["f1_in0"]).abs().amax(1, keepdim=True) > 0.05).float()
    changed = F.max_pool2d(changed, 5, 1, 2)
    w = valid * (0.1 + 0.9 * changed)
    warped = pull(b["f1_in0"], v)
    loss = sum(((blur(warped, s) - blur(b["f0"], s)).abs() * w).sum() / w.sum().clamp(min=1) for s in sigmas)
    occ = (b["f0"].amax(1, keepdim=True) > 0.3).float()
    dx = (v[:, :, 1:] - v[:, :, :-1]).abs() * occ[:, :, 1:] * occ[:, :, :-1]
    dy = (v[:, :, :, 1:] - v[:, :, :, :-1]).abs() * occ[:, :, :, 1:] * occ[:, :, :, :-1]
    return loss + w_smooth * (dx.mean() + dy.mean())


def loss_supervised(net, b):
    v = net(b["f0"], b["f1_in0"])
    m = b["obj"].unsqueeze(1).float()
    return (((v - b["vel"]) ** 2) * m).sum() / m.sum().clamp(min=1)


def train(net, kind, trainset, iters, bs=8, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = random.Random(seed)
    t0 = time.time()
    for it in range(iters):
        b = batch(rng.sample(trainset, bs))
        loss = loss_supervised(net, b) if kind == "supervised" else loss_selfsup(net, b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 200 == 0 or it == iters - 1:
            print(f"  {kind:15s} iter {it:4d} loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
    return net


def evaluate(net, test):
    cell, box, ncell, nbox = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(test), 8):
            b = batch(test[i:i + 8])
            v = net(b["f0"], b["f1_in0"]) if net is not None else torch.zeros_like(b["vel"])
            v, b = v.cpu(), {k: t.cpu() for k, t in b.items()}
            err = (v - b["vel"]).norm(dim=1)
            for j in range(err.shape[0]):
                m = b["moving"][j]
                if m.sum() == 0:
                    continue
                cell.extend(err[j][m].tolist())
                ncell.extend(b["vel"][j].norm(dim=0)[m].tolist())
                for k in b["box_id"][j][m].unique().tolist():
                    mk = m & (b["box_id"][j] == k)
                    box.append((v[j][:, mk].mean(1) - b["vel"][j][:, mk].mean(1)).norm().item())
                    nbox.append(b["vel"][j][:, mk].mean(1).norm().item())
    med = sorted(box)[len(box) // 2]
    return sum(cell) / len(cell), sum(box) / len(box), med, len(box), sorted(nbox)[len(nbox) // 2]


if __name__ == "__main__":
    t0 = time.time()
    if os.path.exists(CACHE):
        trainset, testset = torch.load(CACHE, weights_only=True)
    else:
        trainset = [make_pair(random.Random(100 + i)) for i in range(300)]
        testset = [make_pair(random.Random(99990 + i)) for i in range(60)]
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        torch.save((trainset, testset), CACHE)
    print(f"data: {len(trainset)} train / {len(testset)} held-out pairs ({time.time() - t0:.0f} s), device {DEV}",
          flush=True)
    # convention check of the pull: the TRUE velocity must match the older frame on the moving cells better than v = 0,
    # -v and the swapped axes (inside a box the coverage is uniform, so v = 0 matches there too: no fixed ratio)
    with torch.no_grad():
        b = batch(testset[:16])
        m = b["moving"].unsqueeze(1).float()
        v = b["vel"]
        err = {name: (((pull(b["f1_in0"], vv) - b["f0"]).abs() * m).sum() / m.sum()).item()
               for name, vv in {"v": v, "0": 0 * v, "-v": -v, "swap": v.flip(1)}.items()}
        net0 = CorrFlow().to(DEV)
        v_test = net0(b["f0"], b["f1_in0"])
        assert v_test.shape == v.shape and torch.isfinite(loss_selfsup(net0, b))
    print("pull convention, error on moving cells: " + ", ".join(f"{k} {e:.3f}" for k, e in err.items()))
    assert err["v"] < min(err["0"], err["-v"], err["swap"]), "pull convention"
    results = {"null (v = 0)": None}
    torch.manual_seed(0)
    ss = train(CorrFlow().to(DEV), "self-supervised", trainset, 800)
    results["self-supervised (corr, pull loss)"] = ss
    import copy
    results["self-sup + supervised fine-tune 300"] = train(copy.deepcopy(ss), "supervised", trainset, 300, lr=1e-3)
    torch.manual_seed(0)
    results["supervised from scratch 300"] = train(CorrFlow().to(DEV), "supervised", trainset, 300)
    torch.manual_seed(0)
    results["supervised from scratch 800"] = train(CorrFlow().to(DEV), "supervised", trainset, 800)
    print("\nheld-out | method | mean end-point error on moving-box cells (m/s) | per box mean / median (n) | null median")
    rows = {}
    for name, net in results.items():
        if net is not None:
            net.eval()
        ce, be, bm, n, nm = evaluate(net, testset)
        rows[name] = bm
        print(f"{name:38s} | {ce:6.2f} | {be:6.2f} / {bm:6.2f} ({n}) | {nm:6.2f}")
    null = rows["null (v = 0)"]
    ss_ok = rows["self-supervised (corr, pull loss)"] < 0.5 * null
    print(f"GATE flow, self-supervised (corr): {'PASS' if ss_ok else 'FAIL'} (per-box median < half of null "
          f"{null:.2f}); supervised from scratch 800: {'PASS' if rows['supervised from scratch 800'] < 0.5 * null else 'FAIL'}")
    sys.exit(0 if ss_ok else 1)
