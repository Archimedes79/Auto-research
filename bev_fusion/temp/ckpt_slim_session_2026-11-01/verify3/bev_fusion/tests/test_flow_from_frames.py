"""Gate: can a velocity head recover a real flow field from two consecutive BEV frames? (plan.md section 10.1)

Data: synthetic world sequences (tests/synthetic_world.py), frames 1/12 s apart. Features per frame: the GT layers lane
lines, vehicles, pedestrians as coverage maps - NO velocity channel. The newer frame is ego-pulled back into the older
frame (splat with -dt, no object motion: exact for static content), so the only displacement left is object motion
(up to 15 m/s x 1/12 s = 1.25 m = 2.5 cells). A small CNN (4 x 3x3 convs, dilations 1-2-4-1, receptive field 17 cells)
outputs v_hat (2 channels, m/s, axes of the older frame; last layer zero-initialised).
(a) supervised: masked MSE to the GT velocity on the box cells of the older frame
(b) self-supervised, no velocity labels: the older frame splatted forward with ego motion + v_hat must match the
    observed newer frame; the photometric-style loss is computed on the maps blurred with sigma 3, 1.5, 0 cells
    (coarse to fine: the bilinear splat alone only gives a position gradient within one cell), plus a smoothness term
    of v_hat inside occupied cells
Evaluated on held-out sequences (other seed): mean end-point error (m/s) of v_hat on the cells of moving boxes (speed
> 0.5 m/s), the same per box (mean over its cells), and the null baseline v_hat = 0.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_flow_from_frames.py
"""
import math
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "training")
sys.path.insert(0, "tests")
from lift_pd import Grid  # noqa: E402
from splat import grid_points, splat, splat_bev, splat_positions  # noqa: E402
from synthetic_world import gt_bev, random_scene  # noqa: E402

G = Grid()
DT = 1 / 12
torch.set_num_threads(8)


def make_pair(rng):
    s0 = random_scene(rng)
    s1 = s0.step(DT)
    g0, g1 = gt_bev(s0, G), gt_bev(s1, G)
    f0 = torch.stack([g0["lanes"], g0["vehicles"], g0["pedestrians"]])
    f1 = torch.stack([g1["lanes"], g1["vehicles"], g1["pedestrians"]])
    with torch.no_grad():                                   # newer frame ego-pulled back into the older frame
        f1_in0, _, _ = splat_bev(f1[None], G, -DT, s0.v, s0.omega)
    moving = (g0["box_id"] >= 0) & (g0["velocity"].norm(dim=0) > 0.5)
    return {"f0": f0, "f1": f1, "f1_in0": f1_in0[0], "vel": g0["velocity"], "obj": g0["box_id"] >= 0,
            "box_id": g0["box_id"], "moving": moving, "v": s0.v, "omega": s0.omega}


def batch(pairs):
    out = {}
    for k in ("f0", "f1", "f1_in0", "vel", "obj", "box_id", "moving"):
        out[k] = torch.stack([p[k] for p in pairs])
    out["v"] = torch.tensor([p["v"] for p in pairs])
    out["omega"] = torch.tensor([p["omega"] for p in pairs])
    return out


class FlowNet(nn.Module):
    def __init__(self, cin=6, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(ch, 2, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f0, f1_in0):
        return 20.0 * torch.tanh(self.net(torch.cat([f0, f1_in0], 1)) / 20.0)        # m/s, bounded


CENTRES = grid_points(G).unsqueeze(0)


def forward_splat(f0, v_hat, v, omega):
    B, C, X, Y = f0.shape
    v_obj = v_hat.permute(0, 2, 3, 1).reshape(B, X * Y, 2)
    pos = splat_positions(CENTRES.expand(B, -1, -1), DT, v, omega, v_obj)
    occ = f0.amax(1).reshape(B, X * Y)
    out, mass = splat(f0.reshape(B, C, X * Y), pos, G, 6.0 * occ)
    return out


def blur(x, sigma):
    if sigma <= 0:
        return x
    r = int(math.ceil(2.5 * sigma))
    t = torch.arange(-r, r + 1, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    C = x.shape[1]
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(C, 1, -1, 1), padding=(r, 0), groups=C)
    return F.conv2d(x, k.view(1, 1, 1, -1).expand(C, 1, 1, -1), padding=(0, r), groups=C)


def loss_supervised(net, b):
    v_hat = net(b["f0"], b["f1_in0"])
    m = b["obj"].unsqueeze(1).float()
    return (((v_hat - b["vel"]) ** 2) * m).sum() / m.sum().clamp(min=1)


def loss_selfsup(net, b, sigmas=(3.0, 1.5, 0.0), w_smooth=0.05):
    v_hat = net(b["f0"], b["f1_in0"])
    pred = forward_splat(b["f0"], v_hat, b["v"], b["omega"])
    loss = sum(((blur(pred, s) - blur(b["f1"], s)) ** 2).mean() for s in sigmas)
    occ = (b["f0"].amax(1, keepdim=True) > 0.3).float()
    dx = (v_hat[:, :, 1:] - v_hat[:, :, :-1]).abs() * occ[:, :, 1:] * occ[:, :, :-1]
    dy = (v_hat[:, :, :, 1:] - v_hat[:, :, :, :-1]).abs() * occ[:, :, :, 1:] * occ[:, :, :, :-1]
    return loss * 100 + w_smooth * (dx.mean() + dy.mean())


def evaluate(net, test):
    cell_err, box_err, null_cell, null_box = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(test), 8):
            b = batch(test[i:i + 8])
            v_hat = net(b["f0"], b["f1_in0"]) if net is not None else torch.zeros_like(b["vel"])
            err = (v_hat - b["vel"]).norm(dim=1)
            nul = b["vel"].norm(dim=1)
            for j in range(err.shape[0]):
                m = b["moving"][j]
                if m.sum() == 0:
                    continue
                cell_err.extend(err[j][m].tolist())
                null_cell.extend(nul[j][m].tolist())
                for k in b["box_id"][j][m].unique().tolist():
                    mk = m & (b["box_id"][j] == k)
                    box_err.append((v_hat[j][:, mk].mean(1) - b["vel"][j][:, mk].mean(1)).norm().item())
                    null_box.append(b["vel"][j][:, mk].mean(1).norm().item())
    med = lambda a: sorted(a)[len(a) // 2]   # noqa: E731
    return (sum(cell_err) / len(cell_err), sum(box_err) / len(box_err), med(box_err), len(box_err),
            sum(null_cell) / len(null_cell), sum(null_box) / len(null_box))


def train(kind, trainset, iters=800, bs=8, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    net = FlowNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.time()
    rng = random.Random(seed)
    for it in range(iters):
        b = batch(rng.sample(trainset, bs))
        loss = loss_supervised(net, b) if kind == "supervised" else loss_selfsup(net, b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 200 == 0 or it == iters - 1:
            print(f"  {kind:15s} iter {it:4d} loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
    return net


if __name__ == "__main__":
    t0 = time.time()
    rng = random.Random(100)
    trainset = [make_pair(rng) for _ in range(300)]
    rng = random.Random(9999)
    testset = [make_pair(rng) for _ in range(60)]
    print(f"data: {len(trainset)} train / {len(testset)} held-out pairs ({time.time() - t0:.0f} s)", flush=True)
    rows = {"null (v_hat = 0)": None}
    for kind in ("supervised", "self-supervised"):
        rows[kind] = train(kind, trainset)
    print("\nheld-out pairs | method | mean end-point error on moving-box cells (m/s) | per box: mean / median "
          "(n boxes) | null baseline: cells / boxes")
    res = {}
    for name, net in rows.items():
        ce, be, bm, n, nc, nb = evaluate(net, testset)
        res[name] = (bm, nb)
        print(f"{name:18s} | {ce:6.2f} | {be:6.2f} / {bm:6.2f} ({n}) | {nc:6.2f} / {nb:6.2f}")
    # gates: per-box median error below half of the null baseline; (b) is a known open item (plan 10.1)
    sup_ok = res["supervised"][0] < 0.5 * res["supervised"][1]
    self_ok = res["self-supervised"][0] < 0.5 * res["self-supervised"][1]
    print(f"GATE flow supervised: {'PASS' if sup_ok else 'FAIL'}; self-supervised: {'PASS' if self_ok else 'FAIL (open)'}")
    sys.exit(0 if (sup_ok and self_ok) else 1)
