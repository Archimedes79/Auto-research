"""Is the TRUE motion a minimum of the self-supervised loss? (plan.md 7.3, 2026-10-26)

Finding that motivates it: in tests/test_flow_refine.py the refined velocity fields reach a warp MSE of 0.0013 on the
known cells while the TRUE field gets 0.0080 - per-cell velocities that reshape blobs and stripes beat the real motion,
so the refinement learns to game the loss. Here, without any network: a per-cell velocity field is set to the TRUE
field of each held-out scene of that test and then optimised directly (Adam, 200 steps) on the loss variant; a good
loss keeps it at the truth (small drift), a gameable one moves it away while its own value falls.
Loss variants: (1) warp MSE (the current loss); (2) + MSE of the 2x and 4x average-pooled maps (multi-scale);
(3) MSE of the 4x pooled maps only; (4) warp MSE + edge-aware smoothness (tests/test_flow_refine.py smoothness);
(5) warp MSE with a Gaussian splat (sigma 1 cell) for prediction AND a target blurred the same way; (6) / (7) warp MSE
with the velocity on a COARSER grid (one velocity per 1 x 1 m / 2 x 2 m block, upsampled nearest; initialised with the
occupancy-weighted mean of the true velocity in the block): motion has fewer degrees of freedom than appearance.
Reported: velocity error (m/s) on the object cells and on the static stripes after optimisation, the loss value at the
start (truth) and at the end. Claim checked per variant: the truth is (near) a minimum - the drift on objects stays
< 0.5 m/s AND the loss falls less than 2x from its value at the truth (a large fall with a small drift is the sign of a
gameable loss: sub-cell reshaping).
Result 2026-10-26: none holds - warp MSE falls 20x (drift 0.32), + pooled 16x (0.31), pooled 4x 10x (0.51),
+ smoothness 7x (0.21), Gaussian splat + blurred target 10x (0.63), coarse velocity 1 m 7x (0.52), 2 m 4x (0.88).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_warp_loss_minimum.py
"""
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "training")
sys.path.insert(0, "tests")
from temporal import gaussian_splat, landing, propagate, steps_tensor  # noqa: E402
from test_flow_refine import G, sample, smoothness, true_flow  # noqa: E402


def blur(m, sigma=1.0, r=2):
    k = torch.exp(-torch.arange(-r, r + 1, dtype=m.dtype) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    c = m.shape[1]
    m = F.conv2d(F.pad(m, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return F.conv2d(F.pad(m, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)


def warp(src, steps, v, sigma=0.0):
    if sigma <= 0:
        return propagate(src, steps, v, G)[0]
    B, C, X, Y = src.shape
    S = steps_tensor(steps, B, src.device, src.dtype)
    return gaussian_splat(src.flatten(2), landing(S, v, G), src.new_ones(B, X * Y), G, sigma, 3)[0]


def losses(name, src, tgt, steps, v, known):
    m = known.float()[None, None]
    if name == "warp MSE":
        return ((warp(src, steps, v) - tgt) ** 2 * m).mean()
    if name == "+ pooled 2x, 4x":
        p = warp(src, steps, v)
        return sum((((F.avg_pool2d(p * m, f) - F.avg_pool2d(tgt * m, f)) ** 2).mean()) for f in (1, 2, 4))
    if name == "pooled 4x only":
        p = warp(src, steps, v)
        return ((F.avg_pool2d(p * m, 4) - F.avg_pool2d(tgt * m, 4)) ** 2).mean()
    if name == "+ smoothness":
        return ((warp(src, steps, v) - tgt) ** 2 * m).mean() + 0.05 * smoothness(v, src)
    if name.startswith("coarse velocity"):
        return ((warp(src, steps, v) - tgt) ** 2 * m).mean()
    if name == "Gaussian splat + blurred target":
        return ((warp(src, steps, v, sigma=1.0) - blur(tgt)) ** 2 * m).mean()
    raise ValueError(name)


if __name__ == "__main__":
    test = [sample(random.Random(10_000 + i)) for i in range(12)]
    names = ("warp MSE", "+ pooled 2x, 4x", "pooled 4x only", "+ smoothness", "Gaussian splat + blurred target",
             "coarse velocity 1 m blocks", "coarse velocity 2 m blocks")
    if "--only-coarse" in sys.argv:
        names = names[5:]
    claims = []
    start_err = []
    print("velocity field initialised at the TRUTH, then optimised per cell on the loss (Adam, lr 0.05, 200 steps)")
    for name in names:
        eo, no, es, ns, l0s, l1s = 0.0, 0, 0.0, 0, 0.0, 0.0
        for src, tgt, steps, hole in test:
            vt = true_flow(src)[None]
            known = ~hole
            if name.startswith("coarse velocity"):
                f = 2 if "1 m" in name else 4
                occ = src[:2].max(0).values.clamp(min=1e-3)[None, None]           # objects dominate the block mean
                vc = (F.avg_pool2d(vt * occ, f) / F.avg_pool2d(occ, f)).requires_grad_(True)
                up = lambda c: F.interpolate(c, scale_factor=f, mode="nearest")    # noqa: E731
                param, field = vc, lambda: up(vc)
            else:
                v = vt.clone().requires_grad_(True)
                param, field = v, lambda: v
            opt = torch.optim.Adam([param], lr=0.05)
            l0 = losses(name, src[None], tgt[None], [steps], field().detach(), known).item()
            e0 = (field().detach()[0] - vt[0]).norm(dim=0)
            for _ in range(200):
                loss = losses(name, src[None], tgt[None], [steps], field(), known)
                opt.zero_grad()
                loss.backward()
                opt.step()
            e = (field().detach()[0] - vt[0]).norm(dim=0)
            obj = src[:2].max(0).values > 0.6
            init_o = e0[obj].mean().item() if obj.any() else 0.0
            if name.startswith("coarse velocity") and obj.any():
                start_err.append(init_o)
            stat = (src[2] > 0.5) & (src[:2].max(0).values < 0.1)
            eo, no, es, ns = eo + e[obj].sum().item(), no + int(obj.sum()), es + e[stat].sum().item(), ns + int(stat.sum())
            l0s, l1s = l0s + l0, l1s + loss.item()
        drift_o, drift_s = eo / max(1, no), es / max(1, ns)
        extra = f"   (block mean at the start: {sum(start_err) / len(start_err):.2f} m/s from the truth)" if start_err else ""
        start_err.clear()
        print(f"  {name:34s} drift from the truth: objects {drift_o:5.2f}  stripes {drift_s:5.2f} m/s   "
              f"loss at the truth {l0s / len(test):.5f} -> after {l1s / len(test):.5f}{extra}", flush=True)
        ratio = l0s / max(l1s, 1e-12)
        claims.append((f"{name}: the truth is (near) a minimum - object drift < 0.5 m/s and loss fall < 2x ({ratio:.0f}x)",
                       drift_o < 0.5 and ratio < 2.0))
    print("== claims")
    for n, ok in claims:
        print(f"{'PASS' if ok else 'FAIL'}  {n}")
