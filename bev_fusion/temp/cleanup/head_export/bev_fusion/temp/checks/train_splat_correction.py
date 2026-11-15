"""Can the learned correction term (splat.CorrectionHead) remove the lag of chained splats? Synthetic, train / held-out.

Scenes: 3-6 random objects (cars 4.5 x 1.8 m up to 15 m/s along their heading, pedestrians 0.8 x 0.8 m up to 2 m/s),
two static lane lines, random ego speed (5-20 m/s) and yaw rate (-0.4..0.4 rad/s); 6 camera frames of 1/12 s are
splatted one after the other (a recurrent state). Channels: occupancy, vx, vy (what the heads see), then one coverage
channel per object (up to 6) and per lane line (only for the loss and the evaluation; the heads do not see them).
Velocity head: pass-through of vx, vy (as in check_splat_scenes.py); importance 6 x occupancy.
Trained: only the correction head g_phi (input: occupancy, vx, vy + ego and position channels), MSE between the
splatted object / lane channels after 6 frames and the expected ones (exact constant-turn-rate ego motion, exact object
motion). Evaluated on held-out scenes (other seed): centre error per moving object, IoU, lane lateral error, for
(a) chained, no correction, (b) chained with the trained correction, (c) each frame splatted once from its original
(window, no chaining) - the reference.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/train_splat_correction.py <out.png>
"""
import math
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "training")
from lift_pd import Grid  # noqa: E402
from splat import CorrectionHead, VelocityHead, grid_points, splat, splat_positions  # noqa: E402

G = Grid()
DT = torch.float32
DEV = "cpu"                                                  # the GPU is left to the lane project's training
MAXO, NL, STEPS, TOTAL = 6, 2, 6, 0.5
NCH = 3 + MAXO + NL
SS = 4                                                       # supersampling per cell for the coverage


def ctrv(px, py, dt, v, w):
    th = w * dt
    dx, dy = (v * dt, 0.0) if abs(w) < 1e-12 else (v / w * math.sin(th), v / w * (1 - math.cos(th)))
    qx, qy = px - dx, py - dy
    return math.cos(th) * qx + math.sin(th) * qy, -math.sin(th) * qx + math.cos(th) * qy


_o = (torch.arange(SS, dtype=DT) + 0.5) / SS
_xs = G.x_min + (torch.arange(G.nx, dtype=DT).view(-1, 1) + _o).reshape(-1) * G.res
_ys = G.y_min + (torch.arange(G.ny, dtype=DT).view(-1, 1) + _o).reshape(-1) * G.res
GX, GY = [t.to(DEV) for t in torch.meshgrid(_xs, _ys, indexing="ij")]


def cover(mask):
    return mask.to(DT).view(G.nx, SS, G.ny, SS).mean((1, 3))


def rect(cx, cy, L, W, h):
    u = (GX - cx) * math.cos(h) + (GY - cy) * math.sin(h)
    vv = -(GX - cx) * math.sin(h) + (GY - cy) * math.cos(h)
    return cover((u.abs() <= L / 2) & (vv.abs() <= W / 2))


def segment(a, b, half_width=0.075):
    ax, ay = a
    bx, by = b
    ex, ey = bx - ax, by - ay
    t = (((GX - ax) * ex + (GY - ay) * ey) / (ex * ex + ey * ey)).clamp(0, 1)
    d = torch.hypot(GX - ax - t * ex, GY - ay - t * ey)
    return cover(d <= half_width)


def random_scene(rng):
    objs = []
    for _ in range(rng.randint(3, MAXO)):
        if rng.random() < 0.75:
            L, W, sp = 4.5, 1.8, rng.uniform(0, 15)
        else:
            L, W, sp = 0.8, 0.8, rng.uniform(0, 2)
        h = rng.uniform(-math.pi, math.pi)
        objs.append(((rng.uniform(10, 50), rng.uniform(-18, 18)), L, W, h, (sp * math.cos(h), sp * math.sin(h))))
    lanes = [rng.uniform(-6, -2), rng.uniform(2, 6)]
    return objs, lanes, rng.uniform(5, 20), rng.uniform(-0.4, 0.4)


def render(objs, lanes, dt, v, w):
    """state channels at time dt after the source (dt = 0: the source itself)."""
    covs, occ = [], torch.zeros(G.nx, G.ny, dtype=DT, device=DEV)
    vx, vy = torch.zeros_like(occ), torch.zeros_like(occ)
    for (cx, cy), L, W, h, (ovx, ovy) in objs:
        tx, ty = ctrv(cx + ovx * dt, cy + ovy * dt, dt, v, w)
        c = rect(tx, ty, L, W, h - w * dt)
        covs.append(c)
        m = c > 0
        occ[m] = torch.maximum(occ[m], c[m])
        vx[m], vy[m] = ovx, ovy                                  # source-axis velocity (only used at dt = 0)
    while len(covs) < MAXO:
        covs.append(torch.zeros_like(occ))
    lcov = [segment(ctrv(5.0, y0, dt, v, w), ctrv(58.0, y0, dt, v, w)) for y0 in lanes]
    for lc in lcov:
        occ = torch.maximum(occ, lc)
    return torch.stack([occ, vx, vy] + covs + lcov)[None]


def pass_head():
    head = VelocityHead(cin=3, hidden=4, v_max=1e6).to(DEV)
    w1 = torch.zeros(4, 3, 1, 1)
    w1[0, 1], w1[1, 1], w1[2, 2], w1[3, 2] = 1, -1, 1, -1
    w2 = torch.zeros(2, 4, 1, 1)
    w2[0, 0], w2[0, 1], w2[1, 2], w2[1, 3] = 1, -1, 1, -1
    with torch.no_grad():
        head.net[0].weight.copy_(w1)
        head.net[0].bias.zero_()
        head.net[2].weight.copy_(w2)
        head.net[2].bias.zero_()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


HEAD = pass_head()
CENTRES = grid_points(G, DEV, DT).unsqueeze(0)


def chain(state, v, w, corr=None, steps=STEPS, total=TOTAL):
    X, Y = G.nx, G.ny
    for _ in range(steps):
        seen = state[:, :3]
        v_obj = HEAD(seen).permute(0, 2, 3, 1).reshape(1, X * Y, 2)
        delta = corr(seen.reshape(1, 3, -1), CENTRES, total / steps, v, w) if corr is not None else None
        pos = splat_positions(CENTRES, total / steps, v, w, v_obj, delta)
        state, mass = splat(state.reshape(1, NCH, -1), pos, G, 6.0 * state[:, 0].reshape(1, -1))
    return state, mass


def centroid(m):
    xs, ys = G.xy_centres(DEV, DT)
    s = m.sum()
    return (m.sum(1) @ xs / s).item(), (m.sum(0) @ ys / s).item()


def evaluate(got, exp, objs):
    errs, ious = [], []
    for k, (_, L, W, h, (ovx, ovy)) in enumerate(objs):
        if math.hypot(ovx, ovy) < 0.5 or exp[0, 3 + k].sum() < 1.0 or got[0, 3 + k].sum() < 0.2:
            continue                                             # moving objects inside the grid only
        ex, ey = centroid(exp[0, 3 + k])
        gx, gy = centroid(got[0, 3 + k])
        errs.append(math.hypot(gx - ex, gy - ey))
        a, b = got[0, 3 + k] > 0.5, exp[0, 3 + k] > 0.5
        ious.append((a & b).sum().item() / max(1, (a | b).sum().item()))
    _, ys = G.xy_centres(DEV, DT)
    lat = []
    for j in range(NL):
        ge, ee = got[0, 3 + MAXO + j], exp[0, 3 + MAXO + j]
        for i in range(G.nx):
            if ee[i].sum() > 0.2 and ge[i].sum() > 0.2:
                lat.append(abs((ge[i] @ ys / ge[i].sum() - ee[i] @ ys / ee[i].sum()).item()))
    return errs, ious, lat


if __name__ == "__main__":
    torch.manual_seed(0)
    corr = CorrectionHead(cin=3, hidden=32, d_max=1.0).to(DEV)
    opt = torch.optim.Adam(corr.parameters(), lr=3e-3)
    rng = random.Random(1)
    t0 = time.time()
    n_iter = 400
    for it in range(n_iter):
        objs, lanes, v, w = random_scene(rng)
        src = render(objs, lanes, 0.0, v, w)
        exp = render(objs, lanes, TOTAL, v, w)
        got, _ = chain(src, v, w, corr)
        loss = ((got[:, 3:] - exp[:, 3:]) ** 2).sum() / max(1.0, exp[:, 3:].sum().item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == n_iter - 1:
            print(f"iter {it:4d} loss {loss.item():.4f} ({time.time() - t0:.0f} s)")
    corr.eval()
    torch.set_grad_enabled(False)
    rng = random.Random(12345)                                   # held-out scenes
    res = {"a chained, no correction": ([], [], []), "b chained + learned correction": ([], [], []),
           "c window: each frame once from its original": ([], [], [])}
    example = None
    for s in range(40):
        objs, lanes, v, w = random_scene(rng)
        src = render(objs, lanes, 0.0, v, w)
        exp = render(objs, lanes, TOTAL, v, w)
        outs = {"a chained, no correction": chain(src, v, w)[0],
                "b chained + learned correction": chain(src, v, w, corr)[0],
                "c window: each frame once from its original": chain(src, v, w, None, steps=1)[0]}
        for k, o in outs.items():
            e, i, l = evaluate(o, exp, objs)
            res[k][0].extend(e)
            res[k][1].extend(i)
            res[k][2].extend(l)
        if s == 0:
            example = (src[0, 0], exp[0, 0], outs["a chained, no correction"][0, 0], outs["b chained + learned correction"][0, 0])
    print("\nheld-out: 40 scenes, moving objects | method | mean / median centre error (m) | mean IoU | lane lateral error (m)")
    for k, (e, i, l) in res.items():
        es = sorted(e)
        print(f"{k:45s} | n={len(e):3d} | {sum(e) / len(e):.3f} / {es[len(es) // 2]:.3f} | {sum(i) / len(i):.2f} | "
              f"{sum(l) / max(1, len(l)):.3f}")
    fig, ax = plt.subplots(1, 4, figsize=(14, 4.6))
    for a, (img, t) in zip(ax, zip(example, ["source", "expected after 0.5 s", "chained, no correction",
                                            "chained + learned correction"])):
        a.imshow(img.flip(0).flip(1).cpu().numpy(), cmap="gray", vmin=0, vmax=1, extent=[26, -26, 0, 60])
        a.set_title(t, fontsize=10)
        a.set_xticks([])
    fig.suptitle("learned correction of chained splats - held-out synthetic scene (6 frames, 0.5 s)")
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=90)
    print("saved", sys.argv[1])
