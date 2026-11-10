"""Synthetic check of the window splat (training/splat.py: chain_positions + splat_window, Mirko 2026-10-25):
each of the last K original frames is brought into the current time with ONE splat, its coordinates chained
analytically over the per-frame ego steps (speed and yaw rate may change every frame). Compared with chaining the
resampled maps frame by frame.

World: lane lines and objects with constant world velocity; the ego drives with per-frame speed v_i and yaw rate
omega_i (exact constant turn rate within each frame). Frame k (t_k = k / 12 s, k = 0..5) is rendered in its own ego
frame (positions, headings, velocity channels in its own axes); the expectation is the scene at the newest frame.
Measured per frame age: centre error of the moving objects (m), mean IoU at 0.5, lane lines' lateral error (m), and
unknown cells: the share of the grid with mass 0, and how many of the cells the old frame never saw (outside its
grid) were reported as known (should be 0).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_splat_window.py <out.png>
"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "training")
from lift_pd import Grid  # noqa: E402
from splat import VelocityHead, splat, splat_bev, splat_window  # noqa: E402

G = Grid()
DT = torch.float64
SS = 4
OBJ = [  # world (= frame 0) centre, length, width, heading, world velocity
    ((30.0, 1.75), 4.5, 1.8, 0.0, (14.0, 0.0)),
    ((45.0, -1.75), 4.5, 1.8, math.pi, (-12.0, 0.0)),
    ((22.0, 10.0), 4.5, 1.8, -math.pi / 2, (0.0, -8.0)),
    ((14.0, -7.0), 0.8, 0.8, math.pi / 2, (0.0, 1.5)),
]
LANES = ((5.0, 3.5, 70.0, 3.5), (5.0, -3.5, 70.0, -3.5))    # world segments x0, y0, x1, y1
NCH = 3 + len(OBJ) + len(LANES)
_o = (torch.arange(SS, dtype=DT) + 0.5) / SS
GX, GY = torch.meshgrid(G.x_min + (torch.arange(G.nx, dtype=DT).view(-1, 1) + _o).reshape(-1) * G.res,
                        G.y_min + (torch.arange(G.ny, dtype=DT).view(-1, 1) + _o).reshape(-1) * G.res, indexing="ij")


def cover(m):
    return m.to(DT).view(G.nx, SS, G.ny, SS).mean((1, 3))


def poses(steps):
    """exact ego poses (x, y, heading) in the world after each step, starting at the origin."""
    out = [(0.0, 0.0, 0.0)]
    for dt, v, w in steps:
        x, y, h = out[-1]
        th = w * dt
        lx, ly = (v * dt, 0.0) if abs(w) < 1e-12 else (v / w * math.sin(th), v / w * (1 - math.cos(th)))
        out.append((x + math.cos(h) * lx - math.sin(h) * ly, y + math.sin(h) * lx + math.cos(h) * ly, h + th))
    return out


def to_ego(px, py, pose):
    x, y, h = pose
    qx, qy = px - x, py - y
    return math.cos(h) * qx + math.sin(h) * qy, -math.sin(h) * qx + math.cos(h) * qy


def render(t, pose):
    """map channels of the world at time t seen from the ego pose: occupancy, vx, vy (own axes), objects, lanes."""
    covs, occ = [], torch.zeros(G.nx, G.ny, dtype=DT)
    vx, vy = torch.zeros_like(occ), torch.zeros_like(occ)
    for (cx, cy), L, W, hd, (wvx, wvy) in OBJ:
        ox, oy = to_ego(cx + wvx * t, cy + wvy * t, pose)
        h = hd - pose[2]
        u = (GX - ox) * math.cos(h) + (GY - oy) * math.sin(h)
        vv = -(GX - ox) * math.sin(h) + (GY - oy) * math.cos(h)
        c = cover((u.abs() <= L / 2) & (vv.abs() <= W / 2))
        covs.append(c)
        m = c > 0
        occ[m] = torch.maximum(occ[m], c[m])
        cv, sv = math.cos(pose[2]), math.sin(pose[2])
        vx[m], vy[m] = cv * wvx + sv * wvy, -sv * wvx + cv * wvy
    lanes = []
    for x0, y0, x1, y1 in LANES:
        (ax, ay), (bx, by) = to_ego(x0, y0, pose), to_ego(x1, y1, pose)
        ex, ey = bx - ax, by - ay
        tt = (((GX - ax) * ex + (GY - ay) * ey) / (ex * ex + ey * ey)).clamp(0, 1)
        lanes.append(cover(torch.hypot(GX - ax - tt * ex, GY - ay - tt * ey) <= 0.075))
        occ = torch.maximum(occ, lanes[-1])
    return torch.stack([occ, vx, vy] + covs + lanes)[None]


def head():
    h = VelocityHead(cin=NCH, hidden=4, v_max=1e6).double()
    w1 = torch.zeros(4, NCH, 1, 1, dtype=DT)
    w1[0, 1], w1[1, 1], w1[2, 2], w1[3, 2] = 1, -1, 1, -1
    w2 = torch.zeros(2, 4, 1, 1, dtype=DT)
    w2[0, 0], w2[0, 1], w2[1, 2], w2[1, 3] = 1, -1, 1, -1
    h.net[0].weight.data, h.net[0].bias.data = w1, torch.zeros(4, dtype=DT)
    h.net[2].weight.data, h.net[2].bias.data = w2, torch.zeros(2, dtype=DT)
    return h


def centroid(m):
    xs, ys = G.xy_centres(dtype=DT)
    return (m.sum(1) @ xs / m.sum()).item(), (m.sum(0) @ ys / m.sum()).item()


def measure(got, exp):
    errs, ious = [], []
    for k, (_, _, _, _, vel) in enumerate(OBJ):
        if exp[3 + k].sum() < 1 or got[3 + k].sum() < 0.2 or math.hypot(*vel) < 0.5:
            continue
        a, b = centroid(got[3 + k]), centroid(exp[3 + k])
        errs.append(math.hypot(a[0] - b[0], a[1] - b[1]))
        A, Bm = got[3 + k] > 0.5, exp[3 + k] > 0.5
        ious.append((A & Bm).sum().item() / max(1, (A | Bm).sum().item()))
    _, ys = G.xy_centres(dtype=DT)
    lat = []
    for j in range(len(LANES)):
        ge, ee = got[3 + len(OBJ) + j], exp[3 + len(OBJ) + j]
        for i in range(G.nx):
            if ee[i].sum() > 0.2 and ge[i].sum() > 0.2:
                lat.append(abs((ge[i] @ ys / ge[i].sum() - ee[i] @ ys / ee[i].sum()).item()))
    return sum(errs) / len(errs), sum(ious) / len(ious), sum(lat) / max(1, len(lat))


CASES = {
    "constant turn: 10 m/s, 0.4 rad/s": [(1 / 12, 10.0, 0.4)] * 5,
    "changing: speed 8 -> 14 m/s, yaw rate 0.5 -> -0.1 rad/s": [(1 / 12, 8.0 + 1.5 * i, 0.5 - 0.15 * i) for i in range(5)],
}

GATE = "--no-gate" not in sys.argv

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    hd = head()
    imp = (lambda bev: 6.0 * bev[:, 0])
    print("case | frame age (s) | method | moving-object centre error (m) | IoU | lane error (m) | "
          "unknown cells (mass 0) | never-seen cells reported known")
    pics = []
    for name, steps in CASES.items():
        P = poses(steps)
        ts = [sum(s[0] for s in steps[:k]) for k in range(len(steps) + 1)]
        frames = [render(ts[k], P[k]) for k in range(len(steps) + 1)]
        exp = frames[-1][0]
        win = [(frames[k], steps[k:]) for k in range(len(frames))]
        out_w, mass_w = splat_window(win, G, head=hd, importance=imp, mass_gate=GATE)
        for k in range(len(frames) - 1):
            age = ts[-1] - ts[k]
            e, i, l = measure(out_w[0, k], exp)
            unk = (mass_w[0, k, 0] == 0).double().mean().item()
            # cells of the current grid that lie outside the grid of frame k (never seen by it)
            xs, ys = G.xy_centres(dtype=DT)
            gx, gy = torch.meshgrid(xs, ys, indexing="ij")
            pose_rel = P[k]
            wx = P[-1][0] + math.cos(P[-1][2]) * gx - math.sin(P[-1][2]) * gy
            wy = P[-1][1] + math.sin(P[-1][2]) * gx + math.cos(P[-1][2]) * gy
            qx, qy = wx - pose_rel[0], wy - pose_rel[1]
            kx = math.cos(pose_rel[2]) * qx + math.sin(pose_rel[2]) * qy
            ky = -math.sin(pose_rel[2]) * qx + math.cos(pose_rel[2]) * qy
            never = (kx < G.x_min - G.res) | (kx > G.x_max + G.res) | (ky < G.y_min - G.res) | (ky > G.y_max + G.res)
            wrong = ((mass_w[0, k, 0] > 0) & never).sum().item()
            print(f"{name:52s} | {age:.3f} | window, one splat     | {e:.3f} | {i:.2f} | {l:.3f} | {unk:.3f} | {wrong}")
            if k == 0:
                state = frames[0]
                for dt, v, w in steps:                      # the old way: chain the resampled maps
                    state, mass, _ = splat_bev(state, G, dt, v, w, head=hd, importance=6.0 * state[:, :1])
                ec, ic, lc = measure(state[0], exp)
                print(f"{name:52s} | {age:.3f} | chained map splats    | {ec:.3f} | {ic:.2f} | {lc:.3f} | "
                      f"{(mass[0, 0] == 0).double().mean().item():.3f} | -")
                pics.append((name, frames[0][0, 0], exp[0], out_w[0, 0, 0], state[0, 0], mass_w[0, 0, 0]))
    fig, ax = plt.subplots(len(pics), 5, figsize=(16, 4.4 * len(pics)))
    for r, (name, src, ex, w_, ch, m) in enumerate(pics):
        for c, (img, t, cm) in enumerate([(src, "oldest frame (0.42 s ago)", "gray"), (ex, "expected now", "gray"),
                                          (w_, "window: one splat", "gray"), (ch, "chained map splats", "gray"),
                                          ((m > 0).double(), "known cells of the window splat (mass > 0)", "viridis")]):
            ax[r, c].imshow(img.flip(0).flip(1).numpy(), cmap=cm, vmin=0, vmax=1, extent=[26, -26, 0, 60])
            ax[r, c].set_title(t, fontsize=9)
            ax[r, c].set_xticks([])
            if c == 0:
                ax[r, c].set_ylabel(name, fontsize=8)
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=80)
    print("saved", sys.argv[1])
