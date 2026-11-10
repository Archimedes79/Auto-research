"""Synthetic check of the temporal splat (training/splat.py): do pixels with their own speed vector land where expected?

Scene on the BEV grid (x 0..60 m, y -26..26 m, 0.5 m): two static lane lines and four objects (rectangles). Every map
carries the channels: occupancy, vx, vy (the object's velocity in its cells, 0 elsewhere), then one coverage channel per
object and per lane line, so each is measured on its own channel. The splat moves the map by the ego motion (speed v,
yaw rate omega; translation first order in omega, rotation exact) and by each cell's own velocity, read by the velocity
head from the features (weights set to pass vx, vy through: the real module path).
The expectation is rendered analytically: each object moved by its velocity and transformed with the EXACT
constant-turn-rate ego motion, footprint rotated, area coverage per cell by 8 x 8 supersampling.
Measured: per object the centre error (m) and IoU at 0.5; per lane line the mean lateral error over the x columns;
holes = cells inside an expected object that received no mass.
Variants for chained splats (the same total time in several frames): importance scale (who wins a collision with the
background) and "carry" = each cell keeps its exact sub-cell position as two extra channels (the mean landing position
of what arrived), so the next step starts from it instead of the cell centre.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_splat_scenes.py <out.png>
"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "training")
from lift_pd import Grid  # noqa: E402
from splat import VelocityHead, grid_points, splat, splat_positions  # noqa: E402

G = Grid()
DT = torch.float64
OBJECTS = [  # name, centre (x, y) m, length, width, heading rad, velocity (vx, vy) m/s in the source ego frame
    ("car ahead", (30.0, 1.75), 4.5, 1.8, 0.0, (15.0, 0.0)),
    ("oncoming car", (42.0, -1.75), 4.5, 1.8, math.pi, (-15.0, 0.0)),
    ("crossing car", (22.0, 9.0), 4.5, 1.8, -math.pi / 2, (0.0, -8.0)),
    ("pedestrian", (14.0, -6.0), 0.8, 0.8, math.pi / 2, (0.0, 1.5)),
]
LANES = (3.5, -3.5)                           # static lane lines at y = +3.5 m and -3.5 m, x 5..58 m
NCH = 3 + len(OBJECTS) + len(LANES)


def ctrv_exact(px, py, dt, v, w):
    """Static point of the source frame expressed in the target ego frame (exact constant turn rate)."""
    th = w * dt
    dx, dy = (v * dt, 0.0) if abs(w) < 1e-12 else (v / w * math.sin(th), v / w * (1 - math.cos(th)))
    qx, qy = px - dx, py - dy
    return math.cos(th) * qx + math.sin(th) * qy, -math.sin(th) * qx + math.cos(th) * qy


def render(objects_state, lanes_state, ss=8):
    """Coverage maps (X, Y) per object (cx, cy, L, W, heading) and per lane line (dense point list), supersampled."""
    X, Y = G.nx, G.ny
    o = (torch.arange(ss, dtype=DT) + 0.5) / ss
    xs = G.x_min + (torch.arange(X, dtype=DT).view(X, 1) + o.view(1, ss)).reshape(-1) * G.res
    ys = G.y_min + (torch.arange(Y, dtype=DT).view(Y, 1) + o.view(1, ss)).reshape(-1) * G.res
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    covs = []
    for cx, cy, L, W, h in objects_state:
        u = (gx - cx) * math.cos(h) + (gy - cy) * math.sin(h)
        vv = -(gx - cx) * math.sin(h) + (gy - cy) * math.cos(h)
        covs.append(((u.abs() <= L / 2) & (vv.abs() <= W / 2)).to(DT).view(X, ss, Y, ss).mean((1, 3)))
    lanes = []
    flat = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)
    for pts in lanes_state:                   # 0.15 m wide line
        d = torch.cdist(flat, torch.tensor(pts, dtype=DT)).min(1).values.view(gx.shape)
        lanes.append((d <= 0.075).to(DT).view(X, ss, Y, ss).mean((1, 3)))
    return covs, lanes


def source_scene():
    objs = [(c[0], c[1], L, W, h) for _, c, L, W, h, _ in OBJECTS]
    lane_pts = [[(x, y) for x in torch.arange(5.0, 58.0, 0.05).tolist()] for y in LANES]
    covs, lanes = render(objs, lane_pts)
    occ = torch.zeros(G.nx, G.ny, dtype=DT)
    vx, vy = torch.zeros_like(occ), torch.zeros_like(occ)
    for cov, (_, _, _, _, _, vel) in zip(covs, OBJECTS):
        m = cov > 0
        occ[m] = torch.maximum(occ[m], cov[m])
        vx[m], vy[m] = vel
    for ln in lanes:
        occ = torch.maximum(occ, ln)
    return torch.stack([occ, vx, vy] + covs + lanes)[None], lane_pts


def expected_scene(lane_pts, dt, v, w):
    objs = []
    for _, (cx, cy), L, W, h, (ovx, ovy) in OBJECTS:
        tx, ty = ctrv_exact(cx + ovx * dt, cy + ovy * dt, dt, v, w)
        objs.append((tx, ty, L, W, h - w * dt))
    return render(objs, [[ctrv_exact(x, y, dt, v, w) for x, y in pts] for pts in lane_pts])


def pass_through_head():
    """VelocityHead whose output equals the feature channels 1, 2 (vx, vy): ReLU split into +/- parts."""
    head = VelocityHead(cin=NCH, hidden=4, v_max=1e6).double()
    w1 = torch.zeros(4, NCH, 1, 1, dtype=DT)
    w1[0, 1], w1[1, 1], w1[2, 2], w1[3, 2] = 1, -1, 1, -1
    head.net[0].weight.data, head.net[0].bias.data = w1, torch.zeros(4, dtype=DT)
    w2 = torch.zeros(2, 4, 1, 1, dtype=DT)
    w2[0, 0], w2[0, 1], w2[1, 2], w2[1, 3] = 1, -1, 1, -1
    head.net[2].weight.data, head.net[2].bias.data = w2, torch.zeros(2, dtype=DT)
    return head


def chain(state, steps, dt, v, w, head, imp_scale=6.0, carry=False):
    """Splat the map `steps` times with dt / steps each; carry = keep exact sub-cell positions between steps."""
    B, C, X, Y = state.shape
    centres = grid_points(G, dtype=DT).unsqueeze(0)
    pos_src = centres.clone()
    mass = None
    for _ in range(steps):
        v_obj = head(state).permute(0, 2, 3, 1).reshape(B, X * Y, 2)
        pos_t = splat_positions(pos_src, dt / steps, v, w, v_obj)
        feats = state.reshape(B, C, X * Y)
        if carry:
            feats = torch.cat([feats, pos_t.permute(0, 2, 1)], 1)
        occ = state[:, 0].reshape(B, X * Y)
        imp = torch.log(occ.clamp(min=0) + 1e-3) if imp_scale == "log" else imp_scale * occ
        out, mass = splat(feats, pos_t, G, imp)
        if carry:
            landed = out[:, C:].reshape(B, 2, X * Y).permute(0, 2, 1)
            has = (mass.reshape(B, X * Y, 1) > 0)
            pos_src = torch.where(has, landed, centres)
            out = out[:, :C]
        state = out
    return state, mass


def centroid(m):
    xs, ys = G.xy_centres(dtype=DT)
    s = m.sum()
    return (m.sum(1) @ xs / s).item(), (m.sum(0) @ ys / s).item()


def measure(state, mass, covs_exp, lanes_exp):
    rows = []
    for k, ((oname, *_), ce) in enumerate(zip(OBJECTS, covs_exp)):
        got = state[0, 3 + k]
        ex, ey = centroid(ce)
        gx_, gy_ = centroid(got)
        a, b = got > 0.5, ce > 0.5
        iou = (a & b).sum().item() / max(1, (a | b).sum().item())
        rows.append((oname, math.hypot(gx_ - ex, gy_ - ey), iou, ((mass[0, 0] == 0) & b).sum().item()))
    _, ys = G.xy_centres(dtype=DT)
    lat = []
    for j, le in enumerate(lanes_exp):
        got = state[0, 3 + len(OBJECTS) + j]
        for i in range(G.nx):
            if le[i].sum() > 0.2 and got[i].sum() > 0.2:
                lat.append(abs((got[i] @ ys / got[i].sum() - le[i] @ ys / le[i].sum()).item()))
    return rows, sum(lat) / max(1, len(lat))


def run(dt, v, w, steps=1, imp_scale=6.0, carry=False):
    src, lane_pts = source_scene()
    state, mass = chain(src, steps, dt, v, w, pass_through_head(), imp_scale, carry)
    covs_exp, lanes_exp = expected_scene(lane_pts, dt, v, w)
    rows, lane_err = measure(state, mass, covs_exp, lanes_exp)
    return rows, lane_err, (src[0, 0], state[0, 0], sum(covs_exp) + sum(lanes_exp))


CASES = [  # name, dt, v, omega, steps
    ("straight 15 m/s, dt 0.5 s, 1 step", 0.5, 15.0, 0.0, 1),
    ("turning 10 m/s 0.4 rad/s, 0.5 s, 1 step", 0.5, 10.0, 0.4, 1),
    ("radar -> vision hop, dt 0.04 s", 0.04, 25.0, 0.1, 1),
    ("backwards in time, dt -0.3 s", -0.3, 12.0, -0.3, 1),
    ("straight 13 m/s, 0.5 s in 6 frames", 0.5, 13.0, 0.0, 6),
    ("turning, 0.5 s in 6 frames", 0.5, 10.0, 0.4, 6),
]
VARIANTS = [("normalised, importance 6", 6.0, False), ("no importance", 0.0, False), ("importance 30", 30.0, False), ("weight = occupancy (log)", "log", False)]

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    print("case | variant | centre error per object (m): car ahead / oncoming / crossing / pedestrian | "
          "mean IoU (cars) | lane lateral error (m) | hole cells")
    pics = []
    for name, dt, v, w, steps in CASES:
        for vname, s, carry in (VARIANTS if steps > 1 else VARIANTS[:1]):
            rows, lane_err, pic = run(dt, v, w, steps, s, carry)
            errs = " / ".join(f"{r[1]:.3f}" for r in rows)
            iou = sum(r[2] for r in rows[:3]) / 3
            holes = sum(r[3] for r in rows)
            print(f"{name:38s} | {vname:31s} | {errs} | {iou:.2f} | {lane_err:.3f} | {holes}")
            if vname == VARIANTS[0][0] or vname == VARIANTS[-1][0]:
                pics.append((f"{name}\n{vname}", pic))
    fig, axes = plt.subplots(len(pics), 4, figsize=(13, 3.5 * len(pics)))
    for r, (title, (src, got, exp)) in enumerate(pics):
        for c, (img, sub) in enumerate([(src, "source"), (exp, "expected (exact motion)"), (got, "splatted"),
                                        (got - exp.clamp(max=1), "splatted - expected")]):
            ax = axes[r, c]
            cmap, lim = ("bwr", (-1, 1)) if c == 3 else ("gray", (0, 1))
            ax.imshow(img.flip(0).flip(1).numpy(), cmap=cmap, vmin=lim[0], vmax=lim[1], extent=[26, -26, 0, 60])
            ax.set_title(sub, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([0, 30, 60])
            if c == 0:
                ax.set_ylabel(title, fontsize=7)
    fig.suptitle("temporal splat on synthetic BEV maps: objects with own velocity + ego motion (x forward up, y left)")
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=80)
    print("saved", sys.argv[1])
