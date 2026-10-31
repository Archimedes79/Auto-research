"""Velocity at keyframe k from TWO frames + ego motion + real radar Doppler, fused by networks in a small pyramid
(Mirko, 2026-10-31: "use them [two frames] and ego motion and doppler comparison together to form a flow field hypothesis.
Maybe doing that in a pyramid, but small. Loss could be equal for both. velocity gt, boxes from future and past, all this
could build an intermediate loss for first training before training E2E. Other loss terms may make sense as well, test it.
Make sure that you use networks").

Model (all learned except the fixed geometry: ego alignment, backward warp, ray directions):
    E(F_k), E(F_k-1) moved to k with the ego motion only (exact splat)
    per level l = 2 m, 1 m, 0.5 m (coarse to fine), v_up = the coarser level's velocity (0 at the start):
        W   = E(F_k-1) sampled where each cell of k was at k-1 under v_up (backward warp, p - v_up dt)
        R   = radar raster pooled to l (count, mean Doppler vr, mean ray u, age) and the Doppler comparison of v_up
              (u . v - vr at the cells with a detection)
        v_l = v_up + net_l([E(F_k), W, v_up, R, ray / range of the cell])
    --corr (off since 2026-10-31: a tie with it, 0.93 / 0.99 vs 0.96 / 1.04 m/s on moving vehicles, so the cheaper one;
    Mirko: "Yes, drop it") replaces W by the local correlation C of E(F_k) with W (+-4 / 2 / 1 cells) and its soft frame
    hypothesis h = v_up - sum softmax(beta C) d / dt (no argmax), with the Doppler comparison of h added to R.
The same loss is applied to the velocity of EVERY level (upsampled to 0.5 m, weight 1 each: "equal for both").
Loss terms (--loss a,b,..., and --then x,y --pre f: switch to the second set after the fraction f of the steps, the
"intermediate loss first, then end to end" proxy):
  vel     annotated velocity at k (box cells of each sample weighted to sum 1)
  box     every cell of a box at k lands where the same box is at k-1 (dt -0.5 s) and k+1 (+0.5 s), rigid incl. yaw
  static  |v| on cells outside every box (weight 0.1)
  warp    the RAW maps of k moved with v (ego + v) match k+1 and k-1 (MSE x 3; label-free, the end-to-end signal)
  dop     the radial part of v matches the radar Doppler at cells with a detection (Huber; label-free)
  smooth  edge-aware smoothness of v (weight 0.05; label-free)
Arms: --no-dop (no radar input), --corr (the correlation / hypothesis above instead of the concatenated W; the runs of
plan 7.4 before 2026-10-31 had it on by default and --no-corr as the arm), --levels 1 (2 m only, upsampled), --ray (the heads output (radial, tangential) in the ray frame of the
cell instead of (x, y)), --dop-sim (Doppler simulated from the GT velocity instead of the real radar: the input of
tests/test_flow_explains.py, for comparison only).
Data: nuScenes GT cache (one generator signature, SIG), radar cache (all 5 radars, 0.5 s window, moved to the camera
time of k with the Doppler shift); scene split 1 in 5 held out; samples with a moving vehicle (> 1 m/s) at k.
Run from the bev_fusion folder:
  wsl ~/miniconda3/envs/gcar/bin/python tests/test_flow_fused.py [--loss vel,box,static] [--then warp,dop] [--pre 0.5]
      [--no-dop] [--corr] [--levels 1] [--ray] [--dop-sim] [--quick] [--seed N] [--iters N]
"""
import glob
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "training")
sys.path.insert(0, "tests")
from lift_pd import Grid  # noqa: E402
from temporal import propagate  # noqa: E402


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


QUICK, SEED = "--quick" in sys.argv, arg("--seed", 0)
LOSS = arg("--loss", "vel,box,static").split(",")
THEN = arg("--then", "").split(",") if "--then" in sys.argv else None
PRE = arg("--pre", 0.5)
DOP, CORR, RAY, DOPSIM = "--no-dop" not in sys.argv, "--corr" in sys.argv, "--ray" in sys.argv, "--dop-sim" in sys.argv
LEVELS = arg("--levels", 3)
TAG = "_" + "+".join(LOSS) + (f"_then_{'+'.join(THEN)}{PRE:g}" if THEN else "") + ("" if DOP else "_nodop") \
    + ("_corr" if CORR else "") + (f"_L{LEVELS}" if LEVELS != 3 else "") + ("_ray" if RAY else "") \
    + ("_dopsim" if DOPSIM else "") + (f"_s{SEED}" if SEED else "") + ("_quick" if QUICK else "")
G = Grid(x_min=0.0, x_max=60.0, y_min=-26.0, y_max=26.0, res=0.5)            # 120 x 104, as data/cache/gt
SIG = "962e788566"                                  # the GT generator signature that covers every sample (with radar)
DEV = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 1.5e9 else "cpu"
if DEV == "cuda" and os.environ.get("GPU_FRACTION"):  # a shared GPU: this process can never take more than its share
    torch.cuda.set_per_process_memory_fraction(float(os.environ["GPU_FRACTION"]))
C = 32
STRIDES = (4, 2, 1)[3 - LEVELS:] if LEVELS > 1 else (4,)
RADII = {4: 4, 2: 2, 1: 1}
NR = 5                                             # radar raster: valid, count, vr, u_x, u_y (+ age below)


# ----------------------------------------------------------------------------------------------------------- data
def maps_of(f):
    lanes = np.maximum(f["lane_divider"], f["road_divider"]).astype(np.float32)
    veh = np.maximum(f["vehicle"], f["two_wheeler"]).astype(np.float32)
    return np.stack([lanes, veh, f["pedestrian"].astype(np.float32)])


def yaw_of(T):
    return math.atan2(T[1, 0], T[0, 0])


def box_motion(cur, other):
    """per-cell target displacement (2, X, Y) in m, k's axes, of every observed box of keyframe k to the pose of the same
    instance in the other keyframe (rigid: centre + yaw change), and its mask (tests/test_flow_timed.py)."""
    T, To = cur["meta"]["T_e2g"], other["meta"]["T_e2g"]
    R, t = T[:3, :3], T[:3, 3]
    by_inst = {b["instance"]: b for b in other["meta"]["boxes"]}
    xs, ys = (c.numpy() for c in G.xy_centres())
    px, py = np.meshgrid(xs, ys, indexing="ij")
    inst = cur["instance"]
    D = np.zeros((2,) + inst.shape, np.float32)
    M = np.zeros(inst.shape, np.float32)
    for i, b in enumerate(cur["meta"]["boxes"]):
        cells = inst == i
        o = by_inst.get(b["instance"])
        if not cells.any() or o is None:
            continue
        c = np.asarray(b["center"])[:2]
        co = (R.T @ (np.asarray(o["world"]) - t))[:2]
        dyaw = (o["yaw"] + yaw_of(To)) - (b["yaw"] + yaw_of(T))
        cs, sn = math.cos(dyaw), math.sin(dyaw)
        qx, qy = px[cells] - c[0], py[cells] - c[1]
        D[0][cells] = co[0] + cs * qx - sn * qy - px[cells]
        D[1][cells] = co[1] + sn * qx + cs * qy - py[cells]
        M[cells] = 1.0
    return D, M


def radar_raster(r):
    """radar detections of k (x, y in k's frame at the camera time, psi = ray from the sensor, vr = ego-compensated radial
    speed, dt = age) -> (6, X, Y): valid, log(1 + count), mean vr, mean ray direction (unit), mean age. A fixed input
    encoding (no learning): the cell a detection falls in, means over its detections."""
    ix = np.floor((r["x"] - G.x_min) / G.res).astype(int)
    iy = np.floor((r["y"] - G.y_min) / G.res).astype(int)
    ok = (ix >= 0) & (ix < G.nx) & (iy >= 0) & (iy < G.ny)
    ix, iy = ix[ok], iy[ok]
    idx = ix * G.ny + iy
    N = G.nx * G.ny

    def acc(w):
        return np.bincount(idx, weights=w, minlength=N).reshape(G.nx, G.ny)
    cnt = acc(np.ones(len(idx)))
    m = np.maximum(cnt, 1.0)
    ux, uy = acc(np.cos(r["psi"][ok])), acc(np.sin(r["psi"][ok]))
    un = np.maximum(np.hypot(ux, uy), 1e-6)
    return np.stack([(cnt > 0), np.log1p(cnt), acc(r["vr"][ok]) / m, ux / un * (cnt > 0), uy / un * (cnt > 0),
                     acc(r["dt"][ok]) / m]).astype(np.float32)


RADAR = {}


def item(path):
    """keyframes k-1, k, k+1 of one cached GT sample + the radar of k -> dict, or None without a moving vehicle at k."""
    g = np.load(path, allow_pickle=True)["gt"].item()
    p, c, n = g["prev"], g["cur"], g["next"]
    vel = c["velocity"].astype(np.float32)
    if not ((maps_of(c)[1] > 0.5) & (np.hypot(*vel) > 1.0)).any():
        return None
    token = os.path.basename(path).split("_")[0]
    rf = RADAR.get(token)
    if not rf:
        return None
    Db, Mb = box_motion(c, p)
    Df, Mf = box_motion(c, n)
    static = ((c["instance"] < 0) & (c["ignore"] < 0.5)).astype(np.float32)
    return {"token": token,
            "maps": torch.from_numpy(np.stack([maps_of(p), maps_of(c), maps_of(n)])).half(),
            "vel": torch.from_numpy(vel),
            "inst": torch.from_numpy(c["instance"].astype(np.int16)),
            "D": torch.from_numpy(np.stack([Db, Df])), "M": torch.from_numpy(np.stack([Mb, Mf])).bool(),
            "static": torch.from_numpy(static).bool(),
            "radar": torch.from_numpy(radar_raster(np.load(rf[0]))),
            "s_back": [tuple(map(float, s)) for s in p["meta"]["steps"]],
            "s_fwd": [tuple(map(float, s)) for s in n["meta"]["steps"]],
            "s_prev": [(-float(dt), float(v), float(w)) for dt, v, w in reversed(p["meta"]["steps"])],
            "dt": (float(p["meta"]["dt"]), float(n["meta"]["dt"]))}


def items(test, n):
    path = f"temp/checks/flow_fused_nusc_{'test' if test else 'train'}_{n}.pt"
    if os.path.exists(path):
        return torch.load(path, weights_only=False)                    # our own scratch cache
    out = []
    for f in sorted(os.listdir("data/cache/radar")):                    # token -> its radar file (one per sample)
        RADAR.setdefault(f.split("_")[0], []).append(os.path.join("data/cache/radar", f))
    for f in sorted(glob.glob(f"data/cache/gt/*_0.5_4_{SIG}.npz")):
        scene = np.load(f, allow_pickle=True)["gt"].item()["cur"]["meta"]["scene"]
        if (int(scene[:8], 16) % 5 == 0) != test:
            continue
        it = item(f)
        if it is not None:
            out.append(it)
            if len(out) == n:
                break
    torch.save(out, path)
    return out


def batch_of(data, idx):
    b = [data[i] for i in idx]
    t = {k: torch.stack([x[k] for x in b]).to(DEV).float() for k in ("maps", "vel", "D", "M", "static", "radar")}
    t.update({k: [x[k] for x in b] for k in ("s_back", "s_fwd", "s_prev", "dt")})
    t["dtp"] = torch.tensor([abs(d[0]) for d in t["dt"]], device=DEV).view(-1, 1, 1, 1)   # k-1 -> k (stored as k -> k-1 < 0)
    if DOPSIM:            # the simulated Doppler of tests/test_flow_explains.py: GT radial, origin rays, half the occupied cells
        u0 = ray_dirs(1)[None]
        occ = t["maps"][:, 1, 1:].amax(1) > 0.5                       # vehicles and pedestrians of k
        valid = (occ & (torch.rand_like(occ.float()) < 0.5)).float()
        vr = (u0 * t["vel"]).sum(1) + 0.1 * torch.randn_like(valid)
        t["radar"] = torch.stack([valid, valid * math.log(2.0), vr * valid, u0[:, 0].expand_as(valid) * valid,
                                  u0[:, 1].expand_as(valid) * valid, torch.zeros_like(valid)], 1)
    t["radar_in"] = t["radar"] if DOP else torch.zeros_like(t["radar"])   # --no-dop: the model sees no radar (the loss can)
    return t


# ---------------------------------------------------------------------------------------------------------- model
_RAY = {}


def ray_dirs(s):
    """(2, X/s, Y/s) unit ray direction from the vehicle origin to each cell centre of the level with stride s."""
    if s not in _RAY:
        xs, ys = G.xy_centres()
        x = xs.view(-1, s).mean(1) if s > 1 else xs
        y = ys.view(-1, s).mean(1) if s > 1 else ys
        gx, gy = torch.meshgrid(x, y, indexing="ij")
        r = torch.sqrt(gx ** 2 + gy ** 2).clamp(min=1e-3)
        _RAY[s] = torch.stack([gx / r, gy / r]).to(DEV)
    return _RAY[s]


def coords(s, B):
    """fixed geometry channels (a conv is translation-equivariant and cannot know where the radar ray points):
    ray direction (2) and range / 60 of each cell."""
    u = ray_dirs(s)
    xs, ys = G.xy_centres()
    x = xs.view(-1, s).mean(1) if s > 1 else xs
    y = ys.view(-1, s).mean(1) if s > 1 else ys
    gx, gy = torch.meshgrid(x, y, indexing="ij")
    rng = (torch.sqrt(gx ** 2 + gy ** 2) / 60.0).to(DEV)[None]
    return torch.cat([u, rng], 0)[None].expand(B, -1, -1, -1)


def pool_radar(rad, s):
    """radar raster pooled to stride s: valid fraction, count, mean vr / ray / age over the detections in the coarse cell."""
    if s == 1:
        return rad
    v = rad[:, :1]
    pv = F.avg_pool2d(v, s)
    m = pv.clamp(min=1e-6)
    out = [pv, F.avg_pool2d(rad[:, 1:2], s), F.avg_pool2d(rad[:, 2:3] * v, s) / m]
    u = F.avg_pool2d(rad[:, 3:5] * v, s)
    out.append(u / u.norm(dim=1, keepdim=True).clamp(min=1e-6) * (pv > 0))
    out.append(F.avg_pool2d(rad[:, 5:6] * v, s) / m)
    return torch.cat(out, 1)


def local_corr(a, b, r):
    """a, b (B, c, X, Y) -> (B, (2r+1)^2, X, Y): sum_c a(p) b(p + d) / sqrt(c) for the displacements d of a +-r window."""
    B, c, X, Y = a.shape
    ub = F.unfold(b, 2 * r + 1, padding=r).view(B, c, (2 * r + 1) ** 2, X * Y)
    return (a.flatten(2)[:, :, None] * ub).sum(1).view(B, -1, X, Y) / c ** 0.5


def backward_warp(f, v, dt, s):
    """f (B, c, X/s, Y/s) in k's frame -> f sampled at p - v dt for every cell p of the level (bilinear, zeros outside)."""
    B, _, X, Y = f.shape
    res = G.res * s
    ix = torch.arange(X, device=f.device, dtype=f.dtype).view(1, X, 1)
    iy = torch.arange(Y, device=f.device, dtype=f.dtype).view(1, 1, Y)
    qx = ix - v[:, 0] * dt[:, 0] / res                                 # source position in cells
    qy = iy - v[:, 1] * dt[:, 0] / res
    grid = torch.stack([(qy + 0.5) / Y * 2 - 1, (qx + 0.5) / X * 2 - 1], -1)
    return F.grid_sample(f, grid, align_corners=False, padding_mode="zeros")


def block(cin, cout, dil=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=dil, dilation=dil), nn.GroupNorm(8, cout), nn.ReLU())


class Level(nn.Module):
    """one pyramid level: correlation + soft hypothesis + Doppler comparison -> a learned velocity update."""

    def __init__(self, s):
        super().__init__()
        self.s, self.r = s, RADII[s]
        k = (2 * self.r + 1) ** 2
        cin = C + 3 + 2 + (k + 2 if CORR else C) + ((NR + 1) + 2 if DOP else 0)
        dil = (1, 2, 4) if s == 4 else (1, 2)
        self.proj = nn.Conv2d(C, 16, 1)
        self.log_beta = nn.Parameter(torch.tensor(math.log(40.0)))     # correlation of unit vectors / 4: sharp start
        layers = [block(cin, 64)] + [block(64, 64, d) for d in dil[1:]]
        self.net = nn.Sequential(*layers, nn.Conv2d(64, 2, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        dx, dy = torch.meshgrid(torch.arange(-self.r, self.r + 1.0), torch.arange(-self.r, self.r + 1.0), indexing="ij")
        self.register_buffer("disp", torch.stack([dx.flatten(), dy.flatten()]) * G.res * s)   # (2, K) m, unfold order

    def forward(self, ek, ep, v_up, rad, dt):
        s = self.s
        fk = F.avg_pool2d(ek, s) if s > 1 else ek
        fp = F.avg_pool2d(ep, s) if s > 1 else ep
        w = backward_warp(fp, v_up, dt, s)
        B = ek.shape[0]
        parts = [fk, coords(s, B), v_up / 10.0]
        if CORR:
            c = local_corr(F.normalize(self.proj(fk), dim=1), F.normalize(self.proj(w), dim=1), self.r)
            a = torch.softmax(self.log_beta.exp() * c, 1)                         # soft frame hypothesis: expected displacement
            hyp = v_up - torch.einsum("bkxy,ck->bcxy", a, self.disp) / dt
            parts += [c, hyp / 10.0]
        else:
            hyp = v_up
            parts += [w]
        if DOP:
            r = pool_radar(rad, s)
            valid, vr, u = (r[:, :1] > 0).float(), r[:, 2:3], r[:, 3:5]
            parts += [r, valid * ((u * v_up).sum(1, keepdim=True) - vr) / 10.0,
                      valid * ((u * hyp).sum(1, keepdim=True) - vr) / 10.0]
        o = 10.0 * self.net(torch.cat(parts, 1))
        if RAY:                                                         # (radial, tangential) in the cell's ray frame
            u = ray_dirs(s)[None]
            o = o[:, :1] * u + o[:, 1:] * torch.stack([-u[:, 1], u[:, 0]], 1)
        return (hyp if CORR else v_up) + o


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(block(3, 32), block(32, 32), nn.Conv2d(32, C, 3, padding=1))
        self.levels = nn.ModuleList(Level(s) for s in STRIDES)

    def forward(self, t):
        """-> the velocity of every level (B, 2, X, Y) at 0.5 m, coarse to fine; the last one is the output."""
        m = t["maps"]
        ek = self.enc(m[:, 1])
        ep = propagate(self.enc(m[:, 0]), t["s_prev"], torch.zeros_like(ek[:, :2]), G)[0]    # ego alignment only
        v = ek.new_zeros(ek.shape[0], 2, G.nx // STRIDES[0], G.ny // STRIDES[0])
        outs = []
        for i, lv in enumerate(self.levels):
            if i > 0:
                v = F.interpolate(v, scale_factor=STRIDES[i - 1] // lv.s, mode="bilinear", align_corners=False)
            v = lv(ek, ep, v, t["radar_in"], t["dtp"])
            outs.append(v if lv.s == 1 else F.interpolate(v, scale_factor=lv.s, mode="bilinear", align_corners=False))
        return outs


# ----------------------------------------------------------------------------------------------------------- train
def smoothness(v, feat, alpha=10.0):
    tot = 0.0
    for dim in (2, 3):
        dv = (v.diff(dim=dim) / 10.0).abs().sum(1)
        w = torch.exp(-alpha * feat.diff(dim=dim).abs().sum(1))
        tot = tot + (w * dv).mean()
    return tot


def loss_of(t, v, terms):
    m, loss, parts = t["maps"], 0.0, {}
    B = m.shape[0]
    Mbox = t["M"].amax(1, keepdim=True)
    wbox = Mbox / Mbox.sum((2, 3), keepdim=True).clamp(min=1.0)
    st = t["static"][:, None]
    if "vel" in terms:
        parts["vel"] = (wbox * (v - t["vel"]).norm(dim=1, keepdim=True)).sum() / B
    if "box" in terms:
        for j in (0, 1):
            dt = torch.tensor([d[j] for d in t["dt"]], device=DEV).view(-1, 1, 1, 1)
            Mj = t["M"][:, j:j + 1]
            err = (v * dt - t["D"][:, j]).norm(dim=1, keepdim=True) / dt.abs()
            parts[f"box{j}"] = ((Mj / Mj.sum((2, 3), keepdim=True).clamp(min=1.0)) * err).sum() / B
    if "static" in terms:
        parts["static"] = 0.1 * (st * v.norm(dim=1, keepdim=True)).sum() / st.sum().clamp(min=1.0)
    if "warp" in terms:
        for j, sk, ti in ((0, "s_back", 0), (1, "s_fwd", 2)):
            parts[f"warp{j}"] = 3.0 * ((propagate(m[:, 1], t[sk], v, G)[0] - m[:, ti]) ** 2).mean() * 100.0
    if "dop" in terms:
        r = t["radar"]
        valid = r[:, :1]
        res = (r[:, 3:5] * v).sum(1, keepdim=True) - r[:, 2:3]
        parts["dop"] = (valid * F.huber_loss(res, torch.zeros_like(res), reduction="none", delta=1.0)).sum() \
            / valid.sum().clamp(min=1.0)
    if "smooth" in terms:
        parts["smooth"] = 0.05 * smoothness(v, m[:, 1]) * 100.0
    for x in parts.values():
        loss = loss + x
    return loss, parts


def train(model, data, iters, rng, bs=8, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=iters, pct_start=0.1)
    t0 = time.time()
    for it in range(iters):
        terms = THEN if THEN and it >= PRE * iters else LOSS
        t = batch_of(data, [rng.randrange(len(data)) for _ in range(bs)])
        outs = model(t)
        loss, parts = 0.0, {}
        for v in outs:                                                    # the same loss on every level
            lo, parts = loss_of(t, v, terms)
            loss = loss + lo
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if it % max(1, iters // 10) == 0 or it == iters - 1:
            print(f"    iter {it:5d} loss {loss.item():.4f} (last level: " + ", ".join(f"{k} {float(x):.3f}" for k, x in parts.items())
                  + f")  {time.time() - t0:.0f} s", flush=True)


# -------------------------------------------------------------------------------------------------------- evaluate
def iou(a, b, thr=0.5):
    a, b = a > thr, b > thr
    return (a & b).sum().item(), (a | b).sum().item()


@torch.no_grad()
def evaluate(model, test):
    epe, acc = {}, {}

    def add(k, s, n):
        a = epe.setdefault(k, [0.0, 0])
        a[0] += s
        a[1] += n
    u0 = ray_dirs(1).cpu()
    for x in test:
        t = batch_of([x], [0])
        m = t["maps"]
        outs = model(t)
        v = outs[-1]
        vg, zero = t["vel"], torch.zeros_like(v)
        for name, sk, ti in (("fwd", "s_fwd", 2), ("back", "s_back", 0)):
            for vn, vv in (("v", v), ("GT v", vg), ("v=0", zero)):
                a = acc.setdefault(f"IoU raw {name} {vn}", [0, 0])
                i, u = iou(propagate(m[:, 1], t[sk], vv, G)[0][0, 1], m[0, ti, 1])
                a[0] += i
                a[1] += u
        vgt, maps, inst = x["vel"], x["maps"][1].float(), x["inst"].long()
        veh = maps[1] > 0.5
        mov = veh & (vgt.norm(dim=0) > 1.0)
        for li, vl in enumerate(outs[:-1]):
            add(f"moving, level {li} ({G.res * STRIDES[li]:g} m)", (vl[0].cpu() - vgt).norm(dim=0)[mov].sum().item(), int(mov.sum()))
        vc = v[0].cpu()
        e = (vc - vgt).norm(dim=0)
        masks = {"vehicles": veh, "vehicles moving (> 1 m/s)": mov, "vehicles static (< 0.2 m/s)": veh & (vgt.norm(dim=0) < 0.2),
                 "pedestrians": maps[2] > 0.5, "lanes": (maps[0] > 0.05) & (maps[1:].max(0).values < 0.1)}
        for k, mk in masks.items():
            add(k, e[mk].sum().item(), int(mk.sum()))
        d = vc - vgt                                                    # radial / tangential error on moving vehicles
        add("moving: radial part", (u0 * d).sum(0).abs()[mov].sum().item(), int(mov.sum()))
        add("moving: tangential part", (u0[0] * d[1] - u0[1] * d[0]).abs()[mov].sum().item(), int(mov.sum()))
        hit = x["radar"][0] > 0
        for k in inst.unique().tolist():
            cells = (inst == k) & veh
            if k >= 0 and cells.any():
                err = (vc[:, cells].mean(1) - vgt[:, cells].mean(1)).norm().item()
                add("vehicles per box (mean v)", err, 1)
                if vgt[:, cells].mean(1).norm() > 1.0:
                    add("moving boxes WITH radar points" if (hit & cells).any() else "moving boxes WITHOUT radar points", err, 1)
    res = {"EPE " + k: s / max(1, n) for k, (s, n) in epe.items()}
    res["n moving boxes with / without radar"] = (epe.get("moving boxes WITH radar points", [0, 0])[1],
                                                   epe.get("moving boxes WITHOUT radar points", [0, 0])[1])
    res.update({k: i / max(1, u) for k, (i, u) in acc.items()})
    return res


@torch.no_grad()
def timing(model, x, n=30):
    t = batch_of([x], [0])
    for _ in range(5):
        model(t)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        model(t)
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e3


@torch.no_grad()
def picture(model, test, path, n=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from flow_viz import flow_rgb, wheel

    def overlay(a, b):
        oa, ob = a[1:].max(0).values.clamp(0, 1), b[1:].max(0).values.clamp(0, 1)
        lane = 0.35 * b[0].clamp(0, 1)
        return torch.stack([torch.maximum(oa, lane), torch.maximum(ob, lane), lane], -1).flip(1).numpy()
    cols = ["k (red) / k+1 (green)", "radar Doppler at k (red +, blue -)", "GT velocity at k", "learned velocity at k",
            "k moved with v (red) vs k+1", "k moved with v (red) vs k-1"]
    fig, ax = plt.subplots(n, len(cols) + 1, figsize=(2.3 * (len(cols) + 1), 2.5 * n), squeeze=False)
    for r in range(n):
        x = test[r]
        t = batch_of([x], [0])
        m = t["maps"]
        v = model(t)[-1]
        mp = x["maps"].float()
        occ = (mp[1, 1:].max(0).values > 0.3).float()
        rad = t["radar"][0].cpu()
        dop = torch.stack([(rad[2] / 10).clamp(0, 1), 0.25 * rad[0], (-rad[2] / 10).clamp(0, 1)], -1)
        dop = torch.maximum(dop, 0.3 * mp[1, 1:].max(0).values[..., None]).flip(1).numpy()
        ims = [overlay(mp[1], mp[2]), dop, flow_rgb(x["vel"], occ, 15.0), flow_rgb(v[0].cpu(), occ, 15.0),
               overlay(propagate(m[:, 1], t["s_fwd"], v, G)[0][0].cpu(), mp[2]),
               overlay(propagate(m[:, 1], t["s_back"], v, G)[0][0].cpu(), mp[0])]
        for c, im in enumerate(ims):
            ax[r, c].imshow(im, origin="lower")
            ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=7)
        ax[r, -1].axis("off")
    wheel(ax[0, -1], vmax=15.0)
    fig.suptitle(f"fused flow{TAG.replace('_', ' ')}; nuScenes keyframes 0.5 s apart; hue = direction, brightness = speed "
                 "(15 m/s full); x forward = up; red = moved, green = truth, yellow = both", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def layer0(test):
    """checks of the data and the fixed geometry before any training."""
    gm = torch.cat([((x["D"][1] / x["dt"][1] + x["D"][0] / x["dt"][0]) / 2 - x["vel"]).norm(dim=0)[(x["M"].all(0))
                    & (x["maps"][1, 1] > 0.5)] for x in test])
    print(f"check: box-motion targets vs annotated velocity on vehicle cells: median {gm.median():.3f}, mean {gm.mean():.3f} m/s")
    e, n_hit = [], 0
    for x in test:
        r, vg = x["radar"], x["vel"]
        mov = (r[0] > 0) & (x["maps"][1, 1] > 0.5) & (vg.norm(dim=0) > 1.0)
        e.append(((r[3:5] * vg).sum(0) - r[2]).abs()[mov])
        n_hit += int(mov.sum())
    e = torch.cat(e)
    print(f"check: radar Doppler vs radial GT velocity, cells of moving vehicles with a detection: median {e.median():.2f}, "
          f"mean {e.mean():.2f} m/s ({n_hit} cells in {len(test)} samples)")
    # the backward warp: E(F_k-1) ego-aligned and sampled at p - v_gt dt must match F_k better than with v = 0
    ious = {"GT v": [0, 0], "v = 0": [0, 0]}
    for x in test:
        t = batch_of([x], [0])
        m = t["maps"]
        prev = propagate(m[:, 0], t["s_prev"], torch.zeros_like(m[:, 0, :2]), G)[0]
        for k, vv in (("GT v", t["vel"]), ("v = 0", torch.zeros_like(t["vel"]))):
            i, u = iou(backward_warp(prev, vv, t["dtp"], 1)[0, 1], m[0, 1, 1])
            ious[k][0] += i
            ious[k][1] += u
    print("check: k-1 ego-aligned, backward-warped with GT v vs v = 0, vehicle IoU with k: "
          + ", ".join(f"{k} {i / max(1, u):.3f}" for k, (i, u) in ious.items()), flush=True)


if __name__ == "__main__":
    t0 = time.time()
    n_train, n_test, iters = (300, 16, 600) if QUICK else (1500, 48, 3000)
    iters = arg("--iters", iters)
    data, test = items(False, n_train), items(True, n_test)
    print(f"device {DEV}, seed {SEED}, run{TAG}; {len(data)} training samples, {len(test)} held-out, {iters} steps of 8 "
          f"({time.time() - t0:.0f} s to load)", flush=True)
    layer0(test)
    torch.manual_seed(SEED)
    model = Model().to(DEV)
    print(f"parameters {sum(p.numel() for p in model.parameters()) / 1e3:.0f} k", flush=True)
    train(model, data, iters, random.Random(SEED + 7))
    torch.save(model.state_dict(), f"temp/checks/model_fused{TAG}.pt")
    model.eval()
    r = evaluate(model, test)
    print("== held-out samples (EPE in m/s at keyframe k; IoU of vehicles)")
    for k, x in r.items():
        print(f"  {k:40s} {x:.3f}" if isinstance(x, float) else f"  {k:40s} {x}")
    print(f"  ms per frame (batch 1, {DEV}, GPU shared)   {timing(model, test[0]):.1f}")
    if DEV == "cuda":
        print(f"  peak GPU memory reserved (GB)            {torch.cuda.max_memory_reserved() / 1e9:.2f}")
    claims = [("vehicles: EPE < 1 m/s per cell", r["EPE vehicles"] < 1.0),
              ("moving vehicles: EPE < 1.5 m/s per cell", r["EPE vehicles moving (> 1 m/s)"] < 1.5),
              ("raw move to k+1 reaches >= 90 % of the GT-velocity IoU", r["IoU raw fwd v"] >= 0.9 * r["IoU raw fwd GT v"]),
              ("raw move to k-1 reaches >= 90 % of the GT-velocity IoU", r["IoU raw back v"] >= 0.9 * r["IoU raw back GT v"])]
    print("== claims")
    for name, ok in claims:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    picture(model, test, f"temp/checks/flow_fused{TAG}.png")
    print(f"== {sum(ok for _, ok in claims)} / {len(claims)} claims hold; image temp/checks/flow_fused{TAG}.png "
          f"({time.time() - t0:.0f} s)")
