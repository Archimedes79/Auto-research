"""Two prototypes, measured against the current code on REAL GT frames (CPU, 2 threads, backward only, no optimiser step):
 1. the map losses on stacked matrices (prediction P, target G, weight W, all (B, 12, X, Y) in spec order) with every GT-only quantity precomputed
    ("target pack"): must give the same group losses as losses.map_terms, in far fewer operations
 2. object_head.build_targets with numpy instead of one torch.tensor per field per box: must give the same tensors"""
import glob
import math
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.dirname(ROOT))
torch.set_num_threads(2)
import losses as L                                              # noqa: E402
import object_head as O                                         # noqa: E402
from lift_pd import Grid                                        # noqa: E402

EPS = 1e-6
N = 9
files = sorted(glob.glob(os.path.join(ROOT, "data", "cache", "gt", "*_0.5_4_15efbde2d1.npz")))[::211][:N]
gt = [np.load(f, allow_pickle=True)["gt"].item()["cur"] for f in files]
grid, cfg = Grid(), L.LossConfig(w_obj_vel=1.0)
lm = L.BEVLoss(cfg)
X, Y = grid.nx, grid.ny
T = L.build_layer_targets(gt, cfg, None, 0.0)
spec = lm.spec
idx = {}
for i, (_, kind, _) in enumerate(spec):
    idx.setdefault(kind, []).append(i)
C = len(spec)


def med(fn, n=7):
    fn()
    ts = []
    for _ in range(n):
        a = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - a) * 1000)
    return statistics.median(ts)


def nops(fn):
    with profile(activities=[ProfilerActivity.CPU]) as p:
        fn()
    return sum(e.count for e in p.key_averages())


# ------------------------------------------------------------------ 1. fused map loss
def pack_targets(T, cfg):
    """everything that depends on the GT and the weights only (not on the prediction): the target matrix G, the weight matrix W, the dilated
    line targets, the per-channel sums. Made once per sample set - in a worker, not in the training step."""
    G = torch.cat([T[spec[i][2]] for i in range(C)], 1)                          # (B, 12, X, Y)
    wmap = {"area": T["w_map"], "line": T["w_lines"], "object": T["w_obj"]}
    kinds = [spec[i][1] for i in range(C)]
    W = torch.cat([wmap["line" if k in ("line", "dist") else k].expand(-1, 1, -1, -1) for k in kinds], 1)       # (B, 12, X, Y)
    Wd = W[:, idx["dist"]] * torch.where(G[:, idx["dist"]] < cfg.dist_near, torch.ones(1), torch.full((1,), cfg.dist_w_far))
    lm_idx = idx["line"]
    m = (W[:, lm_idx] > 0).to(G.dtype)
    g = G[:, lm_idx] * m
    k = 2 * cfg.line_tolerance + 1
    pk = {"ao": idx["area"] + idx["object"], "line": lm_idx, "dist": idx["dist"],
          "Wao": W[:, idx["area"] + idx["object"]], "Gao": G[:, idx["area"] + idx["object"]],
          "Wl": W[:, lm_idx], "m": m, "g": g, "dg": F.max_pool2d(g, k, 1, cfg.line_tolerance),
          "Gd": G[:, idx["dist"]], "Wd": Wd}
    s = lambda t: t.sum((0, 2, 3))                                               # noqa: E731
    pk["cnt_ao"], pk["Sg_ao"] = s(pk["Wao"]).clamp(min=EPS), s(pk["Gao"] * pk["Wao"])
    pk["Sg_l"], pk["Sgw_l"] = s(g * pk["Wl"]), s(pk["Wl"])
    pk["cnt_d"] = s(Wd).clamp(min=EPS)
    return pk


def fused_groups(logits, pk, cfg):
    """the three group losses of the sem head from the stacked matrices: 3 slices of the prediction, ~35 tensor operations"""
    s = lambda t: t.sum((0, 2, 3))                                               # noqa: E731
    xa = logits[:, pk["ao"]]                                                     # areas + dense objects (8 channels)
    pa = torch.sigmoid(xa)
    Wa, Ga = pk["Wao"], pk["Gao"]
    bce = s(F.binary_cross_entropy_with_logits(xa, Ga, reduction="none") * Wa) / pk["cnt_ao"]
    ipg = s(pa * Ga * Wa)
    iou = torch.where(pk["Sg_ao"] > 0, 1 - ipg / (s(pa * Wa) + pk["Sg_ao"] - ipg).clamp(min=EPS), torch.zeros_like(ipg))
    pl = torch.sigmoid(logits[:, pk["line"]]) * pk["m"]
    dp = F.max_pool2d(pl, 2 * cfg.line_tolerance + 1, 1, cfg.line_tolerance)
    num = s(pl * pk["dg"] * pk["Wl"]) + s(pk["g"] * dp * pk["Wl"])
    den = s(pl * pk["Wl"]) + pk["Sg_l"]
    dice = torch.where(pk["Sg_l"] > 0, 1 - num / den.clamp(min=EPS), torch.zeros_like(num))
    xd = logits[:, pk["dist"]]
    xd = torch.where(pk["Gd"] >= cfg.dist_max - 1e-4, xd.clamp(max=cfg.dist_max), xd)
    hub = s(F.huber_loss(xd, pk["Gd"], delta=cfg.huber_delta, reduction="none") * pk["Wd"]) / pk["cnt_d"]
    return {"areas": (bce[:5] + cfg.w_area_iou * iou[:5]).mean(), "lines": dice.mean() + hub.mean(), "objects": (bce[5:] + iou[5:]).mean()}


torch.manual_seed(0)
sem = torch.randn(N, C, X, Y, requires_grad=True)
W = {"area": T["w_map"], "line": T["w_lines"], "object": T["w_obj"]}
pk = pack_targets(T, cfg)
_, ref = L.map_terms(sem, T, spec, W, cfg, 1.0)
got = fused_groups(sem, pk, cfg)
for g in ("areas", "lines", "objects"):
    assert abs(float(ref[g]) - float(got[g])) < 1e-5 * max(1.0, abs(float(ref[g]))), (g, float(ref[g]), float(got[g]))
print("1. fused map loss = losses.map_terms:", {g: (round(float(ref[g]), 5), round(float(got[g]), 5)) for g in ref})
g_ref = torch.autograd.grad(sum(ref.values()), sem)[0]
g_got = torch.autograd.grad(sum(got.values()), sem)[0]
print("   gradient max abs difference:", float((g_ref - g_got).abs().max()), " (reference gradient max", float(g_ref.abs().max()), ")")


def cur():
    t, g = L.map_terms(sem, T, spec, W, cfg, 1.0)
    return sum(g.values())


def new():
    return sum(fused_groups(sem, pk, cfg).values())


print(f"   current map_terms : fwd {med(cur):6.1f} ms  fwd+bwd {med(lambda: cur().backward()):6.1f} ms  ops {nops(lambda: cur().backward())}")
print(f"   fused matrices    : fwd {med(new):6.1f} ms  fwd+bwd {med(lambda: new().backward()):6.1f} ms  ops {nops(lambda: new().backward())}")
print(f"   target pack (GT only, once per sample set, would run in a worker): {med(lambda: pack_targets(T, cfg)):.1f} ms")


# ------------------------------------------------------------------ 2. build_targets in numpy
def build_targets_np(boxes_batch, grid, groups=L.NUSC_GROUPS, max_obj=64, min_radius=2):
    B, G_, X_, Y_ = len(boxes_batch), len(groups), grid.nx, grid.ny
    gidx = {name: i for i, (name, _) in enumerate(groups)}
    box_kind = np.array([k == "box" for _, k in groups])
    hm = np.zeros((B, G_, X_, Y_), np.float32)
    ind = np.zeros((B, max_obj), np.int64)
    mask, is_box = np.zeros((B, max_obj), np.float32), np.zeros((B, max_obj), np.float32)
    grp = np.zeros((B, max_obj), np.int64)
    off, size, rot, vel = (np.zeros((B, max_obj, 2), np.float32) for _ in range(4))
    rad, dr = np.zeros((B, max_obj, 1), np.float32), np.zeros((B, max_obj, 1), np.float32)
    for b, boxes in enumerate(boxes_batch):
        if not boxes:
            continue
        g = np.array([gidx[bx["group"]] for bx in boxes])
        cx, cy = np.array([bx["cx"] for bx in boxes]), np.array([bx["cy"] for bx in boxes])
        fx, fy = (cx - grid.x_min) / grid.res - 0.5, (cy - grid.y_min) / grid.res - 0.5
        ix, iy = np.floor(fx + 0.5).astype(np.int64), np.floor(fy + 0.5).astype(np.int64)
        ok = (ix >= 0) & (ix < X_) & (iy >= 0) & (iy < Y_)
        sel = np.nonzero(ok)[0][:max_obj]
        if not len(sel):
            continue
        n = len(sel)
        g, fx, fy, ix, iy = g[sel], fx[sel], fy[sel], ix[sel], iy[sel]
        bb = [boxes[i] for i in sel]
        isb = box_kind[g]
        length = np.array([bx.get("length", 1.0) for bx in bb]) / grid.res
        width = np.array([bx.get("width", 1.0) for bx in bb]) / grid.res
        d = 2 * np.array([bx.get("radius", 1.0) for bx in bb]) / grid.res
        h_, w_ = np.where(isb, length, d), np.where(isb, width, d)
        r = np.array([O.gaussian_radius(float(h), float(w)) for h, w in zip(h_, w_)])
        r = np.maximum(min_radius, r.astype(np.int64))
        sigma = (2 * r + 1) / 6
        for k in range(n):
            x0, x1, y0, y1 = max(0, ix[k] - r[k]), min(X_, ix[k] + r[k] + 1), max(0, iy[k] - r[k]), min(Y_, iy[k] + r[k] + 1)
            gx = np.arange(x0, x1, dtype=np.float32)[:, None]
            gy = np.arange(y0, y1, dtype=np.float32)[None, :]
            peak = np.exp(-((gx - ix[k]) ** 2 + (gy - iy[k]) ** 2) / np.float32(2 * sigma[k] * sigma[k]))
            win = hm[b, g[k], x0:x1, y0:y1]
            np.maximum(win, peak, out=win)
        ind[b, :n], mask[b, :n], grp[b, :n] = ix * Y_ + iy, 1.0, g
        off[b, :n] = np.stack([fx - ix, fy - iy], 1)
        vel[b, :n] = np.stack([[bx["vx"] for bx in bb], [bx["vy"] for bx in bb]], 1)
        yaw = np.array([bx.get("yaw", 0.0) for bx in bb])
        size[b, :n] = np.where(isb[:, None], np.log(np.maximum(np.stack([length, width], 1) * grid.res, 1e-9)), 0.0)
        rot[b, :n] = np.where(isb[:, None], np.stack([np.sin(yaw), np.cos(yaw)], 1), 0.0)
        dr[b, :n, 0] = np.where(isb, ((yaw - O.DIR_OFFSET) % (2 * math.pi) < math.pi).astype(np.float32), 0.0)
        rad[b, :n, 0] = np.where(isb, 0.0, np.log(np.maximum(np.array([bx.get("radius", 1.0) for bx in bb]), 1e-9)))
        is_box[b, :n] = isb
    return {k: torch.from_numpy(v) for k, v in dict(hm=hm, ind=ind, mask=mask, group=grp, off=off, size=size, rot=rot, rad=rad, vel=vel,
                                                     is_box=is_box, dir=dr).items()}


boxes = [[L.head_box(b) for b in bs if L.head_box(b)] for bs in T["boxes"]]
a, b = O.build_targets(boxes, grid, L.NUSC_GROUPS, cfg.max_obj), build_targets_np(boxes, grid, L.NUSC_GROUPS, cfg.max_obj)
bad = [k for k in a if not torch.allclose(a[k].float(), b[k].float(), atol=1e-5)]
print(f"2. build_targets numpy == torch version for all {len(a)} tensors:", not bad, bad,
      f"({sum(len(x) for x in boxes)} boxes in {len(boxes)} frames)")
print(f"   current {med(lambda: O.build_targets(boxes, grid, L.NUSC_GROUPS, cfg.max_obj)):6.1f} ms   numpy {med(lambda: build_targets_np(boxes, grid, L.NUSC_GROUPS, cfg.max_obj)):6.1f} ms   (one dt, 9 frames)")
