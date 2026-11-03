"""Cost of each loss family on REAL GT frames and real-size tensors: CPU forward ms, forward + backward ms, aten operator calls (a proxy for GPU kernel launches).
One dt, 9 samples; the sem and object families run at 3 dt in training, motion and the aux heads at dt 0 only (model.forward)."""
import glob
import os
import statistics
import sys
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.dirname(ROOT))
torch.set_num_threads(2)
import losses as L                                              # noqa: E402
import object_head as O                                         # noqa: E402
from lift_pd import Grid                                        # noqa: E402

N = 9
files = sorted(glob.glob(os.path.join(ROOT, "data", "cache", "gt", "*_0.5_4_15efbde2d1.npz")))[::211][:N]
gt = [np.load(f, allow_pickle=True)["gt"].item()["cur"] for f in files]
grid = Grid()
X, Y = grid.nx, grid.ny
cfg = L.LossConfig(w_obj_vel=1.0)
lm = L.BEVLoss(cfg)
T = L.build_layer_targets(gt, cfg, None, 0.0)
torch.manual_seed(0)
head = O.ObjectHead(8, groups=L.NUSC_GROUPS, direction=True, moving=True)


def leaf(*shape):
    return torch.randn(*shape, requires_grad=True)


def med(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        a = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - a) * 1000)
    return statistics.median(ts)


def ops(fn):
    with profile(activities=[ProfilerActivity.CPU]) as p:
        fn()
    return sum(e.count for e in p.key_averages())


sem, cam, rad = leaf(N, len(lm.spec), X, Y), leaf(N, len(lm.cam_spec), X, Y), leaf(N, len(lm.radar_spec), X, Y)
maps = {k: v.detach().clone().requires_grad_() for k, v in head(torch.randn(N, 8, X, Y)).items()}
vhat = leaf(N, 2, X, Y)
W = {"area": T["w_map"], "line": T["w_lines"], "object": T["w_obj"]}
wc = T["w_cam"]


idx = {}
for i, (_, kind, _) in enumerate(lm.spec):
    idx.setdefault(kind, []).append(i)


def tg(kind):
    return torch.cat([T[lm.spec[i][2]] for i in idx[kind]], 1)


def areas():
    x, t, w = sem[:, idx["area"]], tg("area"), W["area"]
    return (L.soft_bce(x, t, w) + 0.5 * L.soft_iou_loss(torch.sigmoid(x), t, w)).mean()


def areas_bce_only():
    return L.soft_bce(sem[:, idx["area"]], tg("area"), W["area"]).mean()


def lines():
    w = W["line"]
    return L.tolerant_dice_loss(torch.sigmoid(sem[:, idx["line"]]), tg("line"), w, 1).mean() +         L.dist_huber(sem[:, idx["dist"]], tg("dist"), w, cfg.huber_delta, cfg.dist_near, cfg.dist_w_far, cfg.dist_max).mean()


def objects():
    x, t, w = sem[:, idx["object"]], tg("object"), W["object"]
    return (L.soft_bce(x, t, w) + L.soft_iou_loss(torch.sigmoid(x), t, w)).mean()


fams = [
    ("areas   BCE + 0.5 (1-IoU), 5 ch   (x3 dt)", areas),
    ("   same, BCE only (no IoU term)", areas_bce_only),
    ("lines   dilated Dice + Huber, 2 ch (x3 dt)", lines),
    ("dense objects BCE + IoU, 3 ch (x3 dt)", objects),
    ("object head: targets + focal + 8 L1/BCE (x3 dt)", lambda: L.object_head_terms(maps, T["boxes"], T["w_obj"], grid, L.NUSC_GROUPS, cfg.w_obj_vel, cfg.max_obj, cfg.w_obj_mov)[0]["total"]),
    ("motion  L1 on object cells + background (x1)", lambda: L.motion_term(vhat, T, cfg)),
    ("camera aux head, 12 ch (x1)", lambda: sum(L.map_terms(cam, T, lm.cam_spec, {"area": wc, "line": wc, "object": wc * T["w_keep"]}, cfg, 1.0)[1].values())),
    ("radar aux head, objects (x1)", lambda: sum(L.map_terms(rad, T, lm.radar_spec, {"object": T["w_radar"] * T["w_keep"], "stationary": T["w_radar"]}, cfg, 1.0)[1].values())),
]
print(f"{'loss family':52s} {'fwd ms':>8s} {'fwd+bwd':>8s} {'ops fwd+bwd':>12s}")
for name, f in fams:
    fwd = med(f)
    fb = med(lambda: f().backward())
    n = ops(lambda: f().backward())
    print(f"{name:52s} {fwd:8.1f} {fb:8.1f} {n:12d}")
