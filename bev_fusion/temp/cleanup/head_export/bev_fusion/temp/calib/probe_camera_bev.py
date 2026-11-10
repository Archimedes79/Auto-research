"""One-off (t3_improve_loop, why is the camera path at the no-sensor floor?): linear probe on the LIFTED camera BEV features.

Model-free test of the camera information: lift the cached frozen features (context 80 ch x depth probability) of the current frame with the
model's own LiftPD (8 height slices x 80 channels, oversample 4), then fit a LINEAR classifier per BEV cell on N_TRAIN train samples (one per
scene) and measure IoU on N_VAL val samples (one per scene) on the camera-visible, defined cells. Compared with the prior map (the per-cell
mean of the train targets) and with prior + features.  If a linear probe on the lifted features beats the prior clearly, the information is in
the lifted BEV and the training of trunk / heads is what fails; if it does not, the lifted features carry (almost) nothing the prior has not.
Usage (bev_fusion folder): wsl ~/miniconda3/envs/gcar/bin/python temp/calib/probe_camera_bev.py [n_train n_val]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "training")
from camera_encoder import depth_centres  # noqa: E402
from feature_cache import CachedFrontRadarDataset, FeatureStore  # noqa: E402
from lift_pd import Grid, LiftPD  # noqa: E402
from model import Z_LEVELS  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402

torch.set_num_threads(2)
n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 40
n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 24
root = os.path.expanduser("~/data/nuscenes_full")
nusc = NuScenes("v1.0-trainval", dataroot=root, verbose=False)
sp = json.load(open("data/scenes_parts1_6.json"))
grid = Grid(z_levels=Z_LEVELS)
lift = LiftPD(grid, depth_centres(), oversample=4)
store = FeatureStore("data/cache/cam_feats")


def samples(split, n):
    names = set(sp[split])
    sc = [s["token"] for s in nusc.scene if s["name"] in names]
    sc = sc[:: max(1, len(sc) // n)][:n]
    ds = CachedFrontRadarDataset(root, "v1.0-trainval", sc, store=store, radar_cache="data/cache/radar", with_gt=True,
                                 cache_dir="data/cache/gt", nusc=nusc)
    per_scene = {}
    for i, t in enumerate(ds.samples):
        per_scene.setdefault(nusc.get("sample", t)["scene_token"], []).append(i)
    idx = [v[len(v) // 2] for v in per_scene.values()]               # the middle keyframe of each scene
    return ds, idx


TARGETS = ("drivable_area", "walkway", "vehicle", "lane_divider")


def build(split, n):
    ds, idx = samples(split, n)
    X, Y, M = [], {k: [] for k in TARGETS}, []
    for i in idx:
        it = ds[i]
        f = it["frames"][0]
        with torch.no_grad():
            vol = lift(f["ctx"].float()[None], f["prob"].float()[None], f["K"], f["T_cam_car"], (256, 704), it["ida"])[0]   # (640, X, Y)
        g = it["gt"]["cur"]
        m = (np.asarray(g["camera"], np.float32) > 0) & (np.asarray(g["undefined"], np.float32) == 0)
        X.append(vol.reshape(vol.shape[0], -1).T[torch.from_numpy(m.reshape(-1))])
        M.append(torch.from_numpy(np.stack(np.nonzero(m), 1)))        # cell indices (for the prior map)
        for k in TARGETS:
            Y[k].append(torch.from_numpy((np.asarray(g[k], np.float32) > 0.5).reshape(-1)[m.reshape(-1)]))
    return X, Y, M


print(f"building probe data: {n_train} train scenes, {n_val} val scenes")
Xt, Yt, Mt = build("train", n_train)
Xv, Yv, Mv = build("val", n_val)
Xt, Xv = torch.cat(Xt), torch.cat(Xv)
Mt, Mv = torch.cat(Mt), torch.cat(Mv)
print("train cells", len(Xt), "val cells", len(Xv), "features", Xt.shape[1])
mu, sd = Xt.mean(0), Xt.std(0).clamp(min=1e-4)
zsum = lambda X: X.view(len(X), len(Z_LEVELS), 80).sum(1)
Xt_s, Xv_s = (Xt - mu) / sd, (Xv - mu) / sd


def prior_map(Y, M):
    s, c = torch.zeros(grid.nx, grid.ny), torch.zeros(grid.nx, grid.ny)
    s.index_put_((M[:, 0], M[:, 1]), Y.float(), accumulate=True)
    c.index_put_((M[:, 0], M[:, 1]), torch.ones(len(M)), accumulate=True)
    return s, c


def iou(pred, y):
    tp = (pred & y).sum().item()
    return tp / max(1, (pred | y).sum().item())


def best_thr(score, y):
    qs = torch.quantile(score[torch.randperm(len(score))[:200000]], torch.linspace(0.5, 0.995, 40))
    return max(qs.tolist(), key=lambda t: iou(score > t, y))


def fit(Xa, y, steps=300, wd=1e-2):
    w = torch.zeros(Xa.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pos = y.float().mean().clamp(min=1e-3)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        z = Xa @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y.float(), pos_weight=((1 - pos) / pos).clamp(max=20)) + wd * (w ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


print("\n| target | prior map IoU val | probe zsum80 | probe full640 | prior + full640 | positives (val) |")
print("|---|---|---|---|---|---|")
for k in TARGETS:
    yt, yv = torch.cat(Yt[k]), torch.cat(Yv[k])
    s, c = prior_map(yt, Mt)
    pt = (s / c.clamp(min=1))[Mt[:, 0], Mt[:, 1]]
    pv = (s / c.clamp(min=1))[Mv[:, 0], Mv[:, 1]]
    thr_p = best_thr(pt, yt)
    r = [iou(pv > thr_p, yv)]
    for Xa_t, Xa_v in ((zsum(Xt_s * sd + mu) , zsum(Xv_s * sd + mu)), (Xt_s, Xv_s),
                       (torch.cat([Xt_s, torch.logit(pt.clamp(1e-3, 1 - 1e-3))[:, None]], 1), torch.cat([Xv_s, torch.logit(pv.clamp(1e-3, 1 - 1e-3))[:, None]], 1))):
        m2, s2 = Xa_t.mean(0), Xa_t.std(0).clamp(min=1e-4)
        Xa_t, Xa_v = (Xa_t - m2) / s2, (Xa_v - m2) / s2
        w, b = fit(Xa_t, yt)
        thr = best_thr(Xa_t @ w + b, yt)
        r.append(iou((Xa_v @ w + b) > thr, yv))
    print(f"| {k} | " + " | ".join(f"{x:.3f}" for x in r) + f" | {int(yv.sum())} / {len(yv)} |")
