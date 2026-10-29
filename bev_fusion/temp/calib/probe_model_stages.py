"""One-off (t3_improve_loop): WHERE does the camera information get lost inside a trained model?
Linear probes (as probe_camera_bev.py: fitted on N_TRAIN train samples, IoU on N_VAL val samples, camera-visible defined cells) on the
activations of a trained run at five stages: lifted features (before the model), cam_reduce output (camera BEV), fused BEV (after fusion),
trunk output, head input (after head_reduce). The stage where the probe IoU falls to the prior is where the information is lost.
Usage (bev_fusion folder, CPU): python temp/calib/probe_model_stages.py <run dir> [--ckpt-dir D] [n_train n_val]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "training")
from evaluation import evaluate_run as ev  # noqa: E402
from train import forward_loss, make_datasets, make_loader  # noqa: E402

torch.set_num_threads(2)
args = [a for a in sys.argv[1:]]
ckpt_dir = None
if "--ckpt-dir" in args:
    i = args.index("--ckpt-dir")
    ckpt_dir = args[i + 1]
    del args[i:i + 2]
control = ""
if "--control" in args:
    i = args.index("--control")
    control = args[i + 1]
    del args[i:i + 2]
run = args[0]
n_train = int(args[1]) if len(args) > 1 else 24
n_val = int(args[2]) if len(args) > 2 else 16
cfg, model, loss, step, ckpt = ev.load_run(run, "cpu", ckpt_dir)
cfg.batch_size, cfg.workers, cfg.seq_len = 2, 0, 1
if not cfg.radar_cache and os.path.isdir("data/cache/radar"):
    cfg.radar_cache = "data/cache/radar"
train_ds, val_ds = make_datasets(cfg)
print(f"{run}: step {step}, radar_kind {cfg.model.get('radar_kind', 'raster')}")

CAP = {}
orig_cam = model.camera_bevs
model.camera_bevs = lambda batch: CAP.__setitem__("cam_reduce", (r := orig_cam(batch))[:, 0].detach()) or r
if model.fusion is not None:
    model.fusion.register_forward_hook(lambda m, i, o: CAP.__setitem__("fused", o.detach()))
orig_feat = model.trunk.features
model.trunk.features = lambda *a, **k: CAP.__setitem__("trunk", (r := orig_feat(*a, **k)).detach()) or r
if model.head_reduce is not None:
    model.head_reduce.register_forward_hook(lambda m, i, o: CAP.__setitem__("head_in", o.detach()))
TARGETS = ("drivable_area", "walkway", "vehicle")
STAGES = ("cam_reduce", "fused", "trunk", "head_in")


def collect(ds, n):
    sub = ev.val_subset(ds, n)
    loader = make_loader(sub, cfg, False)
    feats = {s: [] for s in STAGES}
    Y = {k: [] for k in TARGETS}
    M = []
    O = {k: [] for k in TARGETS}                                  # the model's own output logit (semantic head) of each target
    spec_idx = ev.layer_index(model.spec)
    with torch.no_grad():
        for batch in loader:
            CAP.clear()
            if control:
                ev.perturb(batch, control)
            out, res, gts = forward_loss(model, loss, batch, 1.0, "cpu", "off", cfg, train_terms=False)
            for b in range(len(batch["tokens"])):
                g = gts[0.0][b]
                m = (np.asarray(g["camera"], np.float32) > 0) & (np.asarray(g["undefined"], np.float32) == 0)
                flat = m.reshape(-1)
                for s in STAGES:
                    if s in CAP:
                        f = CAP[s][b]
                        if f.shape[-2:] != m.shape:
                            f = torch.nn.functional.interpolate(f[None], size=m.shape, mode="bilinear", align_corners=False)[0]
                        feats[s].append(f.reshape(f.shape[0], -1).T[torch.from_numpy(flat)])
                for k in TARGETS:
                    Y[k].append(torch.from_numpy((np.asarray(g[k], np.float32) > 0.5).reshape(-1)[flat]))
                    O[k].append(res["out"][0.0]["sem"][b, spec_idx[k][0]].reshape(-1)[torch.from_numpy(flat)].float())
                M.append(torch.from_numpy(np.stack(np.nonzero(m), 1)))
    return {s: torch.cat(v) for s, v in feats.items() if v}, {k: torch.cat(v) for k, v in Y.items()}, torch.cat(M), {k: torch.cat(v) for k, v in O.items()}


def iou(pred, y):
    return (pred & y).sum().item() / max(1, (pred | y).sum().item())


def best_thr(score, y):
    qs = torch.quantile(score[torch.randperm(len(score))[:200000]], torch.linspace(0.5, 0.995, 40))
    return max(qs.tolist(), key=lambda t: iou(score > t, y))


def fit(X, y, steps=300, wd=1e-2):
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pos = y.float().mean().clamp(min=1e-3)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        l = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y.float(), pos_weight=((1 - pos) / pos).clamp(max=20)) \
            + wd * (w ** 2).mean()
        opt.zero_grad()
        l.backward()
        opt.step()
    return w.detach(), b.detach()


print("collecting train activations ...", flush=True)
Ft, Yt, Mt, Ot = collect(train_ds, n_train)
print("collecting val activations ...", flush=True)
Fv, Yv, Mv, Ov = collect(val_ds, n_val)
print(f"cells: train {len(Mt)}, val {len(Mv)}; stages {list(Ft)}")
print("\n| target | " + " | ".join(f"probe {s}" for s in Ft) + " | positives (val) |")
print("|---|" + "---|" * (len(Ft) + 1))
for k in TARGETS:
    r = []
    for s in Ft:
        mu, sd = Ft[s].mean(0), Ft[s].std(0).clamp(min=1e-4)
        Xt, Xv = (Ft[s] - mu) / sd, (Fv[s] - mu) / sd
        w, b = fit(Xt, Yt[k])
        thr = best_thr(Xt @ w + b, Yt[k])
        r.append(iou((Xv @ w + b) > thr, Yv[k]))
    thr = best_thr(Ot[k], Yt[k])
    r.append(iou(Ov[k] > thr, Yv[k]))
    print(f"| {k} | " + " | ".join(f"{x:.3f}" for x in r) + f" | {int(Yv[k].sum())} / {len(Yv[k])} |")
