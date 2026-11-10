"""One-off (t3_improve_loop): the TRUE no-sensor floor of the detection metric.
A "detector" without any sensor: the same detections in every frame, the top-k local maxima of the per-cell mean object mask of the
train set (smoothed), scores = that density, mean box size, heading along x. Evaluated with the model's own DetMeter on the same 1000
val frames as the study arms (`val_subset(val_ds, 1000)`), dt = 0. Also reports the map floor (per-cell mean map, evaluate_run SegMeter).
Usage (bev_fusion folder, CPU): python temp/calib/prior_detector.py [n_train_samples default 400]
"""
import os
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, ".")
sys.path.insert(0, "training")
from evaluation import bev_probe as ev  # noqa: E402
from evaluation.metrics import DetMeter, report  # noqa: E402
from losses import NUSC_GROUPS  # noqa: E402
from object_head import decode  # noqa: E402
from train import TrainConfig, make_datasets, val_subset  # noqa: E402
import json  # noqa: E402

torch.set_num_threads(2)
n_prior = int(sys.argv[1]) if len(sys.argv) > 1 else 400
cfg = TrainConfig(**json.load(open("runs/t2_ref_s0/config.json")))
cfg.workers = 0
cfg.radar_cache = cfg.radar_cache or "data/cache/radar"
train_ds, val_ds = make_datasets(cfg)
from losses import layer_spec  # noqa: E402
spec = layer_spec(False, True)
pm, _ = ev.prior_maps(train_ds, n_prior, spec)
from lift_pd import Grid  # noqa: E402
grid = Grid()
groups = [g for g, _ in NUSC_GROUPS]
print("groups", groups, "prior layers", [k for k in pm if k in groups or "vehicle" in k or "pedestrian" in k][:6])
G, X, Y = len(groups), grid.nx, grid.ny
hm = torch.zeros(1, G, X, Y)
for gi, name in enumerate(groups):
    m = gaussian_filter(pm[name].astype(np.float64), 2.0)
    hm[0, gi] = torch.from_numpy((m / max(m.max(), 1e-9)).astype(np.float32)).clamp(1e-4, 0.999)
size = torch.cat([torch.full((1, 1, X, Y), 4.6), torch.full((1, 1, X, Y), 1.9)], 1)       # one size map: irrelevant for AP (centre distance)
out = {"hm": hm, "off": torch.zeros(1, 2, X, Y), "vel": torch.zeros(1, 2, X, Y), "size": size, "rad": torch.zeros(1, 1, X, Y),
       "rot": torch.cat([torch.zeros(1, 1, X, Y), torch.ones(1, 1, X, Y)], 1)}
dets = decode(out, grid, NUSC_GROUPS, k=100, threshold=0.05, from_probabilities=True)[0]
print("detections per frame:", len(dets), {g: sum(d["group"] == g for d in dets) for g in groups})
sub = val_subset(val_ds, 1000)
det = DetMeter(classes=groups)
for i in range(len(sub)):
    g = sub[i]["gt"]["cur"]
    det.add(dets, ev.gt_boxes(g), dt=0.0)
flat = report(seg=None, det=det)
key = lambda c, t: flat[f"det/dt+0.00/{c}/all/ap@{t}"]
mAP2 = sum(key(c, t) for c in ("vehicle", "pedestrian") for t in (0.5, 1, 2, 4)) / 8
print(f"\nPRIOR DETECTOR (no sensor, {len(dets)} fixed detections per frame), {len(sub)} val frames:")
print(f"mAP2 {mAP2:.3f}  mAP3 {flat['det/dt+0.00/mean/all/mAP']:.3f}")
for c in ("vehicle", "pedestrian"):
    print(c, " ".join(f"AP@{t} {key(c, t):.3f}" for t in (0.5, 1, 2, 4)))
