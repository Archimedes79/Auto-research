"""Which array convention do the cached GaussianCaR map logits (lane project, val_logit7.npy [N, 7, 200, 200]) have, relative to our
vehicle-frame front grid (x forward 0..60, y left -26..26, 0.5 m)? Throw-away check of the SOA-gap analysis (2026-10-30), CPU.
Candidates: the 8 axis orientations x forward offsets; score = IoU of max(sigmoid(lane), sigmoid(road_segment)) > 0.5 against our
drivable_area GT on the camera cells of N frames. The best mapping must be unambiguous (clear peak), else the reference is not used.
Usage (from the bev_fusion folder): python temp/soa_gap/align_gcar.py [n_frames]"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, "training")
sys.path.insert(0, "evaluation")
os.environ.setdefault("OMP_NUM_THREADS", "2")

from train import TrainConfig, make_datasets  # noqa: E402

HOME = os.path.expanduser("~")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
RES = 0.5
XS = 0.0 + (np.arange(120) + 0.5) * RES                  # our cell centres, forward
YS = -26.0 + (np.arange(104) + 0.5) * RES                 # our cell centres, left
tok = json.load(open(f"{HOME}/exp_gcar_b123/feats/val_tokens.json"))
lg = np.load(f"{HOME}/exp_gcar_b123/feats/val_logit7.npy", mmap_mode="r")
print("logit7", lg.shape, lg.dtype)

raw = json.load(open("runs/t1_r4_soa/config.json"))
cfg = TrainConfig(**raw)
_, val_ds = make_datasets(cfg)
pos = {t: i for i, t in enumerate(val_ds.samples)}
rows = [(i, t) for i, t in enumerate(tok) if t in pos]
print("frames of the cached logits that are samples of our val set:", len(rows), "of", len(tok))
pick = rows[:: max(1, len(rows) // N)][:N]


def sample_grid(arr, fwd, left):
    """arr [200, 200]; fwd/left: metric coordinates of every target cell (X, Y) -> nearest-cell lookup under the candidate mapping."""
    return arr


# candidate: GaussianCaR cell (r, c) centre -> metres a = -50 + (r + .5) .5, b = -50 + (c + .5) .5.  The vehicle-frame coordinate of
# our cell (x forward, y left) is mapped to (a, b) by one of 8 signed permutations, plus an offset of the forward axis.
best = {}
for swap, sa, sb, off in itertools.product((False, True), (1, -1), (1, -1), (-1.88, -0.94, 0.0, 0.94, 1.88)):
    ious = []
    for i, t in pick:
        g = val_ds._gt(t)["cur"]
        drv = np.asarray(g["drivable_area"], np.float32) > 0.5
        cam = np.asarray(g["camera"], np.float32) > 0
        X, Y = np.meshgrid(XS + off, YS, indexing="ij")
        a, b = (Y, X) if swap else (X, Y)
        r = np.clip(np.floor((sa * a + 50.0) / RES).astype(int), 0, 199)
        c = np.clip(np.floor((sb * b + 50.0) / RES).astype(int), 0, 199)
        L = np.asarray(lg[i])
        p = np.maximum(1 / (1 + np.exp(-L[0])), 1 / (1 + np.exp(-L[1])))[r, c] > 0.5
        m = cam
        tp, fp, fn = (p & drv & m).sum(), (p & ~drv & m).sum(), (~p & drv & m).sum()
        ious.append((tp, tp + fp + fn))
    tp = sum(a for a, _ in ious)
    den = sum(b for _, b in ious)
    best[(swap, sa, sb, off)] = tp / max(1, den)
for k, v in sorted(best.items(), key=lambda kv: -kv[1])[:8]:
    print("swap=%s sa=%+d sb=%+d forward offset=%+.2f m  drivable IoU %.3f" % (*k, v))
