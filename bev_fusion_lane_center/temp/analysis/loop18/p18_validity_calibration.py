"""Loop 18 / WP4 (literature gap 3): is the path validity pi of the builder's nodes a calibrated probability that the node lies on a ground-truth lane (within 1 m)?  All valid nodes of the 873 VAL frames (champion f1, M 288 / K0 144).
Reliability table, ECE (10 bins), ECE after an isotonic fit on the first half of the frames applied to the second half.  Usage (gcar): p18_validity_calibration.py"""
import os
import pickle
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402

raw, gts = pickle.load(open(os.path.expanduser("~/exp_gcar_b123/p18/raw_val873.pkl"), "rb"))
PI_, Y, FR = [], [], []
for f, (r, gt) in enumerate(zip(raw, gts)):
    P, _, _ = GE.dense_points(gt, 0.5)
    sel = np.where(r["valid"] > 0)[0]
    if len(sel) == 0 or len(P) == 0:
        continue
    pos = r["pos"] if "pos_rev" not in r else r["pos_rev"]
    d = cKDTree(P).query(pos[sel])[0]
    PI_.append(r["pi"][sel])
    Y.append((d <= 1.0).astype(float))
    FR.append(np.full(len(sel), f))
p, y, fr = np.concatenate(PI_), np.concatenate(Y), np.concatenate(FR)


def ece(p, y, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean()) for k in range(bins) if (b == k).any())


def pav(x, y):
    o = np.argsort(x)
    xs, ys = x[o], y[o].astype(float)
    out = []
    for i in range(len(ys)):
        out.append([ys[i], 1, xs[i], xs[i]])
        while len(out) > 1 and out[-2][0] >= out[-1][0]:
            b2, b1 = out.pop(), out.pop()
            w = b1[1] + b2[1]
            out.append([(b1[0] * b1[1] + b2[0] * b2[1]) / w, w, b1[2], b2[3]])
    return np.array([b[2] for b in out]), np.array([b[0] for b in out])


print(f"{len(p)} valid nodes, {y.mean():.3f} on a lane; mean pi {p.mean():.3f}")
edges = np.linspace(0, 1, 11)
print("pi bin: n / fraction on a lane  " + " | ".join(f"{edges[i]:.1f}-{edges[i + 1]:.1f}: {int(((p >= edges[i]) & (p < edges[i + 1] + (i == 9))).sum())} / {y[(p >= edges[i]) & (p < edges[i + 1] + (i == 9))].mean():.2f}" for i in range(10) if ((p >= edges[i]) & (p < edges[i + 1] + (i == 9))).any()))
h = fr <= 436
xs, ms = pav(p[h], y[h])
print(f"ECE raw {ece(p, y):.3f} (second half {ece(p[~h], y[~h]):.3f}); second half after isotonic fit on the first {ece(np.interp(p[~h], xs, ms), y[~h]):.3f}")
