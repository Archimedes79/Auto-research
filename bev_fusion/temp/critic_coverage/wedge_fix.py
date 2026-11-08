"""Does dropping the (vc . p > 0) clause make the pre-filter exact? Compare blocked(fixed prefilter) with blocked(no prefilter)."""
import sys, math
import numpy as np
from autogt import coverage as C

def wedge_fixed(cam_xy, pts, fp):
    v = fp - cam_xy
    vc = fp.mean(0) - cam_xy
    ang = np.arctan2(vc[0] * v[:, 1] - vc[1] * v[:, 0], vc[0] * v[:, 0] + vc[1] * v[:, 1])
    if ang.max() - ang.min() > 2.6 or (vc @ vc) < 1e-6:
        return np.ones(len(pts), bool)
    lo, hi = v[np.argmin(ang)], v[np.argmax(ang)]
    p = pts - cam_xy
    return ((lo[0] * p[:, 1] - lo[1] * p[:, 0]) >= -1e-9) & ((p[:, 0] * hi[1] - p[:, 1] * hi[0]) >= -1e-9)

rng = np.random.default_rng(3)
orig = C._wedge
bad_orig = bad_fixed = tot = 0
for it in range(6000):
    cam = np.array([rng.uniform(-3, 8), rng.uniform(-4, 4), rng.uniform(1.0, 2.0)]) if it % 2 else np.array([1.7, 0.0, 1.5])
    cx = rng.uniform(cam[0] - 8, cam[0] + 20); cy = rng.uniform(-6, 6)
    yaw = rng.uniform(-math.pi, math.pi) if rng.random() < 0.5 else rng.choice([0, math.pi / 2]) + rng.normal(0, 0.05)
    l = rng.choice([rng.uniform(0.4, 1.0), rng.uniform(3.5, 5.0), rng.uniform(8, 13)]); w = rng.uniform(0.4, 3.0)
    loc = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    c, s = math.cos(yaw), math.sin(yaw)
    fp = loc @ np.array([[c, s], [-s, c]]) + [cx, cy]
    prism = (fp, rng.uniform(-0.5, 0.5), 0)
    prism = (fp, prism[1], prism[1] + rng.uniform(0.5, 4))
    P = np.c_[rng.uniform(max(cam[0], 0), 60, 200), rng.uniform(-26, 26, 200), np.zeros(200)]
    C._wedge = lambda a, b, f: np.ones(len(b), bool); ref = C.blocked(P, cam, [prism])
    C._wedge = orig; a = C.blocked(P, cam, [prism])
    C._wedge = wedge_fixed; b = C.blocked(P, cam, [prism])
    C._wedge = orig
    bad_orig += int((a != ref).sum()); bad_fixed += int((b != ref).sum()); tot += int(ref.sum())
print("blocked total", tot, "cells wrong with current prefilter", bad_orig, "with the clause dropped", bad_fixed)
