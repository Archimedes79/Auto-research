"""Post-check of E251.dist: the rebuilt b123 exact-distance map (~/exp_gcar_b123x/feats) against the stored one and against the 0.5 m-point bound.

    wsl bash tools/env/_py14.sh temp/loop19/check_b123x_dist.py

Gate: 8,414 + 1,706 rows; gt_pts_exact equal to the stored file within 1e-6 m (same points, same order; E251.dist found 2e-14 m of float noise in one val frame, row 825); on every frame the new map is not above the distance to the
stored 0.5 m points (an upper bound of the true distance) by more than 0.01 px. Reports how many frames / pixels of the stored map break that bound (TF32 damage).
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

H = os.path.expanduser("~")
OLD, NEW = f"{H}/exp_crn_b123/feats", f"{H}/exp_gcar_b123x/feats"
cen = -50.0 + (np.arange(200) + 0.5) * 0.5
gx, gy = np.meshgrid(cen, cen, indexing="ij")
pix = np.stack([gx.reshape(-1), gy.reshape(-1)], -1)
ok = True
for part, n_want in (("train", 8414), ("val", 1706)):
    d0 = np.load(f"{OLD}/{part}_gt_dist_exact.npy", mmap_mode="r")
    d1 = np.load(f"{NEW}/{part}_gt_dist_exact.npy", mmap_mode="r")
    p0 = np.load(f"{OLD}/{part}_gt_pts_exact.npy", allow_pickle=True)
    p1 = np.load(f"{NEW}/{part}_gt_pts_exact.npy", allow_pickle=True)
    same_pts = len(p0) == len(p1) == n_want and all(a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-6) for a, b in zip(p0, p1))
    over_new, bad_old, bad_px, diff_frames = 0.0, 0, 0, 0
    for i in range(len(d1)):
        b = np.minimum(cKDTree(np.asarray(p1[i], np.float64)).query(pix)[0] / 0.5, 8.0).astype(np.float16).astype(np.float32).reshape(200, 200)
        a, c = np.asarray(d0[i], np.float32), np.asarray(d1[i], np.float32)
        over_new = max(over_new, float((c - b).max()))
        o = a - b
        if o.max() > 0.01:
            bad_old += 1
            bad_px += int((o > 0.25).sum())
        diff_frames += float(np.abs(a - c).max()) > 0.05
    good = d1.shape[0] == n_want and same_pts and over_new <= 0.01
    ok &= good
    print(f"{part}: rows {d1.shape[0]} (want {n_want}); gt_pts_exact identical {same_pts}; new map over the bound by at most {over_new:.4f} px; "
          f"stored map above the bound in {bad_old} frames ({bad_px} pixels by > 0.25 px); stored vs new differ by > 0.05 px in {diff_frames} frames -> {'ok' if good else 'FAIL'}", flush=True)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
