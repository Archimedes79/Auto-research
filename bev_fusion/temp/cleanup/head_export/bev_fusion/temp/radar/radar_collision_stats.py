"""One-off: how much does a raster of radar detections lose when detections share a cell? (real nuScenes mini, cached multi-sweep)

For windows of 0.1 s (about one sweep per radar) and 0.5 s (about six): detections inside the front grid (x 0..60 m, |y| < 26 m,
0.5 m cells), the share of detections that sit in a cell with another detection (the per-cell mean merges them), and the share
a K-slot cell keeps (the first K by age: newest first) for K = 1, 2, 3, 4, 6 - by range of the cell.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/radar/radar_collision_stats.py
"""
import numpy as np
import torch

frames = torch.load("temp/radar/cache/mini_multisweep.pt")
NX, NY, RES = 120, 104, 0.5
BINS = [("0-20 m", 0, 20), ("20-40 m", 20, 40), ("40-60 m", 40, 61)]
for window in (0.1, 0.5):
    tot = np.zeros(len(BINS))
    shared = np.zeros(len(BINS))
    kept = {K: np.zeros(len(BINS)) for K in (1, 2, 3, 4, 6)}
    n_in, max_cell = [], 0
    for f in frames:
        m = (f["dt"] <= window) & (f["x"] >= 0) & (f["x"] < 60) & (f["y"] >= -26) & (f["y"] < 26)
        x, y, dt = f["x"][m], f["y"][m], f["dt"][m]
        cell = (x / RES).floor().long() * NY + ((y + 26) / RES).floor().long()
        order = torch.argsort(cell * 10 + (dt - dt.min()).clamp(max=0.99) / 1.0, stable=True)   # by cell, newest (small dt) first
        cell_s, rng = cell[order], torch.hypot(x, y)[order]
        _, inv, cnt = torch.unique_consecutive(cell_s, return_inverse=True, return_counts=True)
        start = torch.cumsum(cnt, 0) - cnt
        rank = torch.arange(len(cell_s)) - start[inv]
        n_in.append(len(x))
        max_cell = max(max_cell, int(cnt.max()))
        for i, (_, lo, hi) in enumerate(BINS):
            sel = (rng >= lo) & (rng < hi)
            tot[i] += int(sel.sum())
            shared[i] += int((sel & (cnt[inv] > 1)).sum())
            for K in kept:
                kept[K][i] += int((sel & (rank < K)).sum())
    print(f"window {window} s: {np.median(n_in):.0f} detections per frame inside the front grid (median), most in one cell: {max_cell}")
    print("  share of detections in a cell with another detection (merged by a per-cell mean): " +
          ", ".join(f"{b[0]} {100 * s / t:.1f} %" for b, s, t in zip(BINS, shared, tot)) + f"; all {100 * shared.sum() / tot.sum():.1f} %")
    for K in kept:
        print(f"  K = {K} slots per cell keep: " + ", ".join(f"{b[0]} {100 * k / t:.1f} %" for b, k, t in zip(BINS, kept[K], tot)) +
              f"; all {100 * kept[K].sum() / tot.sum():.1f} %")
