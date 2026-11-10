"""One-off: what does the per-cell MEAN of the simple radar raster lose? (real nuScenes mini, cached multi-sweep, window 0.5 s)

For every detection that shares its 0.5 m cell with another one: the error of reconstructing it from the cell mean - position
(m), distance from the sensor (m), ray direction (deg), radial velocity (m/s), RCS (dB), age (s) - median and 90 % quantile; then
the same for a K = 3 slot code with a remainder (the first 3 detections of a cell, newest first, are kept exactly, the rest are
represented by their mean).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/radar/radar_cell_dispersion.py
"""
import math

import numpy as np
import torch

frames = torch.load("temp/radar/cache/mini_multisweep.pt")
NY, RES = 104, 0.5
PROPS = ["pos (m)", "dist (m)", "dir (deg)", "vr (m/s)", "rcs (dB)", "age (s)"]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def errors(f, K):
    m = (f["dt"] <= 0.5) & (f["x"] >= 0) & (f["x"] < 60) & (f["y"] >= -26) & (f["y"] < 26)
    g = {k: f[k][m] for k in ["x", "y", "r", "psi", "vr", "rcs", "dt"]}
    cell = (g["x"] / RES).floor().long() * NY + ((g["y"] + 26) / RES).floor().long()
    order = torch.argsort(cell * 10 + g["dt"].clamp(0, 0.99), stable=True)
    g = {k: v[order] for k, v in g.items()}
    cell = cell[order]
    _, inv, cnt = torch.unique_consecutive(cell, return_inverse=True, return_counts=True)
    rank = torch.arange(len(cell)) - (torch.cumsum(cnt, 0) - cnt)[inv]
    rem = (rank >= K) if K else torch.ones_like(rank, dtype=torch.bool)       # detections represented by the cell mean (or remainder mean)
    w = rem.float()
    n_rem = torch.zeros(len(cnt)).index_add_(0, inv, w)
    def mean(v):
        return torch.zeros(len(cnt)).index_add_(0, inv, v * w) / n_rem.clamp(min=1)
    # circular mean of the direction
    cm, sm = mean(torch.cos(g["psi"])), mean(torch.sin(g["psi"]))
    psi_bar = torch.atan2(sm, cm)
    e = {"pos (m)": torch.hypot(g["x"] - mean(g["x"])[inv], g["y"] - mean(g["y"])[inv]),
         "dist (m)": (g["r"] - mean(g["r"])[inv]).abs(), "dir (deg)": torch.rad2deg(wrap(g["psi"] - psi_bar[inv])).abs(),
         "vr (m/s)": (g["vr"] - mean(g["vr"])[inv]).abs(), "rcs (dB)": (g["rcs"] - mean(g["rcs"])[inv]).abs(),
         "age (s)": (g["dt"] - mean(g["dt"])[inv]).abs()}
    sel = rem & (cnt[inv] > 1 if K == 0 else n_rem[inv] > 1)
    return {k: v[sel] for k, v in e.items()}, int(sel.sum()), len(cell)


for K, label in ((0, "per-cell mean (the simple raster)"), (3, "3 exact slots + remainder mean")):
    acc = {k: [] for k in PROPS}
    n_sel = n_all = 0
    for f in frames:
        e, ns, na = errors(f, K)
        for k in PROPS:
            acc[k].append(e[k])
        n_sel += ns
        n_all += na
    print(f"{label}: {n_sel} of {n_all} detections ({100 * n_sel / n_all:.1f} %) are represented by a mean shared with others; their error:")
    for k in PROPS:
        v = torch.cat(acc[k]).numpy()
        print(f"   {k:10s} median {np.median(v):7.3f}   90 % {np.quantile(v, 0.9):7.3f}   max {v.max():7.3f}")
