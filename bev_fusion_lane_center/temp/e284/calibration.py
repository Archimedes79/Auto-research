"""scratch (continuation of fill_side.py, Mirko's question of 2026-11-08: how does the smear come about?). The loss is a binary cross-entropy against the SOFT target T, so the best answer of a head that cannot tell two cases apart is the
MEAN of T over the cases it cannot tell apart (p* = E[T | input]): a calibrated blur. Test: the reliability of p. For the cells of the junction (< 8 m) and of the rest, binned by p, the mean T of the cells in the bin. A calibrated hedge has
mean T = p in every bin; a head that is too sure (sharper than it can know) has mean T < p at high p and > p at low p. For VAL seed 0, VAL seed 1, TRAIN seed 0. Also the mean of T where p is in 0.2 - 0.8 (the grey cells).
Reads temp/e284/target_vs_pred.npz and mechanism.npz.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e284/calibration.py"""
import json
import os

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
z = np.load(os.path.join(ROOT, "temp", "e284", "target_vs_pred.npz"))
m = np.load(os.path.join(ROOT, "temp", "e284", "mechanism.npz"))
c = -50.0 + (np.arange(100) + 0.5)
CX, CY = np.meshgrid(c, c, indexing="ij")
inbox = ((np.abs(CX) <= 47.0) & (np.abs(CY) <= 47.0)).reshape(-1)
SETS = {"VAL seed 0": (z["p"][:88], z["t"][:88], z["dj"][:88]), "VAL seed 1": (m["p1"], z["t"][:88], z["dj"][:88]), "TRAIN seed 0": (m["pt"], m["tt"], m["djt"])}
EDGES = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 1.01]
out = {}
for name, (P, T, DJ) in SETS.items():
    P, T, DJ = (np.asarray(a, np.float32) for a in (P, T, DJ))
    out[name] = {}
    for zone, sel in (("junction (< 8 m)", lambda dj: dj < 8.0), ("outside (>= 8 m)", lambda dj: dj >= 8.0)):
        p = np.concatenate([P[i][sel(DJ[i]) & inbox] for i in range(len(P))])
        t = np.concatenate([T[i][sel(DJ[i]) & inbox] for i in range(len(P))])
        rows = []
        for a, b in zip(EDGES[:-1], EDGES[1:]):
            mm = (p >= a) & (p < b)
            if mm.sum() >= 50:
                rows.append((a, b, int(mm.sum()), float(p[mm].mean()), float(t[mm].mean())))
        grey = (p > 0.2) & (p < 0.8)
        out[name][zone] = dict(bins=rows, grey_cells=int(grey.sum()), grey_mean_p=float(p[grey].mean()), grey_mean_T=float(t[grey].mean()), grey_share_T_below_0_1=float((t[grey] < 0.1).mean()))
        print(f"{name:13s} {zone:17s} grey cells (0.2 < p < 0.8) {grey.sum():6d}: mean p {p[grey].mean():.2f}  mean T {t[grey].mean():.2f}  share with T < 0.1: {100 * (t[grey] < 0.1).mean():.0f} %")
        print("      " + "   ".join(f"p {r[3]:.2f} -> T {r[4]:.2f} (n {r[2]})" for r in rows))
json.dump(out, open(os.path.join(ROOT, "temp", "e284", "calibration.json"), "w"), indent=1)
