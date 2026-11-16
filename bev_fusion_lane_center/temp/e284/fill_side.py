"""scratch (continuation of mechanism.py, Mirko's question of 2026-11-08): the fill of the cells the target leaves empty (T < 0.1, within 6 m of a lane point) inside the junction (< 8 m from the junction points) is systematic (both seeds
agree). WHERE does it sit? Per frame (with >= 10 filled cells, p > 0.3): the p-weighted mean position of the fill cells minus the mean position of the junction body (cells with dj < 8 and T >= 0.1), in the ego frame
(x right, y forward): dx < 0 = the fill lies LEFT of the junction body, dy < 0 = nearer to the ego (before the junction). Median over the frames and the share of frames with dx < 0 / dy < 0, for VAL seed 0, VAL seed 1, TRAIN seed 0.
Reads temp/e284/target_vs_pred.npz and mechanism.npz.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh temp/e284/fill_side.py"""
import json
import os

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
z = np.load(os.path.join(ROOT, "temp", "e284", "target_vs_pred.npz"))
m = np.load(os.path.join(ROOT, "temp", "e284", "mechanism.npz"))
c = -50.0 + (np.arange(100) + 0.5)
CX, CY = np.meshgrid(c, c, indexing="ij")
X, Y = CX.reshape(-1), CY.reshape(-1)
inbox = (np.abs(X) <= 47.0) & (np.abs(Y) <= 47.0)
SETS = {"VAL seed 0": (z["p"][:88], z["t"][:88], z["dj"][:88], z["dg"][:88]), "VAL seed 1": (m["p1"], z["t"][:88], z["dj"][:88], z["dg"][:88]), "TRAIN seed 0": (m["pt"], m["tt"], m["djt"], m["dgt"])}
out = {}
for name, (P, T, DJ, DG) in SETS.items():
    P, T, DJ, DG = (np.asarray(a, np.float32) for a in (P, T, DJ, DG))
    dxs, dys, mass = [], [], []
    for i in range(len(P)):
        body = (DJ[i] < 8.0) & inbox & (T[i] >= 0.1)
        fill = (DJ[i] < 8.0) & inbox & (T[i] < 0.1) & (DG[i] < 6.0) & (P[i] > 0.3)
        if fill.sum() < 10 or body.sum() < 10:
            continue
        w = P[i][fill]
        dxs.append(float((w * X[fill]).sum() / w.sum() - X[body].mean()))
        dys.append(float((w * Y[fill]).sum() / w.sum() - Y[body].mean()))
        mass.append(float(w.sum()))
    dxs, dys = np.array(dxs), np.array(dys)
    out[name] = dict(frames_with_fill=len(dxs), dx_median=float(np.median(dxs)), dx_mean=float(dxs.mean()), share_left=float((dxs < 0).mean()), dy_median=float(np.median(dys)), dy_mean=float(dys.mean()), share_before=float((dys < 0).mean()))
    print(f"{name:13s} frames with fill {len(dxs):3d}  dx median {np.median(dxs):+5.2f} m (left of the junction body in {100 * (dxs < 0).mean():3.0f} % of the frames)  dy median {np.median(dys):+5.2f} m (nearer to the ego in {100 * (dys < 0).mean():3.0f} %)")
json.dump(out, open(os.path.join(ROOT, "temp", "e284", "fill_side.json"), "w"), indent=1)
