"""one-off: top-down idea - are lanes = road centre line + a lateral offset? Roads = the carriageways of sd_lib.build_sd (lanes side by side merged,
centre = mean). For every lane point: signed lateral offset to the nearest road point with the same direction; per lane piece: how constant is the offset
(std along the piece) and how close to a lane slot (n_lanes odd/even grid of 3.5 m). VAL, 1 frame in 8."""
import os, sys, pickle
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "training/data")
import sd_lib as SD
H = os.path.expanduser("~")
VAL = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42}
G = pickle.load(open(f"{H}/exp_gcar_b123/graph_gt_val.pkl", "rb"))
sc = np.load(f"{H}/exp_gcar_b123/poses_val.npz")["scene"]
stds, cover, nfr, ncount = [], [], 0, []
for k, g in enumerate(G):
    if int(sc[k]) not in VAL:
        continue
    nfr += 1
    if nfr % 8:
        continue
    sd = SD.build_sd(g, [])
    R, T = [], []
    for poly, n in sd["roads"]:
        p = np.asarray(poly, float)
        if len(p) < 2:
            continue
        t = np.gradient(p, axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        R.append(p); T.append(t); ncount.append(len(np.unique(np.round(n))))
    if not R:
        continue
    R, T = np.concatenate(R), np.concatenate(T)
    tree = cKDTree(R)
    for p, kd in zip(g["pts"], g["kind"]):
        p = np.asarray(p, float)
        if kd != "lane" or len(p) < 4:
            continue
        t = np.gradient(p, axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        d, j = tree.query(p, k=8)
        offs, ok = [], []
        for i in range(len(p)):
            best = None
            for dd, jj in zip(d[i], j[i]):
                if T[jj] @ t[i] > 0.9 and dd < 12:
                    v = p[i] - R[jj]
                    best = T[jj][0] * v[1] - T[jj][1] * v[0]   # signed lateral offset (left positive)
                    break
            ok.append(best is not None); offs.append(best if best is not None else np.nan)
        offs = np.array(offs)
        cover.append(np.mean(ok))
        if np.sum(ok) >= 4:
            stds.append(np.nanstd(offs))
s, c, nc = np.array(stds), np.array(cover), np.array(ncount)
print(f"lane pieces {len(c)}: points with a parallel road within 12 m {c.mean():.3f}")
print(f"offset to the road centre along a piece: std median {np.median(s):.2f} m, p90 {np.percentile(s, 90):.2f} m, share < 0.3 m {np.mean(s < 0.3):.3f}")
print(f"roads whose lane count changes along them: {np.mean(nc > 1):.3f} of {len(nc)}")
