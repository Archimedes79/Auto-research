"""one-off: how well does a cubic Bezier from the two port poses (P1 = P0 + d/3 t0, P2 = P3 - d/3 t1) reproduce the GT connectors? VAL scenes only"""
import pickle, os
import numpy as np
from scipy.spatial import cKDTree
H = os.path.expanduser("~")
VAL = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42}
G = pickle.load(open(f"{H}/exp_gcar_b123/graph_gt_val.pkl", "rb"))
sc = np.load(f"{H}/exp_gcar_b123/poses_val.npz")["scene"]
mean_d, max_d, turn = [], [], []
for k, g in enumerate(G):
    if int(sc[k]) not in VAL:
        continue
    for i, kd in enumerate(g["kind"]):
        p = np.asarray(g["pts"][i], float)
        if kd != "connector" or len(p) < 4 or g["start_open"][i] or g["end_open"][i]:
            continue
        t0 = p[1] - p[0]; t0 /= max(np.linalg.norm(t0), 1e-9)
        t1 = p[-1] - p[-2]; t1 /= max(np.linalg.norm(t1), 1e-9)
        d = np.linalg.norm(p[-1] - p[0])
        if d < 1:
            continue
        P = [p[0], p[0] + d / 3 * t0, p[-1] - d / 3 * t1, p[-1]]
        s = np.linspace(0, 1, 400)[:, None]
        B = (1 - s) ** 3 * P[0] + 3 * (1 - s) ** 2 * s * P[1] + 3 * (1 - s) * s ** 2 * P[2] + s ** 3 * P[3]
        dist = cKDTree(B).query(p)[0]
        mean_d.append(dist.mean()); max_d.append(dist.max())
        turn.append(np.degrees(np.arccos(np.clip(t0 @ t1, -1, 1))))
m, x, tu = map(np.array, (mean_d, max_d, turn))
print(f"closed connectors {len(m)}: mean distance GT -> Bezier median {np.median(m):.2f} m, p90 {np.percentile(m, 90):.2f} m; max distance median {np.median(x):.2f} m, p90 {np.percentile(x, 90):.2f} m")
for lo, hi in ((0, 15), (15, 60), (60, 120), (120, 181)):
    s = (tu >= lo) & (tu < hi)
    if s.any():
        print(f"  turn {lo}-{hi} deg: n {s.sum()}, mean distance median {np.median(m[s]):.2f} m, max distance p90 {np.percentile(x[s], 90):.2f} m")

# upper bound of "2 learned control-point offsets": P0, P3 fixed at the ports, P1 = P0 + a t0, P2 = P3 - b t1 with the two handle lengths a, b fitted (grid search)
fm, fx, ft = [], [], []
grid = np.linspace(0.1, 0.9, 17)
for k, g in enumerate(G):
    if int(sc[k]) not in VAL:
        continue
    for i, kd in enumerate(g["kind"]):
        p = np.asarray(g["pts"][i], float)
        if kd != "connector" or len(p) < 4 or g["start_open"][i] or g["end_open"][i]:
            continue
        t0 = p[1] - p[0]; t0 /= max(np.linalg.norm(t0), 1e-9)
        t1 = p[-1] - p[-2]; t1 /= max(np.linalg.norm(t1), 1e-9)
        d = np.linalg.norm(p[-1] - p[0])
        if d < 1:
            continue
        tr = cKDTree(p)
        s = np.linspace(0, 1, 200)[:, None]
        best = (1e9, 0)
        for a in grid:
            for b in grid:
                P1, P2 = p[0] + a * d * t0, p[-1] - b * d * t1
                B = (1 - s) ** 3 * p[0] + 3 * (1 - s) ** 2 * s * P1 + 3 * (1 - s) * s ** 2 * P2 + s ** 3 * p[-1]
                dist = cKDTree(B).query(p)[0]
                if dist.mean() < best[0]:
                    best = (dist.mean(), dist.max())
        fm.append(best[0]); fx.append(best[1]); ft.append(np.degrees(np.arccos(np.clip(t0 @ t1, -1, 1))))
fm, fx, ft = map(np.array, (fm, fx, ft))
print(f"fitted handles: mean distance median {np.median(fm):.2f} m, p90 {np.percentile(fm, 90):.2f} m; max distance p90 {np.percentile(fx, 90):.2f} m")
for lo, hi in ((0, 15), (15, 60), (60, 120), (120, 181)):
    s = (ft >= lo) & (ft < hi)
    if s.any():
        print(f"  turn {lo}-{hi} deg: mean distance median {np.median(fm[s]):.2f} m, max distance p90 {np.percentile(fx[s], 90):.2f} m")
