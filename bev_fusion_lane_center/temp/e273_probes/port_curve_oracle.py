"""one-off: can EVERY GT piece (lane pieces too, not only connectors) be drawn as ONE cubic Bezier between its two end poses with two fitted handle lengths?
Pieces are clipped at the window (+-50 m) as in the GT; VAL scenes only; 1 frame in 4 (enough for the distribution)."""
import pickle, os
import numpy as np
from scipy.spatial import cKDTree
H = os.path.expanduser("~")
VAL = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42}
G = pickle.load(open(f"{H}/exp_gcar_b123/graph_gt_val.pkl", "rb"))
sc = np.load(f"{H}/exp_gcar_b123/poses_val.npz")["scene"]
grid = np.linspace(0.05, 0.95, 19)
s = np.linspace(0, 1, 300)[:, None]
res = {"lane": [], "connector": []}
nf = 0
for k, g in enumerate(G):
    if int(sc[k]) not in VAL:
        continue
    nf += 1
    if nf % 4:
        continue
    for i, kd in enumerate(g["kind"]):
        p = np.asarray(g["pts"][i], float)
        if len(p) < 4:
            continue
        L = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        t0 = p[1] - p[0]; t0 /= max(np.linalg.norm(t0), 1e-9)
        t1 = p[-1] - p[-2]; t1 /= max(np.linalg.norm(t1), 1e-9)
        d = np.linalg.norm(p[-1] - p[0])
        if d < 1:
            continue
        best = 1e9
        for a in grid:
            for b in grid:
                P1, P2 = p[0] + a * d * t0, p[-1] - b * d * t1
                B = (1 - s) ** 3 * p[0] + 3 * (1 - s) ** 2 * s * P1 + 3 * (1 - s) * s ** 2 * P2 + s ** 3 * p[-1]
                best = min(best, cKDTree(B).query(p)[0].max())
        res[kd].append((L, best))
print("VAL frames used", nf // 4)
for kd, r in res.items():
    r = np.array(r)
    print(f"{kd}: n {len(r)}, max distance GT -> Bezier: median {np.median(r[:, 1]):.2f} m, p90 {np.percentile(r[:, 1], 90):.2f} m, share within 0.5 m {np.mean(r[:, 1] < 0.5):.3f}")
    for lo, hi in ((0, 20), (20, 50), (50, 150)):
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if m.any():
            print(f"   length {lo}-{hi} m: n {m.sum()}, p90 {np.percentile(r[m, 1], 90):.2f} m, within 0.5 m {np.mean(r[m, 1] < 0.5):.3f}")
