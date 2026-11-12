"""one-off: node spacing vs. a cubic Hermite segment (positions + headings of two neighbouring nodes, handles = chord/3, NO learned parameter): max distance to the GT between the nodes. VAL, 1 frame in 8."""
import pickle, os
import numpy as np
from scipy.spatial import cKDTree
H = os.path.expanduser("~")
VAL = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42}
G = pickle.load(open(f"{H}/exp_gcar_b123/graph_gt_val.pkl", "rb"))
sc = np.load(f"{H}/exp_gcar_b123/poses_val.npz")["scene"]
s = np.linspace(0, 1, 40)[:, None]
for step in (4.0, 8.0, 16.0):
    errs = {"lane": [], "connector": []}
    nf = 0
    for k, g in enumerate(G):
        if int(sc[k]) not in VAL:
            continue
        nf += 1
        if nf % 8:
            continue
        for p, kd in zip(g["pts"], g["kind"]):
            p = np.asarray(p, float)
            if len(p) < 4:
                continue
            seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
            a = np.concatenate([[0], np.cumsum(seg)])
            if a[-1] < step:
                continue
            t = np.arange(0, a[-1] + 1e-6, step)
            if a[-1] - t[-1] > 1e-6:
                t = np.append(t, a[-1])
            q = np.stack([np.interp(t, a, p[:, 0]), np.interp(t, a, p[:, 1])], 1)
            d = np.gradient(p, axis=0); d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            h = np.stack([np.interp(t, a, d[:, 0]), np.interp(t, a, d[:, 1])], 1)
            h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-9)
            tr = cKDTree(p)
            for i in range(len(q) - 1):
                c = np.linalg.norm(q[i + 1] - q[i]) / 3
                P = [q[i], q[i] + c * h[i], q[i + 1] - c * h[i + 1], q[i + 1]]
                B = (1 - s) ** 3 * P[0] + 3 * (1 - s) ** 2 * s * P[1] + 3 * (1 - s) * s ** 2 * P[2] + s ** 3 * P[3]
                # distance of the Bezier samples to the GT polyline (both directions would need the GT part between the nodes; this one is enough for a bound)
                errs[kd].append(tr.query(B)[0].max())
    for kd, e in errs.items():
        e = np.array(e)
        print(f"node spacing {step:>4.0f} m, {kd}: segments {len(e)}, max distance p50 {np.median(e):.3f} m, p90 {np.percentile(e, 90):.3f} m, p99 {np.percentile(e, 99):.3f} m")
