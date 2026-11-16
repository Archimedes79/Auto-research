"""one-off: how often do GT lanes cross or overlap inside one cell (a grid head with ONE heading per cell cannot represent that)? VAL scenes only."""
import pickle, os
import numpy as np
H = os.path.expanduser("~")
VAL = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42}
G = pickle.load(open(f"{H}/exp_gcar_b123/graph_gt_val.pkl", "rb"))
sc = np.load(f"{H}/exp_gcar_b123/poses_val.npz")["scene"]


def dense(p, ds=0.25):
    p = np.asarray(p, float)
    if len(p) < 2:
        return p, np.zeros_like(p)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s[-1], ds)
    q = np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)
    d = np.gradient(q, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    return q, d


tot = {r: [0, 0, 0] for r in (0.5, 1.0, 2.0)}   # cells with lane, cells with >= 2 pieces whose headings differ > 30 deg, cells with >= 2 pieces (any)
nf = 0
for k, g in enumerate(G):
    if int(sc[k]) not in VAL:
        continue
    nf += 1
    for res in tot:
        cells = {}
        for pi, p in enumerate(g["pts"]):
            q, d = dense(p)
            m = (np.abs(q) < 48).all(1)
            for (x, y), h in zip(q[m], d[m]):
                key = (int(np.floor(x / res)), int(np.floor(y / res)))
                cells.setdefault(key, {}).setdefault(pi, h)
        for c in cells.values():
            tot[res][0] += 1
            if len(c) >= 2:
                tot[res][2] += 1
                hs = np.array(list(c.values()))
                if (hs @ hs.T).min() < np.cos(np.deg2rad(30)):
                    tot[res][1] += 1
print("VAL frames", nf)
for r, (n, x, a) in tot.items():
    print(f"res {r} m: lane cells {n}, >=2 pieces {a / n:.3f}, crossing (>30 deg) {x / n:.3f}")
