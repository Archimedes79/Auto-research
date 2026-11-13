"""one-off: intersections in the VAL GT lane graph - connector lengths, midpoint anchors that collide, connectors whose ends lie far apart (VAL scenes only)"""
import pickle, os
import numpy as np
H = os.path.expanduser("~")
VAL = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42}
G = pickle.load(open(f"{H}/exp_gcar_b123/graph_gt_val.pkl", "rb"))
sc = np.load(f"{H}/exp_gcar_b123/poses_val.npz")["scene"]


def mid(p):
    p = np.asarray(p, float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    i = min(np.searchsorted(s, L / 2), len(p) - 1)
    d = p[min(i, len(p) - 1)] - p[max(i - 1, 0)]
    return p[i], d / max(np.linalg.norm(d), 1e-9), L


lens = {"lane": [], "connector": []}
coll = {"lane": [0, 0, 0], "connector": [0, 0, 0]}        # pieces, other midpoint within 2 m, within 2 m AND heading within 30 deg
endd = []
nfr = nconn = 0
for k, g in enumerate(G):
    if int(sc[k]) not in VAL:
        continue
    nfr += 1
    M, D, K = [], [], []
    for p, kd in zip(g["pts"], g["kind"]):
        if len(p) < 2:
            continue
        m, d, L = mid(p)
        if not (np.abs(m) < 48).all():
            continue
        M.append(m), D.append(d), K.append(kd)
        lens[kd].append(L)
        if kd == "connector":
            endd.append(np.linalg.norm(np.asarray(p[-1]) - np.asarray(p[0])))
    if len(M) < 2:
        continue
    M, D = np.array(M), np.array(D)
    dd = np.linalg.norm(M[:, None] - M[None], axis=2) + np.eye(len(M)) * 1e9
    cs = D @ D.T
    for i, kd in enumerate(K):
        coll[kd][0] += 1
        coll[kd][1] += int((dd[i] < 2).any())
        coll[kd][2] += int(((dd[i] < 2) & (cs[i] > np.cos(np.deg2rad(30)))).any())
print("VAL frames", nfr)
for kd in lens:
    a = np.array(lens[kd])
    n, c2, c2h = coll[kd]
    print(f"{kd}: pieces {n}, length median {np.median(a):.1f} m, p90 {np.percentile(a, 90):.1f} m; midpoint of another piece within 2 m: {c2 / n:.3f}, and same heading (30 deg): {c2h / n:.3f}")
e = np.array(endd)
print(f"connector start-to-end distance: median {np.median(e):.1f} m, p90 {np.percentile(e, 90):.1f} m, > 20 m {np.mean(e > 20):.3f}")
