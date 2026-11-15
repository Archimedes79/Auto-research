import os, pickle
import numpy as np
from scipy.spatial import cKDTree
ROOT = os.environ["BFLC_ROOT"]
PF = pickle.load(open(os.path.join(ROOT, "temp", "e309", "perframe_s0.pkl"), "rb"))
def rs(p, ds=0.5):
    p = np.asarray(p, float)
    if len(p) < 2: return p
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if s[-1] < 1e-6: return p[:1]
    t = np.arange(0, s[-1] + 1e-9, ds)
    return np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])]
bins = [(0, 10), (10, 20), (20, 35), (35, 48)]
for k in ("B", "e312"):
    cov = {b: [0, 0, 0] for b in bins}; fal = {b: [0, 0] for b in bins}; side = [0, 0, 0, 0]
    for r in PF:
        G = np.vstack([rs(p) for p in r["gt_pieces"] if len(p) > 1] or [np.zeros((0, 2))])
        P = np.vstack([rs(l) for l in r[k]["lines"] if len(l) > 1] or [np.zeros((0, 2))])
        inb = lambda A: A[(np.abs(A[:, 0]) < 48) & (np.abs(A[:, 1]) < 48)] if len(A) else A
        G, P = inb(G), inb(P)
        if len(G) == 0: continue
        dg = cKDTree(P).query(G)[0] if len(P) else np.full(len(G), 99.)
        rg = np.linalg.norm(G, axis=1)
        for b in bins:
            m = (rg >= b[0]) & (rg < b[1])
            cov[b][0] += (dg[m] <= 1).sum(); cov[b][1] += (dg[m] <= 2).sum(); cov[b][2] += m.sum()
        if len(P):
            dp = cKDTree(G).query(P)[0]; rp = np.linalg.norm(P, axis=1)
            for b in bins:
                m = (rp >= b[0]) & (rp < b[1]); fal[b][0] += (dp[m] > 2).sum(); fal[b][1] += m.sum()
    print(k, "GT covered within 1 m / 2 m by distance from the ego (pooled over 873 frames), delivered length > 2 m from GT")
    for b in bins:
        print(f"  {b[0]:2d}-{b[1]:2d} m: cov1 {cov[b][0]/cov[b][2]:.3f} cov2 {cov[b][1]/cov[b][2]:.3f}  GT {cov[b][2]*.5:7.0f} m | false {fal[b][0]/max(fal[b][1],1):.3f} of {fal[b][1]*.5:7.0f} m")
