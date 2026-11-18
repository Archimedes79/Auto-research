"""read-only CPU diagnostic for the E263 card: do the four j1rc seeds miss the same far ground-truth points (shared misses = no room for a weight average) or different ones?"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
import evaluate as EV
reg = EV.registry()

def dense(pl, step=0.25):
    out = []
    for p in pl:
        p = np.asarray(p, float)
        if len(p) == 1:
            out.append(p); continue
        for a, b in zip(p[:-1], p[1:]):
            n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
            out.append(a + (b - a) * np.linspace(0, 1, n + 1)[:, None])
    return np.concatenate(out) if out else np.zeros((0, 2))

names = sys.argv[1:]
G = []
for n in names:
    g, gts, _ = EV.graphs_of(n, reg[n], "val", require_cache=True)
    G.append(g)
for lo, hi, lab in ((0, 20, "near"), (20, 40, "mid"), (40, 1e9, "far")):
    hist = np.zeros(len(names) + 1, int)
    for k, gt in enumerate(gts):
        gp = dense(gt["pts"])
        if len(gp) == 0:
            continue
        rr = np.linalg.norm(gp, axis=1)
        fp = gp[(rr >= lo) & (rr < hi)]
        if len(fp) == 0:
            continue
        c = np.zeros(len(fp), int)
        for g in G:
            q = dense(g[k]["pts"])
            if len(q):
                c += (cKDTree(q).query(fp)[0] <= 1.0)
        hist += np.bincount(c, minlength=len(names) + 1)
    f = hist / hist.sum()
    single = sum(i * f[i] for i in range(len(f))) / len(names)
    print(f"{lab}: share of GT points hit by 0..{len(names)} seeds {np.round(f, 3).tolist()}; mean single recall {single:.3f}, union {1 - f[0]:.3f}, majority (>= {len(names)//2 + 1}) {f[len(names)//2 + 1:].sum():.3f}, disagreement (1..{len(names)-1}) {f[1:len(names)].sum():.3f}")
