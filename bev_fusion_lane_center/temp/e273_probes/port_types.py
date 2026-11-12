"""one-off: what kinds of ports does the GT graph have (where pieces start / end)? pass-through = exactly one in, one out (invisible on the road). VAL scenes, 1 frame in 4."""
import pickle, os, collections
import numpy as np
from scipy.spatial import cKDTree
H = os.path.expanduser("~")
VAL = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42}
G = pickle.load(open(f"{H}/exp_gcar_b123/graph_gt_val.pkl", "rb"))
sc = np.load(f"{H}/exp_gcar_b123/poses_val.npz")["scene"]
c = collections.Counter()
grid = np.linspace(0.05, 0.95, 19)
s = np.linspace(0, 1, 300)[:, None]
fits = []
nf = 0
for k, g in enumerate(G):
    if int(sc[k]) not in VAL:
        continue
    nf += 1
    if nf % 4:
        continue
    pred, succ = collections.defaultdict(list), collections.defaultdict(list)
    for i, j in g["edges"]:
        succ[i].append(j); pred[j].append(i)
    n = len(g["pts"])
    for i in range(n):          # the END port of piece i (and the starts of its successors meet there)
        if g["end_open"][i]:
            c["window edge"] += 1
        elif not succ[i]:
            c["dead end"] += 1
        elif len(succ[i]) >= 2:
            c["split"] += 1
        elif len(pred[succ[i][0]]) >= 2:
            c["merge"] += 1
        else:
            c["pass-through (1 in, 1 out)"] += 1
    for i in range(n):          # start ports without a predecessor
        if not pred[i]:
            c["window edge (start)" if g["start_open"][i] else "lane start (no predecessor)"] += 1
    # chains through pass-through ports, fitted as ONE Bezier
    nxt = {i: succ[i][0] for i in range(n) if len(succ[i]) == 1 and len(pred[succ[i][0]]) == 1 and not g["end_open"][i]}
    heads = [i for i in range(n) if not any(nxt.get(j) == i for j in range(n))]
    for h in heads:
        ch, i = [h], h
        while i in nxt and nxt[i] not in ch:
            i = nxt[i]; ch.append(i)
        if len(ch) < 2:
            continue
        p = np.concatenate([np.asarray(g["pts"][j], float) for j in ch])
        if len(p) < 4:
            continue
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
        fits.append((len(ch), float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()), best))
tot = sum(c.values())
for kd, v in c.most_common():
    print(f"{kd}: {v} ({v / tot:.3f})")
f = np.array(fits)
print(f"chains through pass-through ports: {len(f)}, median {np.median(f[:, 0]):.0f} pieces / {np.median(f[:, 1]):.0f} m; one Bezier: max distance median {np.median(f[:, 2]):.2f} m, p90 {np.percentile(f[:, 2], 90):.2f} m, within 0.5 m {np.mean(f[:, 2] < 0.5):.3f}")
