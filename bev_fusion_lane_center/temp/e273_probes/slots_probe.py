"""one-off: if every coarse cell (query) is pulled to the lines, how many line points does one cell have to hold? Lines in a cell count as different
when their closest points to the cell centre are > 1.5 m apart or their headings differ > 30 deg. VAL, 1 frame in 4."""
import pickle, os, sys
import numpy as np
H = os.path.expanduser("~")
VAL = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42}
G = pickle.load(open(f"{H}/exp_gcar_b123/graph_gt_val.pkl", "rb"))
sc = np.load(f"{H}/exp_gcar_b123/poses_val.npz")["scene"]
for res in (2.0, 3.0, 4.0):
    hist = np.zeros(6, int)
    nf = 0
    for k, g in enumerate(G):
        if int(sc[k]) not in VAL:
            continue
        nf += 1
        if nf % 4:
            continue
        cells = {}
        for pid, p in enumerate(g["pts"]):
            p = np.asarray(p, float)
            if len(p) < 2:
                continue
            seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
            s = np.concatenate([[0], np.cumsum(seg)])
            t = np.arange(0, s[-1], 0.25)
            q = np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)
            d = np.gradient(q, axis=0); d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            m = (np.abs(q) < 48).all(1)
            for x, h in zip(q[m], d[m]):
                key = (int(np.floor(x[0] / res)), int(np.floor(x[1] / res)))
                cells.setdefault(key, {}).setdefault(pid, []).append((x, h))
        for key, lst in cells.items():
            c = (np.array(key) + 0.5) * res
            reps = []
            near = [min(pl, key=lambda e: np.linalg.norm(e[0] - c)) for pl in lst.values()]   # closest point of each piece to the cell centre
            for x, h in sorted(near, key=lambda e: np.linalg.norm(e[0] - c)):
                if all(np.linalg.norm(x - rx) > 1.5 or h @ rh < np.cos(np.deg2rad(30)) for rx, rh in reps):
                    reps.append((x, h))
            hist[min(len(reps), 5)] += 1
    tot = hist[1:].sum()
    print(f"cell {res:.0f} m: lane cells {tot}; distinct lines per cell 1: {hist[1] / tot:.3f}  2: {hist[2] / tot:.3f}  3: {hist[3] / tot:.3f}  4+: {hist[4:].sum() / tot:.3f}")
