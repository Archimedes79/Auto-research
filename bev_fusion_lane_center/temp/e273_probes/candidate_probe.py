"""one-off: candidate generation for the port graph on GT ports. A piece runs from its start port to its end port; candidates of a start port = end ports
in a forward cone. Measures how many candidates and whether the true end is among them (recall), also for top-k by the straight-curve fit. VAL, 1 frame in 4."""
import pickle, os
import numpy as np
H = os.path.expanduser("~")
VAL = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42}
G = pickle.load(open(f"{H}/exp_gcar_b123/graph_gt_val.pkl", "rb"))
sc = np.load(f"{H}/exp_gcar_b123/poses_val.npz")["scene"]
import sys
R, CONE, TURN = float(sys.argv[1]), np.cos(np.deg2rad(float(sys.argv[2]))), np.cos(np.deg2rad(150))
stats = {"pieces": 0, "in_cone": 0, "cand": 0, "frames": 0}
ranks = []
nf = 0
for k, g in enumerate(G):
    if int(sc[k]) not in VAL:
        continue
    nf += 1
    if nf % 4:
        continue
    S, Sh, E, Eh = [], [], [], []
    for p in g["pts"]:
        p = np.asarray(p, float)
        if len(p) < 2:
            continue
        S.append(p[0]); E.append(p[-1])
        a = p[1] - p[0]; b = p[-1] - p[-2]
        Sh.append(a / max(np.linalg.norm(a), 1e-9)); Eh.append(b / max(np.linalg.norm(b), 1e-9))
    S, Sh, E, Eh = map(np.array, (S, Sh, E, Eh))
    n = len(S)
    stats["frames"] += 1
    for i in range(n):
        v = E - S[i]
        d = np.linalg.norm(v, axis=1)
        u = v / np.maximum(d, 1e-9)[:, None]
        ok = (d <= R) & (d > 0.5) & ((u @ Sh[i]) > CONE) & ((Eh @ Sh[i]) > TURN)
        stats["pieces"] += 1
        stats["cand"] += int(ok.sum())
        if ok[i]:
            stats["in_cone"] += 1
            # cheap score: how far the end heading and the chord disagree with a smooth turn (lower = better), plus distance
            score = (1 - u @ Sh[i]) + (1 - (Eh @ Sh[i])) * 0.5 + d / R * 0.2
            order = np.argsort(np.where(ok, score, np.inf))
            ranks.append(int(np.where(order == i)[0][0]) + 1)
        else:
            ranks.append(10 ** 6)
r = np.array(ranks)
print(f"frames {stats['frames']}, pieces {stats['pieces']}, candidates per piece start {stats['cand'] / stats['pieces']:.1f}, per frame {stats['cand'] / stats['frames']:.0f}")
print(f"true end inside the cone (<= {R:.0f} m): {stats['in_cone'] / stats['pieces']:.3f}")
for kk in (1, 3, 5, 8, 12):
    print(f"  true end within the top {kk} by the geometric score: {np.mean(r <= kk):.3f}")
