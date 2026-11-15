import time, numpy as np
import qgraph_eval as QE, qtargets as QT, e346_lib as L
fr = QE.val_frames()
t = time.time()
info = []
for f in fr:
    g = f.lane()
    n = sum(len(p["pts"]) for p in g["pieces"])
    info.append((f.k, len(g["pieces"]), n, len(f.gt["pts"]), sum(len(p) for p in f.gt["pts"])))
    f._lane = None
print("scan", round(time.time() - t, 1), "s")
info = np.array(info)
print("empty lane GT:", info[info[:, 2] == 0][:, 0].tolist(), "empty old GT:", info[info[:, 3] == 0][:, 0].tolist())
print("lane points: min/median/max", info[:, 2].min(), np.median(info[:, 2]), info[:, 2].max(), " pieces max", info[:, 1].max())
edge = [int(k) for k in info[(info[:, 2] == 0) | (info[:, 3] == 0), 0]]
heavy = [int(k) for k in info[np.argsort(-info[:, 2])[:4], 0]]
few = [int(k) for k in info[np.argsort(info[:, 2])[:6], 0]]
print("edge", edge, "heavy", heavy, "light", few)
for arm in ("g0", "g1", "l2", "l1"):
    t = time.time()
    with L.cell_size(L.GRID.get(arm, 2.0)):
        recs = [L.frame_record(arm, fr[k]) for k in sorted(set(edge + heavy + few))]
    pl = L.pool(recs); m = L.numbers(pl)
    print(arm, len(recs), "frames ok", round(time.time() - t, 1), "s", {k: round(m[k], 4) for k in ("topo_F1", "F1@1.0", "lanes_found")})
# timing on a seeded random sample
rng = np.random.RandomState(7)
samp = sorted(rng.choice(len(fr), 40, replace=False).tolist())
for arm in ("l2", "l1"):
    t = time.time()
    with L.cell_size(L.GRID[arm]):
        recs = [L.frame_record(arm, fr[k]) for k in samp]
    dt = time.time() - t
    print(arm, "random 40 frames:", round(dt / 40, 3), "s/frame")
