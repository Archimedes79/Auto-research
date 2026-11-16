import numpy as np, collections
import qgraph_eval as QE, qtargets as QT, e346_lib as L
fr = QE.val_frames()[::29]
res = 2.0
stat = collections.Counter(); dists = []; angs = []
conn_hit = collections.Counter()
nconn = 0; nhit_cross = 0; nhit_par = 0; nhit_any = 0
with L.cell_size(res):
    for f in fr:
        g = f.lane()
        pts, tans, keys = [], [], []
        cells = {}
        kinds = L.point_kinds(g)
        for i, p in enumerate(g["pieces"]):
            q = np.asarray(p["pts"], float).reshape(-1, 2); pts.append(q); tans.append(QT._tangents(q))
            c = QT.cell_of(q); key = c[:, 0] * QT.N + c[:, 1]; keys.append(key)
            for k in np.unique(key): cells.setdefault(int(k), []).append((i, np.flatnonzero(key == k)))
        affx = [set() for _ in pts]; affp = [set() for _ in pts]
        for key, mem in cells.items():
            for x in range(len(mem)):
                for y in range(x + 1, len(mem)):
                    (ia, ja), (ib, jb) = mem[x], mem[y]
                    d = np.linalg.norm(pts[ia][ja][:, None] - pts[ib][jb][None], axis=2)
                    u, v = np.unravel_index(d.argmin(), d.shape)
                    ang = np.degrees(np.arccos(np.clip(float(tans[ia][ja[u]] @ tans[ib][jb[v]]), -1, 1)))
                    ka, kb = kinds[ia][ja[u]], kinds[ib][jb[v]]
                    if ang > 30:
                        stat["cross", ka, kb] += 1; affx[ia].add(key); affx[ib].add(key)
                    elif d[u, v] > 0.3:
                        stat["par", ka, kb] += 1; dists.append(d[u, v]); angs.append(ang); affp[ia].add(key); affp[ib].add(key)
        for i, p in enumerate(g["pieces"]):
            for s in p["segments"]:
                if s["kind"] != "connector": continue
                ks = set(keys[i][s["i0"]: s["i1"] + 1].tolist())
                nconn += 1; hx = bool(affx[i] & ks); hp = bool(affp[i] & ks)
                nhit_cross += hx; nhit_par += (hp and not hx); nhit_any += (hx or hp)
        f._lane = None
print(stat)
print("connectors", nconn, "hit by crossing", nhit_cross, "hit only by parallel offset", nhit_par, "any", nhit_any)
dists = np.array(dists); print("parallel-offset collisions: n", len(dists), "dist quantiles", np.percentile(dists, [0, 25, 50, 75, 100]) if len(dists) else None, "ang", np.percentile(angs, [50, 100]) if len(angs) else None)
