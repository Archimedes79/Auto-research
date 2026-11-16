"""where do E273's 18.4 % / 3.7 % (2 m) and 12.1 % / 1.9 % (1 m) come from: the same cell counting on the OLD graph GT pieces (resampled every 0.5 m) and on the lane GT, per grid, 31 and 218 frames"""
import numpy as np
import graph_eval as GE
import qgraph_eval as QE
import qtargets as QT
import e346_lib as L


def shares(polys, res):
    """(raw share, crossing share) of the occupied cells for a list of (n, 2) polylines"""
    with L.cell_size(res):
        cells = {}
        for i, q in enumerate(polys):
            q = np.asarray(q, float).reshape(-1, 2)
            if len(q) < 2:
                continue
            t = QT._tangents(q)
            c = QT.cell_of(q)
            key = c[:, 0] * QT.N + c[:, 1]
            for k in np.unique(key):
                j = np.flatnonzero(key == k)
                cells.setdefault(int(k), []).append((i, q[j], t[j]))
    raw = cross = 0
    for mem in cells.values():
        if len(mem) < 2:
            continue
        raw += 1
        c = False
        for x in range(len(mem)):
            for y in range(x + 1, len(mem)):
                d = np.linalg.norm(mem[x][1][:, None] - mem[y][1][None], axis=2)
                u, v = np.unravel_index(d.argmin(), d.shape)
                c |= np.degrees(np.arccos(np.clip(float(mem[x][2][u] @ mem[y][2][v]), -1, 1))) > 30
        cross += c
    return raw, cross, len(cells)


for step in (29, 4):
    fr = QE.val_frames()[::step]
    for src in ("old", "lane"):
        for res in (2.0, 1.0):
            tot = np.zeros(3)
            for f in fr:
                polys = [GE.resample(np.asarray(p, float), 0.5) for p in f.gt["pts"]] if src == "old" else [np.asarray(p["pts"]) for p in f.lane()["pieces"]]
                tot += shares(polys, res)
                f._lane = None
            print(f"{len(fr)} frames, {src} GT, {res} m: raw {tot[0] / tot[2]:.4f}  crossing {tot[1] / tot[2]:.4f}", flush=True)
