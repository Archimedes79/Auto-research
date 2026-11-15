"""scratch: stop / exit events and in_intersection points against intersection cells for several dilations"""
import os, sys, time
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "training", "grid"))
import qraster as QR
from test_qtargets import frames
fr = frames(120)
print("frames", len(fr))
for dil in (0.0, 1.0, 2.0, 3.0, 4.0):
    own = within4 = nev = 0
    pin = ptot = 0
    cells = 0
    t = time.perf_counter()
    for g in fr:
        r2 = QR.areas2(g, dilate=dil)[1]
        cells += r2.sum()
        ii, jj = np.nonzero(r2)
        cen = np.stack([-50 + 2 * (ii + 0.5), -50 + 2 * (jj + 0.5)], 1)
        for e in g["events"]:
            if e["type"] in ("stop", "exit"):
                pos = np.asarray(e["pos"], float)
                c = np.clip(np.floor((pos + 50) / 2).astype(int), 0, 49)
                nev += 1
                own += r2[c[0], c[1]] > 0
                within4 += len(cen) > 0 and np.linalg.norm(cen - pos, axis=1).min() <= 4
        for p in g["pieces"]:
            q = np.asarray(p["pts"], float)
            for s in p["segments"]:
                if s["in_intersection"]:
                    pts = q[s["i0"]: s["i1"] + 1]
                    c = np.clip(np.floor((pts + 50) / 2).astype(int), 0, 49)
                    ptot += len(pts); pin += int(r2[c[:, 0], c[:, 1]].sum())
    print(f"dilate {dil}: stop/exit in an intersection cell {own / nev:.3f}, within 4 m {within4 / nev:.3f}; in_intersection points in cells {pin / ptot:.3f}; intersection cells/frame {cells / len(fr):.0f}; areas2 {1e3 * (time.perf_counter() - t) / len(fr):.1f} ms/frame (incl. this loop)")
