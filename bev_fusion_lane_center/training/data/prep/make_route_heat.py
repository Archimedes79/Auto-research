"""Schleife 5 / E60: Soll-Waermekarte der Route (Gauss sigma 0,75 m um die Mittellinien der Routenstuecke) je Sample -> ~/exp_gcar_b123/route_heat_<teil>.npy (uint8 = 255 * Waerme) und route_pts_<teil>.pkl (dichte Soll-Punkte der Route, 1 m, Fahrzeugsystem).
Raster wie die uebrigen Soll-Karten: Zeile = x (nach rechts), Spalte = y (nach vorn), 0,5 m, +-50 m.  Aufruf (gcar): make_route_heat.py"""
import os
import pickle
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc_lib as AL  # noqa: E402

HOME = os.path.expanduser("~")
D = f"{HOME}/exp_gcar_b123"
SIGMA = 0.75


def rasterize(polys, n=200, half=50.0, ds=0.25):
    m = np.zeros((n, n), bool)
    for p in polys:
        q = AL.resample(np.asarray(p, np.float64), ds)
        r = np.round((q[:, 0] + half) / (2 * half / n) - 0.5).astype(int)
        c = np.round((q[:, 1] + half) / (2 * half / n) - 0.5).astype(int)
        ok = (r >= 0) & (r < n) & (c >= 0) & (c < n)
        m[r[ok], c[ok]] = True
    return m


def main():
    for part in ("train", "val"):
        G = pickle.load(open(f"{D}/graph_gt_{part}.pkl", "rb"))
        RT = pickle.load(open(f"{D}/route_gt_{part}.pkl", "rb"))
        N = len(G)
        out = np.lib.format.open_memmap(f"{D}/route_heat_{part}.npy", mode="w+", dtype=np.uint8, shape=(N, 200, 200))
        pts = []
        for i in range(N):
            rt = RT[i]
            if rt is None:
                out[i] = 0
                pts.append(None)
                continue
            polys = [G[i]["pts"][p] for p in rt["route"] if len(G[i]["pts"][p]) >= 2]
            m = rasterize(polys)
            d = distance_transform_edt(~m) * 0.5
            out[i] = np.round(255.0 * np.exp(-(d ** 2) / (2 * SIGMA ** 2))).astype(np.uint8)
            pts.append(np.concatenate([AL.resample(p, 1.0) for p in polys]).astype(np.float32))
            if i % 2000 == 0:
                print(part, i, N, flush=True)
        out.flush()
        pickle.dump(pts, open(f"{D}/route_pts_{part}.pkl", "wb"))
        print(part, "fertig", sum(p is not None for p in pts), "mit Route", flush=True)


if __name__ == "__main__":
    main()
