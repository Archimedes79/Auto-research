"""Soll-Arrays aus dem Soll-Graphen (E31/E32): feste Punktzahl je Spur-Stueck, Nachbarschaftsmatrix, Richtungsraster.
Spuren laenger als CHUNK m werden in gleich lange Teilstuecke zerlegt (Aufloesung); Teilstuecke einer Spur sind in Fahrtrichtung verkettet.
Ausgabe je Teil (train/val) in ~/exp_gcar_<tag>/: gtK_<part>.npz (pts [N,M,K,2] m, valid [N,M], adj [N,M,M] bool, orig [N,M] Nr. des Original-Stuecks) und dir_<part>.npy (fp16 [N,2,200,200]: Fahrtrichtung (dx, dy) auf Spurmitten, 3 Zellen breit).
Aufruf: make_gt_arrays.py <b123|mini>"""
import os
import pickle
import sys

import cv2
import numpy as np

HOME = os.path.expanduser("~")
K, MMAX, CHUNK = 11, 96, 20.0


def resample(p, n):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    q = np.linspace(0, s[-1], n)
    return np.stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])], 1)


def chunks(g):
    """Stuecke -> Teilstuecke (<= CHUNK m) mit Verkettung. Rueckgabe: Liste Polylinien, Kanten, Original-Nr."""
    out, edges, orig = [], [], []
    first, last = {}, {}
    for i, p in enumerate(g["pts"]):
        p = np.asarray(p, np.float64)
        L = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        n = max(int(np.ceil(L / CHUNK)), 1)
        if n == 1:
            parts = [p]
        else:
            q = resample(p, 2 + int(L / 0.5))  # dichte Abtastung, dann gleich lange Teile
            idx = np.linspace(0, len(q) - 1, n + 1).round().astype(int)
            parts = [q[idx[k] : idx[k + 1] + 1] for k in range(n)]
        ids = []
        for part in parts:
            out.append(part)
            orig.append(i)
            ids.append(len(out) - 1)
        edges += [(ids[k], ids[k + 1]) for k in range(len(ids) - 1)]
        first[i], last[i] = ids[0], ids[-1]
    for i, j in g["edges"]:
        edges.append((last[i], first[j]))
    return out, edges, orig


def main(tag):
    d = f"{HOME}/exp_gcar_{tag}"
    for part in ("train", "val"):
        G = pickle.load(open(f"{d}/graph_gt_{part}.pkl", "rb"))
        N = len(G)
        pts = np.zeros((N, MMAX, K, 2), np.float32)
        valid = np.zeros((N, MMAX), bool)
        adj = np.zeros((N, MMAX, MMAX), bool)
        orig = -np.ones((N, MMAX), np.int16)
        dirr = np.zeros((N, 2, 200, 200), np.float16)
        dropped, mx = 0, 0
        for n, g in enumerate(G):
            polys, edges, org = chunks(g)
            mx = max(mx, len(polys))
            dropped += max(len(polys) - MMAX, 0)
            for m, p in enumerate(polys[:MMAX]):
                pts[n, m] = resample(p, K)
                valid[n, m] = True
                orig[n, m] = org[m]
            for a, b in edges:
                if a < MMAX and b < MMAX:
                    adj[n, a, b] = True
            dx, dy = np.zeros((200, 200), np.float32), np.zeros((200, 200), np.float32)
            for p in polys:
                q = resample(p, max(int(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() / 1.0), 2))
                for a, b in zip(q[:-1], q[1:]):
                    v = b - a
                    nv = np.linalg.norm(v)
                    if nv < 1e-6:
                        continue
                    v = v / nv
                    pa = (int(round((a[1] + 50) / 0.5 - 0.5)), int(round((a[0] + 50) / 0.5 - 0.5)))  # (Spalte = y, Zeile = x)
                    pb = (int(round((b[1] + 50) / 0.5 - 0.5)), int(round((b[0] + 50) / 0.5 - 0.5)))
                    cv2.line(dx, pa, pb, float(v[0]), 3)
                    cv2.line(dy, pa, pb, float(v[1]), 3)
            dirr[n, 0], dirr[n, 1] = dx, dy
        np.savez_compressed(f"{d}/gtK_{part}.npz", pts=pts, valid=valid, adj=adj, orig=orig)
        np.save(f"{d}/dir_{part}.npy", dirr)
        print(f"{part}: {N} Samples, Stuecke je Sample im Mittel {valid.sum(1).mean():.1f} (max {mx}, verworfen {dropped}), Kanten {adj.sum((1, 2)).mean():.1f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

