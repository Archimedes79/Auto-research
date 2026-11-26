"""Polylines of the lane graphs as points (D152 follow-up: moved unchanged from evaluation/sparse_lanes/graph_eval.py, which reads them from here: the grid
data and the metric use one definition)."""
import numpy as np


def resample(p, ds):
    """Polylinie (n,2) auf gleichen Abstand ds bringen (Endpunkt bleibt); Rueckgabe (m,2)."""
    p = np.asarray(p, np.float64)
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    if L < 1e-6:
        return p[:1]
    k = max(int(round(L / ds)), 1)
    q = np.linspace(0, L, k + 1)
    return np.stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])], 1)


def dense_points(g, ds=0.5, clip=48.0):
    """Alle Polylinien als dichte Punkte (m, 2) + Fahrtrichtung (m, 2) + Polylinien-Nr; auf |x|,|y| < clip beschnitten (wie die Auswertung in E19)."""
    P, D, I = [], [], []
    for i, p in enumerate(g["pts"]):
        q = resample(p, ds)
        if len(q) < 2:
            continue
        d = np.gradient(q, axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        P.append(q)
        D.append(d)
        I.append(np.full(len(q), i))
    if not P:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, int)
    P, D, I = np.concatenate(P), np.concatenate(D), np.concatenate(I)
    k = (np.abs(P) < clip).all(1)
    return P[k], D[k], I[k]
