"""Zeitliche Stabilitaet ('Flackern') von Graph-Vorhersagen: Vorhersage des Vorgaenger-Bildes per Eigenbewegung ins aktuelle Bild falten und mit der Vorhersage des aktuellen Bildes vergleichen
(Punkt-F1 bei 1 m im gemeinsam sichtbaren Bereich). Als Referenz dasselbe mit dem Soll (Eigenbewegung/Kartenaufloesung begrenzen die Obergrenze unter 1)."""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE
import lf_inputs as LI


def _dense(g):
    P, _, _ = GE.dense_points(g)
    return P


def _stab(Pj, Pi, Q, s):
    """Pj Punkte im System j, Pi Punkte im System i; p_i = Q p_j + s. Rueckgabe [tp_A, |A|, tp_B, |B|]."""
    if len(Pj) == 0 or len(Pi) == 0:
        return np.zeros(4)
    A = Pj @ Q.T + s
    A = A[(np.abs(A) < 47).all(1)]
    back = (Pi - s) @ Q
    cov = (np.abs(back) < 47).all(1)
    B = Pi[cov]
    if len(A) == 0 or len(B) == 0:
        return np.zeros(4)
    tA, tB = cKDTree(A), cKDTree(B)
    return np.array([np.sum(tB.query(A)[0] <= 1.0), len(A), np.sum(tA.query(B)[0] <= 1.0), len(B)])


def _f1(c):
    p, r = c[0] / max(c[1], 1), c[2] / max(c[3], 1)
    return dict(P=float(p), R=float(r), F1=float(2 * p * r / max(p + r, 1e-9)))


def stability(preds, gts, ids, part, gap=1, pose=None):
    """preds/gts: Graphen je Sample (gleiche Reihenfolge wie ids = Zeilen im Teil <part>). Paare nur, wenn der Vorgaenger (gap Bilder zurueck) in ids liegt."""
    pose = pose or LI.PoseTable("b123", gap)
    pos = {int(i): n for n, i in enumerate(ids)}
    cp, cg, npair = np.zeros(4), np.zeros(4), 0
    for n, i in enumerate(ids):
        j = pose.prev(part, int(i))
        if j < 0 or j not in pos:
            continue
        Q, s, _ = pose.rel(part, int(i))
        Q = Q.reshape(2, 2)
        m = pos[j]
        cp += _stab(_dense(preds[m]), _dense(preds[n]), Q, s)
        cg += _stab(_dense(gts[m]), _dense(gts[n]), Q, s)
        npair += 1
    out = dict(pairs=npair, pred=_f1(cp), gt_reference=_f1(cg))
    out["flicker"] = 1 - out["pred"]["F1"]
    return out
