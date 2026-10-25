"""Loop 4 / E44: Schluesselpunkt-gefuehrte Knoten im Graphen (SAM-Road-artig: gelernte Knoten verankern die Topologie).
Eingabe: Graph der regelbasierten Verbindung (pts = Polylinien, edges = (i, j)) + Split-/Merge-Schluesselpunkte des Knoten-Kopfes (node_head.py).
Schritte (Parameter auf VAL gewaehlt):
  snap   : jeder vorhergesagte Split-/Merge-Knoten innerhalb r_snap eines Schluesselpunkts wird dorthin verschoben (Stueckenden / -anfaenge werden gekuerzt bzw. gerade verlaengert)
  create : Schluesselpunkt ohne Knoten in der Naehe: laeuft ein Stueck in < d_c daran vorbei und beginnt / endet ein anderes in < r_b mit passendem Kurs, wird das Stueck am Punkt geteilt und verbunden
  prune  : Knoten ohne Schluesselpunkt in < r_prune verlieren ihre schwaechsten Zusatzkanten (Kurs)
Aufruf (gcar): node_assemble.py --graphs e43/graphs_rule_e41_A --nodes e44_nodes --sets val,test --sweep 1   (schreibt out/loop2/<out>/assemble.json und Graph-Pickles)"""
import bflc_paths as BP  # project paths from the location of this checkout
import argparse
import itertools
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402
import graph_from_dense as GD  # noqa: E402
import node_head as NH  # noqa: E402

OUT = BP.RUNS


# ------------------------------------------------------------------ Geometrie


def arclen(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def project(p, k):
    """naechster Punkt der Polylinie p an k: (Abstand, Bogenlaenge, Punkt)"""
    best = (1e9, 0.0, p[0])
    s = arclen(p)
    for a in range(len(p) - 1):
        d = p[a + 1] - p[a]
        L2 = float(d @ d)
        t = 0.0 if L2 < 1e-12 else float(np.clip(((k - p[a]) @ d) / L2, 0.0, 1.0))
        q = p[a] + t * d
        dist = float(np.linalg.norm(q - k))
        if dist < best[0]:
            best = (dist, float(s[a] + t * np.sqrt(L2)), q)
    return best


def straight(a, b, ds=1.0):
    n = max(int(np.ceil(np.linalg.norm(b - a) / ds)), 1)
    return np.stack([a + (b - a) * (t / n) for t in range(1, n + 1)])


def cut_at(p, s):
    """Polylinie bei Bogenlaenge s teilen -> (vorderer Teil, hinterer Teil), beide mit dem Schnittpunkt."""
    a = arclen(p)
    i = int(np.searchsorted(a, s))
    i = min(max(i, 1), len(p) - 1)
    t = (s - a[i - 1]) / max(a[i] - a[i - 1], 1e-9)
    q = p[i - 1] + t * (p[i] - p[i - 1])
    return np.vstack([p[:i], q[None]]), np.vstack([q[None], p[i:]])


def new_end(p, k):
    d, s, q = project(p, k)
    L = arclen(p)[-1]
    if s >= L - 1e-6:
        return np.vstack([p, straight(p[-1], k)]) if np.linalg.norm(k - p[-1]) > 0.2 else p
    front, _ = cut_at(p, s)
    return np.vstack([front, straight(front[-1], k)]) if np.linalg.norm(k - front[-1]) > 0.2 else front


def new_start(p, k):
    d, s, q = project(p, k)
    if s <= 1e-6:
        if np.linalg.norm(k - p[0]) <= 0.2:
            return p
        ext = np.vstack([k[None], straight(k, p[0])[:-1]])
        return np.vstack([ext, p])
    _, back = cut_at(p, s)
    return np.vstack([k[None], back]) if np.linalg.norm(k - back[0]) > 0.2 else back


def heading_end(p):
    t = p[-1] - p[max(len(p) - 4, 0)]
    return t / max(np.linalg.norm(t), 1e-6)


def heading_start(p):
    t = p[min(3, len(p) - 1)] - p[0]
    return t / max(np.linalg.norm(t), 1e-6)


# ------------------------------------------------------------------ Schritte


def degrees(g):
    n = len(g["pts"])
    out_n, in_n = [[] for _ in range(n)], [[] for _ in range(n)]
    for i, j in g["edges"]:
        out_n[i].append(j)
        in_n[j].append(i)
    return out_n, in_n


def snap(g, S, M, r):
    pts = [np.asarray(p, np.float64).copy() for p in g["pts"]]
    out_n, in_n = degrees(g)
    done_end, done_start = set(), set()
    for i in range(len(pts)):
        if len(out_n[i]) >= 2 and len(S):
            k = S[np.argmin(np.linalg.norm(S[:, :2] - pts[i][-1], axis=1)), :2]
            if np.linalg.norm(k - pts[i][-1]) <= r:
                if i not in done_end:
                    pts[i] = new_end(pts[i], k)
                    done_end.add(i)
                for j in out_n[i]:
                    if j not in done_start:
                        pts[j] = new_start(pts[j], k)
                        done_start.add(j)
    for j in range(len(pts)):
        if len(in_n[j]) >= 2 and len(M):
            k = M[np.argmin(np.linalg.norm(M[:, :2] - pts[j][0], axis=1)), :2]
            if np.linalg.norm(k - pts[j][0]) <= r:
                if j not in done_start:
                    pts[j] = new_start(pts[j], k)
                    done_start.add(j)
                for i in in_n[j]:
                    if i not in done_end:
                        pts[i] = new_end(pts[i], k)
                        done_end.add(i)
    return dict(pts=pts, edges=list(g["edges"]))


def create(g, S, M, r_near, d_c, r_b, cos_min=0.5):
    """fehlende Knoten an Schluesselpunkten schaffen (Stuecke teilen und verbinden)."""
    pts = [np.asarray(p, np.float64).copy() for p in g["pts"]]
    edges = set(map(tuple, g["edges"]))
    out_n, in_n = degrees(dict(pts=pts, edges=list(edges)))
    ends_ = lambda: np.array([p[-1] for p in pts]).reshape(-1, 2)
    starts_ = lambda: np.array([p[0] for p in pts]).reshape(-1, 2)
    n_new = 0
    for kind, K in (("split", S), ("merge", M)):
        for k in K[:, :2] if len(K) else []:
            out_n, in_n = degrees(dict(pts=pts, edges=list(edges)))
            if kind == "split":
                served = [i for i in range(len(pts)) if len(out_n[i]) >= 2 and np.linalg.norm(pts[i][-1] - k) <= r_near]
            else:
                served = [j for j in range(len(pts)) if len(in_n[j]) >= 2 and np.linalg.norm(pts[j][0] - k) <= r_near]
            if served:
                continue
            # Stueck, das am Punkt vorbeilaeuft
            cand = []
            for i, p in enumerate(pts):
                if len(p) < 2:
                    continue
                d, s, q = project(p, k)
                if d <= d_c:
                    cand.append((d, i, s))
            if not cand:
                continue
            cand.sort()
            _, i, s = cand[0]
            L = arclen(pts[i])[-1]
            h = heading_end(cut_at(pts[i], max(s, 1.0))[0]) if s > 1.0 else heading_start(pts[i])
            if kind == "split":
                branches = [j for j in range(len(pts)) if j != i and np.linalg.norm(pts[j][0] - k) <= r_b and float(heading_start(pts[j]) @ h) >= cos_min and (i, j) not in edges]
                if not branches:
                    continue
                if 1.0 < s < L - 1.0:
                    a_, b_ = cut_at(pts[i], s)
                    pts[i] = a_
                    pts.append(b_)
                    nb = len(pts) - 1
                    for j in out_n[i]:  # alte Nachfolger gehen auf den hinteren Teil
                        edges.discard((i, j))
                        edges.add((nb, j))
                    edges.add((i, nb))
                    n_new += 1
                for j in branches:
                    edges.add((i, j))
                    n_new += 1
            else:
                branches = [j for j in range(len(pts)) if j != i and np.linalg.norm(pts[j][-1] - k) <= r_b and float(heading_end(pts[j]) @ h) >= cos_min and (j, i) not in edges]
                if not branches:
                    continue
                if 1.0 < s < L - 1.0:
                    a_, b_ = cut_at(pts[i], s)
                    pts[i] = b_  # i = Empfaenger (Anfang am Knoten); vorderer Teil wird neues Stueck
                    pts.append(a_)
                    na = len(pts) - 1
                    for j in in_n[i]:
                        edges.discard((j, i))
                        edges.add((j, na))
                    edges.add((na, i))
                    n_new += 1
                for j in branches:
                    edges.add((j, i))
                    n_new += 1
    return dict(pts=pts, edges=sorted(edges)), n_new


def prune(g, S, M, r_prune):
    pts = g["pts"]
    out_n, in_n = degrees(g)
    drop = set()
    for i in range(len(pts)):
        if len(out_n[i]) >= 2:
            near = len(S) and np.min(np.linalg.norm(S[:, :2] - pts[i][-1], axis=1)) <= r_prune
            if not near:
                h = heading_end(pts[i])
                sc = sorted(out_n[i], key=lambda j: -float(heading_start(pts[j]) @ h))
                for j in sc[1:]:
                    drop.add((i, j))
        if len(in_n[i]) >= 2:
            near = len(M) and np.min(np.linalg.norm(M[:, :2] - pts[i][0], axis=1)) <= r_prune
            if not near:
                h = heading_start(pts[i])
                sc = sorted(in_n[i], key=lambda j: -float(heading_end(pts[j]) @ h))
                for j in sc[1:]:
                    drop.add((j, i))
    return dict(pts=pts, edges=[e for e in g["edges"] if tuple(e) not in drop])


def repair(g, r_c, cos_min=0.7, border=42.0):
    """Kontinuitaet (LaneGAP-Sicht): innere Sackgassen (Stueck ohne Nachfolger, Ende im Fensterinneren) bzw. Waisen (ohne Vorgaenger, Anfang im Inneren) mit dem naechsten passenden Stueck verbinden."""
    pts = g["pts"]
    edges = set(map(tuple, g["edges"]))
    out_n, in_n = degrees(g)
    n = len(pts)
    for i in range(n):
        if len(pts[i]) < 2:
            continue
        if not out_n[i] and np.abs(pts[i][-1]).max() < border:
            e, h = pts[i][-1], heading_end(pts[i])
            best = None
            for j in range(n):
                if j == i or len(pts[j]) < 2 or (i, j) in edges:
                    continue
                d = float(np.linalg.norm(pts[j][0] - e))
                if d > r_c:
                    continue
                u = (pts[j][0] - e) / max(d, 1e-6)
                if d > 1.0 and float(h @ u) < cos_min:
                    continue
                if float(h @ heading_start(pts[j])) < cos_min:
                    continue
                if best is None or d < best[0]:
                    best = (d, j)
            if best:
                edges.add((i, best[1]))
        if not in_n[i] and np.abs(pts[i][0]).max() < border:
            sp, h = pts[i][0], heading_start(pts[i])
            best = None
            for j in range(n):
                if j == i or len(pts[j]) < 2 or (j, i) in edges:
                    continue
                d = float(np.linalg.norm(pts[j][-1] - sp))
                if d > r_c:
                    continue
                u = (sp - pts[j][-1]) / max(d, 1e-6)
                if d > 1.0 and float(heading_end(pts[j]) @ u) < cos_min:
                    continue
                if float(heading_end(pts[j]) @ h) < cos_min:
                    continue
                if best is None or d < best[0]:
                    best = (d, j)
            if best:
                edges.add((best[1], i))
    return dict(pts=pts, edges=sorted(edges))


def _sp(edges, n):
    out_n, in_n = [[] for _ in range(n)], [[] for _ in range(n)]
    for i, j in edges:
        out_n[i].append(j)
        in_n[j].append(i)
    return out_n, in_n


def cut_nodes(g, S, M, d_c=2.5, r_s=12.0, r_b=12.0, cos_min=0.3):
    """Geometrie erhaltender Knoteneinbau (statt Verschieben): laeuft ein Stueck am Schluesselpunkt vorbei, wird es dort geteilt (Knoten = Ende des vorderen Teils bzw. Anfang des hinteren Teils, wie im Soll);
    der Durchgangs-Nachfolger bleibt am hinteren Teil, die uebrigen Nachfolger bzw. neue Zweige (Stuecke, die in r_b beginnen) werden am Knoten angehaengt.  Merges spiegelbildlich."""
    pts = [np.asarray(p, np.float64).copy() for p in g["pts"]]
    edges = set(map(tuple, g["edges"]))
    n_mod = 0
    for kind, K in (("split", S), ("merge", M)):
        for k in (K[:, :2] if len(K) else []):
            n = len(pts)
            out_n, in_n = _sp(edges, n)
            best = None
            for i, p in enumerate(pts):
                if len(p) < 2:
                    continue
                d, s_, q = project(p, k)
                if d <= d_c and (best is None or d < best[0]):
                    best = (d, i, s_)
            if best is None:
                continue
            _, t, s_ = best
            L = arclen(pts[t])[-1]
            snap_pts, snap_edges, snap_n = pts[t].copy(), set(edges), len(pts)
            if kind == "split":
                if any(len(out_n[i]) >= 2 and np.linalg.norm(pts[i][-1] - k) <= 2.0 for i in range(n)):
                    continue
                succ = list(out_n[t])
                if s_ <= 1.0:
                    continue
                a_, b_ = cut_at(pts[t], min(s_, L - 1e-3))
                h = heading_end(a_)
                extras = [j for j in range(n) if j != t and j not in succ and len(pts[j]) >= 2 and np.linalg.norm(pts[j][0] - k) <= r_b and float(heading_start(pts[j]) @ h) >= cos_min]
                if s_ >= L - 1.0:  # k am Ende: kein Schnitt, nur Zweige anhaengen
                    if (succ and extras) or len(extras) >= 2:
                        for j in extras:
                            edges.add((t, j))
                        n_mod += 1
                    continue
                if np.linalg.norm(pts[t][-1] - k) > r_s and not extras:
                    continue
                if arclen(a_)[-1] < 1.0 or arclen(b_)[-1] < 1.0:
                    continue
                pts[t] = a_
                pts.append(b_)
                b = len(pts) - 1
                for q in succ:
                    edges.discard((t, q))
                edges.add((t, b))
                if succ:
                    hb = heading_end(b_)
                    thr = sorted(succ, key=lambda q: -float(heading_start(pts[q]) @ hb))
                    edges.add((b, thr[0]))
                    for q in thr[1:]:
                        edges.add((t, q))
                for q in extras:
                    edges.add((t, q))
                if sum(1 for e in edges if e[0] == t) < 2:  # kein Abzweig entstanden -> zuruecknehmen
                    pts[t] = snap_pts
                    del pts[snap_n:]
                    edges = snap_edges
                else:
                    n_mod += 1
            else:
                if any(len(in_n[i]) >= 2 and np.linalg.norm(pts[i][0] - k) <= 2.0 for i in range(n)):
                    continue
                preds = list(in_n[t])
                if s_ >= L - 1.0:
                    continue
                if s_ <= 1.0:
                    h = heading_start(pts[t])
                    extras = [j for j in range(n) if j != t and j not in preds and len(pts[j]) >= 2 and np.linalg.norm(pts[j][-1] - k) <= r_b and float(heading_end(pts[j]) @ h) >= cos_min]
                    if (preds and extras) or len(extras) >= 2:
                        for j in extras:
                            edges.add((j, t))
                        n_mod += 1
                    continue
                a_, b_ = cut_at(pts[t], s_)
                h = heading_start(b_)
                extras = [j for j in range(n) if j != t and j not in preds and len(pts[j]) >= 2 and np.linalg.norm(pts[j][-1] - k) <= r_b and float(heading_end(pts[j]) @ h) >= cos_min]
                if np.linalg.norm(pts[t][0] - k) > r_s and not extras:
                    continue
                if arclen(a_)[-1] < 1.0 or arclen(b_)[-1] < 1.0:
                    continue
                pts[t] = a_
                pts.append(b_)
                b = len(pts) - 1
                for q in out_n[t]:
                    edges.discard((t, q))
                    edges.add((b, q))
                edges.add((t, b))
                if preds:
                    ha = heading_start(a_)
                    thr = sorted(preds, key=lambda p_: -float(heading_end(pts[p_]) @ ha))
                    for p_ in thr[1:]:
                        edges.discard((p_, t))
                        edges.add((p_, b))
                for q in extras:
                    edges.add((q, b))
                if sum(1 for e in edges if e[1] == b) < 2:
                    pts[t] = snap_pts
                    del pts[snap_n:]
                    edges = snap_edges
                else:
                    n_mod += 1
    return dict(pts=pts, edges=sorted(edges)), n_mod


def bezier(P0, P1, P2, P3, ds=1.0):
    L = float(np.linalg.norm(P3 - P0))
    n = max(int(L / ds), 4)
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3


def heat_along(heat, pts):
    from scipy.ndimage import map_coordinates

    rc = (np.asarray(pts, np.float64) + 50.0) / 0.5 - 0.5
    return map_coordinates(heat, [rc[:, 0], rc[:, 1]], order=1, mode="nearest")


def reach_len(g, a, b, limit):
    """kuerzeste Pfadlaenge (m) vom Ende des Stuecks a zum Anfang des Stuecks b ueber die Kanten (Breitensuche, Stuecklaengen), np.inf wenn > limit"""
    n = len(g["pts"])
    out_n, _ = _sp(set(map(tuple, g["edges"])), n)
    L = [float(arclen(p)[-1]) if len(p) > 1 else 0.0 for p in g["pts"]]
    best = {a: 0.0}
    stack = [a]
    while stack:
        i = stack.pop()
        for j in out_n[i]:
            d = best[i] + (L[j] if j != b else 0.0)
            if d <= limit and (j not in best or d < best[j]):
                best[j] = d
                stack.append(j)
    return best.get(b, np.inf)


def dir_cos(dirf, curve):
    """mittlere Kosinus-Uebereinstimmung der Kurventangente (Reisericht.) mit der vorhergesagten Richtungskarte laengs der Kurve (Innenpunkte)"""
    from scipy.ndimage import map_coordinates

    if len(curve) < 4:
        return 1.0
    t = np.gradient(np.asarray(curve, np.float64), axis=0)
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    rc = (np.asarray(curve, np.float64) + 50.0) / 0.5 - 0.5
    d0 = map_coordinates(dirf[0], [rc[:, 0], rc[:, 1]], order=1, mode="nearest")
    d1 = map_coordinates(dirf[1], [rc[:, 0], rc[:, 1]], order=1, mode="nearest")
    dn = np.hypot(d0, d1)
    ok = dn > 0.1
    ok[0] = ok[-1] = False
    if ok.sum() < 3:
        return 1.0
    return float(np.mean((t[:, 0] * d0 + t[:, 1] * d1)[ok] / dn[ok]))


def synth_connectors(g, S, M, heat, tau_h, dmin=4.0, dmax=40.0, kmax=2, cos_s=0.17, d_c=2.5, dirf=None, cos_c=-2.0):
    """SeqGrowGraph-artig (Bezier-Mittellinien) / SAM-Road-artig (Knoten): fehlt an einem Split-Schluesselpunkt der Zweig, wird zu einem Merge-Schluesselpunkt voraus ein Verbinder als kubische Bezier-Kurve
    (Endrichtungen = Kurs des Zubringers am Split und des Abgangs-Stuecks am Merge) erzeugt, wenn die Waerme des dichten Kopfes laengs der Kurve im Mittel >= tau_h ist und beide Punkte nicht schon verbunden sind."""
    pts = [np.asarray(p, np.float64).copy() for p in g["pts"]]
    edges = set(map(tuple, g["edges"]))
    n_new = 0
    for sk in (S[:, :2] if len(S) else []):
        n = len(pts)
        out_n, in_n = _sp(edges, n)
        if any(len(out_n[i]) >= 2 and np.linalg.norm(pts[i][-1] - sk) <= 3.0 for i in range(n)):
            continue
        best = None
        for i, p in enumerate(pts):
            if len(p) < 2:
                continue
            d, s_, _ = project(p, sk)
            if d <= d_c and (best is None or d < best[0]):
                best = (d, i, s_)
        if best is None:
            continue
        _, t, s_ = best
        Lt = arclen(pts[t])[-1]
        if s_ < 1.0:
            continue
        hs = heading_end(cut_at(pts[t], min(s_, Lt - 1e-3))[0]) if s_ < Lt - 1e-3 else heading_end(pts[t])
        cands = []
        for mk in (M[:, :2] if len(M) else []):
            v = mk - sk
            D = float(np.linalg.norm(v))
            if D < dmin or D > dmax or float((v / D) @ hs) < cos_s:
                continue
            e_best = None
            for j, q in enumerate(pts):
                if j == t or len(q) < 2:
                    continue
                d, se, _ = project(q, mk)
                if d <= d_c and (e_best is None or d < e_best[0]):
                    e_best = (d, j, se)
            if e_best is None:
                continue
            _, ei, se = e_best
            Le = arclen(pts[ei])[-1]
            hm = heading_start(cut_at(pts[ei], max(se, 1e-3))[1]) if se < Le - 1.0 else heading_end(pts[ei])
            if float(hs @ hm) < -0.3:
                continue
            a_ = D / 3.0
            curve = bezier(sk, sk + a_ * hs, mk - a_ * hm, mk)
            hv = float(np.mean(heat_along(heat, curve[1:-1]))) if len(curve) > 2 else 0.0
            if hv >= tau_h and (dirf is None or cos_c < -1.5 or dir_cos(dirf, curve) >= cos_c):
                cands.append((hv, ei, se, mk, curve))
        cands.sort(key=lambda c: -c[0])
        for hv, ei, se, mk, curve in cands[:kmax]:
            n = len(pts)
            out_n, in_n = _sp(edges, n)
            if reach_len(dict(pts=pts, edges=list(edges)), t, ei, 1.6 * float(np.linalg.norm(mk - sk)) + 6.0) < np.inf:
                continue  # schon verbunden
            Lt = arclen(pts[t])[-1]
            # Zubringer am Split-Punkt teilen (sofern der Punkt innen liegt)
            t_end = t
            if 1.0 < s_ < Lt - 1.0:
                a_, b_ = cut_at(pts[t], s_)
                pts[t] = a_
                pts.append(b_)
                bn = len(pts) - 1
                for q in out_n[t]:
                    edges.discard((t, q))
                    edges.add((bn, q))
                edges.add((t, bn))
            Le = arclen(pts[ei])[-1]
            e_start = ei
            if 1.0 < se < Le - 1.0:
                a2, b2 = cut_at(pts[ei], se)
                pts[ei] = a2
                pts.append(b2)
                bn2 = len(pts) - 1
                for q in out_n[ei]:
                    edges.discard((ei, q))
                    edges.add((bn2, q))
                edges.add((ei, bn2))
                e_start = bn2
            pts.append(curve)
            cn = len(pts) - 1
            edges.add((t_end, cn))
            edges.add((cn, e_start))
            n_new += 1
    return dict(pts=pts, edges=sorted(edges)), n_new


def apply(g, kp, par, heat=None, dirf=None):
    S, M = kp
    h = g
    if par.get("cut", 0) > 0:
        h, _ = cut_nodes(h, S, M, par.get("d_c", 2.5), par.get("r_s", 12.0), par.get("r_b", 12.0), par.get("cos_min", 0.3))
    if par.get("snap", 0) > 0:
        h = snap(h, S, M, par["snap"])
    if par.get("create", 0) > 0:
        h, _ = create(h, S, M, par["snap"] if par.get("snap", 0) > 0 else 6.0, par.get("d_c", 2.5), par.get("r_b", 6.0), par.get("cos_min", 0.5))
        if par.get("snap", 0) > 0:
            h = snap(h, S, M, par["snap"])
    if par.get("repair", 0) > 0:
        h = repair(h, par["repair"], par.get("rep_cos", 0.7))
    if par.get("prune", 0) > 0:
        h = prune(h, S, M, par["prune"])
    if par.get("synth", 0) > 0 and heat is not None:
        h, _ = synth_connectors(h, S, M, heat, par["synth"], par.get("dmin", 4.0), par.get("dmax", 40.0), int(par.get("kmax", 2)), dirf=dirf, cos_c=par.get("cos_c", -2.0))
    return h


def keypoints(prob, tau):
    import torch

    pk = NH.peaks(torch.from_numpy(np.asarray(prob, np.float32))[None].cuda(), tau)[0]
    return pk[0], pk[1]


# ------------------------------------------------------------------ Hauptprogramm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True, help="z. B. e43/graphs_rule_e41_A  (+ _val.pkl / _test.pkl)")
    ap.add_argument("--nodes", default="e44_nodes")
    ap.add_argument("--sets", default="val,test")
    ap.add_argument("--out", default="e44_assemble")
    ap.add_argument("--sweep", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tau_k", type=float, default=0.0, help="0 = auf VAL gewaehlt (Sweep)")
    ap.add_argument("--heat", default="", help="Lauf des dichten Kopfes (prob_<menge>.npy) fuer die Verbinder-Erzeugung")
    ap.add_argument("--mode", default="snap", choices=["snap", "cut"], help="Sweep: Verschieben/Erzeugen (snap) oder geometrieerhaltender Schnitt (cut)")
    ap.add_argument("--par_json", default="", help="feste Parameter {\"tau\":..,\"par\":{..}} auswerten (kein Sweep)")
    ap.add_argument("--oracle", type=int, default=0, help="Obergrenze: statt der gelernten Schluesselpunkte die Soll-Knoten (Split / Merge) verwenden")
    a = ap.parse_args()
    os.makedirs(f"{OUT}/{a.out}", exist_ok=True)
    data = {}
    for st in a.sets.split(","):
        d = pickle.load(open(f"{OUT}/{a.graphs}_{st}.pkl", "rb"))
        prob = np.load(f"{OUT}/{a.nodes}/prob_{st}.npy", mmap_mode="r")
        nid = {int(i): k for k, i in enumerate(np.load(f"{OUT}/{a.nodes}/ids_{st}.npy"))}
        part = "val"
        gts = GD.load_gt_graphs(part, d["ids"])
        data[st] = dict(graphs=d["graphs"], ids=d["ids"], prob=[np.asarray(prob[nid[int(i)]], np.float32) for i in d["ids"]], gts=gts)
        if a.heat:
            hp = np.load(f"{OUT}/{a.heat}/prob_{st}.npy", mmap_mode="r")
            hid = {int(i): k for k, i in enumerate(np.load(f"{OUT}/{a.heat}/ids_{st}.npy"))}
            data[st]["heat"] = [np.asarray(hp[hid[int(i)]], np.float32) for i in d["ids"]]
            if os.path.exists(f"{OUT}/{a.heat}/dir_{st}.npy"):
                dd = np.load(f"{OUT}/{a.heat}/dir_{st}.npy", mmap_mode="r")
                data[st]["dirf"] = [np.asarray(dd[hid[int(i)]], np.float32) for i in d["ids"]]
    res = {}

    kp_cache = {}

    def run(st, par, tau):
        if (st, tau) not in kp_cache:
            if a.oracle:
                kp_cache[(st, tau)] = [tuple(np.concatenate([np.asarray(q, np.float32).reshape(-1, 2), np.ones((len(q), 1), np.float32)], 1) for q in GE.junctions(g)) for g in data[st]["gts"]]
            else:
                kp_cache[(st, tau)] = [keypoints(p, tau) for p in data[st]["prob"]]
        kps = kp_cache[(st, tau)]
        gs = [apply(g, kp, par, (data[st]["heat"][k] if "heat" in data[st] else None), (data[st]["dirf"][k] if "dirf" in data[st] else None)) for k, (g, kp) in enumerate(zip(data[st]["graphs"], kps))]
        return gs, GD.evaluate_parallel(gs, data[st]["gts"], a.workers, True)

    base = {st: GD.evaluate_parallel(data[st]["graphs"], data[st]["gts"], a.workers, True) for st in data}
    res["base"] = base
    line = lambda m: f"F1@1 {m['F1@1.0']:.3f} route F {m['route_f']:.3f} split F1 3/6m {m['split_F1']:.3f}/{m['split6_F1']:.3f} (P {m['split6_P']:.3f} R {m['split6_R']:.3f}) merge F1 3/6m {m['merge_F1']:.3f}/{m['merge6_F1']:.3f} (P {m['merge6_P']:.3f} R {m['merge6_R']:.3f}) dir {m.get('dir_acc', 0):.3f}"
    for st in base:
        print(f"[{st}] Ausgang: {line(base[st])}", flush=True)
    if a.par_json:
        cfgp = json.loads(a.par_json)
        res["best"] = cfgp
        for st in data:
            gs, m = run(st, cfgp["par"], cfgp["tau"])
            res[st] = m
            pickle.dump(dict(graphs=gs, ids=data[st]["ids"]), open(f"{OUT}/{a.out}/graphs_{st}.pkl", "wb"))
            print(f"[{st}] {cfgp}: {line(m)}", flush=True)
    elif a.sweep and a.mode == "cut":
        sw = []

        def score(m):
            return m["split6_F1"] + m["merge6_F1"] + 2.0 * m["route_f"] + 2.0 * m["F1@1.0"]  # Geometrie und Route staerker gewichtet (der Schnitt soll sie nicht verschlechtern)

        def trial(tau, par):
            gs, m = run("val", par, tau)
            sw.append(dict(tau=tau, par=dict(par), val=m, score=score(m)))
            print(f"tau_k {tau} {par}: {line(m)}", flush=True)
            return score(m)

        best_tau = max((0.3, 0.5), key=lambda t: trial(t, dict(cut=1)))
        par = dict(cut=1)
        cur = max(r["score"] for r in sw if r["tau"] == best_tau)
        for key, vals in (("r_s", (8.0, 16.0)), ("r_b", (8.0, 16.0)), ("cos_min", (0.0, 0.6)), ("d_c", (1.5, 4.0))):
            for v in vals:
                sc = trial(best_tau, dict(par, **{key: v}))
                if sc > cur:
                    par, cur = dict(par, **{key: v}), sc
        for key, val in (("repair", 12.0),):
            sc = trial(best_tau, dict(par, **{key: val}))
            if sc > cur:
                par, cur = dict(par, **{key: val}), sc
        best_p = None
        for rp in (6.0, 10.0):
            sc = trial(best_tau, dict(par, prune=rp))
            if sc > cur and (best_p is None or sc > best_p[0]):
                best_p = (sc, rp)
        if best_p:
            par, cur = dict(par, prune=best_p[1]), best_p[0]
        res["sweep"] = sw
        res["best"] = dict(tau=best_tau, par=par)
        print("BESTE (VAL):", res["best"], flush=True)
        for st in data:
            gs, m = run(st, par, best_tau)
            res[st] = m
            pickle.dump(dict(graphs=gs, ids=data[st]["ids"]), open(f"{OUT}/{a.out}/graphs_{st}.pkl", "wb"))
            print(f"[{st}] Schnitt-Aufbau: {line(m)}", flush=True)
    elif a.sweep:
        sw = []

        def score(m):
            return m["split6_F1"] + m["merge6_F1"] + m["route_f"] + m["F1@1.0"]

        def trial(tau, par):
            gs, m = run("val", par, tau)
            sw.append(dict(tau=tau, par=dict(par), val=m, score=score(m)))
            print(f"tau_k {tau} {par}: {line(m)}", flush=True)
            return score(m)

        # Stufe 1: Schwelle des Knoten-Kopfes (mit Verschieben auf 6 m)
        best_tau = max((0.3, 0.4, 0.5), key=lambda t: trial(t, dict(snap=6.0)))
        # Stufe 2: Verschiebe-Radius
        best_snap = max((4.0, 6.0, 8.0, 10.0, 12.0), key=lambda r: trial(best_tau, dict(snap=r)))
        par = dict(snap=best_snap)
        cur = max(r["score"] for r in sw if r["tau"] == best_tau and r["par"] == par)
        # Stufe 3: Zusatzschritte einzeln dazunehmen, wenn sie helfen
        best_c = None
        for rb, cm in ((8.0, 0.5), (12.0, 0.5), (16.0, 0.2)):
            sc = trial(best_tau, dict(par, create=1, r_b=rb, cos_min=cm))
            if sc > cur and (best_c is None or sc > best_c[0]):
                best_c = (sc, dict(create=1, r_b=rb, cos_min=cm))
        if best_c:
            par, cur = dict(par, **best_c[1]), best_c[0]
        for key, val in (("repair", 12.0),):
            sc = trial(best_tau, dict(par, **{key: val}))
            if sc > cur:
                par, cur = dict(par, **{key: val}), sc
        best_p = None
        for rp in (6.0, 8.0, 12.0):
            sc = trial(best_tau, dict(par, prune=rp))
            if sc > cur and (best_p is None or sc > best_p[0]):
                best_p = (sc, rp)
        if best_p:
            par, cur = dict(par, prune=best_p[1]), best_p[0]
        res["sweep"] = sw
        res["best"] = dict(tau=best_tau, par=par)
        print("BESTE (VAL):", res["best"], flush=True)
        for st in data:
            gs, m = run(st, par, best_tau)
            res[st] = m
            pickle.dump(dict(graphs=gs, ids=data[st]["ids"]), open(f"{OUT}/{a.out}/graphs_{st}.pkl", "wb"))
            print(f"[{st}] mit Knoten-Schluesselpunkten: {line(m)}", flush=True)
    json.dump(res, open(f"{OUT}/{a.out}/assemble.json", "w"), indent=1)


if __name__ == "__main__":
    main()
