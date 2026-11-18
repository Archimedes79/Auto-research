"""Auswertung von Spurmitten-Graphen (E27 ff.): Vorhersage und Soll im selben Format.

Graph = dict(pts=[(n_i, 2) float Polylinien in m, Fahrzeug-System, in Fahrtrichtung], edges=[(i, j): Ende von i -> Anfang von j]).
Kennzahlen (je Sample, dann gemittelt bzw. mikro-gemittelt):
  - Geometrie: Punkt-Praezision/Ausbeute/F1 bei 0,5/1,0/1,5 m (dichte Punkte alle 0,5 m, wie E19) und F1 mit 1:1-Zuordnung (doppelte Linien zaehlen nicht, wie E26)
  - Spuren: 'Spuren gefunden' = Anteil der Soll-Spuren, deren Punkte zu >= 80 % innerhalb 1 m der Vorhersage liegen; 'falsche Linien' = Anteil der vorhergesagten Linienlaenge, die weiter als 1,5 m von jedem Soll-Punkt entfernt ist
  - Abzweige: Splits (Stuecke mit >= 2 Nachfolgern; Ort = Ende des Stuecks) und Merges (>= 2 Vorgaenger; Ort = Anfang) gegen die Vorhersage, Toleranz 3 m, Praezision/Ausbeute/F1
  - Route: APLS-aehnlich ueber den gerichteten Punktgraphen (Punkte alle 1 m): Quell-/Zielpaare aus dem Soll (Erreichbarkeit <= 40 m), Vorhersage-Weglaenge gegen Soll-Weglaenge ('route_recall'); umgekehrt Paare der Vorhersage im Soll ('route_precision'); 'route_f' = harmonisches Mittel
  - Richtung: Anteil der Vorhersagepunkte (innerhalb 1,5 m eines Soll-Punkts), deren Fahrtrichtung zum Soll passt (< 60 Grad), nur fuer gerichtete Graphen
"""
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

TAUS = (0.5, 1.0, 1.5)


# ------------------------------------------------------------------ Hilfen


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


@njit(cache=True)
def _greedy(pi, gi, n_p, n_g):
    up = np.zeros(n_p, np.bool_)
    ug = np.zeros(n_g, np.bool_)
    m = 0
    for t in range(pi.shape[0]):
        p, g = pi[t], gi[t]
        if (not up[p]) and (not ug[g]):
            up[p] = True
            ug[g] = True
            m += 1
    return m


def match_1to1(P, G, tG, tau, K=8):
    d, idx = tG.query(P, k=K, distance_upper_bound=tau + 1e-6)
    pi = np.repeat(np.arange(len(P)), K)
    d, gi = d.ravel(), idx.ravel()
    ok = np.isfinite(d)
    pi, gi, d = pi[ok], gi[ok], d[ok]
    o = np.argsort(d, kind="stable")
    return _greedy(pi[o].astype(np.int64), gi[o].astype(np.int64), len(P), len(G))


def point_graph(g, ds=1.0):
    """Gerichteter Punktgraph: Knoten = Punkte alle ds m, Kanten entlang der Polylinien und je Verbindung (Ende i -> Anfang j). Rueckgabe: Koordinaten, Kantenliste, Anfangs-/Endknoten je Polylinie."""
    xy, edges, first, last = [], [], {}, {}
    off = 0
    for i, p in enumerate(g["pts"]):
        q = resample(p, ds)
        if len(q) < 2:
            continue
        xy.append(q)
        n = len(q)
        edges += [(off + k, off + k + 1) for k in range(n - 1)]
        first[i], last[i] = off, off + n - 1
        off += n
    if not xy:
        return np.zeros((0, 2)), [], first, last
    xy = np.concatenate(xy)
    for i, j in g["edges"]:
        if i in last and j in first:
            edges.append((last[i], first[j]))
    return xy, edges, first, last


def _csr(xy, edges):
    if not len(xy):
        return csr_matrix((0, 0))
    r = np.array([a for a, _ in edges] + [], int)
    c = np.array([b for _, b in edges] + [], int)
    w = np.linalg.norm(xy[c] - xy[r], axis=1) + 1e-3 if len(r) else np.zeros(0)
    return csr_matrix((w, (r, c)), shape=(len(xy), len(xy)))


def junctions(g):
    """Split- und Merge-Orte: Ende von Stuecken mit >= 2 Nachfolgern, Anfang von Stuecken mit >= 2 Vorgaengern."""
    n = len(g["pts"])
    out_deg, in_deg = np.zeros(n, int), np.zeros(n, int)
    for i, j in g["edges"]:
        out_deg[i] += 1
        in_deg[j] += 1
    spl = np.array([g["pts"][i][-1] for i in range(n) if out_deg[i] >= 2 and len(g["pts"][i])], np.float64).reshape(-1, 2)
    mrg = np.array([g["pts"][i][0] for i in range(n) if in_deg[i] >= 2 and len(g["pts"][i])], np.float64).reshape(-1, 2)
    return spl, mrg


def _match_points(A, B, tol):
    """Gierige Zuordnung von Punktmengen A (Vorhersage) und B (Soll) innerhalb tol -> Zahl der Paare."""
    if len(A) == 0 or len(B) == 0:
        return 0
    D = np.linalg.norm(A[:, None] - B[None], axis=2)
    used_a, used_b, m = set(), set(), 0
    for flat in np.argsort(D, axis=None):
        a, b = divmod(int(flat), D.shape[1])
        if D[a, b] > tol:
            break
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        m += 1
    return m


def route_score(src, dst, rng, n_src=40, max_len=40.0, snap=2.0, per_src=4):
    """Paare (a, b) im Graph 'src' (Erreichbarkeit <= max_len), Weglaenge in 'src' gegen Weglaenge in 'dst'; Mittelwert von max(0, 1 - |L_dst - L_src| / L_src), nicht erreichbar/kein Knoten nahe: 0."""
    xs, es, _, _ = src
    xd, ed, _, _ = dst
    if len(xs) < 2:
        return np.nan
    As = _csr(xs, es)
    Ad = _csr(xd, ed)
    td = cKDTree(xd) if len(xd) else None
    scores = []
    for a in rng.choice(len(xs), min(n_src, len(xs)), replace=False):
        ds_ = dijkstra(As, directed=True, indices=int(a), limit=max_len)
        cand = np.nonzero(np.isfinite(ds_) & (ds_ >= 10.0))[0]
        if len(cand) == 0:
            continue
        for b in rng.choice(cand, min(per_src, len(cand)), replace=False):
            Ls = ds_[b]
            if td is None:
                scores.append(0.0)
                continue
            da, ia = td.query(xs[a])
            db, ib = td.query(xs[b])
            if da > snap or db > snap:
                scores.append(0.0)
                continue
            Ld = dijkstra(Ad, directed=True, indices=int(ia), limit=2 * max_len)[ib]
            scores.append(0.0 if not np.isfinite(Ld) else max(0.0, 1.0 - abs(Ld - Ls) / Ls))
    return float(np.mean(scores)) if scores else np.nan


# ------------------------------------------------------------------ Auswertung eines Samples


def eval_sample(pred, gt, seed=0, with_route=True, directed=True):
    P, DP, IP = dense_points(pred)
    G, DG, IG = dense_points(gt)
    r = dict(cnt=np.zeros((3, 4)), m1=np.zeros(3), n_p=len(P), n_g=len(G))
    if len(P) and len(G):
        tG, tP = cKDTree(G), cKDTree(P)
        dp, ip = tG.query(P)
        dg = tP.query(G)[0]
        for k, t in enumerate(TAUS):
            r["cnt"][k] = (np.sum(dp <= t), len(P), np.sum(dg <= t), len(G))
            r["m1"][k] = match_1to1(P, G, tG, t)
        # Spuren gefunden / falsche Linienlaenge
        lanes = np.unique(IG)
        cov = [(dg[IG == i] <= 1.0).mean() for i in lanes]
        r["lanes_found"], r["lanes_total"] = int(np.sum(np.array(cov) >= 0.8)), len(lanes)
        r["false_len"], r["pred_len"] = float(np.sum(dp > 1.5)) * 0.5, len(P) * 0.5
        if directed:
            ok = dp <= 1.5
            if ok.any():
                dot = (DP[ok] * DG[ip[ok]]).sum(1)
                r["dir_ok"], r["dir_n"] = int((dot > 0.5).sum()), int(ok.sum())
    else:
        r["cnt"][:, 1], r["cnt"][:, 3] = len(P), len(G)
        r["lanes_found"], r["lanes_total"] = 0, len(np.unique(IG)) if len(G) else 0
        r["false_len"], r["pred_len"] = len(P) * 0.5, len(P) * 0.5
    # Abzweige
    sp_p, mg_p = junctions(pred)
    sp_g, mg_g = junctions(gt)
    r["spl"] = (_match_points(sp_p, sp_g, 3.0), len(sp_p), len(sp_g))
    r["mrg"] = (_match_points(mg_p, mg_g, 3.0), len(mg_p), len(mg_g))
    r["spl6"] = (_match_points(sp_p, sp_g, 6.0), len(sp_p), len(sp_g))
    r["mrg6"] = (_match_points(mg_p, mg_g, 6.0), len(mg_p), len(mg_g))
    # Route
    if with_route:
        rng = np.random.RandomState(seed)
        pg, gg = point_graph(pred), point_graph(gt)
        r["route_rec"] = route_score(gg, pg, rng)
        r["route_prec"] = route_score(pg, gg, rng)
    return r


def aggregate(rs):
    c = sum(r["cnt"] for r in rs)
    m1 = sum(r["m1"] for r in rs)
    out = {}
    for k, t in enumerate(TAUS):
        pr, rc = c[k, 0] / max(c[k, 1], 1), c[k, 2] / max(c[k, 3], 1)
        out[f"P@{t}"], out[f"R@{t}"], out[f"F1@{t}"] = float(pr), float(rc), float(2 * pr * rc / max(pr + rc, 1e-9))
        out[f"F1o2o@{t}"] = float(2 * m1[k] / max(c[k, 1] + c[k, 3], 1))
    out["lanes_found"] = float(sum(r.get("lanes_found", 0) for r in rs) / max(sum(r.get("lanes_total", 0) for r in rs), 1))
    out["false_len_frac"] = float(sum(r.get("false_len", 0) for r in rs) / max(sum(r.get("pred_len", 0) for r in rs), 1e-9))
    for key, nm in (("spl", "split"), ("mrg", "merge"), ("spl6", "split6"), ("mrg6", "merge6")):
        if not all(key in r for r in rs):
            continue
        m = sum(r[key][0] for r in rs)
        np_, ng = sum(r[key][1] for r in rs), sum(r[key][2] for r in rs)
        pr, rc = m / max(np_, 1), m / max(ng, 1)
        out[f"{nm}_P"], out[f"{nm}_R"], out[f"{nm}_F1"] = float(pr), float(rc), float(2 * pr * rc / max(pr + rc, 1e-9))
        out[f"{nm}_n_gt"], out[f"{nm}_n_pred"] = float(ng / len(rs)), float(np_ / len(rs))
    rr = [r["route_rec"] for r in rs if "route_rec" in r and not np.isnan(r["route_rec"])]
    rp = [r["route_prec"] for r in rs if "route_prec" in r and not np.isnan(r["route_prec"])]
    if rr and rp:
        out["route_rec"], out["route_prec"] = float(np.mean(rr)), float(np.mean(rp))
        out["route_f"] = float(2 * out["route_rec"] * out["route_prec"] / max(out["route_rec"] + out["route_prec"], 1e-9))
    dn = sum(r.get("dir_n", 0) for r in rs)
    if dn:
        out["dir_acc"] = float(sum(r.get("dir_ok", 0) for r in rs) / dn)
    out["n_samples"] = len(rs)
    return out


def evaluate(preds, gts, with_route=True, directed=True, seed=0):
    return aggregate([eval_sample(p, g, seed + i, with_route, directed) for i, (p, g) in enumerate(zip(preds, gts))])
