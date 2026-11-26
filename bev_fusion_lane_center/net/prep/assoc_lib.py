"""Schleife 5 / E59-E62: Zuordnung extrahierter Spurmitten zum Kartengraphen und Uebertragung des Navigationsziels (Route) von der Karte auf die extrahierten Spuren.
Karte = Soll-Graph des Samples (Stuecke mit Karten-Token, Fahrtrichtung, Kanten) im Fahrzeugsystem; extrahierter Graph = Ausgabe des Systems (Stuecke, Kanten).
Bausteine: Karten-Stoerungen (Pose-Rauschen, veraltete Karte), Registrierung Graph -> Graph (ICP mit Richtung), Zuordnung (Punkt -> Kartenstueck, Mehrheit je Stueck), Routen-Uebertragung, Metriken."""
import numpy as np
from scipy.spatial import cKDTree

COS_DIR = float(np.cos(np.deg2rad(60.0)))


# ------------------------------------------------------------------ Hilfen


def resample(p, ds=1.0):
    p = np.asarray(p, np.float64)
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return p[:1]
    n = max(int(np.ceil(s[-1] / ds)), 1)
    t = np.linspace(0.0, s[-1], n + 1)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def tangents(p):
    if len(p) < 2:
        return np.tile(np.array([[0.0, 1.0]]), (len(p), 1))
    t = np.gradient(p, axis=0)
    return t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)


def vertices(pieces, tokens=None, ds=1.0):
    """Stuecke (Liste Polylinien) -> V [n,2], T [n,2] (Fahrtrichtung), pid [n], tok [n] (Token oder Stueck-Index)."""
    V, T, pid, tok = [], [], [], []
    for k, p in enumerate(pieces):
        q = resample(p, ds)
        if len(q) < 2:
            continue
        V.append(q)
        T.append(tangents(q))
        pid.append(np.full(len(q), k))
        tok.append(np.array([tokens[k] if tokens is not None else k] * len(q), dtype=object))
    if not V:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, int), np.zeros(0, dtype=object)
    return np.concatenate(V), np.concatenate(T), np.concatenate(pid), np.concatenate(tok)


def rot2(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


# ------------------------------------------------------------------ Karten-Stoerungen


def perturb_map(gt, sigma_m=0.0, drop=0.0, protect=(), rng=None, shift_frac=0.0, shift_m=1.5):
    """Karte mit Pose-Rauschen (dx, dy ~ N(0, sigma), Gierwinkel ~ N(0, (2 sigma Grad)^2) um den Ursprung) und veraltet: Anteil 'drop' der Stuecke entfernt (ausser 'protect'), Anteil shift_frac seitlich verschoben.
    -> dict(pieces, tokens, idx) mit idx = urspruengliche Stueck-Indizes der behaltenen Stuecke."""
    rng = rng or np.random.default_rng(0)
    n = len(gt["pts"])
    keep = [k for k in range(n) if (k in protect) or rng.random() >= drop]
    pieces = [np.asarray(gt["pts"][k], np.float64).copy() for k in keep]
    if shift_frac > 0:
        for i, p in enumerate(pieces):
            if rng.random() < shift_frac and len(p) > 1:
                t = tangents(p)
                nrm = np.stack([-t[:, 1], t[:, 0]], 1)
                pieces[i] = p + nrm * shift_m * (1 if rng.random() < 0.5 else -1)
    if sigma_m > 0:
        d = rng.normal(0.0, sigma_m, 2)
        R = rot2(rng.normal(0.0, 2.0 * sigma_m))
        pieces = [p @ R.T + d for p in pieces]
        true_shift = (d, R)
    else:
        true_shift = (np.zeros(2), np.eye(2))
    new = {k: n_ for n_, k in enumerate(keep)}
    edges = [(new[i], new[j]) for i, j in gt["edges"] if i in new and j in new]
    return dict(pieces=pieces, tokens=[gt["tokens"][k] for k in keep], idx=keep, shift=true_shift, edges=edges)


# ------------------------------------------------------------------ Registrierung Graph -> Graph


def _score(tree, MT, ex_V, ex_T, tr, R, r=1.0):
    q = ex_V @ R.T + tr  # extrahierte Punkte in das (gestoerte) Kartensystem zurueck
    qt = ex_T @ R.T
    d, j = tree.query(q, distance_upper_bound=r)
    ok = np.isfinite(d)
    if not ok.any():
        return 0.0
    cosv = (MT[j[ok]] * qt[ok]).sum(1)
    return float(((cosv >= COS_DIR)).sum()) / len(ex_V)


def register(map_V, map_T, ex_V, ex_T, search=3.0, step=1.0, yaw_search=6.0, yaw_step=2.0, iters=12):
    """Schaetzt die Korrektur, die die extrahierten Punkte ins System der (gestoerten) Karte bringt: x_map = R @ x_ex + t. Grobe Gittersuche (Anteil der Punkte mit Karten-Treffer < 1 m und passender Richtung), dann ICP mit Richtungsbedingung."""
    if len(map_V) < 5 or len(ex_V) < 5:
        return np.eye(2), np.zeros(2), 0.0
    tree = cKDTree(map_V)
    best = (-1.0, np.eye(2), np.zeros(2))
    ds = np.arange(-search, search + 1e-6, step)
    ys = np.arange(-yaw_search, yaw_search + 1e-6, yaw_step)
    for yaw in ys:
        R = rot2(yaw)
        for dx in ds:
            for dy in ds:
                s = _score(tree, map_T, ex_V, ex_T, np.array([dx, dy]), R)
                if s > best[0]:
                    best = (s, R, np.array([dx, dy]))
    s, R, t = best
    for it in range(iters):
        q = ex_V @ R.T + t
        qt = ex_T @ R.T
        d, j = tree.query(q, distance_upper_bound=2.0)
        ok = np.isfinite(d)
        if ok.sum() < 5:
            break
        ok2 = ok.copy()
        ok2[ok] = (map_T[j[ok]] * qt[ok]).sum(1) >= COS_DIR
        if ok2.sum() < 5:
            break
        a, b = q[ok2], map_V[j[ok2]]
        ca, cb = a.mean(0), b.mean(0)
        H = (a - ca).T @ (b - cb)
        U, _, Vt = np.linalg.svd(H)
        dR = Vt.T @ U.T
        if np.linalg.det(dR) < 0:
            Vt[-1] *= -1
            dR = Vt.T @ U.T
        dt = cb - dR @ ca
        R, t = dR @ R, dR @ t + dt
    s2 = _score(tree, map_T, ex_V, ex_T, t, R)
    return R, t, s2


def register_around(map_V, map_T, ex_V, ex_T, R0, t0, search=1.0, step=0.5, yaw_search=3.0, yaw_step=1.5, iters=12):
    """Registrierung mit Vorwissen (zeitliche Verfolgung): die extrahierten Punkte werden zuerst mit der vorhergesagten Ausrichtung (R0, t0) transformiert, danach nur noch ein kleines Fenster abgesucht + ICP.
    -> Gesamtausrichtung x_map = R x_ex + t."""
    if len(map_V) < 5 or len(ex_V) < 5:
        return R0, t0, 0.0
    R1, t1, s = register(map_V, map_T, ex_V @ R0.T + t0, ex_T @ R0.T, search=search, step=step, yaw_search=yaw_search, yaw_step=yaw_step, iters=iters)
    return R1 @ R0, R1 @ t0 + t1, s


# ------------------------------------------------------------------ Zuordnung


def assign_points(map_V, map_T, map_tok, map_pid, q, qt, r=2.0):
    """je Punkt q (im Kartensystem) das naechste Kartenpunkt mit passender Richtung innerhalb r -> Stueck-Index (Index in der Kartenstueck-Liste) oder -1."""
    if len(map_V) == 0 or len(q) == 0:
        return -np.ones(len(q), int), np.full(len(q), np.inf)
    tree = cKDTree(map_V)
    out = -np.ones(len(q), int)
    dist = np.full(len(q), np.inf)
    for i in range(len(q)):
        cand = tree.query_ball_point(q[i], r)
        best = None
        for c in cand:
            if float(map_T[c] @ qt[i]) < COS_DIR:
                continue
            d = float(np.linalg.norm(map_V[c] - q[i]))
            if best is None or d < best[0]:
                best = (d, int(map_pid[c]))
        if best is not None:
            out[i], dist[i] = best[1], best[0]
    return out, dist


def assign_hmm(map_V, map_T, map_pid, edges, q, qt, Epid, r=3.0, w_dir=2.0, stay=0.0, succ_cost=0.3, other=3.0):
    """Zuordnung laengs jeder extrahierten Polylinie als Viterbi-Pfad ueber die Kartenstuecke: Kosten je Punkt = Abstand + w_dir (1 - cos Richtung); Wechsel auf einen Nachfolger 0,3, auf ein nicht verbundenes Stueck 3,0.
    Gegenueber 'naechster Punkt + Mehrheitsfilter' liefert das pfadkonsistente Zuordnungen (Kreuzungen, parallele Spuren)."""
    n = len(q)
    out = -np.ones(n, int)
    if len(map_V) == 0 or n == 0:
        return out
    succ = set(map(tuple, edges))
    tree = cKDTree(map_V)
    for k in np.unique(Epid):
        sel = np.nonzero(Epid == k)[0]
        cands = []
        for i in sel:
            cd = {}
            for c in tree.query_ball_point(q[i], r):
                cs = float(map_T[c] @ qt[i])
                if cs < COS_DIR:
                    continue
                cost = float(np.linalg.norm(map_V[c] - q[i])) + w_dir * (1.0 - cs)
                kk = int(map_pid[c])
                if kk not in cd or cost < cd[kk]:
                    cd[kk] = cost
            cands.append(cd)
        prev_cost, back, pts_idx = {}, [], []
        for ii, cd in enumerate(cands):
            if not cd:
                continue
            cost, bp = {}, {}
            for kk, d in cd.items():
                if not prev_cost:
                    cost[kk], bp[kk] = d, None
                    continue
                best = None
                for pp, c in prev_cost.items():
                    tr = stay if pp == kk else (succ_cost if (pp, kk) in succ else other)
                    v = c + tr + d
                    if best is None or v < best[0]:
                        best = (v, pp)
                cost[kk], bp[kk] = best
            back.append(bp)
            pts_idx.append(ii)
            prev_cost = cost
        if not pts_idx:
            continue
        kk = min(prev_cost, key=prev_cost.get)
        seq = [kk]
        for bp in reversed(back[1:]):
            kk = bp[kk]
            seq.append(kk)
        seq.reverse()
        for ii, kk in zip(pts_idx, seq):
            out[sel[ii]] = kk
    return out


def route_flags(mp, route_tok, q, qt, Epid, r=2.0, tie=0.75, win=2):
    """Geometrische Routen-Markierung (robust gegen sich ueberlappende Kartenstuecke, z. B. Kreuzungs-Verbinder, bei denen 'Token des naechsten Stuecks' mehrdeutig ist):
    Punkt q (Kartensystem) liegt auf der Route, wenn ein richtungsgleicher Punkt eines Routenstuecks innerhalb r liegt UND dieser Abstand hoechstens 'tie' Meter groesser ist als der Abstand zum naechsten
    richtungsgleichen Punkt eines beliebigen Stuecks. Danach Mehrheitsfilter (Fenster 2 win + 1) laengs jeder extrahierten Polylinie."""
    n = len(q)
    if n == 0 or not route_tok:
        return np.zeros(n, bool)
    MV, MT, Mpid, _ = vertices(mp["pieces"])
    if len(MV) == 0:
        return np.zeros(n, bool)
    is_route = np.array([t in route_tok for t in mp["tokens"]])
    m = is_route[Mpid]
    if not m.any():
        return np.zeros(n, bool)
    _, d_any = assign_points(MV, MT, None, Mpid, q, qt, r)
    _, d_rt = assign_points(MV[m], MT[m], None, Mpid[m], q, qt, r)
    on = np.isfinite(d_rt) & (d_rt <= d_any + tie)
    out = on.copy()
    for k in np.unique(Epid):
        sel = np.nonzero(Epid == k)[0]
        for ii, i in enumerate(sel):
            out[i] = on[sel[max(ii - win, 0) : ii + win + 1]].mean() >= 0.5
    return out


def associate(ex, mp, register_graph=True, r_assign=2.0, min_frac=0.4, fixed=None, hmm=False, route_tok=None, tie=0.75):
    """ex: dict(pts, edges) extrahierter Graph; mp: dict(pieces, tokens, idx) (ggf. gestoerte) Karte; route_tok: Menge der Routen-Token (optional -> 'point_route' = geometrische Routen-Markierung je Punkt).
    -> je extrahiertem Stueck: Karten-Token (oder None), plus Punktdaten fuer die Auswertung."""
    EV, ET, Epid, _ = vertices(ex["pts"])
    MV, MT, Mpid, _ = vertices(mp["pieces"])
    R, t, sc = np.eye(2), np.zeros(2), 0.0
    if fixed is not None:  # vorgegebene Ausrichtung (z. B. zeitlich gefiltert): x_map = R x_ex + t
        R, t = fixed
    elif register_graph and len(EV) and len(MV):
        R, t, sc = register(MV, MT, EV, ET)
    q = EV @ R.T + t if len(EV) else EV
    qt = ET @ R.T if len(EV) else ET
    pa, pdist = assign_points(MV, MT, None, Mpid, q, qt, r_assign)
    if hmm:
        pa = assign_hmm(MV, MT, Mpid, mp.get("edges", []), q, qt, Epid)
    # Punkt-Ebene: Mehrheitsfilter (Fenster 5) laengs jeder Polylinie, damit einzelne Ausreisser die Zuordnung nicht kippen
    pa_s = pa.copy()
    for k in range(len(ex["pts"])):
        sel = np.nonzero(Epid == k)[0]
        for ii, i in enumerate(sel):
            w = pa[sel[max(ii - 2, 0) : ii + 3]]
            w = w[w >= 0]
            if len(w) >= 2:
                vals, cnt = np.unique(w, return_counts=True)
                pa_s[i] = vals[np.argmax(cnt)]
    point_tok = [mp["tokens"][int(v)] if v >= 0 else None for v in pa_s]
    piece_tok = {}
    for k in range(len(ex["pts"])):
        sel = np.nonzero(Epid == k)[0]
        if len(sel) == 0:
            piece_tok[k] = None
            continue
        a = pa[sel]
        a = a[a >= 0]
        if len(a) < min_frac * len(sel):
            piece_tok[k] = None
            continue
        vals, cnt = np.unique(a, return_counts=True)
        piece_tok[k] = mp["tokens"][int(vals[np.argmax(cnt)])]
    point_route = route_flags(mp, route_tok, q, qt, Epid, r_assign, tie) if route_tok is not None else None
    return dict(piece_tok=piece_tok, point_tok=point_tok, point_piece=Epid, point_assign=pa, point_dist=pdist, R=R, t=t, score=sc, EV=EV, ET=ET, point_route=point_route)


# ------------------------------------------------------------------ Referenz (Soll-Zuordnung der extrahierten Punkte)


def gt_assign(ex, gt, r=1.5):
    """je extrahiertem Punkt das Soll-Stueck (Token) mit passender Richtung innerhalb r (sonst None) im unveraenderten Soll-Graph."""
    EV, ET, Epid, _ = vertices(ex["pts"])
    GV, GT, Gpid, _ = vertices(gt["pts"])
    pa, _ = assign_points(GV, GT, None, Gpid, EV, ET, r)
    tok = [gt["tokens"][a] if a >= 0 else None for a in pa]
    return EV, ET, Epid, pa, tok
