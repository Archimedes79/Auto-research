"""E26: Punkt-fuer-Punkt-Linienverfolger auf einem Abstandsfeld (Sparsifizierung: dichtes Feld -> duenne Polylinien).

Idee: statt Schwelle + Ausduennen (Pixel-Skelett, dann Graph) wird die Linie wie von Hand nachgefahren:
  1. Saatpunkt = bester noch nicht abgedeckter Talboden des (geglaetteten) Feldes
  2. Schritt:    Richtung aus einem Faecher (+-max_turn) waehlen, der laut Feld (1 m und 2 m voraus) am tiefsten liegt, 1 m vorruecken
  3. Zentrieren: Punkt quer zur Fahrtrichtung in die Mitte des Tals schieben (gewichteter Schwerpunkt des Querschnitts)
  4. Pruefen:    liegt der Punkt auf der Linie (Rohfeld <= t_on)?  Nein: hoechstens gap_max Schritte ueberbruecken, dann Ende
  5. Treffen:    stoesst der Kopf auf eine fremde, fertige Linie: kleiner Winkel -> verschmelzen (Abzweig/Zusammenfuehrung, Anbindung), grosser Winkel -> Kreuzung, weiter
Ausgabe: Polylinien (Knoten alle 1 m), auf Wunsch auf 3 m ausgeduennt. Alles in Pixel (Zeile = x, Spalte = y, Pixelmitte bei ganzen Zahlen), 0,5 m je Pixel.
"""
import math

import numpy as np
from numba import njit
from scipy import ndimage as ndi

PX = 0.5
HALF = 50.0

DEFAULT = dict(
    sig_nav=1.0,  # px: Glaettung des Feldes fuer Richtungswahl und Zentrierung
    t_seed=0.35,  # m: Saatpunkte = lokale Minima des geglaetteten Feldes unter dieser Schwelle
    t_on=0.6,  # m: Linienpruefung am Rohfeld
    step=2.0,  # px (= 1 m)
    max_turn=45.0,  # Grad je Schritt
    n_cand=9,
    lam=0.15,  # m: Strafe fuer Richtungsaenderung (bei max_turn)
    gap_max=2,  # ueberbrueckbare Schritte ohne Linie
    r_cov=1.5,  # px: Reichweite, in der eine fremde Linie als erreicht gilt
    merge_deg=40.0,  # Winkel unter dem Linien verschmelzen (darueber: Kreuzung)
    r_seed=3.0,  # px: Sperrradius fuer neue Saatpunkte um fertige Linien
    ign=3,  # Schritte nach einer Kreuzung ohne Treffer-Pruefung
    max_steps=160,
    margin=1.5,  # px Randabstand
    sig_w=0.3,  # m: Gewichtsbreite der Querschnitts-Zentrierung
    center_px=3.0,  # px: halbe Breite des Querschnitts
    min_len=4.0,  # m: kuerzere Linien verwerfen (ausser an beiden Enden angebunden)
    smooth=1,  # Glaettungsdurchgaenge [1 2 1]/4
    beta=0.06,  # m je Pixel: Zug zur Mittelachse breiter Talboeden (0 = aus); Richtungswahl und Saatpunkte nutzen Feld + beta * (cap - halbe Talbreite)
    cap_px=12.0,  # Pixel: Obergrenze der halben Talbreite fuer den Zug
    t_blob=0.5,  # m: Schwelle, die den Talboden (Blob) fuer die halbe Talbreite abgrenzt
    dup_m=1.0,  # m: Linien, die zu mindestens dup_frac ihrer Laenge so nah an einer laengeren Linie liegen, werden als Duplikat verworfen (0 = aus)
    dup_frac=0.7,
)

REASON = {0: "Feld zu hoch (Ende der Linie)", 1: "auf andere Linie getroffen (verschmolzen)", 2: "Bildrand", 3: "Schrittgrenze", 4: "Speichergrenze"}


@njit(cache=True)
def bil(D, r, c):
    h, w = D.shape
    if r < 0.0:
        r = 0.0
    if r > h - 1.001:
        r = h - 1.001
    if c < 0.0:
        c = 0.0
    if c > w - 1.001:
        c = w - 1.001
    r0 = int(r)
    c0 = int(c)
    fr = r - r0
    fc = c - c0
    return D[r0, c0] * (1 - fr) * (1 - fc) + D[r0 + 1, c0] * fr * (1 - fc) + D[r0, c0 + 1] * (1 - fr) * fc + D[r0 + 1, c0 + 1] * fr * fc


@njit(cache=True)
def _scan(Dn, r, c, th, maxturn, ncand, lam, d1, d2):
    best = 1e9
    bth = th
    for k in range(ncand):
        dth = (k - (ncand - 1) * 0.5) * (2.0 * maxturn / max(ncand - 1, 1))
        t = th + dth
        ur = math.cos(t)
        uc = math.sin(t)
        v = 0.5 * (bil(Dn, r + d1 * ur, c + d1 * uc) + bil(Dn, r + d2 * ur, c + d2 * uc))
        j = v + lam * (dth / maxturn) * (dth / maxturn)
        if j < best:
            best = j
            bth = t
    return bth


@njit(cache=True)
def _center(Dn, r, c, th, half_px, sig):
    ur = math.cos(th)
    uc = math.sin(th)
    sw = 0.0
    sx = 0.0
    n = int(half_px * 2)
    for k in range(-n, n + 1):
        o = 0.5 * k
        v = bil(Dn, r - o * uc, c + o * ur)
        w = math.exp(-0.5 * (v / sig) * (v / sig))
        sw += w
        sx += w * o
    if sw < 1e-9:
        return 0.0
    return sx / sw


@njit(cache=True)
def _init_heading(Dn, r, c, step):
    best = 1e9
    bth = 0.0
    for k in range(16):
        t = k * math.pi / 16.0
        ur = math.cos(t)
        uc = math.sin(t)
        v = 0.25 * (bil(Dn, r + step * ur, c + step * uc) + bil(Dn, r - step * ur, c - step * uc) + bil(Dn, r + 2 * step * ur, c + 2 * step * uc) + bil(Dn, r - 2 * step * ur, c - 2 * step * uc))
        if v < best:
            best = v
            bth = t
    return bth


@njit(cache=True)
def _trace(D, Dn, Dc, r, c, th, line_id, P, pts, pv, vline, vth, n, cov, maxp):
    t_on = P[0]
    step = P[1]
    maxturn = P[2]
    ncand = int(P[3])
    lam = P[4]
    gap_max = int(P[5])
    msin = P[7]
    ign_steps = int(P[9])
    maxsteps = int(P[10])
    margin = P[11]
    sig_w = P[12]
    half = P[13]
    h, w = D.shape
    miss = 0
    ign = 0
    reason = 3
    snap = -1
    for it in range(maxsteps):
        mt = maxturn if miss == 0 else maxturn * 0.4
        bth = _scan(Dc, r, c, th, mt, ncand, lam, step, 2.0 * step)
        ur = math.cos(bth)
        uc = math.sin(bth)
        nr = r + step * ur
        nc = c + step * uc
        off = _center(Dn, nr, nc, bth, half, sig_w)
        if off > 1.5:
            off = 1.5
        if off < -1.5:
            off = -1.5
        nr += off * (-uc)
        nc += off * ur
        if nr < margin or nr > h - 1 - margin or nc < margin or nc > w - 1 - margin:
            reason = 2
            break
        v = bil(D, nr, nc)
        if v > t_on:
            miss += 1
            if miss > gap_max:
                n -= miss - 1
                reason = 0
                break
        else:
            miss = 0
        pts[n, 0] = nr
        pts[n, 1] = nc
        pv[n] = v
        vline[n] = line_id
        vth[n] = bth
        n += 1
        ir = int(nr + 0.5)
        ic = int(nc + 0.5)
        if ign > 0:
            ign -= 1
        else:
            cv = cov[ir, ic]
            if cv > 0:
                j = cv - 1
                if vline[j] != line_id:
                    if abs(math.sin(bth - vth[j])) < msin:
                        snap = j
                        reason = 1
                        break
                    else:
                        ign = ign_steps
        r = nr
        c = nc
        th = bth
        if n >= maxp - 2:
            reason = 4
            break
    return n, reason, snap


@njit(cache=True)
def _stamp(cov, pts, i0, i1, rad):
    ir = int(rad + 1.0)
    h, w = cov.shape
    for i in range(i0, i1):
        r0 = int(pts[i, 0] + 0.5)
        c0 = int(pts[i, 1] + 0.5)
        for dr in range(-ir, ir + 1):
            for dc in range(-ir, ir + 1):
                if dr * dr + dc * dc <= rad * rad:
                    rr = r0 + dr
                    cc = c0 + dc
                    if rr >= 0 and rr < h and cc >= 0 and cc < w and cov[rr, cc] == 0:
                        cov[rr, cc] = i + 1


@njit(cache=True)
def _block(blk, pts, i0, i1, rad):
    ir = int(rad + 1.0)
    h, w = blk.shape
    for i in range(i0, i1):
        r0 = int(pts[i, 0] + 0.5)
        c0 = int(pts[i, 1] + 0.5)
        for dr in range(-ir, ir + 1):
            for dc in range(-ir, ir + 1):
                if dr * dr + dc * dc <= rad * rad:
                    rr = r0 + dr
                    cc = c0 + dc
                    if rr >= 0 and rr < h and cc >= 0 and cc < w:
                        blk[rr, cc] = True


@njit(cache=True)
def trace_all(D, Dn, Dc, sr, sc, P, maxl, maxp):
    h, w = D.shape
    cov = np.zeros((h, w), np.int32)
    blk = np.zeros((h, w), np.bool_)
    pts = np.zeros((maxp, 2), np.float32)
    pv = np.zeros(maxp, np.float32)
    vline = -np.ones(maxp, np.int32)
    vth = np.zeros(maxp, np.float32)
    lines = np.zeros((maxl, 8), np.int32)
    n = 0
    nl = 0
    r_cov = P[6]
    r_seed = P[8]
    for s in range(sr.shape[0]):
        if nl >= maxl or n > maxp - 400:
            break
        if blk[sr[s], sc[s]]:
            continue
        r = float(sr[s])
        c = float(sc[s])
        th0 = _init_heading(Dc, r, c, P[1])
        fs = n
        pts[n, 0] = r
        pts[n, 1] = c
        pv[n] = D[sr[s], sc[s]]
        vline[n] = nl
        vth[n] = th0
        n += 1
        n, rf, snf = _trace(D, Dn, Dc, r, c, th0, nl, P, pts, pv, vline, vth, n, cov, maxp)
        fn = n - fs
        bs = n
        n, rb, snb = _trace(D, Dn, Dc, r, c, th0 + math.pi, nl, P, pts, pv, vline, vth, n, cov, maxp)
        bn = n - bs
        lines[nl, 0] = fs
        lines[nl, 1] = fn
        lines[nl, 2] = bs
        lines[nl, 3] = bn
        lines[nl, 4] = snf
        lines[nl, 5] = snb
        lines[nl, 6] = rf
        lines[nl, 7] = rb
        _stamp(cov, pts, fs, fs + fn, r_cov)
        _stamp(cov, pts, bs, bs + bn, r_cov)
        _block(blk, pts, fs, fs + fn, r_seed)
        _block(blk, pts, bs, bs + bn, r_seed)
        nl += 1
    return pts, pv, vline, vth, lines, nl, n


def _pack(p):
    return np.array([p["t_on"], p["step"], math.radians(p["max_turn"]), p["n_cand"], p["lam"], p["gap_max"], p["r_cov"], math.sin(math.radians(p["merge_deg"])), p["r_seed"], p["ign"], p["max_steps"], p["margin"], p["sig_w"], p["center_px"]], np.float64)


def seeds_from(Dc, Dn, D, t_seed, t_on, win=5):
    mn = ndi.minimum_filter(Dc, size=win, mode="nearest")
    m = (Dc <= mn + 1e-9) & (Dn <= t_seed) & (D <= t_on)
    rr, cc = np.nonzero(m)
    o = np.argsort(Dc[rr, cc], kind="stable")
    return rr[o].astype(np.int64), cc[o].astype(np.int64)


def smooth_poly(a, passes):
    for _ in range(passes):
        if len(a) < 3:
            break
        b = a.copy()
        b[1:-1] = 0.25 * a[:-2] + 0.5 * a[1:-1] + 0.25 * a[2:]
        a = b
    return a


def poly_len_m(a):
    return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum() * PX) if len(a) > 1 else 0.0


def drop_duplicates(polys, meta, dup_m, frac):
    """Kuerzere Linien verwerfen, die zu >= frac ihrer Laenge innerhalb dup_m Metern einer laengeren (schon behaltenen) Linie liegen."""
    from scipy.spatial import cKDTree

    order = np.argsort([-m["length"] for m in meta], kind="stable")
    kept, pts = [], np.zeros((0, 2))
    for k in order:
        a = resample(polys[k], 1.0)
        if len(pts) and (cKDTree(pts).query(a)[0] * PX <= dup_m).mean() >= frac:
            continue
        kept.append(k)
        pts = np.vstack([pts, a])
    kept.sort()
    return [polys[k] for k in kept], [meta[k] for k in kept]


def trace_field(D, params=None, want_debug=False):
    """D: Feld in Metern [200,200] (float). -> dict(polys=[(n,2) Pixel-Koordinaten], info)."""
    p = dict(DEFAULT)
    if params:
        p.update(params)
    D = np.ascontiguousarray(D, np.float32)
    Dn = ndi.gaussian_filter(D, p["sig_nav"], mode="nearest") if p["sig_nav"] > 0 else D.copy()
    if p["beta"] > 0:
        edt = ndi.distance_transform_edt(D <= p["t_blob"])  # halbe Talbreite in Pixeln (0 ausserhalb des Blobs)
        Dc = (Dn + p["beta"] * (p["cap_px"] - np.minimum(edt, p["cap_px"]))).astype(np.float32)
    else:
        Dc = Dn
    sr, sc = seeds_from(Dc, Dn, D, p["t_seed"], p["t_on"])
    P = _pack(p)
    pts, pv, vline, vth, lines, nl, n = trace_all(D, Dn, np.ascontiguousarray(Dc, np.float32), sr, sc, P, 400, 12000)
    polys, meta = [], []
    for l in range(nl):
        fs, fn, bs, bn, snf, snb, rf, rb = lines[l]
        a = np.concatenate([pts[bs : bs + bn][::-1], pts[fs : fs + fn]]).astype(np.float64)
        sn_b, sn_f = snb >= 0, snf >= 0
        if sn_b:
            a = np.vstack([pts[snb][None].astype(np.float64), a])
        if sn_f:
            a = np.vstack([a, pts[snf][None].astype(np.float64)])
        L = poly_len_m(a)
        if L < p["min_len"] and not (sn_b and sn_f and L >= 2.0):
            continue
        a = smooth_poly(a, p["smooth"])
        polys.append(a)
        meta.append(dict(length=L, snap_start=sn_b, snap_end=sn_f, reason_start=int(rb), reason_end=int(rf), seed=(int(round(pts[fs, 0])), int(round(pts[fs, 1]))),
                         id=int(l), snap_start_to=((int(vline[snb]), (float(pts[snb][0]), float(pts[snb][1]))) if sn_b else None),
                         snap_end_to=((int(vline[snf]), (float(pts[snf][0]), float(pts[snf][1]))) if sn_f else None)))
    if p["dup_m"] > 0 and len(polys) > 1:
        polys, meta = drop_duplicates(polys, meta, p["dup_m"], p["dup_frac"])
    info = dict(n_seeds=int(len(sr)), n_traced=int(nl), n_kept=len(polys), meta=meta)
    if want_debug:
        info["seeds"] = (sr, sc)
    return dict(polys=polys, info=info)


# ---------------------------------------------------------------- Umrechnung, Ausduennen, Rasterung


def px_to_m(a):
    return np.stack([-HALF + (a[:, 0] + 0.5) * PX, -HALF + (a[:, 1] + 0.5) * PX], 1)


def resample(a, ds_px):
    """Polylinie (Pixel) in gleichen Bogenlaengenabstand ds (Pixel) bringen; Endpunkt bleibt erhalten."""
    if len(a) < 2:
        return a
    seg = np.linalg.norm(np.diff(a, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    if L < 1e-6:
        return a[:1]
    k = max(int(round(L / ds_px)), 1)
    q = np.linspace(0, L, k + 1)
    return np.stack([np.interp(q, s, a[:, 0]), np.interp(q, s, a[:, 1])], 1)


def polys_points_m(polys, ds_m=0.5, clip=48.0):
    """Polylinien (Pixel) -> dichte Punkte in Metern (Abstand ds_m), auf |x|,|y| < clip beschnitten (wie in der Auswertung des Skeletts)."""
    out = [px_to_m(resample(a, ds_m / PX)) for a in polys if len(a) >= 2]
    if not out:
        return np.zeros((0, 2), np.float32)
    P = np.concatenate(out)
    return P[(np.abs(P) < clip).all(1)].astype(np.float32)


def sparse_nodes(polys, node_m=3.0):
    """Ausgeduennte Knoten (Pixel) je Polylinie im Abstand node_m (Endpunkte bleiben)."""
    return [resample(a, node_m / PX) for a in polys if len(a) >= 2]


def raster(polys, shape=(200, 200)):
    import cv2

    img = np.zeros(shape, np.uint8)
    for a in polys:
        if len(a) < 2:
            continue
        q = np.round(a[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)  # cv2: (x = Spalte, y = Zeile)
        cv2.polylines(img, [q], False, 1, 1)
    return img.astype(bool)
