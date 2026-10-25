"""Loop 7 / E77: SD-map simulator with ADASIS-like quality, derived from the lane-level map graph of a sample.

A standard-definition (SD) navigation map knows ROADS (one centre line per carriageway, sparse shape points, a lane-count attribute, a route = most probable path),
not lane centre lines. build_sd() turns the lane graph of a sample into that representation:
  * lane pieces that run side by side in the same direction are merged into one carriageway; its centre line is the mean of the lane centre lines (lane count = number of merged lanes);
  * the SD route is the centred path of the lane pieces of the driven route (the lane choice is lost, as in a real navigation route).
degrade_sd() adds the map-quality effects of a navigation map: shape-point sparsification (Douglas-Peucker with a tolerance of 0.5-2 m), lane-count errors, missing side roads.
The pose error of the map (translation + yaw) is applied afterwards in raster_sd() / by the caller, like for the HD map.
"""
import numpy as np
from scipy.spatial import cKDTree

import assoc_lib as AL

LAT = 5.5  # lanes whose centre lines are closer than this (laterally) and run in the same direction belong to one carriageway
DS = 2.0


def _par(qa, ta, treeb, qb, tb, lat=LAT):
    """for every point of a: index of the nearest point of b if b runs side by side with a (distance 1.2 .. lat, same direction, offset mostly lateral), else -1."""
    d, j = treeb.query(qa)
    v = qb[j] - qa
    lng = np.abs((v * ta).sum(1))
    ok = (d < lat) & (d > 1.2) & ((ta * tb[j]).sum(1) > 0.9) & (lng < 2.5)
    return np.where(ok, j, -1)


def build_sd(g, route_ids):
    """g: lane graph (pts, tokens, ...), route_ids: piece indices of the driven route (in driving order).
    -> dict(route=[(poly, n)], roads=[(poly, n)]) with poly [K,2] (m, vehicle frame) and n [K] the lane count at every vertex."""
    pcs = {}
    for k, p in enumerate(g["pts"]):
        if len(p) < 2:
            continue
        q = AL.resample(np.asarray(p, np.float64), DS)
        if len(q) >= 2:
            pcs[k] = (q, AL.tangents(q))
    ks = sorted(pcs)
    trees = {k: cKDTree(pcs[k][0]) for k in ks}
    par = {}  # par[(a, b)]: for the points of a, the parallel neighbour point index in b
    nb = {k: [] for k in ks}
    for a in ks:
        qa, ta = pcs[a]
        for b in ks:
            if a == b:
                continue
            qb, tb = pcs[b]
            if np.linalg.norm(qa.mean(0) - qb.mean(0)) > 60:
                continue
            j = _par(qa, ta, trees[b], qb, tb)
            if (j >= 0).sum() >= max(3, 0.3 * len(qa)):
                par[(a, b)] = j
                nb[a].append(b)

    def centred(a, nbrs):
        qa = pcs[a][0]
        acc = qa.copy()
        cnt = np.ones(len(qa))
        for b in nbrs:
            j = par.get((a, b))
            if j is None:
                continue
            m = j >= 0
            acc[m] += pcs[b][0][j[m]]
            cnt[m] += 1
        return acc / cnt[:, None], cnt

    # roads: longest piece first as reference; neighbours that run beside >= 70 % of their length are covered by it
    left = set(ks)
    roads = []
    for a in sorted(ks, key=lambda k: -len(pcs[k][0])):
        if a not in left:
            continue
        nbrs = [b for b in nb[a] if b in left]
        poly, cnt = centred(a, nbrs)
        roads.append((poly, cnt))
        left.discard(a)
        for b in nbrs:
            jb = par.get((b, a))
            if jb is not None and (jb >= 0).mean() >= 0.7:
                left.discard(b)
    route = []
    for a in route_ids:
        if a not in pcs:
            continue
        poly, cnt = centred(a, nb[a])
        route.append((poly, cnt))
    return dict(route=route, roads=roads)


def _dp(p, eps):
    """Douglas-Peucker; returns the indices of the kept vertices."""
    n = len(p)
    if n <= 2:
        return list(range(n))
    keep = [0, n - 1]
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = p[i], p[j]
        ab = b - a
        L = np.linalg.norm(ab)
        seg = p[i + 1 : j] - a
        d = np.abs(seg[:, 0] * ab[1] - seg[:, 1] * ab[0]) / L if L > 1e-9 else np.linalg.norm(seg, axis=1)
        m = int(np.argmax(d))
        if d[m] > eps:
            k = i + 1 + m
            keep.append(k)
            stack += [(i, k), (k, j)]
    return sorted(keep)


def sparsify(poly, cnt, eps):
    """navigation-map shape points: keep the Douglas-Peucker vertices (tolerance eps m); the lane count of the kept vertices is carried along."""
    idx = _dp(poly, eps)
    return poly[idx], cnt[idx]


def degrade_sd(sd, rng, eps_lo=0.5, eps_hi=2.0, p_cnt=0.1, drop=0.0):
    """ADASIS-like map quality: sparse shape points, lane-count errors (+-1 on a road with probability p_cnt), missing side roads (probability `drop` per road; the route is never dropped)."""
    eps = rng.uniform(eps_lo, eps_hi)
    out = dict(route=[], roads=[])
    for key in ("route", "roads"):
        for poly, cnt in sd[key]:
            if key == "roads" and rng.rand() < drop:
                continue
            p, c = sparsify(poly, cnt, eps)
            if rng.rand() < p_cnt:
                c = np.maximum(c + rng.choice([-1, 1]), 1)
            out[key].append((p, c))
    return out


def raster_sd(sd, R=None, d=None, n=200, half=50.0, ds=0.25):
    """SD map -> [5, n, n]: route, roads (incl. the route), direction dx, dy of the roads, lane count / 4 on the roads (row = x, column = y). (R, d): pose error x -> R x + d."""
    out = np.zeros((5, n, n), np.float32)
    for key, ch in (("roads", 1), ("route", 0)):
        for poly, cnt in sd[key]:
            p = np.asarray(poly, np.float64)
            if len(p) < 2:
                continue
            if R is not None:
                p = p @ R.T + d
            seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            if s[-1] < 1e-6:
                continue
            t = np.linspace(0.0, s[-1], max(int(np.ceil(s[-1] / ds)), 1) + 1)
            q = np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)
            nq = np.interp(t, s, np.asarray(cnt, np.float64))
            tg = AL.tangents(q)
            r = np.round((q[:, 0] + half) / (2 * half / n) - 0.5).astype(int)
            c = np.round((q[:, 1] + half) / (2 * half / n) - 0.5).astype(int)
            ok = (r >= 0) & (r < n) & (c >= 0) & (c < n)
            out[ch][r[ok], c[ok]] = 1.0
            if ch == 1:
                out[2][r[ok], c[ok]] = tg[ok, 0]
                out[3][r[ok], c[ok]] = tg[ok, 1]
                out[4][r[ok], c[ok]] = np.clip(nq[ok] / 4.0, 0, 1)
    return out


def as_graph(sd):
    """SD map as a pseudo lane graph (pieces + tokens) for the geometric assignment baseline: roads are the pieces, the route pieces carry the route tokens."""
    pieces, tokens, route_tok = [], [], set()
    for k, (poly, cnt) in enumerate(sd["roads"]):
        pieces.append(np.asarray(poly, np.float64))
        tokens.append(f"road{k}")
    for k, (poly, cnt) in enumerate(sd["route"]):
        pieces.append(np.asarray(poly, np.float64))
        tokens.append(f"route{k}")
        route_tok.add(f"route{k}")
    return dict(pieces=pieces, tokens=tokens), route_tok


# ------------------------------------------------------------------ pose error of a scene (moved here from the loop-11 script p11_route_time.py, 2026-10-24: live code must not import from temp/)
def scene_pose_error(scene, sigma):
    """scene-constant WORLD-frame pose error of the SD map: translation N(0, sigma^2) per axis (m) and yaw N(0, (2 sigma)^2) (deg); returns (w [2], theta_deg)"""
    import zlib

    rs = np.random.RandomState((zlib.crc32(f"scenebias{scene}".encode()) + 3) % (2 ** 32))
    return rs.randn(2) * sigma, rs.randn() * 2.0 * sigma


def sd_raster(sd, row, scene, sigma, P, wrong):
    """SD map raster [5, 200, 200] of a frame with the scene-constant world-frame pose error (ego frame: d = R_ego^T w, rotation theta about the ego origin)"""
    import zlib

    w, th = scene_pose_error(scene, sigma)
    d = P["R"][row].T @ w
    rng = np.random.RandomState((zlib.crc32(f"sd{scene}".encode()) + int(row)) % (2 ** 32))
    dg = degrade_sd(sd, rng)
    return raster_sd(dg, AL.rot2(th), d)
