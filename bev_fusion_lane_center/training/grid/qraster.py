"""Pixel-stage targets of the lane head (E273, card sections 2.1 and 3.1) drawn from the lane GT of autogt (`autogt.lane_gt`, frame "lane": x right, y forward, window |x|, |y| <= 50 m).
Only vectors are stored; these rasters are made in the data loader, per frame, on the CPU.

    t = pixel_targets(g)               # g = one lane GT graph (dict) -> dict of the targets below + ms (total) and ms_parts (per target)
    mask, dist = lines05(g)            # (3, 200, 200) 0.5 m: lane divider, road divider, boundary; mask = the cells a polyline passes through, dist = metres from the cell centre to the nearest line, truncated
    g05, t05 = centre_points(g)        # (G, 2) every point of every piece inside |x|, |y| < 48 and its travel tangent: the format of `data.GridData`, so `losses.grid_loss` takes them unchanged
    heat = events1(g)                  # (6, 100, 100) 1 m: Gaussian heat (peak 1) of split, merge, stop, exit, start, end (cross = split + merge; window events are not drawn)
    road2 = areas2(g)                  # (2, 50, 50) 2 m: road = the cell is covered >= 0.5 by the drivable polygons (4 x 4 sub-cells), intersection = any part of the cell is covered

Frame of every map: row = x, column = y, cell (i, j) has its centre at -n res / 2 + res (i + 0.5) (as `qtargets` and the feature raster of the project).
Empty lists, polylines of one point and degenerate polygons give zeros (dist = trunc), no error. `dilate` (m) grows the area polygons: a loader parameter, default 0 (the GT is not dilated).
Run the self-test: python qraster.py     (real frames of the lane GT store: `test_qraster.py`)
"""
import time

import numpy as np
import shapely
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

import qtargets as QT

LINES = ("lane_divider", "road_divider", "boundary")
EVENTS = ("split", "merge", "stop", "exit", "start", "end")
EVENT_OF = {t: (i,) for i, t in enumerate(EVENTS)}
EVENT_OF["cross"] = (0, 1)                                         # a node that is both a split and a merge


def _polylines(items):
    """-> list of (n, 2) float arrays, n >= 2"""
    out = [np.asarray(it["pts"], float).reshape(-1, 2) for it in items]
    return [p for p in out if len(p) >= 2]


def _cells_on(polys, n, res):
    """bool (n, n): the cells that the polylines pass through. Exact: a segment is cut at every grid line it crosses and each piece belongs to the cell of its midpoint
    (a cell met only in a corner or along an edge of its own is not counted)"""
    u = np.concatenate([p[:-1] for p in polys]) / res + n / 2              # segment ends in cell units
    v = np.concatenate([p[1:] for p in polys]) / res + n / 2
    d = v - u
    S = len(u)
    sid, tt = [np.arange(S), np.arange(S)], [np.zeros(S), np.ones(S)]
    for ax in (0, 1):                                                      # the crossings with the grid lines k = lo .. hi of this axis
        lo = np.floor(np.minimum(u[:, ax], v[:, ax])).astype(int) + 1
        cnt = np.maximum(np.floor(np.maximum(u[:, ax], v[:, ax])).astype(int) - lo + 1, 0)
        s = np.repeat(np.arange(S), cnt)
        k = lo[s] + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        sid.append(s)
        tt.append((k - u[s, ax]) / d[s, ax])
    sid, tt = np.concatenate(sid), np.concatenate(tt)
    o = np.lexsort((tt, sid))
    sid, tt = sid[o], tt[o]
    ok = (sid[1:] == sid[:-1]) & (tt[1:] - tt[:-1] > 1e-9)                 # consecutive crossings of one segment; zero-length pieces (a corner crossing) drop out
    s, t = sid[:-1][ok], 0.5 * (tt[1:] + tt[:-1])[ok]
    c = np.floor(u[s] + t[:, None] * d[s]).astype(int)
    c = c[((c >= 0) & (c < n)).all(1)]
    m = np.zeros((n, n), bool)
    m[c[:, 0], c[:, 1]] = True
    return m


def _samples(polys, step):
    """points every `step` m along the polylines (every vertex and every end included)"""
    a, b = np.concatenate([p[:-1] for p in polys]), np.concatenate([p[1:] for p in polys])
    k = np.maximum(np.ceil(np.linalg.norm(b - a, axis=1) / step).astype(int), 1)
    s = np.repeat(np.arange(len(a)), k)
    f = (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)) / k[s]
    return np.concatenate([a[s] + f[:, None] * (b - a)[s], [p[-1] for p in polys]])


def _distance(pts, n, res, trunc):
    """(n, n) float32: distance from each cell centre to the nearest of `pts`, truncated at `trunc`. Only the cells that can hold less than trunc are queried
    (the dilated cells of the points: a centre within trunc of a point lies at most ceil(trunc / res) cells from the point's cell)"""
    c = -n * res / 2 + res * (np.arange(n) + 0.5)
    near = np.zeros((n, n), bool)
    cell = np.clip(np.floor(pts / res + n / 2).astype(int), 0, n - 1)
    near[cell[:, 0], cell[:, 1]] = True
    i, j = np.nonzero(maximum_filter(near, size=2 * int(np.ceil(trunc / res)) + 1, mode="constant"))
    d = cKDTree(pts).query(np.stack([c[i], c[j]], 1), distance_upper_bound=trunc)[0]       # inf where nothing lies within trunc
    out = np.full((n, n), trunc, np.float32)
    out[i, j] = np.minimum(d, trunc)
    return out


def lines05(g, n=200, res=0.5, trunc=2.0, step=0.1):
    """-> mask (3, n, n) float32, dist (3, n, n) float32 (classes: lane_divider, road_divider, boundary; nearest point of the polylines sampled every `step` m; a class without lines: mask 0, dist = trunc)"""
    mask = np.zeros((len(LINES), n, n), np.float32)
    dist = np.full((len(LINES), n, n), trunc, np.float32)
    for c, name in enumerate(LINES):
        polys = _polylines(g.get("lines", {}).get(name, []))
        if polys:
            mask[c] = _cells_on(polys, n, res)
            dist[c] = _distance(_samples(polys, step), n, res, trunc)
    return mask, dist


def centre_points(g, box=48.0):
    """-> g05 (G, 2), t05 (G, 2) float32: all points of all pieces (every 0.5 m, joins twice) with |x|, |y| < box and their unit travel tangents (central differences), as `GridData`"""
    P, T = [], []
    for p in g.get("pieces", []):
        q = np.asarray(p["pts"], float).reshape(-1, 2)
        if len(q) >= 2:
            P.append(q)
            T.append(QT._tangents(q))
    if not P:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    P, T = np.concatenate(P), np.concatenate(T)
    k = (np.abs(P) < box).all(1)
    return P[k].astype(np.float32), T[k].astype(np.float32)


def events1(g, n=100, res=1.0, sigma=1.5):
    """-> (6, n, n) float32 Gaussian heat maps exp(-d^2 / 2 sigma^2) of the event positions per type (overlapping events of one channel: the maximum, so the peak stays 1)"""
    pos = [[] for _ in EVENTS]
    for e in g.get("events", []):
        for c in EVENT_OF.get(e["type"], ()):                              # window events: not in EVENT_OF
            pos[c].append(e["pos"])
    c = -n * res / 2 + res * (np.arange(n) + 0.5)
    out = np.zeros((len(EVENTS), n, n), np.float32)
    for k, p in enumerate(pos):
        if p:
            p = np.asarray(p, float).reshape(-1, 2)
            d2 = ((c[:, None, None] - p[None, None, :, 0]) ** 2 + (c[None, :, None] - p[None, None, :, 1]) ** 2).min(2)
            out[k] = np.exp(-d2 / (2 * sigma ** 2))
    return out


def _polygons(areas, dilate):
    """shapely polygons (exterior minus holes, optionally grown by `dilate` m) of a list of GT area dicts; broken or empty ones are left out"""
    out = []
    for a in areas:
        ext = np.asarray(a["exterior"], float).reshape(-1, 2)
        if len(ext) < 3:
            continue
        poly = shapely.Polygon(ext, [h for h in (np.asarray(h, float).reshape(-1, 2) for h in a.get("holes", [])) if len(h) >= 3])
        if not poly.is_valid:
            poly = shapely.make_valid(poly)
        if dilate > 0:
            poly = poly.buffer(dilate)
        out += [q for q in shapely.get_parts(poly) if q.geom_type == "Polygon" and q.area > 0]
    return out


def _coverage(polys, n, res, ss):
    """(n, n) share of each cell that lies inside the polygons, from ss x ss sub-cell centres (shapely.contains_xy)"""
    s = -n * res / 2 + (np.arange(n * ss) + 0.5) * (res / ss)
    inside = np.zeros((n * ss, n * ss), bool)
    for p in polys:
        x0, y0, x1, y1 = p.bounds
        i0, i1, j0, j1 = np.searchsorted(s, x0), np.searchsorted(s, x1), np.searchsorted(s, y0), np.searchsorted(s, y1)
        if i1 > i0 and j1 > j0:
            shapely.prepare(p)                                              # 15 x faster than the unprepared test
            X, Y = np.meshgrid(s[i0:i1], s[j0:j1], indexing="ij")
            inside[i0:i1, j0:j1] |= shapely.contains_xy(p, X, Y)
    return inside.reshape(n, ss, n, ss).mean((1, 3))


def _touched(polys, n, res):
    """(n, n) bool: the cell shares area with a polygon (even 1 cm of it); a cell met only along an edge or in a corner does not count"""
    m = np.zeros((n, n), bool)
    for p in polys:
        lo = np.floor((np.array(p.bounds[:2]) + n * res / 2) / res).astype(int).clip(0, n - 1)
        hi = np.floor((np.array(p.bounds[2:]) + n * res / 2) / res).astype(int).clip(0, n - 1)
        i, j = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
        i, j = i.ravel(), j.ravel()
        x, y = -n * res / 2 + res * i, -n * res / 2 + res * j
        box = shapely.box(x, y, x + res, y + res)
        shapely.prepare(p)
        hit = shapely.intersects(p, box)
        hit[hit] = shapely.relate_pattern(p, box[hit], "T********")        # the interiors meet
        m[i[hit], j[hit]] = True
    return m


def areas2(g, n=50, res=2.0, dilate=0.0, ss=4):
    """-> (2, n, n) float32: channel 0 road (cell covered >= 0.5 by the drivable polygons, ss x ss sub-cells), channel 1 intersection (any part of the cell covered)"""
    a = g.get("areas", {})
    out = np.zeros((2, n, n), np.float32)
    road, inter = _polygons(a.get("road", []), dilate), _polygons(a.get("intersection", []), dilate)
    if road:
        out[0] = _coverage(road, n, res, ss) >= 0.5
    if inter:
        out[1] = _touched(inter, n, res)
    return out


def pixel_targets(g, dilate=0.0):
    """all targets of the pixel stage: lines_mask, lines_dist, g05, t05, events1, road2; ms = time in ms (total), ms_parts = per target"""
    parts = {}

    def timed(name, f, *a, **k):
        t0 = time.perf_counter()
        r = f(*a, **k)
        parts[name] = 1e3 * (time.perf_counter() - t0)
        return r

    mask, dist = timed("lines", lines05, g)
    g05, t05 = timed("centre", centre_points, g)
    ev = timed("events", events1, g)
    road = timed("areas", areas2, g, dilate=dilate)
    return dict(lines_mask=mask, lines_dist=dist, g05=g05, t05=t05, events1=ev, road2=road, ms=sum(parts.values()), ms_parts=parts)


def _selftest():
    sq = lambda x0, y0, x1, y1: np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    line = lambda *p: dict(pts=np.array(p, np.float32), start_open=False, end_open=False, corners=[])
    # a straight divider at x = 3.4 from y = -20 to 20: the cells are row 106 (x in [3.0, 3.5]), columns 60 .. 139
    g = dict(lines=dict(lane_divider=[line((3.4, -20.0), (3.4, 20.0))], road_divider=[line((-10.0, -10.0), (10.0, 10.0))], boundary=[]), pieces=[], events=[], areas={})
    mask, dist = lines05(g)
    assert mask.shape == dist.shape == (3, 200, 200) and mask.dtype == dist.dtype == np.float32
    assert set(zip(*np.nonzero(mask[0]))) == {(106, j) for j in range(60, 140)}, "divider cells"
    assert set(zip(*np.nonzero(mask[1]))) == {(80 + k, 80 + k) for k in range(40)}, "a diagonal through the cell corners passes only the diagonal cells"
    assert mask[2].sum() == 0 and (dist[2] == 2.0).all(), "a class without lines: mask 0, dist = trunc"
    xc = -50 + 0.5 * (np.arange(200) + 0.5)
    ref = np.minimum(np.abs(xc[:, None] - 3.4), 2.0) * np.ones((1, 200))                     # exact distance to the infinite line, valid where the nearest line point is a foot point
    inner = (np.abs(xc[None, :]) < 19.0) * np.ones((200, 1), bool)
    err = np.abs(dist[0] - ref)[inner]
    assert err.max() < 0.0085 and (dist[0][inner] >= ref[inner] - 1e-6).all(), err.max()   # sampled every 0.1 m: at most sqrt(0.15^2 + 0.05^2) - 0.15 for the nearest centres
    assert abs(dist[0, 106, 140] - np.hypot(0.15, 0.25)) < 1e-5 and dist[0, 0, 100] == 2.0, "beyond the end: the distance to the end point (y = 20); far away: trunc"
    # random segments against a dense walk along them
    rng = np.random.RandomState(0)
    for _ in range(30):
        a, b = rng.uniform(-45, 45, (2, 2))
        w = a + np.linspace(0, 1, int(np.linalg.norm(b - a) / 1e-4) + 2)[:, None] * (b - a)
        ref = np.zeros((200, 200), bool)
        c = np.floor((w + 50) / 0.5).astype(int)
        ref[c[:, 0], c[:, 1]] = True
        assert (_cells_on([np.stack([a, b])], 200, 0.5) == ref).all(), (a, b)
    # events: a split on a cell centre has the peak 1, a cross is a split and a merge, a window event is not drawn, two peaks of one channel do not add up
    ev = lambda t, x, y: dict(type=t, pos=[x, y], **{"in": [], "out": []})
    g["events"] = [ev("split", 0.5, 0.5), ev("split", 1.5, 0.5), ev("cross", 10.5, 10.5), ev("window", -20.5, 20.5), ev("exit", -20.5, 20.5), ev("stop", 30.5, -30.5)]
    h = events1(g)
    assert h.shape == (6, 100, 100) and abs(h[0, 50, 50] - 1.0) < 1e-6 and abs(h[0, 51, 50] - 1.0) < 1e-6 and abs(h[0, 50, 51] - np.exp(-1 / 4.5)) < 1e-6, "peak 1, maximum not sum"
    assert abs(h[0, 60, 60] - 1) < 1e-6 and abs(h[1, 60, 60] - 1) < 1e-6 and abs(h[3, 29, 70] - 1) < 1e-6 and abs(h[2, 80, 19] - 1) < 1e-6
    assert h[:, 29, 70].sum() == 1.0 and h.max() == 1.0, "the window event adds no peak"
    # areas: a road square [0.5, 9]^2 with the hole [2, 6]^2 against the exact cell coverage (edges on the 0.5 m sub-cell grid, so 4 x 4 sub-cells are exact); the intersection square [1.99, 6] x [0, 6] touches the column x in [0, 2] by 1 cm
    cov = lambda lo, hi: np.clip(np.minimum(hi, 2 * np.arange(-25, 25) + 2) - np.maximum(lo, 2 * np.arange(-25, 25)), 0, 2)
    exp = ((cov(0.5, 9)[:, None] * cov(0.5, 9)[None] - cov(2, 6)[:, None] * cov(2, 6)[None]) / 4 >= 0.5)
    road = dict(exterior=sq(0.5, 0.5, 9, 9), holes=[sq(2, 2, 6, 6)], area=56.0)
    g["areas"] = dict(road=[road], intersection=[dict(exterior=sq(1.99, 0, 6, 6), holes=[], area=24.0)])
    r2 = areas2(g)
    assert r2.shape == (2, 50, 50) and (r2[0] == exp).all() and exp.sum() == 18 and r2[0, 26, 26] == 0 and r2[0, 29, 29] == 0 and r2[0, 25, 25] == 1
    assert set(zip(*np.nonzero(r2[1]))) == {(i, j) for i in (25, 26, 27) for j in (25, 26, 27)}, "any coverage, 1 cm counts"
    g["areas"]["intersection"] = [dict(exterior=sq(2, 0, 6, 6), holes=[], area=24.0)]
    assert set(zip(*np.nonzero(areas2(g)[1]))) == {(i, j) for i in (26, 27) for j in (25, 26, 27)}, "a polygon edge on a cell edge does not mark the neighbour"
    r3 = areas2(g, dilate=2.0)
    assert r3[1].sum() > areas2(g)[1].sum() and r3[0].sum() > r2[0].sum(), "dilate grows the polygons"
    # centre points: the format of GridData (float32, unit tangents), pieces of one point and points beyond the box are left out
    pc = lambda pts: dict(pts=np.array(pts, np.float32))
    g["pieces"] = [pc([(0, y) for y in np.arange(-10, 10.1, 0.5)]), pc([(1.0, 1.0)]), pc([(48.0, 0.0), (49.0, 0.0)]), pc([(40.0, y) for y in np.arange(46.0, 50.1, 0.5)])]
    p, t = centre_points(g)
    assert p.dtype == t.dtype == np.float32 and len(p) == 41 + 4 + 0 and np.allclose(np.linalg.norm(t, axis=1), 1) and np.allclose(t[:41], [0, 1]), (len(p),)
    # empty and degenerate graphs
    e = pixel_targets(dict(lines=dict(lane_divider=[line((1.0, 1.0))], road_divider=[], boundary=[]), pieces=[], events=[],
                           areas=dict(road=[dict(exterior=np.zeros((2, 2)), holes=[], area=0.0), dict(exterior=sq(0, 0, 0, 0), holes=[], area=0.0)], intersection=[])))
    assert not e["lines_mask"].any() and (e["lines_dist"] == 2.0).all() and not e["events1"].any() and not e["road2"].any() and e["g05"].shape == (0, 2)
    print("qraster selftest ok")


if __name__ == "__main__":
    _selftest()
