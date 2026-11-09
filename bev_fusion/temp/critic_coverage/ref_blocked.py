"""Brute-force reference for coverage.blocked: shapely exact 2D clip + z interval, and dense sampling."""
import sys, math
import numpy as np
from shapely.geometry import LineString, Polygon, Point
from autogt import coverage as C

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
NCASE = int(sys.argv[2]) if len(sys.argv) > 2 else 4000


def rand_box(centre_range, cam):
    cx = rng.uniform(*centre_range[0]); cy = rng.uniform(*centre_range[1])
    yaw = rng.uniform(-math.pi, math.pi)
    l = rng.choice([rng.uniform(0.4, 1.0), rng.uniform(3.5, 5.0), rng.uniform(8, 13)])
    w = rng.uniform(0.4, 3.0)
    loc = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    c, s = math.cos(yaw), math.sin(yaw)
    fp = loc @ np.array([[c, s], [-s, c]]) + [cx, cy]
    if rng.random() < 0.5:
        fp = fp[::-1].copy()          # other orientation
    z_lo = rng.uniform(-1.0, 1.0); h = rng.uniform(0.3, 4.0)
    return fp, z_lo, z_lo + h


def ref_exact(cam, p, prism):
    """exact: 2D clip with shapely, then z interval. Returns (blocked, margin) margin = size of the decision (for tolerance)."""
    fp, z_lo, z_hi = prism
    poly = Polygon(fp)
    if poly.contains(Point(p[:2])) or poly.touches(Point(p[:2])):
        return False, 0.0           # the end point is inside / on the footprint: not hidden by this box
    d = p - cam
    seg = LineString([cam[:2], p[:2]])
    inter = seg.intersection(poly)
    if inter.is_empty:
        return False, 1.0
    if inter.geom_type == "Point":
        return False, 0.0
    pts = []
    geoms = [inter] if inter.geom_type == "LineString" else list(inter.geoms)
    L2 = float(d[:2] @ d[:2])
    ts = []
    for gm in geoms:
        if gm.geom_type != "LineString":
            continue
        for q in gm.coords:
            ts.append(((np.array(q) - cam[:2]) @ d[:2]) / L2)
    t0, t1 = min(ts), max(ts)
    if t1 - t0 < 1e-9:
        return False, 0.0
    za, zb = cam[2] + t0 * d[2], cam[2] + t1 * d[2]
    lo, hi = min(za, zb), max(za, zb)
    ov = min(hi, z_hi) - max(lo, z_lo)
    return ov >= 0, ov


def run(kind):
    bad_pre = bad_ref = n_blk = 0
    worst = []
    for it in range(NCASE):
        if kind == "real_cam":
            cam = np.array([1.5, 0.0, 1.6])
        else:
            cam = np.array([rng.uniform(-3, 8), rng.uniform(-4, 4), rng.uniform(1.0, 2.0)])
        prism = rand_box(((-5, 60), (-26, 26)), cam)
        if kind == "near":     # boxes close to the camera, big ones
            prism = rand_box(((cam[0] - 6, cam[0] + 14), (-5, 5)), cam)
        P = np.c_[rng.uniform(max(cam[0], 0), 60, 40), rng.uniform(-26, 26, 40), np.where(rng.random(40) < 0.5, 0.0, rng.uniform(0, 2, 40))]
        a = C.blocked(P, cam, [prism])
        # without prefilter
        old = C._wedge
        C._wedge = lambda c, p, f: np.ones(len(p), bool)
        b = C.blocked(P, cam, [prism])
        C._wedge = old
        for k in range(len(P)):
            r, m = ref_exact(cam, P[k], prism)
            n_blk += bool(r)
            if a[k] != b[k]:
                bad_pre += 1
                worst.append(("prefilter", it, k, bool(a[k]), bool(b[k]), bool(r), cam.tolist(), P[k].tolist(), [prism[0].tolist(), prism[1], prism[2]]))
            if b[k] != r and abs(m) > 1e-6 and m != 0.0 and m != 1.0:
                bad_ref += 1
                worst.append(("nofilter-vs-ref", it, k, bool(b[k]), bool(r), m, cam.tolist(), P[k].tolist(), [prism[0].tolist(), prism[1], prism[2]]))
            elif b[k] != r and (m == 1.0 or m == 0.0 or abs(m) > 1e-6) and False:
                pass
    return bad_pre, bad_ref, n_blk, worst


for kind in ("real_cam", "random_cam", "near"):
    bp, br, nb, w = run(kind)
    print(kind, "prefilter-vs-full mismatches:", bp, "full-vs-reference mismatches:", br, "blocked cases:", nb)
    for x in w[:3]:
        print("   ", x)
