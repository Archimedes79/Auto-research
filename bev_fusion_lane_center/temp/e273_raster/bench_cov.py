"""scratch: contains_xy variants"""
import os, sys, time
import numpy as np, shapely
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "training", "grid"))
import qraster as QR
from test_qtargets import frames
fr = frames(30)
n, res, ss = 50, 2.0, 4
s = -50 + (np.arange(n * ss) + 0.5) * (res / ss)
X, Y = np.meshgrid(s, s, indexing="ij")
def T(f):
    t = time.perf_counter(); r = [f(g) for g in fr]; return 1e3 * (time.perf_counter() - t) / len(fr), r
polys = [QR._polygons(g["areas"]["road"], 0) for g in fr]
def a(i):
    return np.logical_or.reduce([shapely.contains_xy(p, X, Y) for p in polys[i]])
def b(i):
    out = []
    for p in polys[i]:
        shapely.prepare(p); out.append(shapely.contains_xy(p, X, Y))
    return np.logical_or.reduce(out)
def c(i):
    out = []
    for p in polys[i]:
        shapely.prepare(p); out.append(shapely.contains(p, shapely.points(X.ravel(), Y.ravel())).reshape(X.shape))
    return np.logical_or.reduce(out)
for name, f in (("xy", a), ("prep+xy", b), ("prep+points", c)):
    t = time.perf_counter(); r = [f(i) for i in range(len(fr))]; print(name, round(1e3 * (time.perf_counter() - t) / len(fr), 1), "ms")
    if name == "xy": ref = r
    else: print("  same as xy:", all((x == y).all() for x, y in zip(r, ref)))
print("verts", [sum(len(p.exterior.coords) + sum(len(h.coords) for h in p.interiors) for p in ps) for ps in polys[:10]])
# box-based classification
def d(i):
    out = np.zeros((n, n, ss, ss), bool)
    f = (np.arange(ss) + 0.5) * (res / ss)
    for p in polys[i]:
        lo = np.floor((np.array(p.bounds[:2]) + 50) / res).astype(int).clip(0, n - 1); hi = np.floor((np.array(p.bounds[2:]) + 50) / res).astype(int).clip(0, n - 1)
        I, J = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij"); I, J = I.ravel(), J.ravel()
        x, y = -50 + res * I, -50 + res * J
        box = shapely.box(x, y, x + res, y + res)
        shapely.prepare(p)
        full = shapely.contains(p, box)
        part = shapely.intersects(p, box) & ~full
        out[I[full], J[full]] = True
        I, J = I[part], J[part]
        xs = (-50 + res * I)[:, None, None] + f[None, :, None] + 0 * f[None, None, :]
        ys = (-50 + res * J)[:, None, None] + f[None, None, :] + 0 * f[None, :, None]
        out[I, J] |= shapely.contains_xy(p, xs, ys)
    return out.transpose(0, 2, 1, 3).reshape(n * ss, n * ss)
t = time.perf_counter(); r = [d(i) for i in range(len(fr))]; print("boxes", round(1e3 * (time.perf_counter() - t) / len(fr), 1), "ms")
print("  same as xy:", all((x == y).all() for x, y in zip(r, ref)), [int((x != y).sum()) for x, y in zip(r, ref)][:10])
