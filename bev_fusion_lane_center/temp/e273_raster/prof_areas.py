"""scratch: where do the 80 ms of areas2 go"""
import os, sys, time
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "training", "grid"))
import qraster as QR
from test_qtargets import frames
fr = frames(40)
tm = dict(poly=0, cov=0, touch=0)
nv = []
for g in fr:
    t = time.perf_counter(); road = QR._polygons(g["areas"]["road"], 0); inter = QR._polygons(g["areas"]["intersection"], 0); tm["poly"] += time.perf_counter() - t
    t = time.perf_counter(); QR._coverage(road, 50, 2.0, 4); tm["cov"] += time.perf_counter() - t
    t = time.perf_counter(); QR._touched(inter, 50, 2.0); tm["touch"] += time.perf_counter() - t
    nv.append((len(road), len(inter), sum(len(p.exterior.coords) for p in road)))
print({k: round(1e3 * v / len(fr), 1) for k, v in tm.items()}, nv[:8])
