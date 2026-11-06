"""how are the GT centre-line directions distributed (ego frame, x forward)? angle = atan2(dir[1], dir[0])"""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch")]
from evaluation import bev_gtcheck as GC
samples = GC.pick(8, "15efbde2d1", 0)
allang, allx = [], []
for name, s in samples:
    g = s[0.0]
    G = np.asarray(g["centreline"], np.float32) > 0.5
    d = np.asarray(g["centreline_dir"], np.float32)
    ok = G & (np.hypot(d[0], d[1]) > 0.5)
    ang = np.degrees(np.arctan2(d[1], d[0]))[ok]
    allang.append(ang)
    print(name[:8], int(G.sum()), int(ok.sum()), "mask values", np.unique(np.asarray(g["centreline"], np.float32)).tolist()[:4],
          "ang hist 0/90/180/-90:", [int(((ang > lo) & (ang <= hi)).sum()) for lo, hi in ((-45, 45), (45, 135), (135, 180), (-180, -135))],
          "unit len", float(np.hypot(d[0], d[1])[ok].min()), float(np.hypot(d[0], d[1])[ok].max()))
a = np.concatenate(allang)
print("all:", len(a), "share |angle|<45:", float((np.abs(a) < 45).mean()), "share >135:", float((np.abs(a) > 135).mean()), "mean |angle| deg", float(np.abs(a).mean()))
