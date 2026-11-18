import os, sys, time
ROOT = os.environ["BFLC_ROOT"]
for sub_ in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub_))
import importlib, builtins
orig = builtins.__import__
depth = [0]
times = {}
def timed(name, *a, **k):
    t = time.time(); depth[0] += 1
    try:
        return orig(name, *a, **k)
    finally:
        depth[0] -= 1
        dt = time.time() - t
        if depth[0] <= 2 and dt > 0.5:
            print("  " * depth[0], name, round(dt, 2), flush=True)
builtins.__import__ = timed
t0 = time.time()
import lane_tok_data
print("total", time.time() - t0)
