import os, sys, time
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training", "training/grid", "training/data", "training/data/prep", "tools/viz", "evaluation", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_view as QV
import jgt as J
import junction_paint as JP
import qjunction as QJ
toks = []
for sid, (name, tk) in QV.val_scenes().items():
    toks += tk[:2]
    if len(toks) >= 24: break
t0 = time.time(); gs = [QV.load_gt(t) for t in toks]; t1 = time.time()
jgs = [J.junction_gt(g) for g in gs]; t2 = time.time()
ps = [JP.load(t) for t in toks]; t3 = time.time()
out = [QJ.junction_targets(g, p) for g, p in zip(gs, ps)]; t4 = time.time()
n = len(toks)
print(f"{n} frames: load_gt {1000*(t1-t0)/n:.1f} ms, junction_gt {1000*(t2-t1)/n:.1f} ms, paint load {1000*(t3-t2)/n:.1f} ms, junction_targets (incl. junction_gt again) {1000*(t4-t3)/n:.1f} ms")
