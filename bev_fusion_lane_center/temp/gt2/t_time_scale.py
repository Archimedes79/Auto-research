import os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training", "training/grid", "training/data", "training/data/prep", "tools/viz", "evaluation", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
import qgraph_view as QV
import jgt as J
import qjunction as QJ
toks = []
for sid, (name, tk) in QV.val_scenes().items():
    toks += tk[:2]
    if len(toks) >= 24: break
gs = [QV.load_gt(t) for t in toks]
jgs = [J.junction_gt(g) for g in gs]
t0 = time.time()
for jg in jgs:
    QJ.fit_pair_scales(jg)
dt = (time.time() - t0) / len(jgs)
npairs = sum(0 if jg is None else len(jg["pairs"]) for jg in jgs) / len(jgs)
print(f"fit_pair_scales: {1000*dt:.0f} ms per frame, {npairs:.1f} pairs per frame")
