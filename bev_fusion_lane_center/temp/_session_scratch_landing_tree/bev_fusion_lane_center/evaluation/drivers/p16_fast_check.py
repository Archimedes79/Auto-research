"""Does the fast assembly give the same graphs as the original one?  Time of both on the same node arrays.  Usage: p16_fast_check.py [--n 120]"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=120)
a = ap.parse_args()
net, cfg = TM.load_builder(288, 144)
ks = C.val_subset()[: a.n]
ds = PD.BuilderData("val_clean", ks=ks, train=False, with_prev=cfg.temporal)
raw = []
_, gts, _ = PI.predict(net, ds, ks, taus=(0.7,), bs=8, workers=2, raw=raw)
res = {}
for fast in (False, True, False, True):
    t0 = time.time()
    gs = [PI.assemble(r, None, 0.7, cfg, fast=fast, **TM.ASM) for r in raw]
    res[fast] = (gs, (time.time() - t0) / len(raw) * 1e3)
    print(f"fast={fast}: {res[fast][1]:.1f} ms per frame", flush=True)
same = 0
diff = []
for i, (g0, g1) in enumerate(zip(res[False][0], res[True][0])):
    e0 = sorted(map(tuple, g0["edges"]))
    e1 = sorted(map(tuple, g1["edges"]))
    p0 = [np.asarray(p) for p in g0["pts"]]
    p1 = [np.asarray(p) for p in g1["pts"]]
    ok = len(p0) == len(p1) and e0 == e1 and all(a_.shape == b_.shape and np.allclose(a_, b_, atol=1e-6) for a_, b_ in zip(p0, p1))
    same += ok
    if not ok:
        diff.append((i, len(p0), len(p1), sum(len(x) for x in p0), sum(len(x) for x in p1)))
print(f"identical graphs: {same}/{len(raw)}; different: {diff[:10]}")
m0, m1 = C.card(res[False][0], gts), C.card(res[True][0], gts)
print("metrics original:", {k: round(float(m0[k]), 4) for k in ("S", "F1@1.0", "J", "vert100")})
print("metrics fast    :", {k: round(float(m1[k]), 4) for k in ("S", "F1@1.0", "J", "vert100")})
