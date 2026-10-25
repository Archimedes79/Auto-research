"""Loop 18 / R2.3: do two assembly paths give identical graphs, and how long does each take?  (default: fast=True = FastUnion of loop 16b against fast=2 = NumbaUnion)
The raw node outputs of the first --n validation frames of the champion (M 288 / K0 144, tau 0.7, champion assembly settings) are computed once and cached in ~/exp_gcar_b123/p18/raw_val<n>.pkl.
Timing is meaningful only on an idle machine (the assembly is CPU; trainings on the same machine slow it down by a factor of 2 - 3).
Usage (gcar): p18_asm_check.py [--n 873] [--a 1 --b 2] [--reps 3]"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=873)
ap.add_argument("--a", type=int, default=1, help="reference path: 0 = original Union, 1 = FastUnion")
ap.add_argument("--b", type=int, default=2, help="candidate path: 2 = NumbaUnion")
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
cache = os.path.expanduser(f"~/exp_gcar_b123/p18/raw_val{a.n}.pkl")
os.makedirs(os.path.dirname(cache), exist_ok=True)
net, cfg = TM.load_builder(288, 144)
if os.path.exists(cache):
    raw, gts = pickle.load(open(cache, "rb"))
else:
    ks = list(range(a.n)) if a.n >= 873 else C.val_subset()[: a.n]
    ds = PD.BuilderData("val_clean", ks=ks, train=False, with_prev=cfg.temporal)
    raw = []
    _, gts, _ = PI.predict(net, ds, ks, taus=(0.7,), bs=8, workers=2, raw=raw)
    pickle.dump((raw, gts), open(cache, "wb"))
res = {}
for rep in range(a.reps):
    for path in (a.a, a.b):
        fast = {0: False, 1: True}.get(path, path)
        t0 = time.time()
        gs = [PI.assemble(r, None, 0.7, cfg, fast=fast, **TM.ASM) for r in raw]
        ms = (time.time() - t0) / len(raw) * 1e3
        res.setdefault(path, []).append(ms)
        res[("g", path)] = gs
    print(f"rep {rep}: " + "  ".join(f"path {p}: {res[p][-1]:.2f} ms" for p in (a.a, a.b)), flush=True)
same, diff = 0, []
for i, (g0, g1) in enumerate(zip(res[("g", a.a)], res[("g", a.b)])):
    e0, e1 = sorted(map(tuple, g0["edges"])), sorted(map(tuple, g1["edges"]))
    p0, p1 = [np.asarray(p) for p in g0["pts"]], [np.asarray(p) for p in g1["pts"]]
    ok = len(p0) == len(p1) and e0 == e1 and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(p0, p1))
    same += ok
    if not ok:
        diff.append((i, len(p0), len(p1), sum(len(x) for x in p0), sum(len(x) for x in p1)))
print(f"{len(raw)} frames: bit-identical graphs {same}; different {len(diff)} {diff[:8]}")
m0, m1 = C.card(res[("g", a.a)], gts), C.card(res[("g", a.b)], gts)
print("metrics a:", {k: round(float(m0[k]), 5) for k in ("S", "F1@1.0", "J", "vert100")})
print("metrics b:", {k: round(float(m1[k]), 5) for k in ("S", "F1@1.0", "J", "vert100")})
print("median ms per frame: a %.2f  b %.2f  (min over reps: a %.2f b %.2f)" % (np.median(res[a.a]), np.median(res[a.b]), min(res[a.a]), min(res[a.b])))
if diff:
    sys.exit(1)  # the self-test of tools/maintenance/run_all_tests.sh fails when a graph differs
