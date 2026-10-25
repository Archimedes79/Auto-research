"""Loop 18 / R2.3 (one-off): where do the ~13 ms of the assembly go?  Builds the raw node outputs of 120 VAL frames once (cached in ~/exp_gcar_b123/p18/raw_val120.pkl), then cProfile of PI.assemble (fast path, champion settings).
Usage (gcar): p18_asm_profile.py [--n 120] [--rebuild]"""
import argparse
import cProfile
import io
import os
import pickle
import pstats
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
ap.add_argument("--rebuild", action="store_true")
ap.add_argument("--fast", type=int, default=2)
a = ap.parse_args()
cache = os.path.expanduser("~/exp_gcar_b123/p18/raw_val120.pkl")
os.makedirs(os.path.dirname(cache), exist_ok=True)
net, cfg = TM.load_builder(288, 144)
if os.path.exists(cache) and not a.rebuild:
    raw, gts = pickle.load(open(cache, "rb"))
else:
    ks = C.val_subset()[: a.n]
    ds = PD.BuilderData("val_clean", ks=ks, train=False, with_prev=cfg.temporal)
    raw = []
    _, gts, _ = PI.predict(net, ds, ks, taus=(0.7,), bs=8, workers=2, raw=raw)
    pickle.dump((raw, gts), open(cache, "wb"))
for _ in range(2):
    t0 = time.time()
    gs = [PI.assemble(r, None, 0.7, cfg, fast=a.fast, **TM.ASM) for r in raw]
    print(f"assemble fast: {(time.time() - t0) / len(raw) * 1e3:.1f} ms per frame", flush=True)
pr = cProfile.Profile()
pr.enable()
gs = [PI.assemble(r, None, 0.7, cfg, fast=a.fast, **TM.ASM) for r in raw]
pr.disable()
for key in ("tottime", "cumulative"):
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats(key).print_stats(22)
    print("\n".join(s.getvalue().splitlines()[:46]))
