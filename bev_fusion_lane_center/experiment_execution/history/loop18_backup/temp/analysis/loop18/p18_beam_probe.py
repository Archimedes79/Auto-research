"""Loop 18 / WP4 test 2c (one-off): is the far-range recall gap (beyond 40 m) a beam / seed budget? Champion f1 at inference with a larger budget (M / K0), fork pruning 0.3. Usage (gcar): p18_beam_probe.py [--sizes 288:144,400:200,576:288]"""
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
import sparse_metrics as SM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="288:144,400:200,576:288")
ap.add_argument("--n", type=int, default=873)
a = ap.parse_args()
ks = list(range(a.n))
for sz in a.sizes.split(","):
    M, K0 = map(int, sz.split(":"))
    net, cfg = TM.load_builder(M, K0)
    ds = PD.BuilderData("val_clean", ks=ks, train=False, with_prev=cfg.temporal)
    raw = []
    t0 = time.time()
    _, gts, _ = PI.predict(net, ds, ks, taus=(0.7,), bs=8, workers=2, raw=raw)
    fw = time.time() - t0
    gs = [PI.assemble(r, None, 0.7, cfg, fast=2, fork_prune=0.3, **TM.ASM) for r in raw]
    m = SM.card_any(gs, gts)
    print(f"M {M} K0 {K0}: S {m['S']:.3f} Q {SM.quality(m):.3f} J {m['J']:.3f} F1@1 {m['F1@1.0']:.3f} R@1 {m['R@1.0']:.3f} found {m['lanes_found']:.3f} FL {m['false_len_frac']:.3f} v100 {m['vert100']:.1f} (forward+load {fw:.0f} s)", flush=True)
