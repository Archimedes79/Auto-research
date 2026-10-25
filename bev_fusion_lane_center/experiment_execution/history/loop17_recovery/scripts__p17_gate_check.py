"""Loop 17: the untrained gate must reproduce the training-free hold: compare the fused series dark_gate0 (p17_gate.py apply --init_only --out gate0) with dark_maxdecay_0.95 per channel group."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p17_eval as EV  # noqa: E402

D = EV.OUTD
a = np.load(f"{D}/dark_gate0.npy", mmap_mode="r")
b = np.load(f"{D}/dark_maxdecay_0.95.npy", mmap_mode="r")
meta = np.load(f"{D}/dark_gate0.meta.npz")
ph = meta["phase"]
for nm, sl in (("features", slice(0, 39)), ("heat", slice(39, 40)), ("dir", slice(40, 42))):
    d = np.array([np.abs(np.asarray(a[i, sl], np.float32) - np.asarray(b[i, sl], np.float32)).mean() for i in range(len(ph)) if ph[i] == 1])
    s = np.array([np.abs(np.asarray(b[i, sl], np.float32)).mean() for i in range(len(ph)) if ph[i] == 1])
    print(f"{nm:9s} failure frames: mean |gate0 - maxdecay| {d.mean():.5f} (mean |maxdecay| {s.mean():.4f})")
