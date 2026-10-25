"""Post-check of the b456 features (D7 step 4): every row written, finite, value range like b123.

    wsl bash tools/env/_py14.sh temp/loop19/check_b456_features.py

Unwritten rows of a memmap are all zeros, so a row counts as written when its strided sample is not all zero. Value range: standard deviation of a sample of
rows per group against the same statistic of b123 (gate: within 20 %). Prints one line per part and group and PASS / FAIL.
"""
import json
import os

import numpy as np

H = os.path.expanduser("~")
NEW, OLD = f"{H}/exp_gcar_b456/feats", f"{H}/exp_gcar_b123/feats"
ok = True
for part in ("train", "val"):
    n = len(json.load(open(f"{NEW}/{part}_tokens.json")))
    done = sorted(f for f in os.listdir(NEW) if f.startswith(f"{part}_") and f.endswith(".done"))
    print(f"{part}: {n} tokens, done flags {done}")
    for g in ("dec32", "logit7", "fused32"):
        a = np.load(f"{NEW}/{part}_{g}.npy", mmap_mode="r")
        b = np.load(f"{OLD}/{part}_{g}.npy", mmap_mode="r")
        empty = [i for i in range(len(a)) if not np.any(np.asarray(a[i, :, ::25, ::25]))]
        idx = np.linspace(0, len(a) - 1, 60).astype(int)
        s = np.asarray(a[idx], np.float32)
        r = np.asarray(b[np.linspace(0, len(b) - 1, 60).astype(int)], np.float32)
        finite = bool(np.isfinite(s).all())
        ratio = float(s.std() / r.std())
        good = len(a) == n and not empty and finite and 0.8 <= ratio <= 1.2
        ok &= good
        print(f"  {g:7} rows {len(a)} | unwritten {len(empty)} {empty[:5]} | finite {finite} | std {s.std():.3f} vs b123 {r.std():.3f} (ratio {ratio:.3f}) | {'ok' if good else 'FAIL'}")
print("PASS" if ok else "FAIL")
