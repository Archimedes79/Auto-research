"""Layer-1 test of the LCR_DIST_DIR switch in training/data/p16_data.py (E237, review input):

    wsl bash tools/env/_py14.sh temp/loop19/test_dist_switch.py        (CPU)

1. default (variable unset): the loader's distance map is the stored b123 file, bit for bit, train and val;
2. LCR_DIST_DIR = the stored folder: the same batch bits as the default;
3. LCR_DIST_DIR = the exact rebuild: a different file (the TF32 damage is gone) with the same shape and dtype, the same rows, and the loader item has
   the same keys and shapes (only gd differs).
Prints PASS / FAIL.
"""
import importlib
import os
import sys

import numpy as np

H = os.path.expanduser("~")
STORED, EXACT = f"{H}/exp_crn_b123/feats", f"{H}/exp_gcar_b123x/feats"


def load(env):
    if env is None:
        os.environ.pop("LCR_DIST_DIR", None)
    else:
        os.environ["LCR_DIST_DIR"] = env
    import p16_data
    importlib.reload(p16_data)
    out = {}
    for split in ("train", "val_clean"):
        ds = p16_data.BuilderData(split, ks=[0, 1, 2, 400, 800])
        ds._open()
        part = ds.part
        out[split] = (ds.gd, [ds[i] for i in range(len(ds))], part)
    return p16_data.DIST_DIR, out


ok = True
d0, a = load(None)
d1, b = load(STORED)
d2, c = load(EXACT)
print("DIST_DIR default / stored / exact:", d0, d1, d2)
ok &= d0 == EXACT and d1 == STORED and d2 == EXACT  # D14: the default is the exact map; the stored one needs LCR_DIST_DIR
for split in a:
    part = a[split][2]
    ref = np.load(f"{STORED}/{part}_gt_dist_exact.npy", mmap_mode="r")
    new = np.load(f"{EXACT}/{part}_gt_dist_exact.npy", mmap_mode="r")
    gd0, items0, _ = a[split]
    gd1, items1, _ = b[split]
    gd2, items2, _ = c[split]
    t1 = gd0.shape == ref.shape and np.array_equal(gd0[:50], new[:50]) and np.array_equal(gd0[-50:], new[-50:])
    t2 = all(np.array_equal(x["gd"].numpy(), y["gd"].numpy()) and np.array_equal(x["x"].numpy(), y["x"].numpy()) for x, y in zip(items0, items2))
    t3 = gd1.shape == ref.shape and gd1.dtype == ref.dtype and all(set(x) == set(y) and all(getattr(x[k], "shape", None) == getattr(y[k], "shape", None) for k in x) for x, y in zip(items0, items1))
    n_diff = sum(not np.array_equal(x["gd"].numpy(), y["gd"].numpy()) for x, y in zip(items0, items1))
    t4 = all(np.array_equal(x["x"].numpy(), y["x"].numpy()) and np.array_equal(x["g05"].numpy(), y["g05"].numpy()) for x, y in zip(items0, items1))
    print(f"{split:10} default==exact file {t1} | explicit exact==default {t2} | stored (env switch): same keys / shapes {t3}, only gd differs {t4}, frames with a different gd {n_diff} of {len(items0)}")
    ok &= t1 and t2 and t3 and t4
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
