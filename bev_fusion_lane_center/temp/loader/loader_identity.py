"""Are the TRAIN batches of the pair-stage trainers the same for the same seed whatever the number of data workers? (lane lead's question 2026-11-14 about the load of the training steps; CPU only)
The loader and its flags are those of `jends_train.py` (train split, bs 8, shuffle, map_mode "train", pixels False, DEFAULT_JUNCTION, DEFAULT_PSCALE, DEFAULT_PPTS) after `trainer.seed_all(seed)`.
Runs: workers 3 twice (the repeat shows what is NOT reproducible even with equal workers), workers 6, workers 0 (main process); the first N batches; every field compared with torch.equal.
usage: wsl bash tools/env/_py14.sh temp/loader/loader_identity.py [--nb 20] [--seed 0] [--no-ppts] [--no-scales]
(scratch: never imported; the result goes to temp/loader/identity_<seed>.txt and into the E342 card)"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.getcwd()
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import lane_gt_data as LD  # noqa: E402
import trainer as T  # noqa: E402


def run(workers, seed, nb, ppts, scales):
    T.seed_all(seed)
    LD.QData.DEFAULT_JUNCTION = True
    LD.QData.DEFAULT_PSCALE = scales
    LD.QData.DEFAULT_PPTS = ppts
    ds, dl = LD.loader("train", bs=8, workers=workers, shuffle=True, map_mode="train", pixels=False)
    t0 = time.time()
    it = iter(dl)
    out = []
    for _ in range(nb):
        out.append(next(it))
    dt = time.time() - t0
    del it, dl
    return out, dt


def same(a, b):
    if torch.is_tensor(a):
        if not (torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype):
            return False
        if a.is_floating_point():                                                                        # NaN padding (pscale, ppts, lk_scale) is equal to NaN
            return torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(torch.nan_to_num(a, nan=0.0), torch.nan_to_num(b, nan=0.0))
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    if isinstance(a, (list, tuple)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


def compare(A, B):
    """-> {field: number of batches (of len(A)) in which the field differs}"""
    bad = {}
    for a, b in zip(A, B):
        for k in a:
            if k not in b or not same(a[k], b[k]):
                bad[k] = bad.get(k, 0) + 1
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ppts", action="store_true")
    ap.add_argument("--no-scales", action="store_true")
    ap.add_argument("--runs", default="w3a,w3b,w4,w6,w0", help="comma separated: w3a, w3b (repeat of 3), w4, w6, w0")
    a = ap.parse_args()
    torch.set_num_threads(1)
    names = {"w3a": 3, "w3b": 3, "w4": 4, "w6": 6, "w0": 0}
    res = {}
    for name in a.runs.split(","):
        res[name] = run(names[name], a.seed, a.nb, not a.no_ppts, not a.no_scales)
        print(f"{name}: {a.nb} batches in {res[name][1]:.1f} s ({res[name][1] / a.nb:.2f} s per batch), fields {sorted(res[name][0][0])}", flush=True)
    for y in res:
        if y == "w3a" or "w3a" not in res:
            continue
        bad = compare(res["w3a"][0], res[y][0])
        print(f"w3a against {y}: fields that differ (number of batches of {a.nb}): {bad if bad else 'NONE: every field of every batch is bit-identical'}", flush=True)
    if "w3b" in res:
        print("not reproducible even with equal workers (repeat of workers 3):", sorted(compare(res["w3a"][0], res["w3b"][0])) or "none")
