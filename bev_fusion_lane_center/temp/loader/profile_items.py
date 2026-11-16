"""Where does the time of one TRAIN item of the pair-stage loader go? (cProfile of `QData.__getitem__` over N consecutive items, workers 0, CPU only; scratch, lane lead's question 2026-11-14)
usage: wsl bash tools/env/_py14.sh temp/loader/profile_items.py [--n 24] [--no-ppts] [--no-scales] [--top 35]"""
import argparse
import cProfile
import io
import os
import pstats
import sys
import time

import torch

ROOT = os.environ.get("BFLC_ROOT") or os.getcwd()
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import lane_gt_data as LD  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--no-ppts", action="store_true")
    ap.add_argument("--no-scales", action="store_true")
    ap.add_argument("--top", type=int, default=35)
    a = ap.parse_args()
    torch.set_num_threads(1)
    LD.QData.DEFAULT_JUNCTION = True
    LD.QData.DEFAULT_PSCALE = not a.no_scales
    LD.QData.DEFAULT_PPTS = not a.no_ppts
    ds = LD.QData("train", map_mode="train", pixels=False)
    idx = list(range(0, len(ds), max(1, len(ds) // a.n)))[: a.n]
    ds[idx[0]]                                                                                           # warm-up (files, caches)
    t0 = time.time()
    pr = cProfile.Profile()
    pr.enable()
    for i in idx:
        ds[i]
    pr.disable()
    dt = time.time() - t0
    print(f"{len(idx)} items in {dt:.1f} s = {dt / len(idx):.3f} s per item (one process, profiled)")
    for sort in ("cumulative", "tottime"):
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats(sort).print_stats(a.top)
        print(f"--- sorted by {sort}")
        print("\n".join(s.getvalue().splitlines()[:a.top + 8]))
