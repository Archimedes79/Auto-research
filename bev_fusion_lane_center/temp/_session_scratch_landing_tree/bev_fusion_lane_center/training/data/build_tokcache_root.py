"""Build a predicted-node cache (`lane_tokcache.py build`) into ANOTHER cache root, so that a second K never lands next to the K 512 caches (where a call without --K would find two caches and refuse: `lane_ptok_data.cache_dir`).

    wsl bash tools/env/_pyr.sh build_tokcache_root.py --root runs/e383_cache --run runs/e292_a_s0 --split val|train --K 384 [--no-f2] [--limit N] [--workers 2]

The same builder, the same fields and labels; only the root folder differs (`lane_tokcache.CACHE` is set before `build` runs, in this process). Read it with `--cache <root> --K 384` (jtoken_train / jtoken_fast_train; the run's args carry both, so
`jtoken_eval.py` finds it). Test: grid/test_build_tokcache_root.py
"""
import argparse
import os
import sys

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import lane_tokcache as LC  # noqa: E402


def build_into(root, run, split, K, f2=True, limit=0, workers=2):
    """`lane_tokcache.build` with the cache root `root` (relative to the problem folder, or absolute) -> the finished cache directory"""
    keep = LC.CACHE
    LC.CACHE = root if os.path.isabs(root) else os.path.join(ROOT, root)
    try:
        return LC.build(argparse.Namespace(run=run, split=split, K=K, f2=f2, limit=limit, workers=workers))
    finally:
        LC.CACHE = keep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True, choices=("val", "val_clean", "train"))
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--no-f2", dest="f2", action="store_false")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    print(build_into(a.root, a.run, a.split, a.K, a.f2, a.limit, a.workers))
