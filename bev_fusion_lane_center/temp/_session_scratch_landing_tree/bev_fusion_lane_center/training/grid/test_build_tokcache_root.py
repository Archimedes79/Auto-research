"""Test of `data/build_tokcache_root.py`: the builder runs with the given cache root and the module's root is restored afterwards (also after an error); no GPU, no real build (`lane_tokcache.build` is replaced by a recorder).

    wsl bash tools/env/_py14.sh training/grid/test_build_tokcache_root.py"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
import lane_tokcache as LC  # noqa: E402
import build_tokcache_root as BR  # noqa: E402


def main():
    seen = []
    orig, root0 = LC.build, LC.CACHE

    def rec(a):
        seen.append((LC.CACHE, a.run, a.split, a.K, a.f2, a.limit, a.workers))
        return "dir"
    LC.build = rec
    try:
        assert BR.build_into("runs/x_cache", "runs/r", "val", 384, f2=False, workers=1) == "dir"
        assert seen[0] == (os.path.join(BR.ROOT, "runs/x_cache"), "runs/r", "val", 384, False, 0, 1) and LC.CACHE == root0
        BR.build_into(os.path.abspath("/tmp/abs_root"), "runs/r", "train", 512)
        assert seen[1][0] == os.path.abspath("/tmp/abs_root") and LC.CACHE == root0
        LC.build = lambda a: 1 / 0
        try:
            BR.build_into("runs/x_cache", "runs/r", "val", 384)
            raise SystemExit("no error?")
        except ZeroDivisionError:
            assert LC.CACHE == root0, "the root must be restored after an error"
    finally:
        LC.build = orig
    print("test_build_tokcache_root ok")


if __name__ == "__main__":
    main()
