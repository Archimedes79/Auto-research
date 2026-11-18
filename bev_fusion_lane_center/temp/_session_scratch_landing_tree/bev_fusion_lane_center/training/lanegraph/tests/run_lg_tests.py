"""The tests of training/lanegraph; exit code 1 on any failure.

    bash tools/env/_py14.sh training/lanegraph/tests/run_lg_tests.py [--quick]                                # default: seeds 0 1, 24 VAL frames each, every 36th row
    bash tools/env/_py14.sh training/lanegraph/tests/run_lg_tests.py --seeds 0 1 --frames 96 48 --stride 9     # the README's numbers
--quick: the read-out unit tests only (no data, no project code). Every comparison is exact (no tolerance).
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "scripts")]
import test_lg_readout as TR  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--frames", type=int, nargs="+", default=[24], help="frames per seed (one number for all seeds, or one per seed)")
    ap.add_argument("--stride", type=int, default=36, help="every n-th VAL row")
    ap.add_argument("--fuzz", type=int, default=300, help="synthetic frames of the read-out fuzz")
    a = ap.parse_args()
    fails = 0
    for t in TR.TESTS:
        t0 = time.perf_counter()
        try:
            t()
            print(f"ok    {t.__name__} ({time.perf_counter() - t0:.1f} s)")
        except AssertionError as ex:
            fails += 1
            print(f"FAIL  {t.__name__}: {ex}")
    if not a.quick:
        import test_lg_equivalence as TE
        from lg_features import split_rows
        n_fuzz = TE.fuzz(a.fuzz)
        fails += n_fuzz > 0
        print(f"{'ok  ' if not n_fuzz else 'FAIL'}  read-out fuzz: {n_fuzz} of {2 * a.fuzz} synthetic graphs differ from the original read-out")
        got = TE.golden()
        ok = TE.golden_ok(got)
        fails += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  golden digests of the snapped graphs (seed 0, VAL {list(TE.GOLDEN)}): {got}")
        frames = a.frames * len(a.seeds) if len(a.frames) == 1 else a.frames
        for seed, n in zip(a.seeds, frames):
            rows = split_rows("val")[:: a.stride][:n]
            t0 = time.perf_counter()
            bad, count = TE.stages(seed, rows)
            diff = {k: v for k, v in bad.items() if v}
            fails += bool(diff)
            print(f"{'ok  ' if not diff else 'FAIL'}  seed {seed}, {len(rows)} VAL frames (every {a.stride}th row): {len(bad)} quantities identical to the original"
                  f"{'' if not diff else ' EXCEPT ' + str(diff)}; pieces compared {count} ({time.perf_counter() - t0:.0f} s)")
    print("all tests passed" if not fails else f"{fails} FAILED")
    return int(fails > 0)


if __name__ == "__main__":
    sys.exit(main())
