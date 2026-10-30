"""DEPRECATED forwarding stub of the former evaluate_run.py (replaced by the evaluation suite on 2026-10-30).

The command line of the old script is translated to `evaluation/bev_suite.py` (same outputs: eval/metrics<tag>.json, table<tag>.txt,
latency.json, images; the numbers equal the old ones on the same frames). It exists only because `temp/study/lane3.sh` is running and
calls the old command line; DELETE THIS FILE when that lane has ended (the pre-suite code is in `_to_delete/evaluation_pre_suite/`).
New work: `python evaluation/bev_suite.py RUN eval --n 1000` (see evaluation/README.md); the helper functions this module used to
export for diagnostic scripts are in `evaluation/bev_probe.py`.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--prior", type=int, default=0)
    ap.add_argument("--images", type=int, default=0)
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--time-only", action="store_true")
    ap.add_argument("--batch", type=int, default=4, help="ignored: the suite runs batch 1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--on-train", action="store_true")
    ap.add_argument("--control", default="")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--det-a", default="det")
    ap.add_argument("--det-b", default="det")
    ap.add_argument("--dt", type=float, default=0.0)
    a = ap.parse_args(argv)
    print("evaluate_run.py is deprecated: forwarding to evaluation/bev_suite.py", file=sys.stderr)
    import bev_suite
    if a.compare:
        return bev_suite.main(["compare", a.compare[0], a.compare[1], "--boot", str(a.boot), "--det-a", a.det_a, "--det-b", a.det_b,
                               "--dt", str(a.dt)])
    glob = [a.run, "--workers", str(a.workers)] + (["--split", "train"] if a.on_train else []) + \
           (["--control", a.control] if a.control else []) + (["--ckpt-dir", a.ckpt_dir] if a.ckpt_dir else [])
    if a.time_only:
        return bev_suite.main(glob + ["time"])
    ev = ["eval", "--n", str(a.n), "--prior", str(a.prior), "--images", str(a.images)] + (["--time"] if a.time else []) + \
         (["--warm"] if a.warm else [])
    return bev_suite.main(glob + ev)


if __name__ == "__main__":
    raise SystemExit(main())
