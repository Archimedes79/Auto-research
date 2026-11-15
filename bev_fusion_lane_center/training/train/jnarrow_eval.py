"""The evaluation path of a narrowed junction checkpoint (E353): the stored tools of `evaluation/` with a surgery-aware model loader that is installed in THIS process only (no digested file is edited).

    wsl bash tools/env/_py14.sh jnarrow_eval.py dump  --run e353_n1_s0 [--thr 0.5] [--max N]               # `jends_eval.dump`: runs/<run>/ends_val.pkl + ends_head.json, the format of the stored runs
    wsl bash tools/env/_py14.sh jnarrow_eval.py score --run e353_n1_s0 --seed 0 [flags of jends_eval.py score]  # `jends_eval.py score` as it is (it reads the state, it never builds the model; the pyramid line reads 'False' for a changed pyramid)
    wsl bash tools/env/_py14.sh jnarrow_eval.py time  run --runs e324_s_s0,e353_n1_s0 [flags of jtiming.py]     # `jtiming.py run | graph | kernels` with the narrowed run next to the stored one, one process
`jnarrow_train.py dump | score | time ...` is the same call. A checkpoint without `args["narrow"]` (the stored runs, the cp control) loads through the stored loader unchanged, so the dump of an unnarrowed run equals `jends_eval.py dump` frame for frame.
"""
import argparse
import os
import runpy
import sys
from types import SimpleNamespace

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jends_eval as JEV  # noqa: E402
import jnarrow as JN  # noqa: E402


def install():
    """`jends_eval.load_model` (called by `dump` and by `jtiming.py`) becomes the surgery-aware loader, in this process"""
    JEV.load_model = JN.load_model


def dump(argv):
    ap = argparse.ArgumentParser(prog="dump")
    ap.add_argument("--run", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--max", type=int, default=0, help="only the first N frames (a test of the pipeline; the score needs all 873)")
    a = ap.parse_args(argv)
    install()
    JEV.dump(SimpleNamespace(run=a.run, thr=a.thr, max=a.max))


def score(argv):
    """the stored `jends_eval.py score` with its own argument parser (it builds no model: the narrowed state is read as it is)"""
    sys.argv = ["jends_eval.py", "score", *argv]
    runpy.run_path(JEV.__file__, run_name="__main__")


def time_cmd(argv):
    import jtiming as JT
    install()
    sys.argv = ["jtiming.py", *argv]
    JT.main()


COMMANDS = {"dump": dump, "score": score, "time": time_cmd}


def main(argv):
    COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    if sys.argv[1:2] and sys.argv[1] in COMMANDS:
        main(sys.argv[1:])
    else:
        sys.exit(__doc__)
