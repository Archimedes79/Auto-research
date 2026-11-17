"""throw-away check: the read-outs that import jtoken_eval still import, and the CLI parses the old and the new commands"""
import os
import sys

sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
import jmap_eval  # noqa: E402,F401
import jpast_eval  # noqa: E402,F401
import jtoken_eval as E  # noqa: E402

for argv in (["score", "--run", "r", "--seed", "0"], ["score", "--run", "r", "--seed", "0", "--link-thr", "0.3", "--link-thr-long", "0.2", "--keep-thr", "0.6", "--no-none-gate", "--ref-run", "q", "--ref-run-cell", "0.3,0.2,0.6,off"],
             ["score", "--run", "r", "--seed", "0", "--argmax", "--keep-thr", "0.6"], ["anatomy", "--run", "r", "--seed", "0"], ["sweep", "--run", "r", "--seed", "0", "--split", "train", "--grid", "perclass"]):
    E.parse_args(argv)
print("imports and CLI ok:", E.__file__)
