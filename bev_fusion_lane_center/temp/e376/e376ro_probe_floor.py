"""throw-away probe: does `check_argmax` at keep 0.6 (the floor rule of the critic's item 1) pass on the stored E373 VAL dumps? counts the kept sources whose class has p(none) < 0.05 and no dump candidate"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e376ro_jtoken_eval as E  # noqa: E402

for run in ("e373_cb_s0", "e373_cb_s1"):
    a = E.parse_args(["score", "--run", run, "--seed", run[-1], "--argmax", "--keep-thr", "0.6"])
    r = E.read_dump(a, E.RUNS, 0)
    lost = 0
    for f, v in zip(r["fr"], r["valid"]):
        c, keep = f["cand"], E.decode_argmax_variant(f, v, 0.6)["keep"]
        has = np.zeros((2, len(keep)), bool)
        has[c["long"].astype(int), c["src"].astype(int)] = True
        lost += int((keep[None] & ~has & (f["none_cls"] < np.float32(E.KEEP_P))).sum())
    print(run, "kept sources at 0.6 with p(none) < 0.05 and no dump candidate:", lost)
    E.check_argmax(r["fr"], r["valid"], 0.6)
    print(run, "check_argmax at 0.6: pass")
