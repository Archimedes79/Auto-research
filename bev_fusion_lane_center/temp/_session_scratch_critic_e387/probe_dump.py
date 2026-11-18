import os, sys, pickle
import numpy as np
R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(R, "evaluation"))
for run in ("e368_z_s0", "e368_r_s0"):
    for fn in ("token_train_every3.pkl", "token_val.pkl"):
        p = os.path.join(R, "runs", run, fn)
        d = pickle.load(open(p, "rb"))
        m, fr = d["meta"], d["frames"]
        f = fr[0]
        print(run, fn, "meta keys", sorted(m.keys()))
        print("  n frames", len(fr), "every", m.get("every"), "of_run", m.get("of_run"), "mode", m.get("mode"), "steps", m.get("steps"))
        print("  frame keys", sorted(f.keys()))
        print("  cand keys", sorted(f["cand"].keys()), "none_cls", f["none_cls"].shape, f["none_cls"].dtype, "rec keys", sorted(f["rec"].keys()))
