import os, pickle
import numpy as np
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
def P(run):
    fr = pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))
    return [1.0 / (1.0 + np.exp(-f["logit"])) * f["allowed"] for f in fr]
for S, tgt in ((0, 11378), (1, 11849)):
    ps = P(f"e320_a_s{S}")
    for t in (0.109, 0.110, 0.111, 0.112, 0.119, 0.120, 0.121):
        c = sum(int((p >= t).sum()) for p in ps)
        print(f"seed {S} E320 at {t:.3f}: {c} curves ({100 * (c - tgt) / tgt:+.2f} % from {tgt})")
