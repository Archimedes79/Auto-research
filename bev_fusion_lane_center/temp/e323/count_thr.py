"""E323 scratch: delivered connector curves against the pair threshold, from the stored dumps (no scoring), and the threshold at which a run delivers as many curves as E312 does at p >= 0.1."""
import os, pickle, sys
import numpy as np
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
def P(run):
    fr = pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))
    return [1.0 / (1.0 + np.exp(-f["logit"])) * f["allowed"] for f in fr]
ts = np.round(np.arange(0.03, 0.31, 0.001), 3)
for S in (0, 1):
    ref = sum(int((p >= 0.1).sum()) for p in P(f"e312_p_s{S}"))
    print(f"seed {S}: E312 curves at p >= 0.1: {ref}")
    for run in (f"e312_p_s{S}", f"e316_p_s{S}", f"e315_p_s{S}", f"e320_a_s{S}"):
        ps = P(run)
        cnt = np.array([sum(int((p >= t).sum()) for p in ps) for t in ts])
        j = int(np.argmin(np.abs(cnt - ref)))
        print(f"  {run:12s} curves at 0.05 {cnt[int(np.argmin(np.abs(ts - 0.05)))]:6d}  0.1 {cnt[int(np.argmin(np.abs(ts - 0.1)))]:6d}  0.15 {cnt[int(np.argmin(np.abs(ts - 0.15)))]:6d}  0.2 {cnt[int(np.argmin(np.abs(ts - 0.2)))]:6d};  matching threshold {ts[j]:.2f} ({cnt[j]} curves, {100 * (cnt[j] - ref) / ref:+.1f} %)")
